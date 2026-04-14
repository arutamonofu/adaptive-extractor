"""
Power Analysis for LLM Experiments.
Combines robust statistical estimation (Mixed Effects, T-distribution)
with production data loading logic.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from aee.domain.evaluation.matcher import ExperimentMatcher

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VarianceComponents:
    sigma_data: float
    sigma_seed: float
    sigma_d: float
    ratio: float


@dataclass
class PowerAnalysisResult:
    variance: VarianceComponents
    recommended_R: int
    recommended_N: int
    test_choice: str
    shapiro_p: float
    se: float
    se_threshold: float


@dataclass
class SimpleExperiment:
    data: dict

    def __getattr__(self, name: str):
        return self.data.get(name)

    def get(self, key: str, default=None):
        return self.data.get(key, default)


# =============================================================================
# Data Loading & F1 Logic (from Version 2)
# =============================================================================


def load_article_json(filepath: Path) -> dict:
    with open(filepath, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "experiments" in data and data["experiments"]:
            return data["experiments"][0]
        if "extraction" in data:
            ext = data["extraction"]
            if isinstance(ext, dict) and "experiments" in ext and ext["experiments"]:
                return ext["experiments"][0]
            return ext if isinstance(ext, dict) else {}
        if "output" in data:
            return data["output"]
    return data if isinstance(data, dict) else {}


def compute_f1_for_extraction(
    article_file: Path,
    gt_lookup: dict,
    matcher: ExperimentMatcher,
    fields: List[str],
    task_name: str = "nanozymes",
) -> Tuple[str, float]:
    """Required by notebook Cell 3."""
    article_id = article_file.stem
    if article_id not in gt_lookup:
        return article_id, np.nan

    extraction = load_article_json(article_file)
    gt_record = gt_lookup[article_id]

    # Сбор атрибутов GT
    gt_attributes = {
        f: (str(gt_record[f]) if gt_record.get(f) is not None else None) for f in fields
    }

    f1 = matcher.get_optimization_score(
        preds=[SimpleExperiment(data=extraction)],
        gts=[SimpleExperiment(data=gt_attributes)],
        task_name=task_name,
    )
    return article_id, f1


def scan_pilot_directory(pilot_dir: Path) -> Dict[int, Dict[int, List[Path]]]:
    result: Dict[int, Dict[int, List[Path]]] = {}
    for sys_dir in sorted(pilot_dir.glob("system_*")):
        sys_id = int(sys_dir.name.replace("system_", ""))
        result[sys_id] = {}
        for seed_dir in sorted(sys_dir.glob("seed_*")):
            seed_id = int(seed_dir.name.replace("seed_", ""))
            result[sys_id][seed_id] = sorted(seed_dir.glob("*.json"))
    return result


# =============================================================================
# Advanced Math (from Version 1)
# =============================================================================


def estimate_variance_mixed(df: pd.DataFrame) -> Tuple[float, float]:
    """Оценка компонентов дисперсии через смешанную модель."""
    try:
        model = smf.mixedlm("f1 ~ config", df, groups=df["article_id"])
        fit = model.fit()
        # Дисперсия между статьями (random intercept) и остаточная (seeds)
        sigma_data = np.sqrt(max(fit.cov_re.iloc[0, 0], 1e-10))
        sigma_seed = np.sqrt(max(fit.scale, 1e-10))
        return sigma_data, sigma_seed
    except Exception:
        # Fallback если модель не сошлась
        sigma_data = df.groupby("article_id")["f1"].mean().std()
        sigma_seed = df.groupby(["article_id", "config"])["f1"].std().mean()
        return sigma_data or 0.1, sigma_seed or 0.05


def bootstrap_sigma_d(d: np.ndarray, n_boot: int = 1000) -> float:
    if len(d) < 2:
        return 0.0
    samples = [
        np.std(np.random.choice(d, size=len(d), replace=True), ddof=1)
        for _ in range(n_boot)
    ]
    return float(np.mean(samples))


def calculate_required_n(
    sigma_d: float, delta: float, alpha: float, power: float
) -> int:
    """Итеративный расчет N через t-распределение (точнее чем Z-test)."""
    n = 10.0
    for _ in range(20):
        df_deg = max(1.0, n - 1.0)
        t_alpha = stats.t.ppf(1 - alpha / 2, df_deg)
        t_beta = stats.t.ppf(power, df_deg)
        n_new = ((t_alpha + t_beta) * sigma_d / delta) ** 2
        if abs(n_new - n) < 0.5:
            break
        n = n_new
    return max(3, int(np.ceil(n)))


# =============================================================================
# Main Analysis API
# =============================================================================


def perform_power_analysis(
    f1_df: pd.DataFrame,
    delta: float = 0.03,
    alpha: float = 0.05,
    power: float = 0.8,
    seed_variance_threshold: float = 0.1,
) -> PowerAnalysisResult:

    # 1. Валидация
    n_articles = f1_df["article_id"].nunique()
    if n_articles < 3:
        raise ValueError(f"Too few articles ({n_articles}). Need at least 3 for pilot.")

    # 2. Оценка дисперсий
    sigma_data, sigma_seed = estimate_variance_mixed(f1_df)

    # 3. Расчет sigma_d (парные разности)
    pivot = f1_df.groupby(["article_id", "config"])["f1"].mean().unstack().dropna()
    if pivot.shape[1] < 2:
        raise ValueError("Need 2 different configs in data")

    d = (pivot.iloc[:, 0] - pivot.iloc[:, 1]).values
    sigma_d = bootstrap_sigma_d(d)

    ratio = sigma_seed / sigma_data if sigma_data > 0 else float("inf")

    # 4. Рекомендации
    rec_N = calculate_required_n(sigma_d, delta, alpha, power)
    # R = (sigma_seed^2 / sigma_data^2) * (1-thr)/thr
    r_val = (ratio**2) * (1 - seed_variance_threshold) / seed_variance_threshold
    rec_R = max(3, int(np.ceil(r_val))) if not math.isinf(ratio) else 5

    # 5. Тесты
    _, p_norm = stats.shapiro(d) if len(d) >= 3 else (0, 1.0)
    test_choice = "paired t-test" if p_norm > 0.05 else "Wilcoxon"

    # 6. Стандартная ошибка (SE)
    se = sigma_d / np.sqrt(rec_N)
    t_alpha = stats.t.ppf(1 - alpha / 2, rec_N - 1)
    t_beta = stats.t.ppf(power, rec_N - 1)
    se_threshold = delta / (t_alpha + t_beta)

    return PowerAnalysisResult(
        variance=VarianceComponents(sigma_data, sigma_seed, sigma_d, ratio),
        recommended_R=rec_R,
        recommended_N=rec_N,
        test_choice=test_choice,
        shapiro_p=p_norm,
        se=se,
        se_threshold=se_threshold,
    )


def compute_power_curve(
    n_range: np.ndarray, sigma_d: float, delta: float, alpha: float
) -> np.ndarray:
    powers = []
    for n in n_range:
        if n < 2:
            powers.append(0)
            continue
        # Используем t-распределение для кривой (согласованность с N)
        df_deg = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df_deg)
        # Нецентральное t-распределение
        nc_param = delta / (sigma_d / np.sqrt(n))
        power = 1 - stats.t.cdf(t_crit, df_deg, loc=nc_param)
        powers.append(power)
    return np.array(powers)
