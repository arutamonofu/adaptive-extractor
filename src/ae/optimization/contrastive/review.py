import json
import logging
from typing import List, Optional
from pathlib import Path
import click

from ae.optimization.contrastive.models import (
    AnalysisResult,
    HumanDecision,
    ReviewSession,
    VerifiedRule,
    Discrepancy,
)

logger = logging.getLogger(__name__)


class HumanReviewCLI:
    """Интерактивный CLI-помощник для разрешения неопределенностей разметки."""

    def __init__(
        self,
        analysis_result: AnalysisResult,
        session_path: Optional[Path] = None,
    ):
        self.analysis_result = analysis_result
        self.session_path = session_path or Path(f"data/analysis/{analysis_result.task_name}_review.json")
        self.decisions: List[HumanDecision] = []
        self.resolved_rules: List[VerifiedRule] = []
        self.skipped_count = 0

    def run(self, auto_skip: bool = False, auto_accept_majority: bool = False) -> ReviewSession:
        """Запускает сессию интерактивного опроса пользователя."""
        # 1. Приветствие и Сводка
        num_rules = len(self.analysis_result.verified_rules)
        num_discrepancies = len(self.analysis_result.discrepancies)
        
        click.echo("=" * 70)
        click.echo(f" Contrastive Analysis Review Session for Task: {self.analysis_result.task_name}")
        click.echo(f" Found: {num_rules} verified rules, {num_discrepancies} discrepancies to resolve.")
        click.echo("=" * 70)

        if num_discrepancies == 0:
            click.echo("No discrepancies found. Session completed automatically.")
            return ReviewSession(
                decisions=[],
                resolved_rules=[],
                skipped_count=0
            )

        # Попытка возобновления сессии при сбое
        if self.session_path.exists():
            try:
                with open(self.session_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                loaded_session = ReviewSession.model_validate(session_data)
                self.decisions = loaded_session.decisions
                self.resolved_rules = loaded_session.resolved_rules
                self.skipped_count = loaded_session.skipped_count
                click.echo(f"Loaded existing session from {self.session_path}. Resuming...")
            except Exception as e:
                click.echo(f"Warning: Failed to load previous session: {e}. Starting fresh.")

        already_resolved_ids = {d.discrepancy_id for d in self.decisions}

        for idx, disc in enumerate(self.analysis_result.discrepancies):
            if disc.discrepancy_id in already_resolved_ids:
                continue

            click.echo(f"\n======================================================================")
            click.echo(f"  Противоречие {idx + 1} из {num_discrepancies}: Уровень {disc.level}" + (f", Поле '{disc.field_name}'" if disc.field_name else ""))
            click.echo(f"======================================================================")
            click.echo(f"  Консенсус: {int(disc.consensus_ratio * 100)}%")
            click.echo(f"  Проблема: {disc.problem_description}")
            click.echo(f"  [A] {disc.variant_a}")
            click.echo(f"  [B] {disc.variant_b}")
            click.echo(f"  Примеры документов с конфликтами: {', '.join(disc.example_documents)}")
            click.echo(f"----------------------------------------------------------------------")

            decision = None

            if auto_skip:
                click.echo("  [Auto-skip active] Skipping discrepancy...")
                decision = HumanDecision(
                    discrepancy_id=disc.discrepancy_id,
                    action="skip"
                )
            elif auto_accept_majority:
                # auto-accept-majority: Variant A is assumed to have consensus_ratio.
                # If consensus_ratio >= 0.5, accept A, otherwise accept B
                action = "accept_a" if disc.consensus_ratio >= 0.5 else "accept_b"
                click.echo(f"  [Auto-accept-majority active] Selecting {action} (consensus: {int(disc.consensus_ratio * 100)}%)...")
                decision = HumanDecision(
                    discrepancy_id=disc.discrepancy_id,
                    action=action
                )
            else:
                choice = click.prompt(
                    "  Выберите действие [A / B / C (Свой вариант) / S (Пропустить)]",
                    type=click.Choice(["A", "B", "C", "S", "a", "b", "c", "s"], case_sensitive=False)
                ).upper()

                if choice == "A":
                    decision = HumanDecision(
                        discrepancy_id=disc.discrepancy_id,
                        action="accept_a"
                    )
                elif choice == "B":
                    decision = HumanDecision(
                        discrepancy_id=disc.discrepancy_id,
                        action="accept_b"
                    )
                elif choice == "C":
                    custom_text = click.prompt("  Введите ваше правило (на английском)")
                    decision = HumanDecision(
                        discrepancy_id=disc.discrepancy_id,
                        action="custom_rule",
                        custom_rule_text=custom_text
                    )
                else:
                    decision = HumanDecision(
                        discrepancy_id=disc.discrepancy_id,
                        action="skip"
                    )

            # Добавляем решение и создаем правило
            self.decisions.append(decision)
            
            if decision.action == "accept_a":
                self.resolved_rules.append(VerifiedRule(
                    rule_id=f"rule_resolved_{disc.discrepancy_id}",
                    level=disc.level,
                    field_name=disc.field_name,
                    rule_text=disc.variant_a,
                    evidence_count=max(1, int(disc.consensus_ratio * len(disc.example_documents))),
                    evidence_examples=disc.example_documents[:2]
                ))
            elif decision.action == "accept_b":
                self.resolved_rules.append(VerifiedRule(
                    rule_id=f"rule_resolved_{disc.discrepancy_id}",
                    level=disc.level,
                    field_name=disc.field_name,
                    rule_text=disc.variant_b,
                    evidence_count=max(1, int((1.0 - disc.consensus_ratio) * len(disc.example_documents))),
                    evidence_examples=disc.example_documents[:2]
                ))
            elif decision.action == "custom_rule" and decision.custom_rule_text:
                self.resolved_rules.append(VerifiedRule(
                    rule_id=f"rule_resolved_{disc.discrepancy_id}",
                    level=disc.level,
                    field_name=disc.field_name,
                    rule_text=decision.custom_rule_text,
                    evidence_count=1,
                    evidence_examples=disc.example_documents[:1]
                ))
            else:
                self.skipped_count += 1

            # Сохраняем промежуточный результат после каждого шага
            self.save_session()

        click.echo("\nSession completed! Decisions saved.")
        return ReviewSession(
            decisions=self.decisions,
            resolved_rules=self.resolved_rules,
            skipped_count=self.skipped_count
        )

    def save_session(self) -> None:
        """Сохраняет текущую сессию разбора в JSON файл."""
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        session = ReviewSession(
            decisions=self.decisions,
            resolved_rules=self.resolved_rules,
            skipped_count=self.skipped_count
        )
        with open(self.session_path, "w", encoding="utf-8") as f:
            f.write(session.model_dump_json(indent=2))


def merge_review_into_result(analysis_result: AnalysisResult, session: ReviewSession) -> AnalysisResult:
    """Объединяет правила, полученные в результате разбора человеком, с исходным набором."""
    # Чтобы избежать дублирования правил, проверим по rule_id
    existing_rule_ids = {rule.rule_id for rule in analysis_result.verified_rules}
    for rule in session.resolved_rules:
        if rule.rule_id not in existing_rule_ids:
            analysis_result.verified_rules.append(rule)
            existing_rule_ids.add(rule.rule_id)
    return analysis_result
