# scripts/predict_fewshot.py

import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm
import dspy
from dspy.teleprompt import LabeledFewShot

from aee.core.logging import setup_logging
from aee.core.config import settings
from aee.llm import setup_student
from aee.tasks import TASK_REGISTRY
from aee.agents import UniversalExtractor
from aee.utils.io import load_ground_truth
from aee.utils.dataset import create_dataset_from_ids

logger = setup_logging()

# --- СПИСОК ID ДОКУМЕНТОВ ДЛЯ ПРИМЕРОВ ---
# Впишите сюда имена файлов (без .json), которые вы хотите использовать как примеры.
# Желательно брать их из train_pool.
FEW_SHOT_IDS = [
    "acsanm.2c05400",  # Замените на реальные ID из вашего трейна
    "03067319.2019.1599875"   # Замените на реальные ID из вашего трейна
]

def main():
    parser = argparse.ArgumentParser(description="Run Static Few-Shot Inference.")
    parser.add_argument("--task", type=str, default="nanozymes")
    parser.add_argument("--input", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="data/results/static_fewshot")
    parser.add_argument("--shots", type=int, default=2, help="Number of examples to use")
    args = parser.parse_args()

    # 1. Setup LLM & Task
    setup_student()
    task_conf = TASK_REGISTRY.get(args.task)
    
    # 2. Load Examples for Prompt (DEMOS)
    logger.info(f"Loading {args.shots} examples for the prompt...")
    
    # Грузим GT для примеров
    gt_path = Path(f"data/ground_truth/{args.task}.csv")
    gt_data = load_ground_truth(gt_path, task_conf["row_converter"])
    
    # Создаем датасет из конкретных ID (ваши примеры)
    # Используем create_dataset_from_ids, который вы уже написали для optimize.py
    demos_dataset = create_dataset_from_ids(
        processed_dir=Path("data/processed"),
        gt_data=gt_data,
        task_conf=task_conf,
        allowed_ids=FEW_SHOT_IDS[:args.shots], # Берем ровно столько, сколько просили
        limit=args.shots
    )
    
    if len(demos_dataset) < args.shots:
        logger.error(f"Could not load enough examples! Found {len(demos_dataset)}, expected {args.shots}.")
        return

    # 3. Create & Compile Agent
    logger.info("Injecting examples into Static Prompt...")
    
    # Базовый агент (Zero-shot)
    agent = UniversalExtractor(task_conf["signature"])
    
    # LabeledFewShot просто берет примеры и добавляет их в контекст
    teleprompter = LabeledFewShot(k=args.shots)
    fewshot_agent = teleprompter.compile(agent, trainset=demos_dataset)

    # 4. Run Prediction (Inference)
    # Тут используем test set или то, что лежит в input папке
    # ВАЖНО: Не запускайте на тех же файлах, которые в FEW_SHOT_IDS (это data leakage)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob("*.json"))
    # Фильтруем, чтобы не предсказывать на примерах, которые уже в промпте
    files = [f for f in files if f.stem not in FEW_SHOT_IDS]
    
    logger.info(f"Running inference on {len(files)} documents...")

    for json_file in tqdm(files):
        try:
            # Читаем документ
            with open(json_file, "r") as f:
                doc_data = json.load(f)
                text = doc_data["text_content"]
            
            # Запуск агента
            pred = fewshot_agent(document_text=text)
            
            # Сохранение
            result = {
                "source_metadata": doc_data.get("metadata", {}),
                "extraction": pred.extracted_data.model_dump(),
                "reasoning": pred.reasoning
            }
            
            out_path = output_dir / f"{json_file.stem}_result_fewshot.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed on {json_file.name}: {e}")

if __name__ == "__main__":
    main()