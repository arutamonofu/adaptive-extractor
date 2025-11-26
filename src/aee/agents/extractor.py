# src/aee/agents/extractor.py

import dspy

class UniversalExtractor(dspy.Module):
    """
    Универсальный агент извлечения.
    Не зависит от конкретной схемы данных — она передается через signature_class.
    """
    def __init__(self, signature_class: type[dspy.Signature]):
        super().__init__()
        
        # Мы используем ChainOfThought, чтобы модель сначала "подумала" (Reasoning),
        # а потом заполнила Pydantic-модель, указанную в OutputField сигнатуры.
        self.prog = dspy.ChainOfThought(signature_class)

    def forward(self, document_text: str) -> dspy.Prediction:
        """
        Запуск агента.
        
        Args:
            document_text: Текст статьи.
            
        Returns:
            dspy.Prediction: Объект с полями:
                .reasoning (str) - ход мыслей
                .extracted_data (Pydantic Model) - результат
        """
        return self.prog(document_text=document_text)