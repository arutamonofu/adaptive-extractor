"""Custom exceptions for Adaptive Extractor.

This module defines domain-specific exceptions that provide clear error
handling across all components of the application.
"""

from typing import List


class AEException(Exception):
    """Base exception for all Adaptive Extractor errors."""
    pass


# Configuration-Related Exceptions
class ConfigurationError(AEException):
    """Base exception for configuration-related errors."""
    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid."""

    def __init__(self, config_path: str, errors: List[str]):
        self.config_path = config_path
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(
            f"Invalid configuration in {config_path}:\n  - {error_list}"
        )


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Missing required configuration: {key}")


# Task-Related Exceptions
class TaskError(AEException):
    """Base exception for task-related errors."""
    pass


class TaskNotFoundError(TaskError):
    """Raised when a requested task is not registered."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        super().__init__(f"Task '{task_name}' not found in registry")


class TaskValidationError(TaskError):
    """Raised when a task definition fails validation."""

    def __init__(self, task_name: str, errors: List[str]):
        self.task_name = task_name
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(
            f"Task '{task_name}' validation failed:\n  - {error_list}"
        )


# Storage-Related Exceptions
class StorageError(AEException):
    """Base exception for storage-related errors."""
    pass


class RepositoryError(StorageError):
    """Raised when repository operations fail."""

    def __init__(self, repository: str, operation: str, reason: str):
        self.repository = repository
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"{repository} {operation} failed: {reason}"
        )


class DataNotFoundError(StorageError):
    """Raised when requested data cannot be found in storage."""

    def __init__(self, data_type: str, identifier: str):
        self.data_type = data_type
        self.identifier = identifier
        super().__init__(f"{data_type} not found: {identifier}")


class DataValidationError(StorageError):
    """Raised when data validation fails."""

    def __init__(self, data_type: str, errors: List[str]):
        self.data_type = data_type
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(
            f"{data_type} validation failed:\n  - {error_list}"
        )


class InvalidDataFormatError(StorageError):
    """Raised when data format is invalid or corrupted."""

    def __init__(self, file_path: str, expected_format: str):
        self.file_path = file_path
        self.expected_format = expected_format
        super().__init__(
            f"Invalid data format in {file_path}, expected {expected_format}"
        )


class AgentNotFoundError(StorageError):
    """Raised when a requested agent cannot be found in repository."""

    def __init__(self, agent_path: str):
        self.agent_path = agent_path
        super().__init__(f"Agent not found: {agent_path}")


class InvalidAgentError(StorageError):
    """Raised when an agent file is invalid or corrupted."""

    def __init__(self, agent_path: str, reason: str):
        self.agent_path = agent_path
        self.reason = reason
        super().__init__(f"Invalid agent at {agent_path}: {reason}")


# LLM-Related Exceptions
class LLMError(AEException):
    """Base exception for LLM provider errors."""
    pass


class LLMProviderError(LLMError):
    """Raised when LLM provider encounters an error."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"LLM provider '{provider}' error: {reason}")


# Gemini-Specific Ingestion Exceptions
class GeminiAPIError(LLMProviderError):
    """Base exception for Gemini API errors in ingestion."""

    def __init__(self, reason: str, stage: str = "unknown"):
        self.stage = stage
        super().__init__(provider="gemini", reason=f"[{stage}] {reason}")


class GeminiBlockedError(GeminiAPIError):
    """Raised when Gemini blocks generation due to content filters or copyright recitation."""

    def __init__(
        self,
        reason: str,
        stage: str = "unknown",
        finish_reason: str = "UNKNOWN",
        block_reason: str | None = None,
        safety_ratings: list | None = None,
    ):
        self.finish_reason = finish_reason
        self.block_reason = block_reason
        self.safety_ratings = safety_ratings
        super().__init__(reason=reason, stage=stage)


class GeminiRecitationError(GeminiBlockedError):
    """Raised when Gemini blocks generation with FinishReason.RECITATION."""

    def __init__(
        self,
        stage: str = "unknown",
        details: str = "Generation blocked by Gemini recitation filter (copyrighted/verbatim text detected).",
    ):
        super().__init__(
            reason=details,
            stage=stage,
            finish_reason="RECITATION",
        )


class GeminiSafetyError(GeminiBlockedError):
    """Raised when Gemini blocks generation due to safety, blocklist, or prohibited content."""

    def __init__(
        self,
        stage: str = "unknown",
        finish_reason: str = "SAFETY",
        block_reason: str | None = None,
        safety_ratings: list | None = None,
    ):
        details = (
            f"Generation blocked by safety filter (finish_reason={finish_reason}"
            + (f", block_reason={block_reason}" if block_reason else "")
            + ")"
        )
        super().__init__(
            reason=details,
            stage=stage,
            finish_reason=finish_reason,
            block_reason=block_reason,
            safety_ratings=safety_ratings,
        )


class GeminiEmptyResponseError(GeminiAPIError):
    """Raised when Gemini returns an empty response without candidates or content."""

    def __init__(
        self,
        stage: str = "unknown",
        details: str = "Gemini returned empty response with no content candidates.",
    ):
        super().__init__(reason=details, stage=stage)


class GeminiMaxTokensError(GeminiAPIError):
    """Raised when Gemini generation hits max output tokens and leaves incomplete content."""

    def __init__(
        self,
        stage: str = "unknown",
        details: str = "Gemini generation truncated due to MAX_TOKENS limit.",
    ):
        super().__init__(reason=details, stage=stage)


# Document Ingestion (Parser) Exceptions
class ParserError(AEException):
    """Raised when document parsing fails."""

    def __init__(self, parser: str, file_path: str, reason: str):
        self.parser = parser
        self.file_path = file_path
        self.reason = reason
        super().__init__(
            f"Parser '{parser}' failed for {file_path}: {reason}"
        )


# Prompt Optimization-Related Exceptions
class OptimizationError(AEException):
    """Raised when syntactic prompt optimization (MIPROv2) fails."""
    pass


class DatasetBuildError(OptimizationError):
    """Raised when dataset building fails."""
    pass


class InvalidExperimentError(OptimizationError):
    """Raised when experiment data is invalid or incomplete."""
    pass


class MatchingError(OptimizationError):
    """Raised when experiment matching fails."""
    pass


# Use Case & Execution Exceptions
class UseCaseExecutionError(AEException):
    """Raised when a high-level use case fails to execute."""

    def __init__(self, use_case: str, reason: str):
        self.use_case = use_case
        self.reason = reason
        super().__init__(f"{use_case} failed: {reason}")
