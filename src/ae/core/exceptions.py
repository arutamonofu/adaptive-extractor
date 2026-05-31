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


# Optimization-Related Exceptions
class OptimizationError(AEException):
    """Raised when agent prompt optimization (MIPROv2) fails."""
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
