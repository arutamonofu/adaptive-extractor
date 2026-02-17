"""Custom exceptions for AutoEvoExtractor.

This module defines domain-specific exceptions that provide clear error
handling across all layers of the application.
"""

from typing import List


class AEEException(Exception):
    """Base exception for all AutoEvoExtractor errors."""
    pass


# Domain Layer Exceptions
class DomainException(AEEException):
    """Base exception for domain layer errors."""
    pass


class TaskNotFoundError(DomainException):
    """Raised when a requested task is not registered."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        super().__init__(f"Task '{task_name}' not found in registry")


class TaskValidationError(DomainException):
    """Raised when a task definition fails validation."""

    def __init__(self, task_name: str, errors: List[str]):
        self.task_name = task_name
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(
            f"Task '{task_name}' validation failed:\n  - {error_list}"
        )


class InvalidExperimentError(DomainException):
    """Raised when experiment data is invalid or incomplete."""
    pass


class MatchingError(DomainException):
    """Raised when experiment matching fails."""
    pass


# Application Layer Exceptions
class ApplicationException(AEEException):
    """Base exception for application layer errors."""
    pass


class UseCaseExecutionError(ApplicationException):
    """Raised when a use case fails to execute."""

    def __init__(self, use_case: str, reason: str):
        self.use_case = use_case
        self.reason = reason
        super().__init__(f"{use_case} failed: {reason}")


class DatasetBuildError(ApplicationException):
    """Raised when dataset building fails."""
    pass


class AgentNotFoundError(ApplicationException):
    """Raised when a requested agent cannot be found."""

    def __init__(self, agent_path: str):
        self.agent_path = agent_path
        super().__init__(f"Agent not found: {agent_path}")


class InvalidAgentError(ApplicationException):
    """Raised when an agent file is invalid or corrupted."""

    def __init__(self, agent_path: str, reason: str):
        self.agent_path = agent_path
        self.reason = reason
        super().__init__(f"Invalid agent at {agent_path}: {reason}")


# Infrastructure Layer Exceptions
class InfrastructureException(AEEException):
    """Base exception for infrastructure layer errors."""
    pass


class LLMProviderError(InfrastructureException):
    """Raised when LLM provider encounters an error."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"LLM provider '{provider}' error: {reason}")


class ParserError(InfrastructureException):
    """Raised when document parsing fails."""

    def __init__(self, parser: str, file_path: str, reason: str):
        self.parser = parser
        self.file_path = file_path
        self.reason = reason
        super().__init__(
            f"Parser '{parser}' failed for {file_path}: {reason}"
        )


class RepositoryError(InfrastructureException):
    """Raised when repository operations fail."""

    def __init__(self, repository: str, operation: str, reason: str):
        self.repository = repository
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"{repository} {operation} failed: {reason}"
        )


class StorageError(InfrastructureException):
    """Raised when storage operations fail."""
    pass


class OptimizationError(InfrastructureException):
    """Raised when optimization (DSPy) fails."""
    pass


# Configuration Exceptions
class ConfigurationException(AEEException):
    """Base exception for configuration errors."""
    pass


class InvalidConfigError(ConfigurationException):
    """Raised when configuration is invalid."""

    def __init__(self, config_path: str, errors: List[str]):
        self.config_path = config_path
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(
            f"Invalid configuration in {config_path}:\n  - {error_list}"
        )


class MissingConfigError(ConfigurationException):
    """Raised when required configuration is missing."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Missing required configuration: {key}")


# Data Exceptions
class DataException(AEEException):
    """Base exception for data-related errors."""
    pass


class DataNotFoundError(DataException):
    """Raised when requested data cannot be found."""

    def __init__(self, data_type: str, identifier: str):
        self.data_type = data_type
        self.identifier = identifier
        super().__init__(f"{data_type} not found: {identifier}")


class DataValidationError(DataException):
    """Raised when data validation fails."""

    def __init__(self, data_type: str, errors: List[str]):
        self.data_type = data_type
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(
            f"{data_type} validation failed:\n  - {error_list}"
        )


class InvalidDataFormatError(DataException):
    """Raised when data format is invalid or corrupted."""

    def __init__(self, file_path: str, expected_format: str):
        self.file_path = file_path
        self.expected_format = expected_format
        super().__init__(
            f"Invalid data format in {file_path}, expected {expected_format}"
        )
