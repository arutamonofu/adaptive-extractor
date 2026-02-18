"""Cache configuration utilities for DSPy.

This module provides utilities for configuring DSPy's built-in caching
system with persistent disk storage.
"""

import logging
from pathlib import Path
from typing import Optional

from aee.infrastructure.config.settings import CacheConfig

logger = logging.getLogger(__name__)


def setup_dspy_cache(
    config: Optional[CacheConfig] = None,
    cache_dir: Optional[str] = None,
    enable_disk_cache: bool = True,
    enable_memory_cache: bool = True,
) -> None:
    """Configure DSPy cache for improved performance and cost reduction.

    This function sets up both in-memory and on-disk caching for DSPy LLM calls.
    The cache is shared across all DSPy programs and persists between runs.

    Args:
        config: Cache configuration from settings. If None, uses hardcoded defaults.
        cache_dir: Directory for disk cache. Defaults to settings.dspy_cache_dir
                   or ~/.dspy_cache if not set.
        enable_disk_cache: Whether to enable persistent disk caching.
        enable_memory_cache: Whether to enable fast in-memory caching.

    Example:
        ```python
        from aee.infrastructure.cache import setup_dspy_cache
        from aee.infrastructure.config import settings

        # Setup cache with settings from config
        setup_dspy_cache(config=settings.cache)

        # Or use custom configuration
        setup_dspy_cache(
            cache_dir="/path/to/cache",
            enable_disk_cache=True,
        )
        ```

    Note:
        - Disk cache persists between program runs
        - Memory cache is faster but cleared on exit
        - Both caches can be enabled simultaneously
        - Cache is automatically used by all DSPy LLM calls
    """
    import dspy

    # Use default cache directory from settings if not specified
    if cache_dir is None:
        from aee.infrastructure.config import settings
        cache_dir = settings.dspy_cache_dir or str(Path.home() / ".dspy_cache")

    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Get cache limits from config or use fallback defaults
    if config:
        disk_size_limit = config.disk_size_limit_bytes
        memory_max_entries = config.memory_max_entries
    else:
        # Fallback defaults if no config provided
        disk_size_limit = 30_000_000_000  # 30 GB
        memory_max_entries = 1_000_000

    # Configure DSPy cache
    dspy.configure_cache(
        enable_disk_cache=enable_disk_cache,
        enable_memory_cache=enable_memory_cache,
        disk_cache_dir=cache_dir,
        disk_size_limit_bytes=disk_size_limit,
        memory_max_entries=memory_max_entries,
    )

    logger.info(
        f"DSPy cache configured: disk={enable_disk_cache} ({cache_dir}), "
        f"memory={enable_memory_cache} (max {memory_max_entries:,} entries)"
    )


def clear_dspy_cache(cache_dir: Optional[str] = None) -> None:
    """Clear DSPy disk cache.

    Args:
        cache_dir: Cache directory to clear. If None, uses default ~/.dspy_cache.

    Example:
        ```python
        from aee.infrastructure.cache import clear_dspy_cache

        # Clear default cache
        clear_dspy_cache()

        # Clear custom cache
        clear_dspy_cache("/path/to/cache")
        ```
    """
    import shutil

    if cache_dir is None:
        cache_dir = str(Path.home() / ".dspy_cache")

    cache_path = Path(cache_dir)

    if cache_path.exists():
        shutil.rmtree(cache_path)
        logger.info(f"Cleared DSPy cache at {cache_dir}")
    else:
        logger.debug(f"No cache found at {cache_dir}")


def get_cache_stats(cache_dir: Optional[str] = None) -> dict:
    """Get DSPy cache statistics.

    Args:
        cache_dir: Cache directory. If None, uses default ~/.dspy_cache.

    Returns:
        Dictionary with cache statistics:
        - exists: Whether cache directory exists
        - size_bytes: Total cache size in bytes
        - size_human: Human-readable size
        - num_files: Number of cached entries
    """

    def human_readable_size(size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    if cache_dir is None:
        cache_dir = str(Path.home() / ".dspy_cache")

    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "size_human": "0 B",
            "num_files": 0,
        }

    # Calculate cache size
    total_size = 0
    num_files = 0

    for file_path in cache_path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            num_files += 1

    return {
        "exists": True,
        "size_bytes": total_size,
        "size_human": human_readable_size(total_size),
        "num_files": num_files,
    }
