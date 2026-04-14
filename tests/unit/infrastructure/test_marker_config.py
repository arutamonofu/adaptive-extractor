"""Unit tests for Marker configuration module.

Tests cover:
- Configuration dictionary generation
- Key settings presence and values
- Processor list configuration
- Device settings
"""

from aee.infrastructure.parsers.marker_config import (
    CUSTOM_PROCESSORS,
    FORCE_OCR,
    LLM_SERVICE,
    TORCH_DEVICE,
    USE_LLM,
    OllamaService_ollama_model,
    get_custom_processors,
    get_marker_config_dict,
    get_torch_device,
)


class TestMarkerConfigDict:
    """Tests for get_marker_config_dict() function."""

    def test_returns_dict(self):
        """Test that get_marker_config_dict returns a dictionary."""
        config = get_marker_config_dict()
        assert isinstance(config, dict)

    def test_core_settings_present(self):
        """Test that core settings are present in config dict."""
        config = get_marker_config_dict()

        assert config["output_format"] == "markdown"
        assert config["page_range"] is None
        assert config["paginate_output"] is False
        assert config["debug"] is False
        assert config["disable_tqdm"] is False

    def test_ocr_settings_present(self):
        """Test that OCR settings are present in config dict."""
        config = get_marker_config_dict()

        assert config["force_ocr"] is True
        assert config["strip_existing_ocr"] is True
        assert config["disable_ocr"] is False
        assert config["keep_chars"] is False

    def test_llm_settings_present(self):
        """Test that LLM settings are present in config dict."""
        config = get_marker_config_dict()

        assert config["use_llm"] is True
        assert config["LLMEquationProcessor_use_llm"] is True
        assert config["LLMMathBlockProcessor_use_llm"] is True
        assert config["LLMTableProcessor_use_llm"] is True
        assert config["LLMTableMergeProcessor_use_llm"] is True

    def test_ollama_settings_injected(self):
        """Test that Ollama settings are injected when LLM_SERVICE is 'ollama'."""
        config = get_marker_config_dict()

        assert config["llm_service"] == "marker.services.ollama.OllamaService"
        assert config["ollama_base_url"] == "https://aicltr.itmo.ru/ollama"
        assert config["ollama_model"] == "qwen2.5vl:72b"

    def test_ollama_settings_override(self):
        """Test that Ollama settings can be overridden."""
        config = get_marker_config_dict(
            ollama_base_url="http://custom-url:11434",
            ollama_model="custom-model:latest",
        )

        assert config["ollama_base_url"] == "http://custom-url:11434"
        assert config["ollama_model"] == "custom-model:latest"

    def test_output_dir_override(self):
        """Test that output directory can be overridden."""
        config = get_marker_config_dict(output_dir="/custom/output")

        assert config["output_dir"] == "/custom/output"

    def test_builder_settings_present(self):
        """Test that builder settings are present in config dict."""
        config = get_marker_config_dict()

        # DocumentBuilder
        assert config["lowres_image_dpi"] == 256
        assert config["highres_image_dpi"] == 600

        # LayoutBuilder
        assert config["max_expand_frac"] == 0.04

        # LineBuilder
        assert config["min_document_ocr_threshold"] == 0.7
        assert config["detection_line_min_confidence"] == 0.7

        # OcrBuilder
        assert config["ocr_task_name"] == "ocr_with_boxes"
        assert config["drop_repeated_text"] is True

        # StructureBuilder
        assert config["gap_threshold"] == 0.04
        assert config["list_gap_threshold"] == 0.08

    def test_processor_settings_present(self):
        """Test that processor settings are present in config dict."""
        config = get_marker_config_dict()

        # EquationProcessor
        assert config["equation_model_max_length"] == 2048

        # LLMEquationProcessor
        assert config["LLMEquationProcessor_max_concurrency"] == 1
        assert config["LLMEquationProcessor_image_expansion_ratio"] == 0.06
        assert config["LLMEquationProcessor_min_equation_height"] == 0.025

        # LLMMathBlockProcessor
        assert config["LLMMathBlockProcessor_max_concurrency"] == 1
        assert config["LLMMathBlockProcessor_image_expansion_ratio"] == 0.04
        assert config["LLMMathBlockProcessor_inlinemath_min_ratio"] == 0.35

        # TableProcessor
        assert config["table_row_split_threshold"] == 0.55
        assert config["table_pdftext_workers"] == 1
        assert config["drop_repeated_table_text"] is True

        # LLMTableProcessor
        assert config["LLMTableProcessor_max_concurrency"] == 1
        assert config["LLMTableProcessor_image_expansion_ratio"] == 0.05
        assert config["LLMTableProcessor_max_rows_per_batch"] == 70
        assert config["LLMTableProcessor_max_table_rows"] == 200
        assert config["LLMTableProcessor_max_table_iterations"] == 2

        # LLMTableMergeProcessor
        assert config["LLMTableMergeProcessor_max_concurrency"] == 1
        assert config["LLMTableMergeProcessor_table_height_threshold"] == 0.65
        assert config["LLMTableMergeProcessor_column_gap_threshold"] == 45

    def test_renderer_settings_present(self):
        """Test that renderer settings are present in config dict."""
        config = get_marker_config_dict()

        assert config["disable_links"] is True
        assert config["extract_images"] is False  # DISABLE_IMAGE_EXTRACTION = True
        assert config["html_tables_in_markdown"] is True
        assert config["keep_pageheader_in_output"] is False
        assert config["keep_pagefooter_in_output"] is False
        assert config["add_block_ids"] is False
        assert config["page_separator"] == "-" * 48


class TestCustomProcessors:
    """Tests for custom processor list."""

    def test_returns_list(self):
        """Test that get_custom_processors returns a list."""
        processors = get_custom_processors()
        assert isinstance(processors, list)

    def test_returns_copy(self):
        """Test that get_custom_processors returns a copy (not the original)."""
        processors1 = get_custom_processors()
        processors2 = get_custom_processors()

        # Modifying one should not affect the other
        processors1.append("test.processor")
        assert "test.processor" not in processors2

    def test_core_processors_present(self):
        """Test that core processors are present."""
        processors = get_custom_processors()

        assert "marker.processors.order.OrderProcessor" in processors
        assert "marker.processors.text.TextProcessor" in processors
        assert "marker.processors.table.TableProcessor" in processors
        assert "marker.processors.equation.EquationProcessor" in processors

    def test_llm_processors_present(self):
        """Test that LLM processors are present."""
        processors = get_custom_processors()

        assert "marker.processors.llm.llm_table.LLMTableProcessor" in processors
        assert "marker.processors.llm.llm_table_merge.LLMTableMergeProcessor" in processors
        assert "marker.processors.llm.llm_equation.LLMEquationProcessor" in processors
        assert "marker.processors.llm.llm_mathblock.LLMMathBlockProcessor" in processors

    def test_processor_count(self):
        """Test the expected number of processors."""
        processors = get_custom_processors()

        # Count should match CUSTOM_PROCESSORS length (excluding commented out)
        expected_count = len([p for p in CUSTOM_PROCESSORS if not p.strip().startswith("#")])
        assert len(processors) == expected_count


class TestDeviceSettings:
    """Tests for device-related settings."""

    def test_torch_device(self):
        """Test that TORCH_DEVICE is set correctly based on CUDA availability."""
        import torch
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert TORCH_DEVICE == expected_device
        assert get_torch_device() == expected_device

    def test_llm_service(self):
        """Test that LLM_SERVICE is set to ollama."""
        assert LLM_SERVICE == "ollama"

    def test_ollama_model(self):
        """Test that Ollama model is qwen2.5vl:72b."""
        assert OllamaService_ollama_model == "qwen2.5vl:72b"


class TestConstants:
    """Tests for individual constants."""

    def test_force_ocr(self):
        """Test FORCE_OCR setting."""
        assert FORCE_OCR is True

    def test_use_llm(self):
        """Test USE_LLM setting."""
        assert USE_LLM is True

    def test_disable_image_extraction(self):
        """Test DISABLE_IMAGE_EXTRACTION setting."""
        from aee.infrastructure.parsers.marker_config import DISABLE_IMAGE_EXTRACTION
        assert DISABLE_IMAGE_EXTRACTION is True

    def test_redo_inline_math(self):
        """Test REDO_INLINE_MATH setting."""
        from aee.infrastructure.parsers.marker_config import REDO_INLINE_MATH
        assert REDO_INLINE_MATH is True
