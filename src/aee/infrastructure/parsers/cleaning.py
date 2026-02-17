# src/aee/ingestion/cleaning.py
"""Text cleaning utilities for extracted PDF content."""

import re
from typing import Dict, Pattern

class TextCleaner:
    """Post-processing utilities to clean and normalize raw text extracted from PDFs.
    
    Handles OCR artifacts, broken Unicode, ligatures, and scientific formatting.
    """

    # Ligature mappings from Unicode characters to ASCII equivalents
    _LIGATURE_MAP: Dict[str, str] = {
        "ﬂ": "fl",
        "ﬁ": "fi",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st"
    }

    # Common scientific terms that need fixing
    _SCIENTIFIC_FIXES: Dict[str, str] = {
        "p H": "pH",
        "K m": "Km",
        "V max": "Vmax",
        "C min": "Cmin",
        "C max": "Cmax",
        "et al .": "et al.",
    }

    # Regex pattern to match broken Unicode sequences like /uniFB01
    _BROKEN_UNICODE_RE: Pattern = re.compile(r"/uni([0-9A-Fa-f]{4})")

    # Dynamically create a regex to stitch broken words ONLY if they contain a ligature.
    # Pattern: [Letter] + [Spaces] + [Ligature] + [Spaces] + [Letter]
    # Example: "signi" + " " + "ﬁ" + " " + "cant" -> "signiﬁcant"
    _LIGATURES_STR: str = "".join(_LIGATURE_MAP.keys())
    _LIGATURE_STITCH_RE: Pattern = re.compile(
        rf"([a-zA-Z])\s+([{_LIGATURES_STR}])\s+([a-zA-Z])",
        re.IGNORECASE
    )

    @classmethod
    def _decode_hex_match(cls, match: re.Match) -> str:
        """Helper to convert hex code (FB01) to unicode char (ﬁ).
        
        Args:
            match: Regex match object containing the hex code.
            
        Returns:
            str: Decoded unicode character or original string if decoding fails.
        """
        try:
            char_code = int(match.group(1), 16)
            return chr(char_code)
        except (ValueError, OverflowError):
            # Return original string if decoding fails
            return match.group(0)

    @classmethod
    def clean_docling_markdown(cls, text: str) -> str:
        """Main entry point for text cleaning.
        
        Applies a cascade of cleaning rules in a safe order.
        
        Args:
            text: Input text to clean. Can be None or empty.
            
        Returns:
            str: Cleaned text. Returns empty string if input is None or empty.
        """
        if not text:
            return ""

        # Decode raw hex strings from PDF parser artifacts.
        # Example: "signi/uniFB01cant" -> "signiﬁcant"
        # Example: "E /uniFB03 cacy"   -> "E ﬃ cacy" (note the spaces)
        text = cls._BROKEN_UNICODE_RE.sub(cls._decode_hex_match, text)

        # We stitch words while the ligature is still a unique Unicode character.
        # This prevents accidental merging of valid ASCII words (e.g., "Sci fi").
        # Example: "E ﬃ cacy" -> "Eﬃcacy"
        text = cls._LIGATURE_STITCH_RE.sub(r"\1\2\3", text)

        # Expand ligatures to standard ASCII characters.
        # Example: "Eﬃcacy" -> "Efficacy"
        # Using translate for better performance with multiple character replacements
        translation_table = str.maketrans(cls._LIGATURE_MAP)
        text = text.translate(translation_table)

        # Fix specific scientific terms.
        # Using translate for better performance with multiple string replacements
        for bad, good in cls._SCIENTIFIC_FIXES.items():
            text = text.replace(bad, good)

        return text