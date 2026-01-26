# src/aee/ingestion/cleaning.py

import re
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Post-processing utilities to clean and normalize raw text extracted from PDFs.
    Handles OCR artifacts, broken Unicode, ligatures, and scientific formatting.
    """

    _LIGATURE_MAP = {
        "ﬂ": "fl", "ﬁ": "fi", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "ﬅ": "ft", "ﬆ": "st"
    }

    _SCIENTIFIC_FIXES = {
        "p H": "pH",
        "K m": "Km",
        "V max": "Vmax",
        "C min": "Cmin",
        "C max": "Cmax",
        "et al .": "et al.",
    }

    _BROKEN_UNICODE_RE = re.compile(r"/uni([0-9A-Fa-f]{4})")

    # Dynamically create a regex to stitch broken words ONLY if they contain a ligature.
    # Pattern: [Letter] + [Spaces] + [Ligature] + [Spaces] + [Letter]
    # Example: "signi" + " " + "ﬁ" + " " + "cant" -> "signiﬁcant"
    _LIGATURES_STR = "".join(_LIGATURE_MAP.keys())
    _LIGATURE_STITCH_RE = re.compile(
        rf"([a-zA-Z])\s+([{_LIGATURES_STR}])\s+([a-zA-Z])", 
        re.IGNORECASE
    )

    @classmethod
    def _decode_hex_match(cls, match: re.Match) -> str:
        """Helper to convert hex code (FB01) to unicode char (ﬁ)."""
        try:
            char_code = int(match.group(1), 16)
            return chr(char_code)
        except Exception:
            # Return original string if decoding fails
            return match.group(0)

    @classmethod
    def clean_docling_markdown(cls, text: str) -> str:
        """
        Main entry point for text cleaning.
        Applies a cascade of cleaning rules in a safe order.
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
        for ligature, ascii_val in cls._LIGATURE_MAP.items():
            if ligature in text:
                text = text.replace(ligature, ascii_val)

        # Fix specific scientific terms.
        for bad, good in cls._SCIENTIFIC_FIXES.items():
            if bad in text:
                text = text.replace(bad, good)

        return text