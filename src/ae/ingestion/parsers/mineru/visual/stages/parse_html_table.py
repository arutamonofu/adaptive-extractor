"""Helper for parsing HTML tables into structured dicts."""

from __future__ import annotations

import re
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def parse_html_table(html_str: str) -> Dict[str, Any]:
    """Parse HTML table string into structured dict conforming to VLM output structure."""
    if not html_str or not html_str.strip():
        return {
            "status": "failed",
            "tables": [],
            "warnings": ["Empty HTML table body provided"]
        }

    try:
        # Find all <tr>...</tr> blocks
        tr_blocks = re.findall(r'<tr.*?>(.*?)</tr>', html_str, re.DOTALL | re.IGNORECASE)
        rows_data = []
        for tr in tr_blocks:
            # Find all <td...>...</td> or <th...>...</th> blocks inside the row
            cells = re.findall(r'<(td|th).*?>(.*?)</\1>', tr, re.DOTALL | re.IGNORECASE)
            row = []
            for cell_type, cell_content in cells:
                # Remove inner HTML tags (like <sub>, <sup>, etc.)
                cleaned_content = re.sub(r'<.*?>', '', cell_content).strip()
                # Clean whitespace
                cleaned_content = " ".join(cleaned_content.split())
                row.append(cleaned_content)
            if row:
                rows_data.append(row)
                
        if not rows_data:
            return {
                "status": "failed",
                "tables": [],
                "warnings": ["Could not parse rows from HTML table"]
            }
            
        columns = rows_data[0]
        rows = rows_data[1:]
        
        # Format columns as List[Dict[str, Any]]
        # In VLM response: "columns": [{"name": "catalyst", "unit": null}, ...]
        formatted_columns = []
        for col_name in columns:
            # Try to extract unit in parentheses if present, e.g. "Km (mM)"
            unit_match = re.search(r'\((.*?)\)', col_name)
            unit = unit_match.group(1).strip() if unit_match else None
            name = re.sub(r'\s*\(.*?\)\s*', '', col_name).strip()
            formatted_columns.append({
                "name": name or col_name,
                "unit": unit
            })

        return {
            "status": "success",
            "tables": [
                {
                    "panel": None,
                    "chart_type": "table",
                    "series_name": None,
                    "columns": formatted_columns,
                    "rows": rows
                }
            ],
            "warnings": []
        }
    except Exception as e:
        logger.error(f"Failed to parse HTML table: {e}", exc_info=True)
        return {
            "status": "failed",
            "tables": [],
            "warnings": [f"Error parsing HTML table: {e}"]
        }
