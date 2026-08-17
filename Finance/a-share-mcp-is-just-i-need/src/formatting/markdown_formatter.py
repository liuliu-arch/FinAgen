"""
Markdown formatting utilities for A-Share MCP Server.
"""
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Configuration
# Common number of trading days per year. Max rows to display in Markdown output
MAX_MARKDOWN_ROWS = 250


def format_df_to_markdown(df: pd.DataFrame, max_rows: int = None) -> str:
    """Formats a Pandas DataFrame to a Markdown string with row truncation.

    Args:
        df: The DataFrame to format
        max_rows: Maximum rows to include in output. Defaults to MAX_MARKDOWN_ROWS if None.

    Returns:
        A markdown formatted string representation of the DataFrame
    """
    if df.empty:
        logger.warning("Attempted to format an empty DataFrame to Markdown.")
        return "(No data available to display)"

    # Default max_rows to the configured limit if not provided
    if max_rows is None:
        max_rows = MAX_MARKDOWN_ROWS
        logger.debug(f"max_rows defaulted to MAX_MARKDOWN_ROWS: {max_rows}")

    original_rows = df.shape[0]  # Only need original_rows now
    truncated = False
    truncation_notes = []

    # Determine the actual number of rows to display, capped by max_rows
    rows_to_show = min(original_rows, max_rows)

    # Always apply the row limit
    df_display = df.head(rows_to_show)

    # Check if actual row truncation occurred (only if original_rows > rows_to_show)
    if original_rows > rows_to_show:
        truncation_notes.append(
            f"rows truncated to the limit of {rows_to_show} (from {original_rows})")
        truncated = True

    try:
        markdown_table = df_display.to_markdown(index=False)
    except Exception as e:
        logger.error(
            f"Error converting DataFrame to Markdown: {e}", exc_info=True)
        return "Error: Could not format data into Markdown table."

    if truncated:
        # Note: 'truncated' is now only True if rows were truncated
        notes = "; ".join(truncation_notes)
        logger.debug(
            f"Markdown table generated with truncation notes: {notes}")
        return f"Note: Data truncated ({notes}).\n\n{markdown_table}"
    else:
        logger.debug("Markdown table generated without truncation.")
        return markdown_table
