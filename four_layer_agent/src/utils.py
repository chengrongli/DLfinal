"""
Utility functions for the four-layer architecture.
"""

import logging
import re


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.

    Args:
        text: Raw text content

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple empty lines -> double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces -> single

    # Clean up common PDF artifacts
    text = re.sub(r'\f', '', text)  # Form feed characters
    text = re.sub(r'\x0c', '', text)  # Another form feed variant

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    return text.strip()
