"""
QBRZoom - The Quarterly Business Review Risk Assessment Tool
============================================================

Submodule: project
---------------

This file contains project level settings.
"""

from os import environ as os_environ
from pathlib import Path


CACHE = Path(__file__).parent.parent / 'cache'
CACHE.mkdir(exist_ok=True)


def set_project_cache() -> None:
    """Set cache to project level instead of using the global cache folders.
    """

    folder_string_ = str(CACHE)
    os_environ["HF_HOME"] = folder_string_
    os_environ["SPACY_HOME"] = folder_string_
    os_environ["SPACY_DATA_PATH"] = folder_string_
    os_environ['SENTENCE_TRANSFORMERS_HOME '] = folder_string_
    os_environ['TORCH_HOME '] = folder_string_