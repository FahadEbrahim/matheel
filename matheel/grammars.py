"""Explicit provisioning and offline access to Tree-sitter grammars."""

import os
from functools import cache

from tree_sitter import Parser
from tree_sitter_language_pack import (
    Error,
    LanguageRegistry,
    cache_dir,
    detect_language_from_content,
    has_language,
    prefetch,
)
from tree_sitter_language_pack._native import ProcessConfig


SUPPORTED_GRAMMARS = (
    "java", "python", "c", "cpp", "go", "javascript", "typescript", "kotlin",
    "scala", "swift", "solidity", "dart", "php", "ruby", "rust", "csharp",
    "lua", "julia", "r", "objc",
)
_LIBS_DIR_ENV = "TREE_SITTER_LANGUAGE_PACK_LIBS_DIR"


class GrammarUnavailableError(ValueError):
    """A known grammar could not be loaded from local storage."""


@cache
def _local_registry():
    # The standalone registry never downloads. Its Python constructor finds
    # dynamic libraries through LIBS_DIR rather than the download cache API.
    if not os.environ.get(_LIBS_DIR_ENV, "").strip():
        os.environ[_LIBS_DIR_ENV] = cache_dir()
    return LanguageRegistry.new()


def get_language(language):
    registry = _local_registry()
    try:
        return registry.get_language(language)
    except Error as exc:
        if not registry.has_language(language):
            raise ValueError(f"Unsupported Tree-sitter language: {language!r}.") from exc
        raise GrammarUnavailableError(
            f"Tree-sitter grammar {language!r} is unavailable in {os.environ[_LIBS_DIR_ENV]}. "
            f"Run 'matheel grammars {language}' with network access, then use the resulting "
            "cache when running offline."
        ) from exc


def get_parser(language):
    return Parser(get_language(language))


def process_code(text, language, **options):
    # The public ProcessConfig wraps the native type, while the standalone
    # registry's process() takes the native type directly.
    get_language(language)
    return _local_registry().process(text, ProcessConfig(language=language, **options))


def detect_code_language(text):
    shebang = detect_language_from_content(text)
    if shebang:
        return shebang

    candidates = []
    registry = _local_registry()
    for language in sorted(SUPPORTED_GRAMMARS):
        if not registry.has_parser(language):
            continue
        result = process_code(text, language, structure=True, imports=True)
        candidates.append((
            result.metrics.error_count,
            -(len(result.structure) + len(result.imports)),
            -result.metrics.node_count,
            language,
        ))
    if not candidates:
        raise GrammarUnavailableError(
            "No code grammars are available locally. Run 'matheel grammars' with "
            "network access, then use the resulting cache when running offline."
        )
    return min(candidates)[-1]


def provision_grammars(languages=()):
    """Download the selected grammars, or all Matheel metric grammars."""
    selected = list(dict.fromkeys(name.strip().lower() for name in (languages or SUPPORTED_GRAMMARS)))
    for language in selected:
        if not has_language(language):
            raise ValueError(f"Unsupported Tree-sitter language: {language!r}.")
    try:
        prefetch(selected)
        # Confirm the downloaded libraries load from the same directory that
        # a fresh process will use for offline access.
        for language in selected:
            get_language(language)
    except Error as exc:
        raise ValueError(f"Could not provision Tree-sitter grammars: {exc}") from exc
    return selected, cache_dir()
