from pathlib import Path


def custom_algorithm_template(name="custom_similarity", include_prepare=True):
    module_name = _safe_identifier(name or "custom_similarity")
    prepare_block = ""
    context_note = "dataset_context=None, "
    if include_prepare:
        prepare_block = f'''def prepare_dataset(dataset, prepared_texts=None, **kwargs):
    """Build optional reusable context before dataset scoring."""
    _ = kwargs
    files = getattr(dataset, "files", None)
    return {{
        "algorithm": "{module_name}",
        "file_count": len(files) if files is not None else 0,
        "prepared_texts": dict(prepared_texts or {{}}),
    }}


'''
    return f'''"""Matheel custom similarity algorithm template.

Edit score_pair, keep the function importable, and run it with:

    matheel compare sample_pairs.zip --algorithm-path this_file.py

The function must return a finite numeric score where larger means more similar.
"""

from __future__ import annotations


{prepare_block}def score_pair(code_a, code_b, {context_note}row=None, **kwargs):
    """Return a similarity score for two source-code strings."""
    _ = (dataset_context, row, kwargs)
    left = str(code_a or "").strip()
    right = str(code_b or "").strip()
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 1.0 if left == right else 0.0
'''


def write_custom_algorithm_template(output_path, name=None, overwrite=False, include_prepare=True):
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Custom algorithm template already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        custom_algorithm_template(
            name=name or target.stem,
            include_prepare=include_prepare,
        ),
        encoding="utf-8",
    )
    return target


def _safe_identifier(value):
    text = "".join(character if character.isalnum() or character == "_" else "_" for character in str(value))
    text = text.strip("_")
    if not text or text[0].isdigit():
        text = f"algorithm_{text}"
    return text
