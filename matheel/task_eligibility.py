"""Predeclared method exclusions, independent of runtime failures."""

from collections.abc import Mapping


def normalize_task_eligibility(value=None):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("task_eligibility must map pair/retrieval to eligibility declarations.")
    result = {}
    for task, declaration in value.items():
        if task not in {"pair", "retrieval"}:
            raise ValueError("task_eligibility keys must be pair or retrieval.")
        if not isinstance(declaration, Mapping) or set(declaration) - {"status", "reason"}:
            raise ValueError("Task eligibility requires status and an optional reason.")
        status = declaration.get("status")
        reason = str(declaration.get("reason") or "").strip()
        if status not in {"eligible", "not_applicable"}:
            raise ValueError("Task eligibility status must be eligible or not_applicable.")
        if status == "not_applicable" and not reason:
            raise ValueError("A not_applicable declaration requires a reason.")
        result[task] = {"status": status, **({"reason": reason} if reason else {})}
    return result


def normalize_dataset_exclusions(value=None):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("dataset_exclusions must map dataset names to nonempty reasons.")
    result = {}
    for name, reason in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(reason, str) or not reason.strip():
            raise ValueError("dataset_exclusions requires nonempty dataset names and reasons.")
        result[name.strip()] = reason.strip()
    return result


def method_exclusion_reason(method, dataset):
    """Return the declared reason, or None when this method is applicable."""
    exclusions = normalize_dataset_exclusions(method.get("dataset_exclusions"))
    if dataset["name"] in exclusions:
        return exclusions[dataset["name"]]
    declaration = normalize_task_eligibility(method.get("task_eligibility")).get(dataset["task_family"], {})
    return declaration.get("reason") if declaration.get("status") == "not_applicable" else None
