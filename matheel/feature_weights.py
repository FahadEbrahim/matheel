_DEFAULT_LEGACY_WEIGHTS = {
    "semantic": 0.7,
    "levenshtein": 0.3,
    "jaro_winkler": 0.0,
}


def available_default_features():
    return ("semantic", "levenshtein", "jaro_winkler", "code_metric")


def _validate_weight_name(name):
    key = str(name or "").strip()
    if not key:
        raise ValueError("Feature names must be non-empty.")
    return key


def _validate_weight_value(name, value):
    numeric_value = float(value)
    if numeric_value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric_value


def legacy_feature_weights(weight_semantic, weight_levenshtein, weight_jaro_winkler, code_metric_weight=0.0):
    return {
        "semantic": _validate_weight_value("Ws", weight_semantic),
        "levenshtein": _validate_weight_value("Wl", weight_levenshtein),
        "jaro_winkler": _validate_weight_value("Wj", weight_jaro_winkler),
        "code_metric": _validate_weight_value("code_metric_weight", code_metric_weight),
    }


def default_feature_weights(code_metric_weight=0.0):
    resolved = dict(_DEFAULT_LEGACY_WEIGHTS)
    numeric_code_metric = _validate_weight_value("code_metric_weight", code_metric_weight)
    if numeric_code_metric > 0:
        resolved["code_metric"] = numeric_code_metric
    return normalize_feature_weights(resolved)


def _iter_weight_entries(feature_weights):
    if feature_weights is None:
        return []
    if isinstance(feature_weights, dict):
        return list(feature_weights.items())
    if isinstance(feature_weights, str):
        parts = [part.strip() for part in feature_weights.split(",") if part.strip()]
        entries = []
        for part in parts:
            if "=" not in part:
                raise ValueError(f"Feature weight entries must look like name=value. Got: {part}")
            name, value = part.split("=", 1)
            entries.append((name.strip(), value.strip()))
        return entries

    entries = []
    for item in feature_weights:
        if isinstance(item, str):
            if "=" not in item:
                raise ValueError(f"Feature weight entries must look like name=value. Got: {item}")
            name, value = item.split("=", 1)
            entries.append((name.strip(), value.strip()))
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2:
            entries.append((item[0], item[1]))
            continue
        raise ValueError(f"Unsupported feature weight entry: {item!r}")
    return entries


def parse_feature_weights(feature_weights):
    parsed = {}
    for raw_name, raw_value in _iter_weight_entries(feature_weights):
        name = _validate_weight_name(raw_name)
        parsed[name] = _validate_weight_value(name, raw_value)
    return parsed


def normalize_feature_weights(feature_weights):
    validated = {}
    total = 0.0
    for raw_name, raw_value in (feature_weights or {}).items():
        name = _validate_weight_name(raw_name)
        numeric_value = _validate_weight_value(name, raw_value)
        validated[name] = numeric_value
        total += numeric_value

    if total <= 0:
        return {name: 0.0 for name in validated}
    return {name: value / total for name, value in validated.items()}


def resolve_feature_weights(
    weight_semantic=None,
    weight_levenshtein=None,
    weight_jaro_winkler=None,
    code_metric_weight=0.0,
    feature_weights=None,
):
    parsed_feature_weights = parse_feature_weights(feature_weights)
    if parsed_feature_weights:
        if (
            "code_metric" not in parsed_feature_weights
            and _validate_weight_value("code_metric_weight", code_metric_weight) > 0
        ):
            parsed_feature_weights["code_metric"] = _validate_weight_value("code_metric_weight", code_metric_weight)
        return normalize_feature_weights(parsed_feature_weights)

    if any(value is not None for value in (weight_semantic, weight_levenshtein, weight_jaro_winkler)):
        resolved = legacy_feature_weights(
            _DEFAULT_LEGACY_WEIGHTS["semantic"] if weight_semantic is None else weight_semantic,
            _DEFAULT_LEGACY_WEIGHTS["levenshtein"] if weight_levenshtein is None else weight_levenshtein,
            _DEFAULT_LEGACY_WEIGHTS["jaro_winkler"] if weight_jaro_winkler is None else weight_jaro_winkler,
            code_metric_weight=code_metric_weight,
        )
        return normalize_feature_weights(resolved)

    return default_feature_weights(code_metric_weight=code_metric_weight)


def combine_weighted_scores(feature_scores, feature_weights):
    score = 0.0
    for name, value in (feature_scores or {}).items():
        score += float(feature_weights.get(name, 0.0)) * float(value)
    return float(score)
