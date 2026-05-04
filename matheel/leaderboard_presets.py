from copy import deepcopy


_LEADERBOARD_ALGORITHM_PRESETS = {
    "Balanced": {
        "name": "Balanced",
        "description": "Embedding-heavy default with a lexical stabilizer.",
        "similarity_options": {
            "feature_weights": {"semantic": 0.7, "levenshtein": 0.3},
            "code_metric": "codebleu",
            "code_metric_weight": 0.0,
        },
    },
    "Lexical Only": {
        "name": "Lexical Only",
        "description": "Offline lexical baseline combining edit distance, Winnowing, and GST.",
        "similarity_options": {
            "feature_weights": {"levenshtein": 0.5, "winnowing": 0.25, "gst": 0.25},
            "code_metric": "codebleu",
            "code_metric_weight": 0.0,
        },
    },
    "Embedding Only": {
        "name": "Embedding Only",
        "description": "Semantic embedding score only.",
        "similarity_options": {
            "feature_weights": {"semantic": 1.0},
            "code_metric": "codebleu",
            "code_metric_weight": 0.0,
        },
    },
    "Code-Aware": {
        "name": "Code-Aware",
        "description": "Embedding, edit distance, and code metric blend.",
        "similarity_options": {
            "feature_weights": {"semantic": 0.5, "levenshtein": 0.25, "code_metric": 0.25},
            "code_metric": "codebleu",
            "code_metric_weight": 0.25,
        },
    },
    "Jaro-Winkler": {
        "name": "Jaro-Winkler",
        "description": "String-similarity baseline focused on short edits and reordered names.",
        "similarity_options": {
            "feature_weights": {"jaro_winkler": 1.0},
            "code_metric": "codebleu",
            "code_metric_weight": 0.0,
        },
    },
    "Winnowing": {
        "name": "Winnowing",
        "description": "Token fingerprint baseline for shared local fragments.",
        "similarity_options": {
            "feature_weights": {"winnowing": 1.0},
            "code_metric": "codebleu",
            "code_metric_weight": 0.0,
        },
    },
    "GST": {
        "name": "GST",
        "description": "Greedy String Tiling token-overlap baseline.",
        "similarity_options": {
            "feature_weights": {"gst": 1.0},
            "code_metric": "codebleu",
            "code_metric_weight": 0.0,
        },
    },
    "CodeBLEU": {
        "name": "CodeBLEU",
        "description": "CodeBLEU-only code-aware baseline.",
        "similarity_options": {
            "feature_weights": {"code_metric": 1.0},
            "code_metric": "codebleu",
            "code_metric_weight": 1.0,
        },
    },
}


def available_leaderboard_algorithm_presets():
    return tuple(_LEADERBOARD_ALGORITHM_PRESETS)


def get_leaderboard_algorithm_preset(name):
    key = _resolve_preset_key(name)
    return deepcopy(_LEADERBOARD_ALGORITHM_PRESETS[key])


def register_leaderboard_algorithm_preset(name, preset, overwrite=False):
    key = _normalize_preset_name(name)
    if not overwrite and key in _LEADERBOARD_ALGORITHM_PRESETS:
        raise ValueError(f"Leaderboard algorithm preset already exists: {key}")
    payload = _normalize_preset_payload(key, preset)
    _LEADERBOARD_ALGORITHM_PRESETS[key] = payload
    return deepcopy(payload)


def leaderboard_algorithm_preset_configs(names=None):
    selected = available_leaderboard_algorithm_presets() if names is None else tuple(names)
    configs = []
    for name in selected:
        preset = get_leaderboard_algorithm_preset(name)
        configs.append({"name": preset["name"], **deepcopy(preset["similarity_options"])})
    return configs


def _resolve_preset_key(name):
    requested = _normalize_preset_name(name)
    for key in _LEADERBOARD_ALGORITHM_PRESETS:
        if key.lower() == requested.lower():
            return key
    supported = ", ".join(available_leaderboard_algorithm_presets())
    raise KeyError(f"Unknown leaderboard algorithm preset: {name}. Supported presets: {supported}")


def _normalize_preset_name(name):
    text = str(name or "").strip()
    if not text:
        raise ValueError("Leaderboard algorithm preset name must be non-empty.")
    return text


def _normalize_preset_payload(name, preset):
    if not isinstance(preset, dict):
        raise ValueError("Leaderboard algorithm preset must be a mapping.")
    options = dict(preset.get("similarity_options") or preset.get("options") or {})
    if not options:
        raise ValueError("Leaderboard algorithm preset must define similarity_options.")
    return {
        "name": str(preset.get("name") or name),
        "description": str(preset.get("description") or ""),
        "similarity_options": deepcopy(options),
    }
