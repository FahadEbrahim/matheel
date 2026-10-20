"""Controlled robustness diagnostics; these cases do not infer plagiarism labels."""

import ast
from dataclasses import asdict, dataclass
import keyword
import random

from ._experiments import (
    content_id, experiment_metadata, experiment_runs, score_experiment_pairs, text_hash,
    write_experiment_artifacts,
)


@dataclass(frozen=True)
class RobustnessCase:
    case_id: str
    transformation: str
    baseline_reference: str
    baseline_candidate: str
    reference: str
    candidate: str
    provenance: dict


def _ast(text):
    return ast.dump(ast.parse(text, type_comments=True), include_attributes=False)


def _case(name, before_left, before_right, after_left, after_right, *, seed, validation, **params):
    provenance = {
        "seed": seed, "parameters": params, "validation": validation,
        "source_sha256": {
            "baseline_reference": text_hash(before_left),
            "baseline_candidate": text_hash(before_right),
            "reference": text_hash(after_left), "candidate": text_hash(after_right),
        },
        "relation": "controlled diagnostic; no dataset labels inferred",
    }
    return RobustnessCase(
        case_id=content_id({"transformation": name, **provenance}), transformation=name,
        baseline_reference=before_left, baseline_candidate=before_right,
        reference=after_left, candidate=after_right, provenance=provenance,
    )


def python_robustness_case(
    source, transformation, *, seed=0, identifier_map=None, validator=None, validation_name=None,
):
    """Transform Python text without running it.

    Formatting/comments are checked for AST equality, not arbitrary observational
    equivalence. Renaming changes binding-sensitive syntax and therefore requires
    an explicit caller validator and a recorded description of its assertions.
    This function only calls that supplied validator; it never executes source.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    tree = ast.parse(source, type_comments=True)
    params = {}
    if transformation == "formatting":
        candidate = ast.unparse(tree) + "\n"
    elif transformation == "comments":
        candidate = f"# Controlled diagnostic comment; seed={seed}\n{source}\n# End diagnostic\n"
    elif transformation == "identifier_renaming":
        mapping = dict(identifier_map or {})
        if not mapping:
            raise ValueError("Identifier renaming requires a nonempty identifier_map.")
        if any(not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name)
               for pair in mapping.items() for name in pair):
            raise ValueError("Identifier mappings must contain valid Python identifiers.")
        if validator is None or not callable(validator) or not str(validation_name or "").strip():
            raise ValueError("Renaming requires a validator and a named fixture assertion.")
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
        if not set(mapping).issubset(names):
            raise ValueError("Every renamed identifier must occur in the source.")

        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                node.id = mapping.get(node.id, node.id)
                return node

            def visit_arg(self, node):
                node.arg = mapping.get(node.arg, node.arg)
                return node

        candidate = ast.unparse(ast.fix_missing_locations(Rename().visit(tree))) + "\n"
        if validator(source, candidate) is not True:
            raise ValueError("The supplied renaming fixture assertions did not pass.")
        validation = {"kind": "caller_fixture_assertions", "assertions": str(validation_name),
                      "scope": "Only the declared fixtures; not a general semantic proof"}
        params["identifier_map"] = mapping
    else:
        raise ValueError("Supported Python transformations: formatting, comments, identifier_renaming.")
    if transformation != "identifier_renaming":
        if _ast(source) != _ast(candidate):
            raise ValueError("Transformation changed the Python AST.")
        validation = {"kind": "python_ast_equal",
                      "scope": "AST equality; source introspection and line numbers can differ"}
    return _case(transformation, source, source, source, candidate,
                 seed=seed, validation=validation, **params)


_SOURCE = '''def sum_positive( values ) :
    """Sum strictly positive values."""
    total=0
    for value in values:
        if value>0:
            total+=value
    return total

def range_width(values):
    if not values:
        return 0
    return max(values) - min(values)
'''
_OTHER = '''def count_even(values):
    count = 0
    for number in values:
        if number % 2 == 0:
            count += 1
    return count
'''
_STARTER = '''# Shared instructor-provided helper; diagnostic fixture only.
def provided_scale(values):
    return [entry * 2 for entry in values]

'''


def _assert_known_fixture(text, *, positive=False, even=False, width=False, starter=False):
    # Only called on the fixed source constants above and their generated ASTs.
    # User-supplied source is never routed through this execution helper.
    namespace = {}
    exec(compile(text, "<matheel-controlled-fixture>", "exec"), namespace)
    samples = [([], 0, 0, 0), ([-3, 2, 5, 0], 7, 2, 8), ([4, 4], 8, 2, 0)]
    for values, expected_sum, expected_even, expected_width in samples:
        if positive and namespace["sum_positive"](values) != expected_sum:
            raise AssertionError("sum_positive fixture failed")
        if even and namespace["count_even"](values) != expected_even:
            raise AssertionError("count_even fixture failed")
        if width and namespace["range_width"](values) != expected_width:
            raise AssertionError("range_width fixture failed")
        if starter and namespace["provided_scale"](values) != [value * 2 for value in values]:
            raise AssertionError("provided_scale fixture failed")
    return True


def controlled_robustness_cases(seed=0):
    """Six deterministic Python diagnostics with explicit known-fixture checks.

    Partial copying and common starter code deliberately change the compared
    content. They are diagnostic configurations, never ground-truth positives.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    _assert_known_fixture(_SOURCE, positive=True, width=True)
    _assert_known_fixture(_OTHER, even=True)
    cases = [_case("identity", _SOURCE, _SOURCE, _SOURCE, _SOURCE, seed=seed,
                   validation={"kind": "fixed_fixture_assertions", "assertions": "sum/range examples"})]
    for transformation in ("formatting", "comments"):
        case = python_robustness_case(_SOURCE, transformation, seed=seed)
        _assert_known_fixture(case.candidate, positive=True, width=True)
        cases.append(case)
    suffix = random.Random(seed).randrange(1_000_000)

    def validate_rename(before, after):
        return (_assert_known_fixture(before, positive=True, width=True)
                and _assert_known_fixture(after, positive=True, width=True))

    cases.append(python_robustness_case(
        _SOURCE, "identifier_renaming", seed=seed,
        identifier_map={"total": f"aggregate_{suffix}", "value": f"element_{suffix}"},
        validator=validate_rename,
        validation_name="sum_positive and range_width: empty, mixed-sign, repeated-value fixtures",
    ))
    copied = ast.get_source_segment(_SOURCE, ast.parse(_SOURCE).body[0]) + "\n"
    partially_copied = _OTHER + "\n" + copied
    _assert_known_fixture(partially_copied, positive=True, even=True)
    cases.append(_case(
        "partial_copy", _SOURCE, _OTHER, _SOURCE, partially_copied, seed=seed,
        validation={"kind": "fixed_fixture_assertions", "assertions": "count_even/sum_positive examples"},
        copied_source_sha256=text_hash(copied), copied_utf8_bytes=len(copied.encode("utf-8")),
        copied_source_fraction=len(copied.encode("utf-8")) / len(_SOURCE.encode("utf-8")),
        description="Append one verbatim function from reference to a different controlled program",
    ))
    _assert_known_fixture(_STARTER + _SOURCE, positive=True, width=True, starter=True)
    _assert_known_fixture(_STARTER + _OTHER, even=True, starter=True)
    cases.append(_case(
        "shared_starter_code", _SOURCE, _OTHER, _STARTER + _SOURCE, _STARTER + _OTHER,
        seed=seed, validation={"kind": "fixed_fixture_assertions",
                              "assertions": "Original program and shared helper behavior fixtures"},
        starter_sha256=text_hash(_STARTER), starter_utf8_bytes=len(_STARTER.encode("utf-8")),
        description="Prepend the same provided helper to both otherwise distinct fixtures",
    ))
    return cases


def run_robustness_experiment(runs, *, cases=None, seed=0, output_dir=None):
    runs = experiment_runs(runs)
    cases = list(controlled_robustness_cases(seed) if cases is None else cases)
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique case IDs.")
    records = [asdict(case) for case in cases]
    pairs = []
    for case in cases:
        if not case.provenance.get("validation"):
            raise ValueError("Every diagnostic case requires explicit validation provenance.")
        expected_hashes = {role: text_hash(getattr(case, role)) for role in (
            "baseline_reference", "baseline_candidate", "reference", "candidate",
        )}
        if expected_hashes != case.provenance.get("source_sha256"):
            raise ValueError("Case source hashes do not match its recorded provenance.")
        if case.case_id != content_id({"transformation": case.transformation, **case.provenance}):
            raise ValueError("Case identity does not match its recorded provenance.")
        pairs.extend([
            (case.case_id + ":baseline", case.baseline_reference, case.baseline_candidate),
            (case.case_id + ":transformed", case.reference, case.candidate),
        ])
    report = {**experiment_metadata("robustness", runs, records), "cases": records, "scores": [],
              "interpretation": "Score changes on controlled fixtures, not plagiarism classifications."}
    for run in runs:
        error = None
        try:
            scores = score_experiment_pairs(pairs, run)
        except Exception as exc:
            scores, error = {}, f"{type(exc).__name__}: {exc}"
        for case in cases:
            before = scores.get(case.case_id + ":baseline")
            after = scores.get(case.case_id + ":transformed")
            report["scores"].append({
                "run_name": run["run_name"], "run_id": content_id(run), "case_id": case.case_id,
                "transformation": case.transformation, "baseline_score": before,
                "transformed_score": after, "score_delta": after - before if error is None else None,
                "status": "error" if error else "ok", "error": error,
            })
    if output_dir is not None:
        return report, write_experiment_artifacts(report, output_dir)
    return report, {}
