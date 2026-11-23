"""Extract typed references from a user-provided BigCloneEval v1.1 H2 database."""

import csv
import os
import shutil
import tempfile
from pathlib import Path

from ..clone_discovery import CloneReference, SourceSpan, write_clone_references
from .common import DetectorError, file_sha256, positive_number, required_file, run_local

VERSION = "1.1"
H2_SHA256 = "6096b735ac4af70fa730e4387582c599bcc347548fd2a5c50c2e7a0b52d5d287"
_RECOVERY_CODE = "The database is read only [90097-176]"
_REFERENCE_SQL = """
    SELECT a.type, a.name, a.startline, a.endline,
           b.type, b.name, b.startline, b.endline,
           c.syntactic_type, c.functionality_id, c.function_id_one, c.function_id_two,
           CASE WHEN a.project IS NULL OR b.project IS NULL THEN 0 WHEN a.project = b.project THEN 1 ELSE 2 END,
           c.min_size, c.min_pretty_size, c.min_tokens, c.similarity_line, c.similarity_token
      FROM clones c
      JOIN functions a ON c.function_id_one = a.id
      JOIN functions b ON c.function_id_two = b.id
     WHERE c.internal = FALSE AND c.syntactic_type IN (1, 2, 3)
     ORDER BY c.functionality_id, c.function_id_one, c.function_id_two
"""


def _clone_type(syntactic, line_similarity, token_similarity):
    if syntactic == "1":
        return "type1"
    if syntactic == "2":
        return "type2_consistent"
    lower = min(float(line_similarity), float(token_similarity))
    for threshold, clone_type in (
        (1.0, "type2_blind"), (0.9, "type3_very_strong"), (0.7, "type3_strong"),
        (0.5, "type3_moderate"),
    ):
        if lower >= threshold:
            return clone_type
    return "weak_type3_type4"


def _read_native_references(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        if len(next(rows, [])) != 18:
            raise DetectorError("BigCloneEval reference export has an unexpected schema.")
        for row in rows:
            if len(row) != 18 or row[8] not in {"1", "2", "3"} or row[12] not in {"0", "1", "2"}:
                raise DetectorError("BigCloneEval reference export contains a malformed row.")
            left_type, left_name, left_start, left_end, right_type, right_name, right_start, right_end = row[:8]
            syntactic, functionality, left_func, right_func, scope, size, pretty_size, tokens, line, token = row[8:]
            if left_type not in {"selected", "default", "sample"} or right_type not in {"selected", "default", "sample"}:
                raise DetectorError("BigCloneEval reference export has an unknown IJaDataset subdirectory.")
            yield CloneReference(
                SourceSpan(f"{left_type}/{left_name}", left_start, left_end),
                SourceSpan(f"{right_type}/{right_name}", right_start, right_end),
                clone_type=_clone_type(syntactic, line, token),
                project_scope=("unknown", "intra_project", "inter_project")[int(scope)],
                pretty_lines=pretty_size, tokens=tokens, line_similarity=line or None, token_similarity=token or None,
                reference_id=f"{functionality}:{min(int(left_func), int(right_func))}:{max(int(left_func), int(right_func))}",
                original_lines=size,
            )


def _export_h2(root, database, script, raw, java, jar, heap_mb, timeout_seconds, max_result_bytes):
    database_name = database.name.removesuffix(".h2.db")

    def export(driver_path, *, read_only):
        mode = ";ACCESS_MODE_DATA=r" if read_only else ""
        run_local(
            [java, f"-Xmx{heap_mb}m", "-cp", jar, "org.h2.tools.RunScript", "-url",
             f"jdbc:h2:{driver_path};IFEXISTS=TRUE{mode}", "-user", "sa", "-password", "", "-script", script],
            cwd=root, timeout_seconds=timeout_seconds, result_paths=(raw,), max_result_bytes=max_result_bytes,
        )

    try:
        export(database.with_name(database_name), read_only=True)
    except DetectorError:
        log_path = root / "process.log"
        if not log_path.is_file():
            raise
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if _RECOVERY_CODE not in log or "PageLog.recover" not in log:
            raise
        # H2 must write to archives taken before a clean database shutdown.
        # Its page recovery may never change the user-provided benchmark bytes.
        try:
            shutil.copyfile(database, root / database.name)
        except OSError as exc:
            raise DetectorError("Could not stage a private H2 recovery copy in temporary storage.") from exc
        export(root / database_name, read_only=False)


def extract_references(
    database_path, output_path, *, java_path, h2_jar, max_references=None,
    timeout_seconds=600, max_result_bytes=2_000_000_000, heap_mb=2048,
):
    """Export known-pair references with BigCloneEval's BOTH strata and native size minima.

    The pinned H2 jar comes from the upstream BigCloneEval v1.1 tree. The H2
    database is opened read-only. An archive that needs H2 log recovery is
    recovered in private temporary storage. A limit selects the first
    references in a defined ID order and describes a separate scope.
    """
    java = required_file(java_path, "Java executable")
    jar = required_file(h2_jar, "BigCloneEval H2 JAR")
    if file_sha256(jar) != H2_SHA256:
        raise DetectorError("BigCloneEval requires the H2 1.3.176 JAR pinned by v1.1.")
    database = required_file(database_path, "BigCloneEval local H2 database")
    if not database.name.endswith(".h2.db") or ";" in str(database):
        raise DetectorError("BigCloneEval requires a local .h2.db database path without semicolons.")
    heap_mb = positive_number(heap_mb, "heap_mb", integer=True)
    sql = " ".join(_REFERENCE_SQL.split())
    if max_references is not None:
        max_references = positive_number(max_references, "max_references", integer=True)
        sql += f" LIMIT {max_references}"
    output = Path(output_path)
    if output.resolve() in {database, jar, java}:
        raise DetectorError("BigCloneEval output must be distinct from its supplied inputs.")
    with tempfile.TemporaryDirectory(prefix="matheel-bce-") as directory:
        root = Path(directory)
        script, raw = root / "extract.sql", root / "raw.csv"
        script.write_text(f"CALL CSVWRITE('raw.csv', '{sql}');\n", encoding="utf-8")
        _export_h2(root, database, script, raw, java, jar, heap_mb, timeout_seconds, max_result_bytes)
        # Publish only a complete normalized manifest. Staging in the same parent
        # allows the atomic rename on platforms with multiple mounted volumes.
        handle = tempfile.NamedTemporaryFile(dir=output.parent, prefix=".matheel-bce-", delete=False)
        handle.close()
        pending = Path(handle.name)
        try:
            write_clone_references(pending, _read_native_references(raw))
            os.replace(pending, output)
        finally:
            pending.unlink(missing_ok=True)
    return output
