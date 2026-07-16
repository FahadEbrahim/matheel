import os

import matheel.reproducibility as reproducibility


def test_directory_fingerprint_is_independent_of_walk_order(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a" / "one.py").write_text("print(1)\n", encoding="utf-8")
    (root / "b" / "two.py").write_text("print(2)\n", encoding="utf-8")
    walk_rows = list(os.walk(root))

    expected = reproducibility.fingerprint_source(root)
    monkeypatch.setattr(
        reproducibility.os,
        "walk",
        lambda _root: iter(reversed(walk_rows)),
    )

    assert reproducibility.fingerprint_source(root) == expected
