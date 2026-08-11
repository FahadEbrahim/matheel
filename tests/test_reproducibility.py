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


def test_reproducibility_snapshot_redacts_nested_credentials(monkeypatch):
    monkeypatch.setattr(
        reproducibility,
        "_safe_package_version",
        lambda name: "1.2.3" if name in {"bert-score", "umap-learn"} else None,
    )

    snapshot = reproducibility.collect_reproducibility_snapshot(
        run_configs=[
            {
                "token": "SECRET_TOKEN",
                "headers": {"Authorization": "Bearer SECRET_HEADER"},
                "provider_api_key": "SECRET_API_KEY",
            }
        ]
    )

    config = snapshot["run_configs"][0]
    assert config["token"] == "<redacted>"
    assert config["headers"]["Authorization"] == "<redacted>"
    assert config["provider_api_key"] == "<redacted>"
    assert snapshot["packages"]["bert-score"] == "1.2.3"
    assert snapshot["packages"]["umap-learn"] == "1.2.3"
