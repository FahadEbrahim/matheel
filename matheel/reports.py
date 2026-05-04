import html
from pathlib import Path

import pandas as pd

from .leaderboard import leaderboard_payload


def benchmark_report_html(report, title=None, artifact_links=None):
    payload = leaderboard_payload(report)
    report_title = str(title or payload["metadata"].get("name") or "Matheel Benchmark Report")
    aggregate = pd.DataFrame(payload["aggregate"])
    per_dataset = pd.DataFrame(payload["per_dataset"])
    cards = payload.get("cards", {})
    links = _sanitize_artifact_links(artifact_links or {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(report_title)}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #ffffff; }}
    header {{ padding: 24px 28px; border-bottom: 1px solid #d1d5db; background: #f9fafb; }}
    main {{ padding: 20px 28px 32px; }}
    h1 {{ font-size: 1.6rem; margin: 0 0 8px; }}
    h2 {{ font-size: 1.15rem; margin: 28px 0 10px; }}
    h3 {{ font-size: 1rem; margin: 16px 0 6px; }}
    .meta {{ color: #4b5563; margin: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: #ffffff; }}
    .card dl {{ display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 5px 10px; margin: 0; }}
    dt {{ color: #4b5563; font-weight: 600; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-bottom: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .links ul {{ margin: 0; padding-left: 18px; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(report_title)}</h1>
    <p class="meta">schema={payload.get("schema_version", 1)}; seed={html.escape(str(payload["metadata"].get("seed") or ""))}</p>
  </header>
  <main>
    <section>
      <h2>Aggregate Ranking</h2>
      {_table_html(aggregate)}
    </section>
    <section>
      <h2>Per-Dataset Ranking</h2>
      {_table_html(per_dataset)}
    </section>
    <section>
      <h2>Dataset Cards</h2>
      <div class="grid">{_cards_html(cards.get("datasets", []))}</div>
    </section>
    <section>
      <h2>Algorithm Cards</h2>
      <div class="grid">{_cards_html(cards.get("algorithms", []))}</div>
    </section>
    <section class="links">
      <h2>Artifacts</h2>
      {_artifact_links_html(links)}
    </section>
  </main>
</body>
</html>
"""


def write_benchmark_report(report, output_path, title=None, artifact_links=None):
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        benchmark_report_html(report, title=title, artifact_links=artifact_links),
        encoding="utf-8",
    )
    return target


def _table_html(frame):
    if frame.empty:
        return "<p>No rows.</p>"
    return frame.to_html(index=False, escape=True)


def _cards_html(cards):
    if not cards:
        return '<p class="card">No cards.</p>'
    return "\n".join(_card_html(card) for card in cards)


def _card_html(card):
    rows = []
    for key in (
        "name",
        "card_type",
        "task_family",
        "dataset_kind",
        "algorithm_kind",
        "license",
        "package",
        "package_version",
    ):
        value = card.get(key)
        if value not in (None, ""):
            rows.append(f"<dt>{html.escape(key)}</dt><dd>{html.escape(str(value))}</dd>")
    fingerprint = card.get("fingerprint") or {}
    sha = fingerprint.get("sha256")
    if sha:
        rows.append(f"<dt>sha256</dt><dd><code>{html.escape(str(sha))}</code></dd>")
    counts = card.get("counts") or {}
    if counts:
        count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        rows.append(f"<dt>counts</dt><dd>{html.escape(count_text)}</dd>")
    if not rows:
        rows.append("<dt>card</dt><dd>empty</dd>")
    return f'<article class="card"><dl>{"".join(rows)}</dl></article>'


def _artifact_links_html(links):
    if not links:
        return "<p>No linked artifacts.</p>"
    items = "\n".join(
        f"<li>{html.escape(str(name))}: <code>{html.escape(str(path))}</code></li>"
        for name, path in sorted(links.items())
    )
    return f"<ul>{items}</ul>"


def _sanitize_artifact_links(links):
    sanitized = {}
    for name, value in dict(links).items():
        if value in (None, ""):
            continue
        sanitized[str(name)] = Path(str(value)).name
    return sanitized
