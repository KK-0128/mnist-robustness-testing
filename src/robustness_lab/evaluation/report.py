"""Result persistence: timestamped JSON / CSV / Markdown outputs."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def save_results(
    results: dict,
    output_dir: str | Path,
    tag: str = "",
) -> Path:
    """Write ``results`` as ``<tag>_<timestamp>.{json,csv,md}`` and return the JSON path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stem = f"{tag}_{stamp}" if tag else stamp

    json_path = out / f"{stem}.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = out / f"{stem}.csv"
    _write_csv(csv_path, results)

    md_path = out / f"{stem}.md"
    md_path.write_text(_to_markdown(results), encoding="utf-8")

    log.info("Saved results: %s (json), %s (csv), %s (md)", json_path, csv_path, md_path)
    return json_path


def _write_csv(path: Path, results: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "epsilon", "accuracy"])
        writer.writerow(["clean", "", results.get("clean_accuracy")])
        for eps, acc in (results.get("fgsm_accuracy") or {}).items():
            writer.writerow(["fgsm", eps, acc])
        for eps, acc in (results.get("pgd_accuracy") or {}).items():
            writer.writerow(["pgd", eps, acc])
        if results.get("cw_accuracy") is not None:
            writer.writerow(["cw", "", results["cw_accuracy"]])
        if results.get("smoothed_accuracy") is not None:
            writer.writerow(["smoothed", "", results["smoothed_accuracy"]])
        if results.get("certified_fraction_0_5") is not None:
            writer.writerow(
                ["certified_fraction(>=0.5)", "", results["certified_fraction_0_5"]]
            )


def _to_markdown(results: dict) -> str:
    lines = [
        "# Robustness Benchmark Report",
        "",
        f"- defense: `{results.get('defense')}`",
        f"- device: `{results.get('device')}`",
        f"- clean accuracy: {results.get('clean_accuracy'):.4f}",
    ]
    if results.get("cw_accuracy") is not None:
        lines.append(f"- CW L2 accuracy: {results['cw_accuracy']:.4f}")
    if results.get("smoothed_accuracy") is not None:
        lines.append(f"- smoothed accuracy: {results['smoothed_accuracy']:.4f}")
    if results.get("certified_fraction_0_5") is not None:
        lines.append(
            f"- certified fraction (radius >= 0.5): {results['certified_fraction_0_5']:.4f}"
        )
    lines += ["", "| epsilon | FGSM acc | PGD acc |", "|---|---|---|"]
    eps_keys = sorted(
        set(results.get("fgsm_accuracy") or {}) | set(results.get("pgd_accuracy") or {})
    )
    for eps in eps_keys:
        f_acc = (results.get("fgsm_accuracy") or {}).get(eps, "N/A")
        p_acc = (results.get("pgd_accuracy") or {}).get(eps, "N/A")
        f_acc = f"{f_acc:.4f}" if isinstance(f_acc, float) else f_acc
        p_acc = f"{p_acc:.4f}" if isinstance(p_acc, float) else p_acc
        lines.append(f"| {eps} | {f_acc} | {p_acc} |")
    return "\n".join(lines) + "\n"
