"""Turn raw training logs into a clean Markdown summary.

The training scripts print tqdm progress bars (lots of noise) plus one
clean line per epoch of the form:

    head_only epoch 001/5 train_loss=1.0196 val_loss=0.9534 val_f1=0.5956
    fine_tune epoch 012/30 train_loss=0.1203 val_loss=0.3456 val_f1=0.8123

This works for both the binary and the three-class tasks (same line shape).
The script strips the tqdm noise and the dataset/config banners, keeping the
per-epoch lines and the run header, and writes a per-run section with an
epoch table.

Usage:
    # three-class (default output / title)
    python scripts/log_to_markdown.py results/logs/innovation_run_*.log

    # explicit output + title (used by the pipeline scripts)
    python scripts/log_to_markdown.py results/logs/binary_run_X.log \
        --output results/run_binary.md --title "Binary Training Log"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

EPOCH_RE = re.compile(
    r"(head_only|fine_tune|train) epoch (\d+)/(\d+) "
    r"train_loss=([\d.]+) val_loss=([\d.]+) val_f1=([\d.]+)"
)
HEADER_KEYS = ("Task", "Model", "Seed", "Class weights", "Selection")
RESPLIT_RE = re.compile(r"\[data\] After resplit -> (.*)")
TRAIN_BANNER_RE = re.compile(r">>> train (\S+)(?: 3class)? seed (\d+)")


def parse_log(path: Path) -> list[dict]:
    """Parse one log file into a list of per-run dicts.

    A single pipeline log contains several training runs (one per
    model/seed). Runs are delimited by the ">>> train <model> seed <n>"
    banner emitted by the pipeline scripts, or by the per-script
    "Model:/Seed:" header when a log holds a single run.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    runs: list[dict] = []
    current: dict | None = None

    def _new_run(model: str = "?", seed: str = "?") -> dict:
        return {
            "header": {"Model": model, "Seed": seed},
            "resplit": "",
            "epochs": [],
            "file": path.name,
        }

    for line in text.splitlines():
        banner = TRAIN_BANNER_RE.search(line)
        if banner:
            current = _new_run(banner.group(1), banner.group(2))
            runs.append(current)
            continue

        if current is None:
            current = _new_run()
            runs.append(current)

        for key in HEADER_KEYS:
            m = re.match(rf"\s*{re.escape(key)}\s*:\s*(.+)", line)
            if m:
                current["header"][key] = m.group(1).strip()
        rm = RESPLIT_RE.search(line)
        if rm:
            current["resplit"] = rm.group(1).strip()
        em = EPOCH_RE.search(line)
        if em:
            phase, ep, total, tl, vl, vf = em.groups()
            current["epochs"].append(
                (phase, int(ep), int(total), float(tl), float(vl), float(vf))
            )

    # Drop empty leading runs (e.g. banners before any epoch / header noise).
    return [r for r in runs if r["epochs"] or any(
        k not in ("Model", "Seed") for k in r["header"]
    )]


def render(runs: list[dict], title: str) -> str:
    out: list[str] = [f"# {title}\n"]
    out.append(
        "Captured per-epoch training output (tqdm progress bars stripped).\n"
    )
    for run in runs:
        h = run["header"]
        out.append(f"\n## {h.get('Model', '?')} — seed {h.get('Seed', '?')}\n")
        out.append("| Field | Value |")
        out.append("|---|---|")
        for k in ("Task", "Class weights", "Selection"):
            if k in h:
                out.append(f"| {k} | {h[k]} |")
        if run["resplit"]:
            out.append(f"| Resplit | {run['resplit']} |")
        out.append(f"| Source log | `{run['file']}` |")
        out.append("")

        if not run["epochs"]:
            out.append("_No completed epochs captured yet._\n")
            continue

        out.append("| Phase | Epoch | Train loss | Val loss | Val F1 |")
        out.append("|---|---|---|---|---|")
        best_f1 = -1.0
        best_row = None
        for phase, ep, total, tl, vl, vf in run["epochs"]:
            out.append(f"| {phase} | {ep}/{total} | {tl:.4f} | {vl:.4f} | {vf:.4f} |")
            # Best epoch is judged on the final/main phase (fine_tune for
            # transfer models, train for the custom CNN baseline).
            if phase in ("fine_tune", "train") and vf > best_f1:
                best_f1 = vf
                best_row = (ep, vf)
        out.append("")
        if best_row:
            out.append(
                f"**Best epoch by val F1:** epoch {best_row[0]} "
                f"(val F1 = {best_row[1]:.4f})\n"
            )
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Markdown training log from raw logs.")
    parser.add_argument("logs", nargs="+", type=Path, help="One or more raw training log files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/three_class_training_log.md"),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Three-Class Training Log",
        help="Document title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: list[dict] = []
    for log_path in args.logs:
        if log_path.exists():
            runs.extend(parse_log(log_path))
        else:
            print(f"WARNING: log not found, skipping: {log_path}")
    runs.sort(key=lambda r: (r["header"].get("Model", ""), str(r["header"].get("Seed", ""))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(runs, args.title), encoding="utf-8")
    print(f"Wrote {args.output}  ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
