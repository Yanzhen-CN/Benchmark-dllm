from __future__ import annotations

from pathlib import Path

from ...metrics.sudoku_editing import compute_sudoku_editing_metrics


def render_editing_dataset(*, dataset_name: str, records, out_dir: Path, **_):
    if dataset_name != "editable_sudoku4":
        return {}
    relevant = []
    for sample, generation in records:
        if sample.meta.get("editable_sudoku"):
            relevant.append(compute_sudoku_editing_metrics(sample, generation, size=4))
    if not relevant: return {}
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "editing.png"
    labels = ["opportunities", "replacements", "corrections", "harmful"]
    keys = ["editing_opportunities", "editing_replacements", "editing_corrections",
            "editing_harmful_replacements"]
    values = [sum(float(item.get(key, 0.0)) for item in relevant) for key in keys]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bars = ax.bar(labels, values, color=["#6B7280", "#2563EB", "#15803D", "#B91C1C"])
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_ylabel("Cell events")
    ax.set_title("LLaDA2.1 Editable Sudoku")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return {"editing": str(path)}
