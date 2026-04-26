"""
eda_subset.py
Focused EDA on the final 8-class subset for Plant Disease Classification.

Reads:
    data/metadata/plantvillage_subset_metadata.csv
    data/metadata/plantdoc_subset_metadata.csv
    configs/class_subset_v1.json

Outputs:
    outputs/figures/final_subset_class_distribution.png
    outputs/figures/plantvillage_sample_grid.png
    outputs/figures/plantdoc_sample_grid.png
    outputs/results/final_subset_eda_summary.md

Usage (from WSL):
    cp /mnt/host/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification/src/data/eda_subset.py ~/
    python ~/eda_subset.py
"""

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for WSL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/mnt/host/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification")

PV_META_CSV  = REPO_ROOT / "data" / "metadata" / "plantvillage_subset_metadata.csv"
PD_META_CSV  = REPO_ROOT / "data" / "metadata" / "plantdoc_subset_metadata.csv"
SUBSET_JSON  = REPO_ROOT / "configs" / "class_subset_v1.json"
FIGURES_DIR  = REPO_ROOT / "outputs" / "figures"
RESULTS_DIR  = REPO_ROOT / "outputs" / "results"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Count helpers
# ---------------------------------------------------------------------------

def count_by_label(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["unified_label"]] += 1
    return dict(sorted(counts.items()))


def group_by_label(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["unified_label"]].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def try_load_image(path: str) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception:
        return None


def pick_sample_images(
    groups: dict[str, list[dict]],
    labels: list[str],
    n: int = 3,
) -> dict[str, list[Image.Image]]:
    """
    For each label, attempt to load up to n sample images.
    Returns a dict of label -> list of PIL Images (may be shorter than n).
    """
    samples: dict[str, list[Image.Image]] = {}
    skip_total = 0
    for label in labels:
        rows = groups.get(label, [])
        loaded = []
        for row in rows:
            if len(loaded) >= n:
                break
            img = try_load_image(row["image_path"])
            if img is not None:
                loaded.append(img)
            else:
                skip_total += 1
        samples[label] = loaded
    if skip_total:
        print(f"  [!] Skipped {skip_total} unloadable images during sampling.")
    return samples


# ---------------------------------------------------------------------------
# Image property statistics
# ---------------------------------------------------------------------------

def image_size_stats(
    rows: list[dict],
    sample_limit: int = 200,
) -> dict:
    """
    Sample up to sample_limit images per dataset and collect size stats.
    Returns a dict with width/height min/max/mean.
    """
    widths, heights = [], []
    skip = 0
    for row in rows[:sample_limit]:
        img = try_load_image(row["image_path"])
        if img is None:
            skip += 1
            continue
        w, h = img.size
        widths.append(w)
        heights.append(h)

    if not widths:
        return {}

    return {
        "sampled":    len(widths),
        "skipped":    skip,
        "width_min":  min(widths),
        "width_max":  max(widths),
        "width_mean": round(sum(widths) / len(widths)),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_mean": round(sum(heights) / len(heights)),
    }


def extension_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        ext = Path(row["image_path"]).suffix.lower()
        counts[ext] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Plotting: class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    pv_counts: dict[str, int],
    pd_counts: dict[str, int],
    labels: list[str],
    out_path: Path,
) -> None:
    short_labels = [lbl.replace("_", "\n") for lbl in labels]
    pv_vals = [pv_counts.get(lbl, 0) for lbl in labels]
    pd_vals = [pd_counts.get(lbl, 0) for lbl in labels]

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar([i - width / 2 for i in x], pv_vals, width,
                   label="PlantVillage", color="#4C72B0", alpha=0.88)
    bars2 = ax.bar([i + width / 2 for i in x], pd_vals, width,
                   label="PlantDoc", color="#DD8452", alpha=0.88)

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylabel("Number of Images")
    ax.set_title("Final Subset - Class Distribution: PlantVillage vs PlantDoc",
                 fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate bar tops.
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=7.5, color="#4C72B0")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=7.5, color="#DD8452")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plotting: sample image grid
# ---------------------------------------------------------------------------

def plot_sample_grid(
    samples: dict[str, list[Image.Image]],
    labels: list[str],
    dataset_name: str,
    out_path: Path,
    n_cols: int = 3,
) -> None:
    n_rows = len(labels)
    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.0))
    fig.suptitle(f"{dataset_name} - Sample Images per Class",
                 fontsize=13, y=1.01)

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.55, wspace=0.1)

    for row_idx, label in enumerate(labels):
        imgs = samples.get(label, [])
        short = label.replace("_", " ")
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.axis("off")
            if col_idx < len(imgs):
                ax.imshow(imgs[col_idx])
            else:
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=8, color="#999999",
                        transform=ax.transAxes)
            if col_idx == 0:
                ax.set_ylabel(short, fontsize=7.5, rotation=0,
                              labelpad=70, va="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_eda_summary(
    labels: list[str],
    pv_counts: dict[str, int],
    pd_counts: dict[str, int],
    pv_stats: dict,
    pd_stats: dict,
    pv_exts: dict[str, int],
    pd_exts: dict[str, int],
    out_path: Path,
) -> None:
    pv_total = sum(pv_counts.values())
    pd_total = sum(pd_counts.values())

    lines = [
        "# Final Subset EDA Summary",
        "",
        "**Datasets:** PlantVillage (in-domain) vs PlantDoc (cross-dataset generalization test)",
        f"**Classes in subset:** {len(labels)}",
        "",
        "---",
        "",
        "## Class Distribution",
        "",
        f"| Class | PlantVillage | PlantDoc | PV / PD ratio |",
        "|---|---|---|---|",
    ]
    for lbl in labels:
        pv = pv_counts.get(lbl, 0)
        pd = pd_counts.get(lbl, 0)
        ratio = f"{pv / pd:.1f}x" if pd > 0 else "-"
        lines.append(f"| {lbl} | {pv} | {pd} | {ratio} |")

    lines += [
        "",
        f"**Total PlantVillage images:** {pv_total}",
        f"**Total PlantDoc images:** {pd_total}",
        f"**Overall PV / PD ratio:** {pv_total / pd_total:.1f}x",
        "",
        "PlantVillage is substantially larger for every class. "
        "PlantDoc counts range from ~79 to ~180 images per class, "
        "which is sufficient for evaluation but not for training.",
        "",
        "---",
        "",
        "## Image Size Statistics",
        f"*(based on first {pv_stats.get('sampled', 0)} PlantVillage "
        f"and {pd_stats.get('sampled', 0)} PlantDoc images sampled)*",
        "",
        "| Property | PlantVillage | PlantDoc |",
        "|---|---|---|",
    ]

    def stat_row(label, pv_key, pd_key):
        return f"| {label} | {pv_stats.get(pv_key, '-')} | {pd_stats.get(pd_key, '-')} |"

    lines += [
        stat_row("Width min (px)",  "width_min",   "width_min"),
        stat_row("Width max (px)",  "width_max",   "width_max"),
        stat_row("Width mean (px)", "width_mean",  "width_mean"),
        stat_row("Height min (px)", "height_min",  "height_min"),
        stat_row("Height max (px)", "height_max",  "height_max"),
        stat_row("Height mean (px)","height_mean", "height_mean"),
        "",
        "---",
        "",
        "## File Extensions",
        "",
        "**PlantVillage:** " + ", ".join(f"`{k}` ({v})" for k, v in pv_exts.items()),
        "",
        "**PlantDoc:** " + ", ".join(f"`{k}` ({v})" for k, v in pd_exts.items()),
        "",
        "---",
        "",
        "## Visual Style Differences",
        "",
        "- **PlantVillage** images are lab-collected against uniform or plain "
        "backgrounds. Lighting is controlled and consistent. Images are typically "
        "close-up leaf shots with minimal background clutter.",
        "- **PlantDoc** images are field-collected with natural backgrounds "
        "(soil, other plants, sky). Lighting varies significantly. Images often "
        "include multiple leaves, plant stems, or distracting background elements.",
        "- These differences make PlantDoc a realistic generalization test: "
        "a model that over-relies on background or lighting cues from PlantVillage "
        "will likely degrade on PlantDoc.",
        "",
        "---",
        "",
        "## Class Balance Notes",
        "",
        "- Within PlantVillage, classes are reasonably balanced for this subset "
        "(range: ~985 to ~2,127 images). No class is severely under-represented.",
        "- Within PlantDoc, class sizes are smaller and less balanced "
        "(range: ~79 to ~180 images). `tomato_early_blight` has the fewest images "
        "and should be monitored in evaluation.",
        "- The large PV/PD ratio (~12.6x overall) is expected: PlantVillage was "
        "purpose-built for training while PlantDoc was scraped for generalization testing.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Final Subset EDA ===\n")

    # Load data.
    print("Loading metadata...")
    pv_rows = load_csv(PV_META_CSV)
    pd_rows = load_csv(PD_META_CSV)
    subset  = load_json(SUBSET_JSON)
    labels  = [e["unified_label"] for e in subset]
    print(f"  PlantVillage rows : {len(pv_rows)}")
    print(f"  PlantDoc rows     : {len(pd_rows)}")
    print(f"  Classes           : {len(labels)}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # A. Count summaries.
    pv_counts = count_by_label(pv_rows)
    pd_counts = count_by_label(pd_rows)

    print("\n--- A. Count Summaries ---")
    print(f"\n  {'Class':<35} {'PlantVillage':>13} {'PlantDoc':>10}")
    print(f"  {'-' * 60}")
    for lbl in labels:
        print(f"  {lbl:<35} {pv_counts.get(lbl, 0):>13} {pd_counts.get(lbl, 0):>10}")
    print(f"  {'-' * 60}")
    print(f"  {'TOTAL':<35} {sum(pv_counts.values()):>13} {sum(pd_counts.values()):>10}")

    # B. Class distribution plot.
    print("\n--- B. Class Distribution Plot ---")
    plot_class_distribution(
        pv_counts, pd_counts, labels,
        FIGURES_DIR / "final_subset_class_distribution.png",
    )

    # C. Sample image grids.
    print("\n--- C. Sample Image Grids ---")
    pv_groups = group_by_label(pv_rows)
    pd_groups = group_by_label(pd_rows)

    print("  Loading PlantVillage samples...")
    pv_samples = pick_sample_images(pv_groups, labels, n=3)
    plot_sample_grid(pv_samples, labels, "PlantVillage",
                     FIGURES_DIR / "plantvillage_sample_grid.png")

    print("  Loading PlantDoc samples...")
    pd_samples = pick_sample_images(pd_groups, labels, n=3)
    plot_sample_grid(pd_samples, labels, "PlantDoc",
                     FIGURES_DIR / "plantdoc_sample_grid.png")

    # D. Image property summary.
    print("\n--- D. Image Property Statistics ---")
    print("  Sampling PlantVillage image sizes (first 200)...")
    pv_stats = image_size_stats(pv_rows, sample_limit=200)
    print("  Sampling PlantDoc image sizes (first 200)...")
    pd_stats = image_size_stats(pd_rows, sample_limit=200)

    pv_exts = extension_counts(pv_rows)
    pd_exts = extension_counts(pd_rows)

    print(f"\n  {'Property':<20} {'PlantVillage':>14} {'PlantDoc':>12}")
    print(f"  {'-' * 48}")
    for key, label in [
        ("width_mean",  "Width mean (px)"),
        ("width_min",   "Width min (px)"),
        ("width_max",   "Width max (px)"),
        ("height_mean", "Height mean (px)"),
        ("height_min",  "Height min (px)"),
        ("height_max",  "Height max (px)"),
    ]:
        print(f"  {label:<20} {str(pv_stats.get(key, '-')):>14} "
              f"{str(pd_stats.get(key, '-')):>12}")

    print(f"\n  PlantVillage extensions : {pv_exts}")
    print(f"  PlantDoc extensions     : {pd_exts}")

    # E. Markdown summary.
    print("\n--- E. EDA Summary Markdown ---")
    write_eda_summary(
        labels, pv_counts, pd_counts,
        pv_stats, pd_stats, pv_exts, pd_exts,
        RESULTS_DIR / "final_subset_eda_summary.md",
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
