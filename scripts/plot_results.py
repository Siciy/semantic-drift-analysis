import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from experiment import LABELS, run_lock


SINGLE = ["qwen_only", "smollm_only"]
ALTERNATING = ["alternate_qwen_first", "alternate_smollm_first"]
CHAIN_ROUND = ["passage_id", "schedule", "round"]


def round_summary(scores, group_by):
    return scores.groupby(group_by, sort=False, as_index=False).agg(
        n_expected=("status", "size"),
        n_reached=("reached", "sum"),
        n_valid=("valid", "sum"),
        n_invalid=("invalid", "sum"),
        n_blocked=("blocked", "sum"),
        n_pending=("pending", "sum"),
        n_unchanged=("is_unchanged", "sum"),
        n_commentary=("is_commentary", "sum"),
        n_truncated=("is_truncated", "sum"),
        n_encoder_limit=("over_encoder_limit", "sum"),
        n_similarity=("similarity", "count"),
        mean_similarity=("similarity", "mean"),
        n_length=("length_ratio", "count"),
        mean_length_ratio=("length_ratio", "mean"),
    )


def load_retention(run_dir, annotations_path):
    key = pd.read_csv(run_dir / "annotation_key.csv", keep_default_na=False)
    annotations = pd.read_csv(annotations_path, keep_default_na=False)
    labels = annotations[["annotation_id", "label"]].copy()
    labels["label"] = labels["label"].astype(str).str.strip().str.lower()
    unknown = set(labels["label"]) - LABELS - {""}
    if unknown:
        raise ValueError(f"Unknown claim labels: {sorted(unknown)}")
    if set(labels["annotation_id"]) - set(key["annotation_id"]):
        raise ValueError("The annotation CSV contains IDs outside this run")

    rows = key.merge(labels, on="annotation_id", how="left", validate="one_to_one")
    rows["reviewed"] = rows["label"].isin(LABELS)
    rows["preserved"] = rows["label"].eq("preserved")
    rows["contradicted"] = rows["label"].eq("contradicted")
    rows["missing"] = rows["label"].eq("missing")
    retention = rows.groupby(CHAIN_ROUND, as_index=False).agg(
        n_claims=("claim_id", "size"), n_reviewed=("reviewed", "sum"),
        n_preserved=("preserved", "sum"), n_contradicted=("contradicted", "sum"), n_missing=("missing", "sum"),
    )
    complete = retention["n_reviewed"].eq(retention["n_claims"])
    retention["retention"] = (retention["n_preserved"] / retention["n_claims"]).where(complete)
    return retention


def paired_comparison(scores):
    final = scores.loc[scores["round"].eq(scores["final_round"])]
    schedules = SINGLE + ALTERNATING
    rows = []
    for metric in ("similarity", "length_ratio", "retention"):
        values = final.pivot(index="passage_id", columns="schedule", values=metric)
        complete = values.reindex(columns=schedules).dropna()
        solo = complete[SINGLE].mean(axis=1)
        mixed = complete[ALTERNATING].mean(axis=1)
        rows.append({
            "metric": metric, "n_passages_expected": len(values), "n_passages_paired": len(complete),
            "mean_single_model": solo.mean(), "mean_alternating": mixed.mean(),
            "mean_paired_difference_alternating_minus_single": (mixed - solo).mean(),
        })
    return pd.DataFrame(rows)


def timing_summary(scores):
    attempts = scores[scores["reached"] & scores["round"].gt(0)].copy()
    attempts["warm_seconds"] = attempts["generation_seconds"].where(attempts["warmup"].eq(False))
    result = attempts.groupby("model_id", as_index=False).agg(
        n_reached=("reached", "size"), n_valid=("valid", "sum"), n_truncated=("is_truncated", "sum"),
        n_warmed_up=("warm_seconds", "count"), mean_warmed_up_seconds=("warm_seconds", "mean"),
        total_model_load_seconds=("model_load_seconds", "sum"),
    )
    result["truncation_rate"] = result["n_truncated"] / result["n_reached"]
    return result


def plot_rounds(frame, metric, target, title, y_label, *, percentage=False, baseline=None):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    if baseline is not None:
        ax.axhline(baseline, color="#777777", linestyle=":", linewidth=1)
    for label, curve in frame.groupby("schedule_label", sort=False):
        if curve[metric].notna().any():
            ax.plot(curve["round"], curve[metric], marker="o", markersize=4, linewidth=1.7, label=label)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No scored passages", ha="center", va="center", transform=ax.transAxes)
    ax.set(title=title, xlabel="Rewrite round", ylabel=y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    if metric == "mean_retention":
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(sorted(frame["round"].unique()))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def generate_reports(run_dir, annotations_path):
    scores = pd.read_csv(run_dir / "scores.csv")
    scores["reached"] = scores["status"].isin({"original", "valid", "invalid"})
    scores["valid"] = scores["status"].isin({"original", "valid"})
    for column in ("invalid", "blocked", "pending"):
        scores[column] = scores["status"].eq(column)
    for column in ("unchanged", "commentary", "truncated"):
        scores[f"is_{column}"] = scores[column].eq(True)
    scores["over_encoder_limit"] = scores["similarity_status"].eq("encoder_limit")
    group_by = ["schedule", "schedule_label", "round"]
    summary = round_summary(scores, group_by)
    summary.to_csv(run_dir / "round_summary.csv", index=False)
    round_summary(scores, ["group", *group_by]).to_csv(run_dir / "group_round_summary.csv", index=False)
    timing_summary(scores).to_csv(run_dir / "timing_summary.csv", index=False)

    retention = load_retention(run_dir, annotations_path)
    retention.to_csv(run_dir / "claim_retention.csv", index=False)
    scores = scores.merge(retention[CHAIN_ROUND + ["retention"]], on=CHAIN_ROUND, how="left", validate="one_to_one")
    if scores["split"].eq("main").all():
        paired_comparison(scores).to_csv(run_dir / "paired_final_summary.csv", index=False)
    sampled = scores["retention_checkpoint"]
    final = scores["round"].eq(scores["final_round"])
    retention_rows = pd.concat([scores[sampled].assign(cohort="checkpoint_subset"), scores[final].assign(cohort="all_final")])
    retention_summary = retention_rows.groupby(["cohort", *group_by], as_index=False, sort=False).agg(
        n_expected=("status", "size"), n_valid=("valid", "sum"),
        n_annotated=("retention", "count"), mean_retention=("retention", "mean"),
    )
    retention_summary.to_csv(run_dir / "retention_summary.csv", index=False)

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    last_reached = scores.loc[scores["reached"], "round"].max()
    plot_data = summary[summary["round"].le(last_reached)]
    plot_rounds(plot_data, "mean_similarity", figures / "similarity.png",
                "Semantic similarity to the original", "Mean cosine similarity")
    plot_rounds(plot_data, "mean_length_ratio", figures / "length.png",
                "Passage length across rewrites", "Mean length relative to original", percentage=True, baseline=1.0)
    checkpoints = retention_summary[retention_summary["cohort"].eq("checkpoint_subset")]
    if checkpoints["mean_retention"].notna().any():
        plot_rounds(checkpoints, "mean_retention", figures / "retention.png",
                    "Information retention at checkpoints", "Mean share of claims preserved", percentage=True)
    else:
        (figures / "retention.png").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Plot scores and summarize coverage from a saved run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--annotations", type=Path, help="Use an adjudicated CSV in place of the default template.")
    args = parser.parse_args()
    with run_lock(args.run_dir):
        generate_reports(args.run_dir, args.annotations or args.run_dir / "annotations.csv")
    print(f"Wrote figures and summary tables in {args.run_dir}")


if __name__ == "__main__":
    main()
