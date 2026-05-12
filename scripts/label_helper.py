from __future__ import annotations

import random
import shutil
from pathlib import Path

import pandas as pd
import typer

from src.data.integrity import deduplicate_paths
from src.settings.config import load_app_config


app = typer.Typer()


@app.command()
def main(
    n: int = typer.Option(200, help="Number of crops to sample."),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
) -> None:
    cfg = load_app_config()
    raw_dir = cfg.paths.data_dir
    labels_dir = cfg.paths.labels_dir
    labels_dir.mkdir(parents=True, exist_ok=True)

    all_crops = sorted(raw_dir.glob("*.png"))
    unique_crops, dedup_report = deduplicate_paths(all_crops)
    typer.echo(
        f"Deduplication: {dedup_report.n_input} -> {dedup_report.n_unique} "
        f"unique crops; {dedup_report.n_removed} duplicate copies skipped."
    )

    if n > len(unique_crops):
        typer.echo(f"Only {len(unique_crops)} unique crops available - sampling all.")
        n = len(unique_crops)

    rng = random.Random(seed)
    sample = sorted(rng.sample(unique_crops, n), key=lambda p: p.name)

    # Copy sampled crops into a dedicated folder so the file viewer shows just these.
    to_label_dir = labels_dir / "to_label"
    if to_label_dir.exists():
        shutil.rmtree(to_label_dir)
    to_label_dir.mkdir()
    for crop in sample:
        shutil.copy2(crop, to_label_dir / crop.name)

    # Pre-populate the CSV with crop_ids and empty label columns.
    csv_path = labels_dir / "validation.csv"
    df = pd.DataFrame(
        {
            "crop_id": [c.stem for c in sample],
            "should_keep": "",
            "violation_reason": "",
            "notes": "",
        }
    )
    df.to_csv(csv_path, index=False)

    typer.echo(f"OK {n} crops copied to {to_label_dir}")
    typer.echo(f"OK pre-populated CSV at {csv_path}")
    typer.echo("Open both side-by-side and fill in the labels.")


if __name__ == "__main__":
    app()
