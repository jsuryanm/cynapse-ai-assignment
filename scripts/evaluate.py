from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.data.labels import load_validation_csv
from src.logger.custom_logger import configure_logger, logger
from src.pipeline.evaluation_pipeline import EvaluationResult, evaluate_run
from src.settings.config import load_app_config


app = typer.Typer(help="Evaluate a curation pipeline run against ground-truth labels.")
console = Console()


def _find_latest_run(runs_dir: Path) -> Path:
    """Return the most recently modified run folder under runs_dir."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run folders found in {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _print_overall(result: EvaluationResult) -> None:
    cm = result.confusion
    console.print()
    console.rule("[bold]Overall Performance")

    table = Table(show_header=False, box=None)
    table.add_column("metric", style="cyan")
    table.add_column("value", style="magenta", justify="right")
    table.add_row("Crops evaluated", f"{result.n_evaluated}")
    table.add_row("Labels without decisions", f"{result.n_unmatched_labels}")
    table.add_row("", "")
    table.add_row("True positives (correctly kept)", f"{cm.tp}")
    table.add_row("False negatives (good crop lost)", f"{cm.fn}")
    table.add_row("True negatives (correctly rejected)", f"{cm.tn}")
    table.add_row("False positives (bad crop leaked)", f"{cm.fp}")
    table.add_row("", "")
    table.add_row("Accuracy", f"{cm.accuracy:.2%}")
    table.add_row("Precision (keep)", f"{cm.precision_keep:.2%}")
    table.add_row("Recall (keep)", f"{cm.recall_keep:.2%}")
    table.add_row("F1 (keep)", f"{cm.f1_keep:.2%}")
    coverage_style = "bold green" if cm.coverage >= 0.90 else "bold yellow"
    table.add_row(
        "Coverage (recall on rejects)",
        f"[{coverage_style}]{cm.coverage:.2%}[/{coverage_style}]",
    )
    console.print(table)


def _print_violations(result: EvaluationResult) -> None:
    console.print()
    console.rule("[bold]Per-Violation Coverage")

    if not result.violations:
        console.print("[yellow]No violation labels in the validation set.[/yellow]")
        return

    table = Table()
    table.add_column("Violation", style="cyan")
    table.add_column("Labeled", justify="right")
    table.add_column("Rejected", justify="right")
    table.add_column("Coverage", justify="right", style="magenta")
    table.add_column("At expected stage", justify="right")
    table.add_column("Stage breakdown", style="dim")

    for v in result.violations:
        breakdown = ", ".join(f"{stage}:{n}" for stage, n in v.stage_breakdown.items())
        table.add_row(
            v.violation,
            str(v.n_labeled),
            str(v.n_rejected),
            f"{v.coverage:.2%}",
            f"{v.coverage_at_expected_stage:.2%}",
            breakdown or "—",
        )
    console.print(table)


@app.command()
def main(
    run_folder: Optional[Path] = typer.Option(
        None,
        help="Path to a run folder. Defaults to the most recent run.",
    ),
    labels: Optional[Path] = typer.Option(
        None,
        help="Validation labels CSV. Defaults to paths.labels_dir/validation.csv.",
    ),
    log_level: str = typer.Option("INFO", help="Console log level."),
) -> None:
    """Evaluate a pipeline run against ground-truth labels."""
    configure_logger(log_level=log_level)
    app_config = load_app_config()

    if run_folder is None:
        run_folder = _find_latest_run(app_config.paths.runs_dir)
        logger.info("Using latest run: {}", run_folder.name)
    if labels is None:
        labels = app_config.paths.labels_dir / "validation.csv"

    decisions_path = run_folder / "decisions.parquet"
    label_list = load_validation_csv(labels)
    result = evaluate_run(decisions_path, label_list)

    _print_overall(result)
    _print_violations(result)

    out_path = run_folder / "evaluation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    console.print()
    console.print(f"[green]Wrote {out_path}[/green]")


if __name__ == "__main__":
    app()
