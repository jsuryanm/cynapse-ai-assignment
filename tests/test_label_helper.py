from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import label_helper


def _write_crop(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _patch_config(monkeypatch, data_dir: Path, labels_dir: Path) -> None:
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            data_dir=data_dir,
            labels_dir=labels_dir,
        )
    )
    monkeypatch.setattr(label_helper, "load_app_config", lambda: cfg)


def test_label_helper_samples_unique_content_only(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    labels_dir = tmp_path / "labels"
    raw_dir.mkdir()
    _patch_config(monkeypatch, raw_dir, labels_dir)

    _write_crop(raw_dir / "crop (1).png", b"same-image")
    _write_crop(raw_dir / "crop (2).png", b"same-image")
    _write_crop(raw_dir / "crop (3).png", b"different-image")

    label_helper.main(n=3, seed=42)

    df = pd.read_csv(labels_dir / "validation.csv")
    copied = sorted(p.stem for p in (labels_dir / "to_label").glob("*.png"))

    assert df["crop_id"].tolist() == ["crop (1)", "crop (3)"]
    assert copied == ["crop (1)", "crop (3)"]


def test_label_helper_clamps_sample_size_to_unique_count(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    raw_dir = tmp_path / "raw"
    labels_dir = tmp_path / "labels"
    raw_dir.mkdir()
    _patch_config(monkeypatch, raw_dir, labels_dir)

    _write_crop(raw_dir / "crop (10).png", b"one")
    _write_crop(raw_dir / "crop (11).png", b"one")

    label_helper.main(n=5, seed=42)

    df = pd.read_csv(labels_dir / "validation.csv")
    out = capsys.readouterr().out

    assert df["crop_id"].tolist() == ["crop (10)"]
    assert "Only 1 unique crops available" in out
    assert "Deduplication: 2 -> 1 unique crops; 1 duplicate copies skipped." in out
