"""Download the Open-i / Indiana University Chest X-ray dataset.

Pulls the official NLM bulk archives (reports XML + PNG images), with a
Kaggle-CLI fallback if the official host is unreachable from your network.

Usage:
    python -m src.data.download_openi --out data/raw
    python -m src.data.download_openi --out data/raw --images        # also pull images (~2GB)
    python -m src.data.download_openi --out data/raw --source kaggle  # force Kaggle mirror

Note: some sandboxed / CI environments block outbound requests to
openi.nlm.nih.gov. If this script fails with a connection error, either
run it from an unrestricted machine, or download the two archives manually
from https://openi.nlm.nih.gov/imgs/collections/ (or the Kaggle mirror at
https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) and
place them in the output directory before re-running with --skip-download
to just extract.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(dest)


def extract_tgz(archive: Path, out_dir: Path) -> None:
    print(f"[extract] {archive} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(out_dir)


def download_via_kaggle(dataset: str, out_dir: Path) -> None:
    """Requires `pip install kaggle` and a configured ~/.kaggle/kaggle.json token."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"]
    print(f"[kaggle] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source", choices=["official", "kaggle"], default="official",
        help="Where to pull the dataset from.",
    )
    parser.add_argument("--images", action="store_true", help="Also download the image archive.")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip fetching; just extract archives already present in --out.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "kaggle":
        download_via_kaggle(cfg["source"]["kaggle_dataset"], out_dir)
        return

    reports_archive = out_dir / "NLMCXR_reports.tgz"
    if not args.skip_download:
        try:
            download_file(cfg["source"]["reports_url"], reports_archive)
        except requests.exceptions.RequestException as e:
            print(
                f"[error] Could not reach {cfg['source']['reports_url']}: {e}\n"
                "This network may block openi.nlm.nih.gov. Try --source kaggle, "
                "or download the archives manually (see module docstring) and "
                "re-run with --skip-download.",
                file=sys.stderr,
            )
            sys.exit(1)
    if reports_archive.exists():
        extract_tgz(reports_archive, out_dir / "reports")

    if args.images:
        images_archive = out_dir / "NLMCXR_png.tgz"
        if not args.skip_download:
            download_file(cfg["source"]["images_url"], images_archive)
        if images_archive.exists():
            extract_tgz(images_archive, out_dir / "images")

    print("[done] Raw data ready under", out_dir)


if __name__ == "__main__":
    main()
