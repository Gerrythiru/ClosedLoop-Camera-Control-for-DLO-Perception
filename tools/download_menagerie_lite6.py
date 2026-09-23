from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


TREE_API_URL = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/main?recursive=1"
RAW_BASE_URL = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"
MODEL_DIR = "ufactory_lite6"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "codex-lite6-simulation-builder"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_file(repo_path: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE_URL}/{repo_path}"
    target.write_bytes(read_url(url))


def download_official_lite6(force: bool = False) -> Path:
    root = project_root()
    vendor_root = root / "vendor" / "mujoco_menagerie"
    vendor_model = vendor_root / MODEL_DIR

    if vendor_model.exists() and not force:
        print(f"Using existing vendor model: {vendor_model}")
        return vendor_model

    print(f"Reading official MuJoCo Menagerie tree from {TREE_API_URL}", flush=True)
    tree_payload = json.loads(read_url(TREE_API_URL).decode("utf-8"))
    file_paths = [
        item["path"]
        for item in tree_payload.get("tree", [])
        if item.get("type") == "blob" and item.get("path", "").startswith(f"{MODEL_DIR}/")
    ]
    if not file_paths:
        raise FileNotFoundError(f"Could not find {MODEL_DIR} files in official repository tree.")

    vendor_root.mkdir(parents=True, exist_ok=True)
    for repo_path in sorted(file_paths):
        relative = Path(repo_path).relative_to(MODEL_DIR)
        target = vendor_model / relative
        print(f"Downloading {repo_path}", flush=True)
        download_file(repo_path, target)

    print("Downloading LICENSE", flush=True)
    download_file("LICENSE", vendor_root / "LICENSE")

    files = [path for path in vendor_model.rglob("*") if path.is_file()]
    print(f"Downloaded {len(files)} official Lite 6 files to {vendor_model}")
    print(f"Copied Menagerie license to {vendor_root / 'LICENSE'}")
    return vendor_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official UFactory Lite 6 MuJoCo Menagerie files.")
    parser.add_argument("--force", action="store_true", help="Overwrite files in the vendor directory.")
    args = parser.parse_args()

    try:
        download_official_lite6(force=args.force)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
