"""Download the publisher's original train/eval Parquet files without edits."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
REPO = "Dingdong-Inc/FreshRetailNet-50K"
# Dataset-card commit v1.0. Pinning the revision makes future downloads reproducible.
REVISION = "01a4be637d72261d180f14a96c3c1aec8c0c1e1e"
EXPECTED = {
    "train.parquet": (106_436_287, "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d"),
    "eval.parquet": (8_440_124, "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e"),
}

for filename, (expected_size, expected_hash) in EXPECTED.items():
    local_path = DATA / filename
    if local_path.exists():
        path = local_path
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
                                    filename=f"data/{filename}", revision=REVISION,
                                    local_dir=ROOT))
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"{filename}: expected {expected_size:,} bytes, received {actual_size:,}.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise RuntimeError(f"{filename}: SHA-256 mismatch; refusing to use this file.")
    print(f"Verified original {filename}: {actual_size:,} bytes; SHA-256 {digest}")
