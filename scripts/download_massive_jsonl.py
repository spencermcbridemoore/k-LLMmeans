"""Download MASSIVE 1.0 en-US.jsonl from Amazon S3 into data_loaders/massive.jsonl (~40 MB tarball)."""
from __future__ import annotations

import io
import os
import shutil
import sys
import tarfile
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO_ROOT, "data_loaders", "massive.jsonl")
TAR_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
MEMBER = "1.0/data/en-US.jsonl"


def main() -> int:
    if os.path.isfile(OUT):
        print(f"Already exists: {OUT}")
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"Downloading {TAR_URL} ...")
    raw = urllib.request.urlopen(TAR_URL, timeout=300).read()
    print(f"Extracting {MEMBER} ...")
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    src = tf.extractfile(MEMBER)
    if src is None:
        print("Missing member in tarball", file=sys.stderr)
        return 1
    with open(OUT, "wb") as f:
        shutil.copyfileobj(src, f)
    print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
