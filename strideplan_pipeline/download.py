"""Download the official Endomondo export using Python on Windows or Linux."""

import argparse
import shutil
import urllib.request
from pathlib import Path


URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/endomondoHR_proper.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/endomondoHR_proper.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".part")
    print("Downloading about 4.9 GB. An interrupted download can be restarted with the same command.", flush=True)
    with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as stream:
        expected_size = response.headers.get("Content-Length")
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if expected_size is not None and partial.stat().st_size != int(expected_size):
        raise OSError("Incomplete download; rerun the command")
    partial.replace(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
