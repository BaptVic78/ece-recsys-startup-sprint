"""Download the BMClab 2019 daily running file (CC BY 4.0) and verify its checksum."""

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

# Afonseca, Watanabe & Duarte (2022), figshare 10.6084/m9.figshare.16620238, version 5.
FILES = {
    "run_ww_2019_d.parquet": ("https://ndownloader.figshare.com/files/33801866", "e10b8ba19ec1340b14b90049f945c259"),
}


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name, output):
    url, checksum = FILES[name]
    target = output / name
    if target.exists():
        if md5(target) != checksum:
            raise OSError(f"{target} exists but its checksum is wrong; delete it and rerun")
        print(f"{target} already present")
        return target
    output.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if md5(partial) != checksum:
        partial.unlink()
        raise OSError(f"Checksum mismatch for {name}; rerun the command")
    partial.replace(target)
    print(f"Saved {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/bmclab"))
    args = parser.parse_args()
    for name in FILES:
        download(name, args.output)


if __name__ == "__main__":
    main()
