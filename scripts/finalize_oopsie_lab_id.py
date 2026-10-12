#!/usr/bin/env python3
"""Replace the local-collection lab-id placeholder after Oopsie initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from oopsie_data_tools.utils import contributor_config


PENDING_LAB_ID = "pending_local_collection"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    args = parser.parse_args()

    lab_id, _ = contributor_config.read_contributor_config()
    updated = 0
    skipped = 0
    for path in sorted(args.samples_dir.rglob("*.h5")):
        with h5py.File(path, "r+") as episode:
            current = str(episode.attrs.get("lab_id", ""))
            if current != PENDING_LAB_ID:
                skipped += 1
                continue
            episode.attrs["lab_id"] = lab_id
            updated += 1
            print(f"Updated {path}")

    print(f"Finished: updated={updated}, unchanged={skipped}, lab_id={lab_id}")


if __name__ == "__main__":
    main()
