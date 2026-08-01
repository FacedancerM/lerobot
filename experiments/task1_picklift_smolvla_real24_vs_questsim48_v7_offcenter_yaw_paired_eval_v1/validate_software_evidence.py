from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> dict:
    rows = {}
    for line in (root / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        rows[name] = digest
    if any(sha256_file(root / name) != digest for name, digest in rows.items()):
        raise RuntimeError("software evidence hash mismatch")
    dry = json.loads((root / "dry_run.json").read_text(encoding="utf-8"))
    if dry["status"] != "software_dry_run_passed_hardware_not_accessed" or any(dry["hardware_access"].values()):
        raise RuntimeError("software evidence claims hardware access")
    if dry["fake_protocol"]["trials_exercised"] != 24:
        raise RuntimeError("dry-run trial count mismatch")
    if any(not row["finite"] or row["shape"] != [1, 6] for row in dry["checkpoint_adapter_smoke"].values()):
        raise RuntimeError("checkpoint adapter smoke mismatch")
    return {"status": "independent_validation_passed", "evidence_root": str(root), "hashes_sha256": sha256_file(root / "hashes.sha256"), "hardware_access": dry["hardware_access"], "trials": 24}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.evidence_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists(): raise RuntimeError("refusing to overwrite validation")
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__": main()
