from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

def find_runs(paths: List[Path]) -> List[Dict[str, Any]]:
    runs = []
    for root in paths:
        for manifest_path in sorted(root.rglob("manifest_*.json")):
            suffix = manifest_path.stem[len("manifest_"):]
            folder = manifest_path.parent
            runs.append({
                "folder": folder,
                "suffix": suffix,
                "manifest_path": manifest_path,
                "summaries_path": folder / f"summaries_{suffix}.json",
                "turns_path": folder / f"turns_{suffix}.jsonl",
            })
    return runs

def load_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Read the manifest, summaries, and turn rows for one run."""
    manifest = json.loads(run["manifest_path"].read_text())
    run["status"] = manifest.get("status", "missing")
    run["mode"] = manifest.get("arguments", {}).get("mode", "?")
    run["split"] = manifest.get("arguments", {}).get("split", "?")

    run["summaries"] = (
        json.loads(run["summaries_path"].read_text())
        if run["summaries_path"].exists() else []
    )
    run["turns"] = (
        [json.loads(line) for line in run["turns_path"].read_text().splitlines() if line.strip()]
        if run["turns_path"].exists() else []
    )
    return run

def apply_validity_gate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only validated runs; print why every other run was excluded."""
    kept = []
    print(f"{'Folder':<45} {'Mode':<6} {'Split':<9} {'Status':<16} Used?")
    print("-" * 85)
    for run in runs:
        ok = run["status"] == "validated"
        print(f"{str(run['folder']):<45} {run['mode']:<6} {run['split']:<9} "
              f"{run['status']:<16} {'yes' if ok else 'NO'}")
        if ok:
            kept.append(run)
    print(f"\n{len(kept)} of {len(runs)} runs passed the validity gate.\n")
    return kept

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="Result folders to search")
    args = parser.parse_args()

    runs = [load_run(r) for r in find_runs(args.paths)]
    if not runs:
        raise SystemExit("No manifest_*.json files found under the given folders.")
    runs = apply_validity_gate(runs)
    if not runs:
        raise SystemExit("No validated runs; nothing to analyze.")

    # Quick look at what we loaded (pieces 2-4 replace this).
    for run in runs:
        print(f"{run['folder']}: {len(run['summaries'])} summaries, {len(run['turns'])} turn rows")


if __name__ == "__main__":
    main()


