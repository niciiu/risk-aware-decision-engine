#!/usr/bin/env python3
"""List all files under a directory (recursive) with simple filters.

Usage examples:
  python decision_intelligence_framework/scripts/list_all_files.py --root .
  python decision_intelligence_framework/scripts/list_all_files.py --root . --extensions .py,.md --output files.txt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional


def parse_extensions(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [p if p.startswith(".") else f".{p}" for p in parts]


def list_files(root: Path, extensions: Optional[Iterable[str]] = None, max_depth: Optional[int] = None, include_hidden: bool = False, follow_symlinks: bool = False) -> List[Path]:
    root = root.resolve()
    results: List[Path] = []
    ext_set = set(ext.lower() for ext in extensions) if extensions else None

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        rel = Path(dirpath).resolve().relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if max_depth is not None and depth > max_depth:
            dirnames.clear()
            continue

        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]

        for fn in filenames:
            if not include_hidden and fn.startswith('.'):
                continue
            p = Path(dirpath) / fn
            if ext_set is not None:
                if p.suffix.lower() not in ext_set:
                    continue
            results.append(p)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="List files recursively with simple filters.")
    parser.add_argument("--root", default=".", help="Root directory to list (default: current dir)")
    parser.add_argument("--extensions", help="Comma-separated extensions to include (e.g. .py,.md or py,md)")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum recursion depth (0 = only root)")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files and directories")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow symbolic links")
    parser.add_argument("--output", help="Write list to output file (one path per line)")

    args = parser.parse_args()

    root = Path(args.root)
    exts = parse_extensions(args.extensions)

    files = list_files(root=root, extensions=exts, max_depth=args.max_depth, include_hidden=args.include_hidden, follow_symlinks=args.follow_symlinks)

    files_sorted = sorted(files)
    for p in files_sorted:
        try:
            print(p.resolve().relative_to(root.resolve()))
        except Exception:
            print(p)

    summary = f"\nFound {len(files_sorted)} file(s) under {root}\n"
    print(summary)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text("\n".join(str(p) for p in files_sorted))
        print(f"Wrote list to {out_path}")


if __name__ == "__main__":
    main()
