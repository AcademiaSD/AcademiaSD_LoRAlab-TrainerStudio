# -*- coding: utf-8 -*-
"""Migrate legacy root folders into Models, Outputs and Cache_Data."""
import argparse
import sys
from pathlib import Path

import project_layout


def print_result(result):
    print("Root: {}".format(result["root"]))
    for item in result["moves"]:
        print("[{}] {}: {} -> {}".format(item.get("status", ""), item.get("kind", ""), item.get("old", ""), item.get("new", "")))
    for item in result["collisions"]:
        line = "[{}] {}: {} -> {}".format(item.get("status", ""), item.get("kind", ""), item.get("old", ""), item.get("new", ""))
        if item.get("error"):
            line += " | " + item["error"]
        print(line)
    for path in result.get("rewrites") or []:
        print("[rewritten] {}".format(path))
    print("Moves: {} | collisions: {} | rewrites: {}".format(
        len(result["moves"]), len(result["collisions"]), len(result.get("rewrites") or [])))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(project_layout.PROJECT_ROOT))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    group.add_argument("--rollback", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.rollback:
        result = project_layout.rollback(root)
    else:
        result = project_layout.ensure_layout(root, dry_run=not args.apply)
    print_result(result)
    return 2 if result["collisions"] else 0


if __name__ == "__main__":
    sys.exit(main())
