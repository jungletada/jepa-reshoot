#!/usr/bin/env python3
"""Check out pinned official dependencies without installing or loading models."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def git_output(directory, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True
    ).strip()


def check_existing(destination, source):
    if not (destination / ".git").exists():
        raise RuntimeError(f"Existing directory is not a Git checkout: {destination}")
    if Path(git_output(destination, "rev-parse", "--show-toplevel")).resolve() != destination.resolve():
        raise RuntimeError(f"Unexpected checkout root: {destination}")
    if git_output(destination, "remote", "get-url", "origin").rstrip("/") != source["repository"]:
        raise RuntimeError(f"Origin differs from sources.json: {destination}")
    if git_output(destination, "rev-parse", "HEAD") != source["commit"]:
        raise RuntimeError(f"Commit differs from sources.json: {destination}; existing files were preserved")
    if git_output(destination, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"Tracked files have local changes: {destination}; existing files were preserved")


def checkout(source, dry_run):
    destination = ROOT / "external" / source["directory"]
    if destination.exists():
        check_existing(destination, source)
        print(f"Ready: {destination.relative_to(ROOT)} @ {source['commit'][:12]}", flush=True)
        return

    commands = [
        ["git", "init", "--quiet"],
        ["git", "remote", "add", "origin", source["repository"]],
        ["git", "fetch", "--depth=1", "origin", source["commit"]],
        ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
    ]
    print(f"Prepare: {destination.relative_to(ROOT)} @ {source['commit'][:12]}", flush=True)
    if dry_run:
        for command in commands:
            print(f"  {shlex.join(command)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for command in commands:
            subprocess.run(command, cwd=temporary, check=True)
        check_existing(temporary, source)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Ready: {destination.relative_to(ROOT)}", flush=True)


def main():
    sources = json.loads((ROOT / "external" / "sources.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="*", help="wan (default), vggt, sam3, or all")
    parser.add_argument("--dry-run", action="store_true", help="Show source preparation commands without changing files")
    args = parser.parse_args()
    components = args.components or ["wan"]
    unknown = set(components) - (set(sources) | {"all"})
    if unknown:
        parser.error(f"Unknown components: {', '.join(sorted(unknown))}")
    if "all" in components:
        components = list(sources)
    try:
        for component in dict.fromkeys(components):
            checkout(sources[component], args.dry_run)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
        parser.exit(1, f"Source preparation failed: {error}\n")


if __name__ == "__main__":
    main()
