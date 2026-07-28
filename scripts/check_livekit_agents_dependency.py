#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONSTRAINT_PATTERN = re.compile(r"^livekit-agents==(\d+)\.(\d+)\.(\d+)$")
REQUIREMENT_PATTERN = re.compile(r"""["'](livekit-agents[^"']*)["']""")


def livekit_requirements(path: Path) -> list[str]:
    return REQUIREMENT_PATTERN.findall(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-installed",
        action="store_true",
        help="also verify the installed livekit-agents version",
    )
    args = parser.parse_args()

    livekit_constraints = livekit_requirements(ROOT / "pyproject.toml")
    if len(livekit_constraints) != 1:
        print(
            "expected exactly one root livekit-agents constraint",
            file=sys.stderr,
        )
        return 1

    match = CONSTRAINT_PATTERN.fullmatch(livekit_constraints[0])
    if match is None:
        print(
            "root constraint must use livekit-agents==<major>.<minor>.<patch>",
            file=sys.stderr,
        )
        return 1

    major, minor, patch = (int(part) for part in match.groups())
    pinned_version = f"{major}.{minor}.{patch}"
    expected_range = f"livekit-agents>={pinned_version},<{major}.{minor + 1}"

    errors: list[str] = []
    package_files = sorted((ROOT / "livekit-plugins").glob("*/pyproject.toml"))
    if not package_files:
        errors.append("no plugin pyproject.toml files found")

    for package_file in package_files:
        livekit_dependencies = livekit_requirements(package_file)
        if livekit_dependencies != [expected_range]:
            relative_path = package_file.relative_to(ROOT)
            errors.append(
                f"{relative_path}: expected {expected_range!r}, "
                f"found {livekit_dependencies!r}"
            )

    if args.check_installed:
        try:
            installed_version = importlib.metadata.version("livekit-agents")
        except importlib.metadata.PackageNotFoundError:
            errors.append("livekit-agents is not installed")
        else:
            if installed_version != pinned_version:
                errors.append(
                    "installed livekit-agents version "
                    f"{installed_version} does not match {pinned_version}"
                )

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    installed_note = " and installed environment" if args.check_installed else ""
    print(
        f"livekit-agents {pinned_version} is consistent across "
        f"{len(package_files)} plugins{installed_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
