"""Verify that scripts has no imports from the legacy Scripts package."""

from __future__ import annotations

import ast
from pathlib import Path


NEW_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]


def legacy_imports(root: Path) -> list[str]:
    violations = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            violations.append(f"{path.relative_to(root)}:{error.lineno}: invalid syntax")
            continue
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = [node.module]
            for name in imported:
                if name == "Scripts" or name.startswith("Scripts."):
                    violations.append(
                        f"{path.relative_to(root)}:{node.lineno}: imports {name}")
    return violations


def main() -> None:
    violations = legacy_imports(NEW_FRAMEWORK_ROOT)
    if violations:
        raise RuntimeError(
            "scripts must remain independent from legacy Scripts:\n- " +
            "\n- ".join(violations))
    print("scripts import boundary is clean: no legacy Scripts imports")


if __name__ == "__main__":
    main()
