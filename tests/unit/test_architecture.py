"""Layer rules (clean architecture). Only `app.di` (the composition root) may know `app.infra`.

api ──► application ──► domain ◄── infra
                            ▲
                   di wires everything
"""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"

# layer -> packages it must NOT import
FORBIDDEN = {
    "domain": ("app.application", "app.infra", "app.api", "app.di"),
    "application": ("app.infra", "app.api", "app.di"),
    "api": ("app.infra",),
    "infra": ("app.application", "app.api", "app.di"),
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_import_forbidden_layers(layer: str):
    violations = [
        f"{path.relative_to(APP.parent)} imports {module}"
        for path in (APP / layer).rglob("*.py")
        for module in _imports(path)
        if module.startswith(FORBIDDEN[layer])
    ]
    assert not violations, "\n".join(violations)
