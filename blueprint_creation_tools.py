"""Blueprint Creation Tools

Simplified placeholders for creating Unreal Engine Blueprints.
Actual implementations would use the Unreal Python API.
"""

from typing import List


def create_blueprint_actor(name: str, path: str) -> str:
    """Pretend to create a Blueprint actor."""
    return f"Blueprint '{name}' would be created at '{path}'."


def create_blueprint_with_components(name: str, path: str, components: List[str]) -> str:
    """Pretend to create a Blueprint with the given components."""
    comps = ", ".join(components)
    return f"Blueprint '{name}' with components [{comps}] would be created at '{path}'."

