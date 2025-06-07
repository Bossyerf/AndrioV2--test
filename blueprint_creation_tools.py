"""Blueprint Creation Tools for AndrioV2
=====================================

This simplified implementation exposes functions for creating common
Blueprint assets via the ``upyrc`` remote execution API. The original
source was recovered from bytecode and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

try:
    from upyrc import upyre
except Exception:  # pragma: no cover - optional dependency
    upyre = None  # type: ignore


@dataclass
class _ConfigWrapper:
    project_path: Path | None = None
    config: upyre.RemoteExecutionConfig | None = None  # type: ignore

    def ensure(self) -> None:
        if self.config is not None or upyre is None:
            return
        if self.project_path and self.project_path.exists():
            try:
                self.config = upyre.RemoteExecutionConfig.from_uproject_path(str(self.project_path))
            except Exception:
                self.config = None
        if self.config is None:
            self.config = upyre.RemoteExecutionConfig(
                multicast_group=("239.0.0.1", 6766),
                multicast_bind_address="0.0.0.0",
            )


class BlueprintCreationTools:
    """Create various Blueprint assets in UE5."""

    def __init__(self, project_path: str | None = None) -> None:
        self._config = _ConfigWrapper(Path(project_path) if project_path else None)

    # ------------------------------------------------------------------
    def _execute(self, code: str, action: str) -> Dict[str, str]:
        """Execute Python code in UE5 via remote execution."""
        self._config.ensure()
        cfg = self._config.config
        if upyre is None or cfg is None:
            return {"success": False, "message": "upyrc not available", "action": action, "output": ""}
        try:
            with upyre.PythonRemoteConnection(cfg) as conn:
                result = conn.execute_python_command(code, exec_type=upyre.ExecTypes.EXECUTE_STATEMENT, raise_exc=False)
        except Exception as exc:  # pragma: no cover - remote execution not tested
            return {"success": False, "message": f"Remote execution error: {exc}", "action": action, "output": ""}

        out = "".join(o.get("output", "") for o in result.output)
        success = result.success and "SUCCESS" in out
        return {"success": success, "message": action, "output": out, "action": action}

    # ------------------------------------------------------------------
    def create_blueprint_actor(self, blueprint_name: str, package_path: str, parent_class: str = "Actor") -> Dict[str, str]:
        """Create a simple Blueprint actor."""
        cmd = f"""import unreal\nfactory = unreal.BlueprintFactory()\nif '{parent_class}' == 'Pawn':\n    factory.set_editor_property('parent_class', unreal.Pawn)\nelif '{parent_class}' == 'Character':\n    factory.set_editor_property('parent_class', unreal.Character)\nelse:\n    factory.set_editor_property('parent_class', unreal.Actor)\nasset_tools = unreal.AssetToolsHelpers.get_asset_tools()\nblueprint = asset_tools.create_asset(asset_name='{blueprint_name}', package_path='{package_path}', asset_class=unreal.Blueprint, factory=factory)\nif blueprint:\n    print(f'Created Blueprint: {blueprint.get_name()}')\n    print(f'Path: {blueprint.get_path_name()}')\n    print('SUCCESS')\nelse:\n    print('Failed to create Blueprint')\n    print('FAILED')"""
        return self._execute(cmd, "create_blueprint_actor")

    # ------------------------------------------------------------------
    def create_blueprint_with_mesh(self, blueprint_name: str, package_path: str, mesh_path: str) -> Dict[str, str]:
        """Create a Blueprint with a StaticMeshComponent."""
        cmd = f"""import unreal\nfactory = unreal.BlueprintFactory()\nfactory.set_editor_property('parent_class', unreal.Actor)\nasset_tools = unreal.AssetToolsHelpers.get_asset_tools()\nblueprint = asset_tools.create_asset(asset_name='{blueprint_name}', package_path='{package_path}', asset_class=unreal.Blueprint, factory=factory)\nif blueprint:\n    mesh_component = blueprint.get_blueprint_generated_class().get_default_object().add_component(unreal.StaticMeshComponent, 'StaticMeshComponent')\n    if mesh_component:\n        mesh_asset = unreal.EditorAssetLibrary.load_asset('{mesh_path}')\n        if mesh_asset:\n            mesh_component.set_static_mesh(mesh_asset)\n            print('Added Static Mesh Component')\n    print('SUCCESS')\nelse:\n    print('Failed to create Blueprint')\n    print('FAILED')"""
        return self._execute(cmd, "create_blueprint_with_mesh")

    # ------------------------------------------------------------------
    def create_blueprint_from_template(self, blueprint_name: str, package_path: str, template_type: str = "Actor") -> Dict[str, str]:
        """Create a Blueprint using common templates."""
        cmd = f"""import unreal\nif '{template_type}' == 'Widget':\n    factory = unreal.WidgetBlueprintFactory()\nelse:\n    factory = unreal.BlueprintFactory()\n    if '{template_type}' == 'Pawn':\n        factory.set_editor_property('parent_class', unreal.Pawn)\n    elif '{template_type}' == 'Character':\n        factory.set_editor_property('parent_class', unreal.Character)\n    elif '{template_type}' == 'GameMode':\n        factory.set_editor_property('parent_class', unreal.GameModeBase)\n    elif '{template_type}' == 'PlayerController':\n        factory.set_editor_property('parent_class', unreal.PlayerController)\n    else:\n        factory.set_editor_property('parent_class', unreal.Actor)\nasset_tools = unreal.AssetToolsHelpers.get_asset_tools()\nif '{template_type}' == 'Widget':\n    blueprint = asset_tools.create_asset(asset_name='{blueprint_name}', package_path='{package_path}', asset_class=unreal.WidgetBlueprint, factory=factory)\nelse:\n    blueprint = asset_tools.create_asset(asset_name='{blueprint_name}', package_path='{package_path}', asset_class=unreal.Blueprint, factory=factory)\nif blueprint:\n    print(f'Created {template_type} Blueprint: {blueprint.get_name()}')\n    print(f'Path: {blueprint.get_path_name()}')\n    print('SUCCESS')\nelse:\n    print('Failed to create Blueprint')\n    print('FAILED')"""
        return self._execute(cmd, "create_blueprint_from_template")
