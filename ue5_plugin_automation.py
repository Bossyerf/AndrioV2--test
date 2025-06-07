"""UE5 Plugin and Settings Automation System for AndrioV2
=======================================================

This module programmatically enables/disables plugins and updates
configuration in a UE5 project. The implementation is simplified from the
original bytecode but retains the key functionality described in the docs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class PluginInfo:
    name: str
    path: Path
    enabled: bool = False


@dataclass
class UProjectConfig:
    path: Path
    data: Dict = field(default_factory=dict)

    def load(self) -> None:
        self.data = _load_json(self.path)

    def save(self) -> None:
        _save_json(self.path, self.data)

    @property
    def plugins(self) -> List[Dict]:
        return self.data.setdefault("Plugins", [])


class UE5PluginAutomation:
    """Automation helpers for UE5 plugin management."""

    def __init__(self, project_path: str | None = None) -> None:
        self.project_path = self._find_project_path(Path(project_path) if project_path else Path.cwd())
        self.project_config = UProjectConfig(self.project_path) if self.project_path else None
        if self.project_config:
            self.project_config.load()
        self.available_plugins: List[PluginInfo] = []
        if self.project_path:
            self._discover_plugins()

    # ------------------------------------------------------------------
    def _find_project_path(self, start: Path) -> Path | None:
        if start.is_file() and start.suffix == ".uproject":
            return start
        for parent in [start, *start.parents]:
            for file in parent.glob("*.uproject"):
                return file
        return None

    # ------------------------------------------------------------------
    def _discover_plugins(self) -> None:
        project_plugins = self.project_path.parent / "Plugins" if self.project_path else None
        engine_plugins = Path(os.environ.get("UE_ENGINE_PATH", "")) / "Engine/Plugins" if os.environ.get("UE_ENGINE_PATH") else None
        for plugins_dir in filter(None, [project_plugins, engine_plugins]):
            if plugins_dir and plugins_dir.exists():
                for root, _, files in os.walk(plugins_dir):
                    for file in files:
                        if file.endswith(".uplugin"):
                            path = Path(root) / file
                            self.available_plugins.append(PluginInfo(name=path.stem, path=path))

    # ------------------------------------------------------------------
    def list_plugins(self) -> List[PluginInfo]:
        return sorted(self.available_plugins, key=lambda p: p.name)

    # ------------------------------------------------------------------
    def _is_enabled(self, name: str) -> bool:
        if not self.project_config:
            return False
        return any(p.get("Name") == name and p.get("Enabled", False) for p in self.project_config.plugins)

    # ------------------------------------------------------------------
    def enable_plugin(self, name: str) -> bool:
        if not self.project_config:
            return False
        if self._is_enabled(name):
            return True
        self.project_config.plugins.append({"Name": name, "Enabled": True})
        self.project_config.save()
        return True

    # ------------------------------------------------------------------
    def disable_plugin(self, name: str) -> bool:
        if not self.project_config:
            return False
        before = len(self.project_config.plugins)
        self.project_config.data["Plugins"] = [p for p in self.project_config.plugins if p.get("Name") != name]
        self.project_config.save()
        return len(self.project_config.plugins) < before

    # ------------------------------------------------------------------
    def set_disable_engine_plugins_by_default(self, enabled: bool) -> None:
        if not self.project_config:
            return
        self.project_config.data["DisableEnginePluginsByDefault"] = not enabled
        self.project_config.save()
