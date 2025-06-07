"""UE5 Remote Python Execution Setup
=================================

Utility helpers to configure remote execution for a UE5 project. This code
is reconstructed from bytecode and documentation references.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

try:
    from upyrc import upyre
except Exception:  # pragma: no cover - optional dependency
    upyre = None  # type: ignore


class UE5RemoteExecutionSetup:
    """Manage UE5 remote Python execution configuration."""

    def __init__(self, project_path: str | None = None) -> None:
        self.project_path = Path(project_path) if project_path else None
        self.config_dir = self.project_path.parent / "Config" if self.project_path else None
        self.remote_settings = {
            "RemoteExecutionMulticastGroupEndpoint": "239.0.0.1:6766",
            "RemoteExecutionMulticastBindAddress": "0.0.0.0",
            "RemoteExecutionMulticastTtl": 1,
        }

    # ------------------------------------------------------------------
    def enable_python_plugin_in_project(self) -> bool:
        if not self.project_path or not self.project_path.exists():
            logger.error("Project file not found: %s", self.project_path)
            return False
        try:
            data = json.loads(self.project_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        plugins = data.setdefault("Plugins", [])
        if not any(p.get("Name") == "PythonScriptPlugin" for p in plugins):
            plugins.append({"Name": "PythonScriptPlugin", "Enabled": True})
            self.project_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True

    # ------------------------------------------------------------------
    def configure_remote_execution_settings(self) -> bool:
        if not self.config_dir or not self.config_dir.exists():
            logger.error("Config directory not found: %s", self.config_dir)
            return False
        engine_ini = self.config_dir / "DefaultEngine.ini"
        settings = [
            "[\/Script\/PythonScriptPlugin.PythonScriptPluginSettings]",
            "bRemoteExecution=True",
            f"RemoteExecutionMulticastGroupEndpoint={self.remote_settings['RemoteExecutionMulticastGroupEndpoint']}",
            f"RemoteExecutionMulticastBindAddress={self.remote_settings['RemoteExecutionMulticastBindAddress']}",
            f"RemoteExecutionMulticastTtl={self.remote_settings['RemoteExecutionMulticastTtl']}",
            "bDeveloperMode=True",
        ]
        try:
            content = engine_ini.read_text(encoding="utf-8") if engine_ini.exists() else ""
            if "PythonScriptPluginSettings" in content:
                engine_ini.write_text(content + "\n" + "\n".join(settings), encoding="utf-8")
            else:
                engine_ini.write_text("\n".join(settings), encoding="utf-8")
            return True
        except Exception as exc:
            logger.error("Failed to configure settings: %s", exc)
            return False

    # ------------------------------------------------------------------
    def test_remote_connection(self) -> Dict[str, str]:
        if upyre is None:
            return {"success": False, "message": "upyrc not installed"}
        cfg = upyre.RemoteExecutionConfig(
            multicast_group=("239.0.0.1", 6766),
            multicast_bind_address="0.0.0.0",
        )
        try:
            with upyre.PythonRemoteConnection(cfg) as conn:
                result = conn.execute_python_command("print('Remote execution test successful!')", exec_type=upyre.ExecTypes.EXECUTE_STATEMENT, raise_exc=False)
        except Exception as exc:  # pragma: no cover - remote execution not tested
            return {"success": False, "message": f"Remote connection test failed: {exc}"}
        out = "".join(o.get("output", "") for o in result.output)
        return {"success": result.success, "message": out.strip()}
