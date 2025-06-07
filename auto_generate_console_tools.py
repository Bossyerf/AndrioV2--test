"""Auto-Generate UE5 Console Tools
=================================

This module parses ``ConsoleHelp.html`` and generates Python functions for
executing UE5 console commands via the ``upyrc`` remote execution package.
The original source was lost, but the compiled bytecode contained enough
information to recreate a simplified version for developers to modify.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:  # ``upyrc`` may not be installed in all environments
    from upyrc import upyre
except Exception:  # pragma: no cover - optional dependency
    upyre = None  # type: ignore


@dataclass
class ConsoleCommand:
    """Representation of a console command extracted from HTML."""

    name: str
    help: str
    cmd_type: str


class ConsoleToolGenerator:
    """Generate Python wrappers for UE5 console commands."""

    def __init__(self, html_file: str, project_path: str | None = None) -> None:
        self.html_file = Path(html_file)
        self.project_path = Path(project_path) if project_path else None
        self.commands: List[ConsoleCommand] = []
        self.generated_tools: List[str] = []
        self._config: upyre.RemoteExecutionConfig | None = None  # type: ignore

    # ------------------------------------------------------------------
    def _setup_config(self) -> None:
        """Initialise ``upyre.RemoteExecutionConfig`` if possible."""
        if self._config is not None or upyre is None:
            return

        if self.project_path and self.project_path.exists():
            try:
                self._config = upyre.RemoteExecutionConfig.from_uproject_path(
                    str(self.project_path)
                )
            except Exception:
                self._config = None

        if self._config is None:
            self._config = upyre.RemoteExecutionConfig(
                multicast_group=("239.0.0.1", 6766),
                multicast_bind_address="0.0.0.0",
            )

    # ------------------------------------------------------------------
    def _execute_ue5_command(self, command: str, command_name: str) -> Dict[str, str]:
        """Execute a Python command in UE5 via ``upyrc``."""
        self._setup_config()
        if upyre is None or self._config is None:
            return {
                "success": False,
                "message": "upyrc not available",
                "command": command_name,
                "output": "",
            }

        try:
            with upyre.PythonRemoteConnection(self._config) as conn:
                result = conn.execute_python_command(
                    command,
                    exec_type=upyre.ExecTypes.EXECUTE_STATEMENT,
                    raise_exc=False,
                )
        except Exception as exc:  # pragma: no cover - remote execution not tested
            return {
                "success": False,
                "message": f"Remote execution error: {exc}",
                "command": command_name,
                "output": "",
            }

        output_text = "".join(o.get("output", "") for o in result.output)
        success = result.success and "SUCCESS" in output_text
        return {
            "success": success,
            "message": f"{command_name} executed successfully" if success else f"{command_name} execution failed",
            "command": command_name,
            "output": output_text,
        }

    # ------------------------------------------------------------------
    def parse_console_help(self) -> List[ConsoleCommand]:
        """Parse ``ConsoleHelp.html`` and return a list of commands."""
        if not self.html_file.exists():
            raise FileNotFoundError(self.html_file)

        pattern = re.compile(r'{name:\s*"(?P<name>[^"]+)",\s*help:\s*"(?P<help>[^"]*)",\s*type:\s*"(?P<type>Cmd|Exec|Var)"}')
        text = self.html_file.read_text(encoding="utf-8")
        matches = pattern.findall(text)
        self.commands = [ConsoleCommand(name=m[0], help=m[1], cmd_type=m[2]) for m in matches]
        return self.commands

    # ------------------------------------------------------------------
    @staticmethod
    def _is_safe_command(command: ConsoleCommand) -> bool:
        """Determine whether a command is considered safe."""
        safe_patterns = [
            r"^stat\.",
            r"^show",
            r"^dump",
            r"^list",
            r"^get",
            r"^print",
            r"^log",
            r"stats?$",
            r"info$",
            r"help$",
            r"version$",
        ]
        risky_patterns = [
            r"quit",
            r"exit",
            r"shutdown",
            r"restart",
            r"delete",
            r"remove",
            r"destroy",
            r"kill",
            r"crash",
            r"force",
            r"reset",
            r"clear",
            r"flush",
            r"reload",
            r"compile",
            r"build",
            r"cook",
            r"package",
            r"^AddWork$",
            r"work",
            r"thread",
            r"memory",
            r"gc\.",
            r"malloc",
            r"free",
            r"alloc",
            r"pool",
        ]
        name = command.name.lower()
        for pat in risky_patterns:
            if re.search(pat, name):
                return False
        return any(re.search(pat, name) for pat in safe_patterns)

    # ------------------------------------------------------------------
    def _sanitize_function_name(self, command_name: str) -> str:
        func = re.sub(r"[^a-zA-Z0-9_]", "_", command_name)
        if func and func[0].isdigit():
            func = f"cmd_{func}"
        return func.lower()

    # ------------------------------------------------------------------
    def generate_tool_function(self, command: ConsoleCommand) -> Tuple[str, str]:
        """Return function name and source code for a console command."""
        func_name = self._sanitize_function_name(command.name)
        code = (
            f"def {func_name}():\n"
            f"    \"\"\"\n"
            f"    {command.help}\n"
            f"    Command: {command.name}\n"
            f"    Type: {command.cmd_type}\n"
            f"    \"\"\"\n"
            f"    command = '''import unreal; unreal.SystemLibrary.execute_console_command(None, \"{command.name}\"); print(\"✅ {command.name} executed\"); print(\"SUCCESS\")'''\n"
            f"    return _execute_ue5_command(command, \"{command.name}\")\n"
        )
        return func_name, code

    # ------------------------------------------------------------------
    def generate_all_tools(self, safe_only: bool = True, output_file: str = "generated_console_tools.py") -> List[str]:
        """Generate a Python module with console command wrappers."""
        if not self.commands:
            self.parse_console_help()

        functions = []
        for cmd in self.commands:
            if safe_only and not self._is_safe_command(cmd):
                continue
            fname, code = self.generate_tool_function(cmd)
            functions.append(code)
            self.generated_tools.append(fname)

        header = [
            '"""Auto-Generated UE5 Console Tools"""',
            'from __future__ import annotations',
            'from upyrc import upyre',
            'from .auto_generate_console_tools import _execute_ue5_command',
            '',
        ]
        Path(output_file).write_text("\n".join(header + functions), encoding="utf-8")
        return self.generated_tools

    # ------------------------------------------------------------------
    def test_tools_systematically(self, delay_seconds: int = 2, max_tests: int | None = None) -> Dict[str, int]:
        """Import generated tools and execute them sequentially."""
        results = {"successful": 0, "failed": 0}
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "generated_console_tools", "generated_console_tools.py"
            )
            generated_module = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(generated_module)  # type: ignore

            for idx, name in enumerate(self.generated_tools):
                if max_tests is not None and idx >= max_tests:
                    break
                tool_func = getattr(generated_module, name, None)
                if callable(tool_func):
                    res = tool_func()
                    if res.get("success"):
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                    time.sleep(delay_seconds)
        except Exception:
            pass
        return results


if __name__ == "__main__":  # pragma: no cover - manual utility
    gen = ConsoleToolGenerator("ConsoleHelp.html")
    gen.parse_console_help()
    gen.generate_all_tools(safe_only=True)
    stats = gen.test_tools_systematically(max_tests=5)
    print(f"Tested {len(gen.generated_tools)} tools -> {stats}")
