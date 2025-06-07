"""UE5 Console Command Tools for AndrioV2
======================================

A small selection of console command helpers demonstrating remote execution
via ``upyrc``. The original module included many more commands; only a
sample is recreated here for clarity.
"""

from __future__ import annotations

from typing import Dict

try:
    from upyrc import upyre
except Exception:  # pragma: no cover - optional dependency
    upyre = None  # type: ignore


_CONFIG: upyre.RemoteExecutionConfig | None = None  # type: ignore


def _setup_config() -> None:
    global _CONFIG
    if _CONFIG is not None or upyre is None:
        return
    _CONFIG = upyre.RemoteExecutionConfig(
        multicast_group=("239.0.0.1", 6766),
        multicast_bind_address="0.0.0.0",
    )


def _execute_command(command: str, name: str) -> Dict[str, str]:
    _setup_config()
    if upyre is None or _CONFIG is None:
        return {"success": False, "message": "upyrc not available", "output": "", "command": name}
    try:
        with upyre.PythonRemoteConnection(_CONFIG) as conn:
            result = conn.execute_python_command(command, exec_type=upyre.ExecTypes.EXECUTE_STATEMENT, raise_exc=False)
    except Exception as exc:  # pragma: no cover - remote execution not tested
        return {"success": False, "message": f"Remote execution error: {exc}", "output": "", "command": name}

    out = "".join(o.get("output", "") for o in result.output)
    ok = result.success and "SUCCESS" in out
    return {"success": ok, "message": name, "output": out, "command": name}


# ----------------------------------------------------------------------
# Example console commands
# ----------------------------------------------------------------------

def stat_fps() -> Dict[str, str]:
    """Toggle FPS statistics display."""
    cmd = "import unreal; unreal.SystemLibrary.execute_console_command(None, 'stat fps'); print('FPS statistics display toggled'); print('SUCCESS')"
    return _execute_command(cmd, "stat.fps")


def dump_gpu_stats() -> Dict[str, str]:
    """Dump GPU statistics to the log."""
    cmd = "import unreal; unreal.SystemLibrary.execute_console_command(None, 'DumpGPU'); print('GPU statistics dumped to log'); print('SUCCESS')"
    return _execute_command(cmd, "DumpGPU")


def list_loaded_assets() -> Dict[str, str]:
    """List all currently loaded assets."""
    cmd = "import unreal; unreal.SystemLibrary.execute_console_command(None, 'AssetManager.DumpLoadedAssets'); print('Loaded assets list dumped to log'); print('SUCCESS')"
    return _execute_command(cmd, "AssetManager.DumpLoadedAssets")


def toggle_wireframe() -> Dict[str, str]:
    """Toggle wireframe rendering."""
    cmd = "import unreal; unreal.SystemLibrary.execute_console_command(None, 'showflag.wireframe'); print('Wireframe rendering mode toggled'); print('SUCCESS')"
    return _execute_command(cmd, "showflag.wireframe")


def memory_report() -> Dict[str, str]:
    """Generate a memory usage report."""
    cmd = "import unreal; unreal.SystemLibrary.execute_console_command(None, 'MemReport'); print('Memory report generated and logged'); print('SUCCESS')"
    return _execute_command(cmd, "MemReport")


def get_all_actors_in_level() -> Dict[str, str]:
    """Return a list of actors currently in the level."""
    cmd = (
        "import unreal; actors = unreal.EditorLevelLibrary.get_all_level_actors(); "
        "print(f'Total Actors in Level: {len(actors)}'); "
        "[print(f'{a.get_name()} ({a.__class__.__name__}) - Location: {a.get_actor_location()}') for a in actors[:20]]; "
        "print('ACTOR_LIST_COMPLETE')"
    )
    return _execute_command(cmd, "get_all_actors_in_level")


def map_coordinates_from_origin(radius: float = 1000.0, step: float = 100.0) -> Dict[str, str]:
    """Map actor coordinates from world origin."""
    cmd = (
        "import unreal; actors = unreal.EditorLevelLibrary.get_all_level_actors(); "
        "print(f'Coordinate Mapping from Origin (0,0,0)'); "
        "print(f'Found {len(actors)} actors'); "
        "[print(f'{a.get_name()} ({a.__class__.__name__}) - Exact: ({a.get_actor_location().x:.1f}, {a.get_actor_location().y:.1f}, {a.get_actor_location().z:.1f}) - Grid: ({round(a.get_actor_location().x/100)*100}, {round(a.get_actor_location().y/100)*100}, {round(a.get_actor_location().z/100)*100}) - Distance: {(a.get_actor_location().x**2 + a.get_actor_location().y**2 + a.get_actor_location().z**2)**0.5:.1f}') for a in actors]; "
        "print('COORDINATE_MAPPING_COMPLETE')"
    )
    return _execute_command(cmd, "map_coordinates_from_origin")
