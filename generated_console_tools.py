"""Auto-Generated UE5 Console Tools
This module normally contains hundreds of automatically generated wrappers
for console commands. Only a few sample functions are provided here.
"""

from __future__ import annotations

from upyrc import upyre
from .auto_generate_console_tools import _execute_ue5_command


def a_checkretargetsourceassetdata():
    """Checks if Anim Sequences and Pose Assets RetargetSourceAsset is valid."""
    command = "import unreal; unreal.SystemLibrary.execute_console_command(None, 'a.CheckRetargetSourceAssetData'); print('✅ a.CheckRetargetSourceAssetData executed'); print('SUCCESS')"
    return _execute_ue5_command(command, "a.CheckRetargetSourceAssetData")


def stat_fps():
    """Display FPS statistics."""
    command = "import unreal; unreal.SystemLibrary.execute_console_command(None, 'stat fps'); print('✅ stat fps executed'); print('SUCCESS')"
    return _execute_ue5_command(command, "stat fps")
