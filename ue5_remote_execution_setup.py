"""UE5 Remote Execution Setup

This module contains helper functions to outline how to enable
remote Python execution in Unreal Engine. In this simplified
version the functions only return instructional text.
"""


def instructions() -> str:
    return (
        "Enable Remote Execution in your UE project via:\n"
        "1. Edit > Project Settings > Python\n"
        "2. Check 'Remote Execution' and set the multicast endpoint to 239.0.0.1:6766"
    )

