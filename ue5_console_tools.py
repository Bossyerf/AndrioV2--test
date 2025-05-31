"""
UE5 Console Command Tools for AndrioV2
Random selection of 5 console commands to test remote execution
"""

from upyrc import upyre
import os

class UE5ConsoleTools:
    def __init__(self, project_path=None):
        """Initialize UE5 console command tools with remote execution"""
        self.project_path = project_path or r"D:\Andrios Output\UnrealProjects\blueprintexperiment\blueprintexperiment.uproject"
        self.config = None
        self._setup_config()
    
    def _setup_config(self):
        """Setup the remote execution configuration"""
        try:
            if os.path.exists(self.project_path):
                self.config = upyre.RemoteExecutionConfig.from_uproject_path(self.project_path)
            else:
                # Fallback to manual config
                self.config = upyre.RemoteExecutionConfig(
                    multicast_group=("239.0.0.1", 6766),
                    multicast_bind_address="0.0.0.0"
                )
        except Exception as e:
            # Fallback to manual config
            self.config = upyre.RemoteExecutionConfig(
                multicast_group=("239.0.0.1", 6766),
                multicast_bind_address="0.0.0.0"
            )
    
    def stat_fps(self):
        """Show FPS and frame time statistics"""
        command = '''import unreal; unreal.SystemLibrary.execute_console_command(None, "stat fps"); print("✅ FPS statistics display toggled"); print("SUCCESS")'''
        return self._execute_command(command, "FPS Statistics")
    
    def dump_gpu_stats(self):
        """Dump GPU rendering statistics to log"""
        command = '''import unreal; unreal.SystemLibrary.execute_console_command(None, "DumpGPU"); print("✅ GPU statistics dumped to log"); print("SUCCESS")'''
        return self._execute_command(command, "GPU Statistics Dump")
    
    def list_loaded_assets(self):
        """List all currently loaded assets"""
        command = '''import unreal; unreal.SystemLibrary.execute_console_command(None, "AssetManager.DumpLoadedAssets"); print("✅ Loaded assets list dumped to log"); print("SUCCESS")'''
        return self._execute_command(command, "List Loaded Assets")
    
    def toggle_wireframe(self):
        """Toggle wireframe rendering mode"""
        command = '''import unreal; unreal.SystemLibrary.execute_console_command(None, "showflag.wireframe"); print("✅ Wireframe rendering mode toggled"); print("SUCCESS")'''
        return self._execute_command(command, "Toggle Wireframe")
    
    def memory_report(self):
        """Generate detailed memory usage report"""
        command = '''import unreal; unreal.SystemLibrary.execute_console_command(None, "MemReport"); print("✅ Memory report generated and logged"); print("SUCCESS")'''
        return self._execute_command(command, "Memory Report")
    
    def _execute_command(self, command, command_name):
        """Execute a command via UE5 remote execution"""
        try:
            with upyre.PythonRemoteConnection(self.config) as conn:
                result = conn.execute_python_command(
                    command,
                    exec_type=upyre.ExecTypes.EXECUTE_STATEMENT,
                    raise_exc=False
                )
                
                if result.success:
                    # Check if the command was successful
                    output_text = ""
                    for output_item in result.output:
                        output_text += output_item.get('output', '')
                    
                    if "SUCCESS" in output_text:
                        return {
                            "success": True,
                            "message": f"{command_name} executed successfully",
                            "output": output_text,
                            "command": command_name
                        }
                    else:
                        return {
                            "success": False,
                            "message": f"{command_name} execution completed but may have issues",
                            "output": output_text,
                            "command": command_name
                        }
                else:
                    return {
                        "success": False,
                        "message": f"{command_name} execution failed",
                        "output": str(result.output),
                        "command": command_name
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "message": f"Remote execution error for {command_name}: {str(e)}",
                "output": "",
                "command": command_name
            }

# Convenience functions for AndrioV2
def show_fps_stats():
    """Show FPS and frame time statistics"""
    tools = UE5ConsoleTools()
    return tools.stat_fps()

def dump_gpu_statistics():
    """Dump GPU rendering statistics to log"""
    tools = UE5ConsoleTools()
    return tools.dump_gpu_stats()

def list_all_loaded_assets():
    """List all currently loaded assets"""
    tools = UE5ConsoleTools()
    return tools.list_loaded_assets()

def toggle_wireframe_mode():
    """Toggle wireframe rendering mode"""
    tools = UE5ConsoleTools()
    return tools.toggle_wireframe()

def generate_memory_report():
    """Generate detailed memory usage report"""
    tools = UE5ConsoleTools()
    return tools.memory_report() 