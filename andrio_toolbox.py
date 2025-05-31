# andrio_toolbox.py
# AndrioV2 Toolbox - Native tools for the current architecture
# Extracted and adapted from the original AndrioV2 tools

import os
import subprocess
import platform
import shutil
import json
import time
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil
import logging

# Import Blueprint creation tools
from blueprint_creation_tools import BlueprintCreationTools

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
ANDRIO_OUTPUT_DIR = "D:\\Andrios Output"

def ensure_andrio_output_dir():
    """Ensure the Andrio output directory exists."""
    if not os.path.exists(ANDRIO_OUTPUT_DIR):
        os.makedirs(ANDRIO_OUTPUT_DIR, exist_ok=True)
    return ANDRIO_OUTPUT_DIR

# ==================== BLUEPRINT CREATION TOOLS ====================

class BlueprintTools:
    """Blueprint creation tools wrapper for AndrioV2"""
    
    def __init__(self):
        self.blueprint_tools = BlueprintCreationTools()
    
    def create_blueprint_actor(self, blueprint_name: str, package_path: str = "/Game/", parent_class: str = "Actor") -> str:
        """Create a basic Blueprint Actor"""
        try:
            result = self.blueprint_tools.create_blueprint_actor(blueprint_name, package_path, parent_class)
            if result.get('success'):
                return f"✅ Created Blueprint Actor: {blueprint_name}\n📁 Path: {package_path}\n🎯 Parent: {parent_class}"
            else:
                return f"❌ Failed to create Blueprint: {result.get('message', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error creating Blueprint Actor: {str(e)}"
    
    def create_blueprint_with_mesh(self, blueprint_name: str, mesh_path: str = None, package_path: str = "/Game/") -> str:
        """Create a Blueprint with a Static Mesh Component"""
        try:
            result = self.blueprint_tools.create_blueprint_with_mesh(blueprint_name, mesh_path, package_path)
            if result.get('success'):
                mesh_info = f"\n🎨 Mesh: {mesh_path}" if mesh_path else "\n🎨 Mesh: Default (none specified)"
                return f"✅ Created Blueprint with Mesh: {blueprint_name}\n📁 Path: {package_path}{mesh_info}"
            else:
                return f"❌ Failed to create Blueprint with mesh: {result.get('message', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error creating Blueprint with mesh: {str(e)}"
    
    def create_blueprint_from_template(self, blueprint_name: str, template_type: str = "Basic", package_path: str = "/Game/") -> str:
        """Create a Blueprint from a template (Basic, Pawn, Character, GameMode, PlayerController, Widget)"""
        try:
            result = self.blueprint_tools.create_blueprint_from_template(blueprint_name, template_type, package_path)
            if result.get('success'):
                return f"✅ Created {template_type} Blueprint: {blueprint_name}\n📁 Path: {package_path}\n📋 Template: {template_type}"
            else:
                return f"❌ Failed to create Blueprint from template: {result.get('message', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error creating Blueprint from template: {str(e)}"

# ==================== FILE OPERATIONS TOOLS ====================

class FileOperationsTools:
    """File and directory management tools for AndrioV2"""
    
    @staticmethod
    def show_andrio_output() -> str:
        """Show the contents of Andrio's centralized output directory."""
        try:
            output_dir = ensure_andrio_output_dir()
            
            if not os.path.exists(output_dir):
                return f"❌ Andrio output directory not found: {output_dir}"
            
            result = [f"📁 Andrio Output Directory: {output_dir}\n"]
            
            # List all items in the output directory
            items = []
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    # Count items in subdirectory
                    try:
                        subitem_count = len(os.listdir(item_path))
                        items.append(f"📁 {item}/ ({subitem_count} items)")
                    except:
                        items.append(f"📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    items.append(f"📄 {item} ({size} bytes)")
            
            if items:
                result.extend(items)
            else:
                result.append("📭 Directory is empty")
            
            return "\n".join(result)
        except Exception as e:
            return f"❌ Error accessing Andrio output directory: {e}"

    @staticmethod
    def create_andrio_workspace(workspace_name: str) -> str:
        """Create a new workspace folder in Andrio's output directory."""
        try:
            output_dir = ensure_andrio_output_dir()
            workspace_path = os.path.join(output_dir, workspace_name)
            
            if os.path.exists(workspace_path):
                return f"❌ Workspace '{workspace_name}' already exists at {workspace_path}"
            
            os.makedirs(workspace_path, exist_ok=True)
            return f"✅ Created workspace: {workspace_path}"
        except Exception as e:
            return f"❌ Error creating workspace: {e}"

    @staticmethod
    def list_drives() -> str:
        """List all available drives on the system."""
        if platform.system() == "Windows":
            drives = []
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append(drive)
            return "Available drives:\n" + "\n".join(drives)
        else:
            return "Drive listing is Windows-specific. Use list_files('/') for Unix systems."

    @staticmethod
    def list_files(directory: str = ".") -> str:
        """List files and folders in a directory."""
        try:
            items = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    items.append(f"📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    items.append(f"📄 {item} ({size} bytes)")
            
            return f"Contents of {directory}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing files: {e}"

    @staticmethod
    def read_file(filepath: str, encoding: str = "utf-8") -> str:
        """Read the contents of a text file."""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            return f"Contents of {filepath}:\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def write_file(filepath: str, content: str, mode: str = "w") -> str:
        """Write content to a file. Mode can be 'w' (overwrite) or 'a' (append)."""
        try:
            with open(filepath, mode, encoding="utf-8") as f:
                f.write(content)
            action = "Written to" if mode == "w" else "Appended to"
            return f"{action} {filepath} ({len(content)} characters)"
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def find_files(pattern: str, directory: str = ".") -> str:
        """Find files matching a pattern (supports wildcards like *.txt, *.py)."""
        try:
            search_path = os.path.join(directory, pattern)
            matches = glob.glob(search_path, recursive=True)
            
            if not matches:
                return f"No files found matching '{pattern}' in {directory}"
            
            return f"Found {len(matches)} files:\n" + "\n".join(matches)
        except Exception as e:
            return f"Error finding files: {e}"

    @staticmethod
    def file_info(filepath: str) -> str:
        """Get detailed information about a file or directory."""
        try:
            if not os.path.exists(filepath):
                return f"File/directory not found: {filepath}"
            
            stat = os.stat(filepath)
            info = []
            info.append(f"Path: {os.path.abspath(filepath)}")
            info.append(f"Type: {'Directory' if os.path.isdir(filepath) else 'File'}")
            info.append(f"Size: {stat.st_size} bytes")
            info.append(f"Created: {time.ctime(stat.st_ctime)}")
            info.append(f"Modified: {time.ctime(stat.st_mtime)}")
            
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(filepath)
                info.append(f"Extension: {ext}")
            
            return "\n".join(info)
        except Exception as e:
            return f"Error getting file info: {e}"

# ==================== EPIC LAUNCHER TOOLS ====================

class EpicLauncherTools:
    """Epic Games Launcher automation tools for AndrioV2"""
    
    @staticmethod
    def is_epic_launcher_running() -> bool:
        """Check if Epic Games Launcher is currently running."""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'EpicGamesLauncher.exe':
                    return True
            return False
        except:
            return False

    @staticmethod
    def launch_epic_games_launcher() -> str:
        """Launch the Epic Games Launcher and wait for it to start."""
        try:
            if platform.system() == "Windows":
                # Check if already running
                if EpicLauncherTools.is_epic_launcher_running():
                    # Try to bring Epic Launcher to foreground if it's hidden
                    try:
                        import win32gui
                        import win32con
                        
                        def enum_windows_callback(hwnd, windows):
                            if win32gui.IsWindowVisible(hwnd):
                                window_text = win32gui.GetWindowText(hwnd)
                                if "Epic Games Launcher" in window_text:
                                    windows.append(hwnd)
                            return True
                        
                        windows = []
                        win32gui.EnumWindows(enum_windows_callback, windows)
                        
                        if windows:
                            # Bring the first Epic window to foreground
                            hwnd = windows[0]
                            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                            win32gui.SetForegroundWindow(hwnd)
                            return "✅ Epic Games Launcher brought to foreground!"
                        else:
                            return "⚠️ Epic Games Launcher is running but window not found (may be in system tray)"
                            
                    except ImportError:
                        return "⚠️ Epic Games Launcher is running but can't bring to foreground (missing win32gui)"
                    except Exception as e:
                        return f"⚠️ Epic Games Launcher is running but can't bring to foreground: {e}"
                
                # Find Epic Games Launcher executable
                possible_paths = [
                    r"E:\Epic Games\Launcher\Portal\Binaries\Win64\EpicGamesLauncher.exe",
                    r"C:\Program Files (x86)\Epic Games\Launcher\Portal\Binaries\Win64\EpicGamesLauncher.exe",
                    r"C:\Program Files\Epic Games\Launcher\Portal\Binaries\Win64\EpicGamesLauncher.exe"
                ]
                
                epic_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        epic_path = path
                        break
                
                if not epic_path:
                    # Try to find it in registry or common locations
                    import winreg
                    try:
                        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Epic Games\EpicGamesLauncher")
                        install_location = winreg.QueryValueEx(key, "AppDataPath")[0]
                        epic_path = os.path.join(install_location, "Portal", "Binaries", "Win64", "EpicGamesLauncher.exe")
                        winreg.CloseKey(key)
                    except:
                        pass
                
                if epic_path and os.path.exists(epic_path):
                    subprocess.Popen([epic_path])
                    
                    # Wait for launcher to start
                    for i in range(15):  # Wait up to 15 seconds
                        time.sleep(1)
                        if EpicLauncherTools.is_epic_launcher_running():
                            return f"✅ Epic Games Launcher launched successfully from {epic_path}!"
                    
                    return f"⚠️ Epic Games Launcher started but may still be loading... (from {epic_path})"
                else:
                    return "❌ Epic Games Launcher executable not found. Please install Epic Games Launcher."
            else:
                return "❌ Epic Games Launcher is Windows-only"
        except Exception as e:
            return f"❌ Error launching Epic Games Launcher: {e}"

    @staticmethod
    def close_epic_games_launcher() -> str:
        """Close the Epic Games Launcher."""
        try:
            if platform.system() == "Windows":
                # Check if running
                if not EpicLauncherTools.is_epic_launcher_running():
                    return "✅ Epic Games Launcher is not running"
                
                # Close Epic Games Launcher process
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", "EpicGamesLauncher.exe"], 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    return "✅ Epic Games Launcher closed successfully!"
                else:
                    return "❌ Epic Games Launcher was not running or could not be closed"
            else:
                return "❌ Epic Games Launcher is Windows-only"
        except Exception as e:
            return f"❌ Error closing Epic Games Launcher: {e}"

    @staticmethod
    def check_epic_launcher_status() -> str:
        """Check the current status of Epic Games Launcher."""
        try:
            is_running = EpicLauncherTools.is_epic_launcher_running()
            status = "🟢 Running" if is_running else "🔴 Not Running"
            return f"Epic Games Launcher Status: {status}"
        except Exception as e:
            return f"❌ Error checking Epic Launcher status: {e}"

# ==================== UNREAL ENGINE TOOLS ====================

class UnrealEngineTools:
    """Unreal Engine development tools for AndrioV2"""
    
    @staticmethod
    def get_ue_installations() -> List[str]:
        """Get list of available UE installations."""
        installations = [
            "E:\\UE_5.6",
            "E:\\UE_5.5"
        ]
        return [path for path in installations if os.path.exists(path)]

    @staticmethod
    def create_unreal_project(project_name: str, template: str = "ThirdPersonBP", custom_path: str = None) -> str:
        """Create a new Unreal Engine project from a template."""
        try:
            if platform.system() != "Windows":
                return "❌ Unreal Engine project creation is Windows-only"
            
            # Ensure Andrio output directory exists
            ensure_andrio_output_dir()
            
            # Determine project path
            if custom_path:
                project_path = custom_path
            else:
                projects_dir = os.path.join(ANDRIO_OUTPUT_DIR, "UnrealProjects")
                os.makedirs(projects_dir, exist_ok=True)
                project_path = os.path.join(projects_dir, project_name)
            
            # Find available UE installation
            ue_installations = UnrealEngineTools.get_ue_installations()
            if not ue_installations:
                return "❌ No Unreal Engine installation found. Available paths checked: E:\\UE_5.6, E:\\UE_5.5"
            
            ue_path = ue_installations[0]  # Use first available
            
            # Template mapping
            template_map = {
                "ThirdPersonBP": "TP_ThirdPersonBP",
                "ThirdPerson": "TP_ThirdPerson", 
                "BlankBP": "TP_BlankBP",
                "Blank": "TP_Blank",
                "AEC_BlankBP": "TP_AEC_BlankBP"
            }
            
            if template not in template_map:
                available = ", ".join(template_map.keys())
                return f"❌ Invalid template '{template}'. Available: {available}"
            
            template_path = os.path.join(ue_path, "Templates", template_map[template])
            
            if not os.path.exists(template_path):
                return f"❌ Template '{template}' not found at {template_path}"
            
            # Check if project path already exists
            if os.path.exists(project_path):
                return f"❌ Project path already exists: {project_path}"
            
            # Create project directory
            os.makedirs(project_path, exist_ok=True)
            
            # Copy template to project location with progress tracking
            files_copied = 0
            total_files = 0
            
            # Count total files first
            for root, dirs, files in os.walk(template_path):
                total_files += len(files)
            
            if total_files == 0:
                return f"❌ Template directory is empty: {template_path}"
            
            # Copy files with error handling
            try:
                for item in os.listdir(template_path):
                    source = os.path.join(template_path, item)
                    destination = os.path.join(project_path, item)
                    
                    if os.path.isdir(source):
                        shutil.copytree(source, destination)
                        # Count files in copied directory
                        for root, dirs, files in os.walk(destination):
                            files_copied += len(files)
                    else:
                        shutil.copy2(source, destination)
                        files_copied += 1
                        
            except Exception as copy_error:
                # Clean up partial copy
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)
                return f"❌ Error copying template files: {copy_error}"
            
            # Find and rename the .uproject file
            old_uproject = None
            for file in os.listdir(project_path):
                if file.endswith('.uproject'):
                    old_uproject = file
                    break
            
            if old_uproject:
                old_path = os.path.join(project_path, old_uproject)
                new_path = os.path.join(project_path, f"{project_name}.uproject")
                try:
                    os.rename(old_path, new_path)
                except Exception as rename_error:
                    return f"❌ Error renaming project file: {rename_error}"
            else:
                return f"❌ No .uproject file found in template"
            
            # Update project file with correct name
            uproject_path = os.path.join(project_path, f"{project_name}.uproject")
            try:
                if os.path.exists(uproject_path):
                    with open(uproject_path, 'r') as f:
                        project_data = json.load(f)
                    
                    # Update project metadata
                    project_data["Description"] = f"Generated project: {project_name}"
                    project_data["Category"] = "Games"
                    
                    with open(uproject_path, 'w') as f:
                        json.dump(project_data, f, indent=4)
            except Exception as json_error:
                return f"⚠️ Project created but failed to update metadata: {json_error}"
            
            # Verify project was created successfully
            if not os.path.exists(uproject_path):
                return f"❌ Project creation failed - .uproject file not found"
            
            # Count final files to verify copy
            final_files = 0
            for root, dirs, files in os.walk(project_path):
                final_files += len(files)
            
            return f"✅ Project '{project_name}' created successfully!\n📁 Location: {project_path}\n📊 Files copied: {final_files}\n🎮 Template: {template}\n📝 Project file: {uproject_path}"
            
        except Exception as e:
            return f"❌ Unexpected error creating project: {str(e)}"

    @staticmethod
    def open_unreal_project(project_path: str) -> str:
        """Open an existing Unreal Engine project."""
        try:
            if platform.system() != "Windows":
                return "❌ Unreal Engine is Windows-only"
            
            # Handle different path formats
            if not project_path.endswith('.uproject'):
                # If it's a directory, look for .uproject file inside
                if os.path.isdir(project_path):
                    uproject_files = [f for f in os.listdir(project_path) if f.endswith('.uproject')]
                    if uproject_files:
                        project_path = os.path.join(project_path, uproject_files[0])
                    else:
                        return f"❌ No .uproject file found in directory: {project_path}"
                else:
                    return f"❌ Invalid project path. Must be a .uproject file or directory containing one: {project_path}"
            
            if not os.path.exists(project_path):
                # Try to find the project in common locations
                project_name = os.path.splitext(os.path.basename(project_path))[0]
                
                # Check Andrio output directory
                andrio_project_path = os.path.join(ANDRIO_OUTPUT_DIR, "UnrealProjects", project_name, f"{project_name}.uproject")
                if os.path.exists(andrio_project_path):
                    project_path = andrio_project_path
                else:
                    return f"❌ Project file not found: {project_path}\n💡 Searched in: {andrio_project_path}"
            
            # Find available UE installation
            ue_installations = UnrealEngineTools.get_ue_installations()
            if not ue_installations:
                return "❌ No Unreal Engine installation found"
            
            editor_path = os.path.join(ue_installations[0], "Engine", "Binaries", "Win64", "UnrealEditor.exe")
            
            if not os.path.exists(editor_path):
                return f"❌ Unreal Editor not found at {editor_path}"
            
            # Verify project file is valid
            try:
                with open(project_path, 'r') as f:
                    project_data = json.load(f)
                    engine_version = project_data.get("EngineAssociation", "Unknown")
            except Exception as e:
                return f"❌ Invalid or corrupted project file: {e}"
            
            # Launch the project
            try:
                process = subprocess.Popen([editor_path, project_path])
                
                # Wait a moment to see if it starts successfully
                time.sleep(3)
                
                # Check if process is still running (didn't crash immediately)
                if process.poll() is None:
                    return f"✅ Opening project: {os.path.basename(project_path)}\n📁 Location: {project_path}\n🎮 Engine Version: {engine_version}\n🚀 Editor PID: {process.pid}"
                else:
                    return f"❌ Unreal Editor crashed immediately. Return code: {process.returncode}"
                    
            except Exception as launch_error:
                return f"❌ Error launching Unreal Editor: {launch_error}"
            
        except Exception as e:
            return f"❌ Unexpected error opening project: {str(e)}"

    @staticmethod
    def list_unreal_templates() -> str:
        """List available Unreal Engine project templates."""
        try:
            ue_installations = UnrealEngineTools.get_ue_installations()
            if not ue_installations:
                return "❌ No Unreal Engine installation found"
            
            templates_dir = os.path.join(ue_installations[0], "Templates")
            if not os.path.exists(templates_dir):
                return f"❌ Templates directory not found: {templates_dir}"
            
            templates = []
            template_map = {
                "TP_ThirdPersonBP": "ThirdPersonBP - Third Person Blueprint template",
                "TP_ThirdPerson": "ThirdPerson - Third Person C++ template",
                "TP_BlankBP": "BlankBP - Blank Blueprint template",
                "TP_Blank": "Blank - Blank C++ template",
                "TP_AEC_BlankBP": "AEC_BlankBP - Architecture Blueprint template"
            }
            
            for item in os.listdir(templates_dir):
                if os.path.isdir(os.path.join(templates_dir, item)):
                    description = template_map.get(item, f"{item} - Template")
                    templates.append(f"  📋 {description}")
            
            if templates:
                return "🎮 Available Unreal Engine Templates:\n" + "\n".join(templates)
            else:
                return "❌ No templates found"
                
        except Exception as e:
            return f"❌ Error listing templates: {str(e)}"

    @staticmethod
    def get_unreal_engine_info() -> str:
        """Get information about available Unreal Engine installations."""
        try:
            installations = UnrealEngineTools.get_ue_installations()
            
            if not installations:
                return "❌ No Unreal Engine installations found"
            
            info = ["🎮 Unreal Engine Installations:\n"]
            
            for installation in installations:
                version = os.path.basename(installation)
                engine_path = os.path.join(installation, "Engine")
                editor_path = os.path.join(engine_path, "Binaries", "Win64", "UnrealEditor.exe")
                
                status = "✅ Ready" if os.path.exists(editor_path) else "❌ Incomplete"
                info.append(f"  📁 {version}: {installation} ({status})")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"❌ Error getting UE info: {str(e)}"

# ==================== TOOLBOX INTEGRATION ====================

class AndrioToolbox:
    """Main toolbox class that provides access to all tools"""
    
    def __init__(self):
        self.file_ops = FileOperationsTools()
        self.epic_launcher = EpicLauncherTools()
        self.unreal_engine = UnrealEngineTools()
        self.blueprint_tools = BlueprintTools()
        
        # Create tools dictionary for easy access
        self.tools = self.get_all_tools()
    
    def get_all_tools(self) -> Dict[str, Any]:
        """Get a dictionary of all available tools"""
        return {
            # File Operations
            "show_andrio_output": self.file_ops.show_andrio_output,
            "create_andrio_workspace": self.file_ops.create_andrio_workspace,
            "list_drives": self.file_ops.list_drives,
            "list_files": self.file_ops.list_files,
            "read_file": self.file_ops.read_file,
            "write_file": self.file_ops.write_file,
            "find_files": self.file_ops.find_files,
            "file_info": self.file_ops.file_info,
            
            # Epic Launcher
            "launch_epic_games_launcher": self.epic_launcher.launch_epic_games_launcher,
            "close_epic_games_launcher": self.epic_launcher.close_epic_games_launcher,
            "check_epic_launcher_status": self.epic_launcher.check_epic_launcher_status,
            
            # Unreal Engine
            "create_unreal_project": self.unreal_engine.create_unreal_project,
            "open_unreal_project": self.unreal_engine.open_unreal_project,
            "list_unreal_templates": self.unreal_engine.list_unreal_templates,
            "get_unreal_engine_info": self.unreal_engine.get_unreal_engine_info,
            
            # Blueprint Creation
            "create_blueprint_actor": self.blueprint_tools.create_blueprint_actor,
            "create_blueprint_with_mesh": self.blueprint_tools.create_blueprint_with_mesh,
            "create_blueprint_from_template": self.blueprint_tools.create_blueprint_from_template,
        }
    
    def get_tools_summary(self) -> str:
        """Get a formatted summary of all available tools"""
        descriptions = self.get_tool_descriptions()
        
        summary = ["🛠️ Available Tools (19 total):\n"]
        
        # File Operations
        summary.append("📁 File Operations (8 tools):")
        file_tools = [
            "show_andrio_output", "create_andrio_workspace", "list_drives", "list_files",
            "read_file", "write_file", "find_files", "file_info"
        ]
        for tool in file_tools:
            summary.append(f"  • {tool}: {descriptions[tool]}")
        
        summary.append("\n🎮 Epic Launcher (3 tools):")
        epic_tools = [
            "launch_epic_games_launcher", "close_epic_games_launcher", "check_epic_launcher_status"
        ]
        for tool in epic_tools:
            summary.append(f"  • {tool}: {descriptions[tool]}")
        
        summary.append("\n🏗️ Unreal Engine (4 tools):")
        ue_tools = [
            "create_unreal_project", "open_unreal_project", "list_unreal_templates", 
            "get_unreal_engine_info"
        ]
        for tool in ue_tools:
            summary.append(f"  • {tool}: {descriptions[tool]}")
        
        summary.append("\n🎨 Blueprint Creation (3 tools):")
        blueprint_tools = [
            "create_blueprint_actor", "create_blueprint_with_mesh", "create_blueprint_from_template"
        ]
        for tool in blueprint_tools:
            summary.append(f"  • {tool}: {descriptions[tool]}")
        
        summary.append("\n💡 Usage: Call any tool by name, e.g., create_blueprint_actor('MyActor')")
        
        return "\n".join(summary)
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools"""
        return {
            "show_andrio_output": "Show contents of Andrio's output directory",
            "create_andrio_workspace": "Create a new workspace folder",
            "list_drives": "List all available system drives",
            "list_files": "List files and folders in a directory",
            "read_file": "Read the contents of a text file",
            "write_file": "Write content to a file",
            "find_files": "Find files matching a pattern",
            "file_info": "Get detailed information about a file",
            "launch_epic_games_launcher": "Launch the Epic Games Launcher",
            "close_epic_games_launcher": "Close the Epic Games Launcher",
            "check_epic_launcher_status": "Check Epic Launcher status",
            "create_unreal_project": "Create a new Unreal Engine project",
            "open_unreal_project": "Open an existing UE project",
            "list_unreal_templates": "List available UE project templates",
            "get_unreal_engine_info": "Get UE installation information",
            "create_blueprint_actor": "Create a basic Blueprint Actor",
            "create_blueprint_with_mesh": "Create a Blueprint with a Static Mesh Component",
            "create_blueprint_from_template": "Create a Blueprint from a template",
        } 