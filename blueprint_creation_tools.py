"""
Blueprint Creation Tools for AndrioV2
Working Blueprint creation tools using upyrc for UE5 remote execution
"""

from upyrc import upyre
import os

class BlueprintCreationTools:
    def __init__(self, project_path=None):
        """Initialize Blueprint creation tools with UE5 remote execution"""
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
    
    def create_blueprint_actor(self, blueprint_name, package_path="/Game/", parent_class="Actor"):
        """Create a basic Blueprint Actor"""
        command = f'''
try:
    import unreal
    
    # Create a Blueprint Factory
    factory = unreal.BlueprintFactory()
    
    # Set parent class
    if "{parent_class}" == "Actor":
        factory.set_editor_property('parent_class', unreal.Actor)
    elif "{parent_class}" == "Pawn":
        factory.set_editor_property('parent_class', unreal.Pawn)
    elif "{parent_class}" == "Character":
        factory.set_editor_property('parent_class', unreal.Character)
    else:
        factory.set_editor_property('parent_class', unreal.Actor)
    
    # Create the asset
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    blueprint = asset_tools.create_asset(
        asset_name='{blueprint_name}',
        package_path='{package_path}',
        asset_class=unreal.Blueprint,
        factory=factory
    )
    
    if blueprint:
        print(f"✅ Created Blueprint: {{blueprint.get_name()}}")
        print(f"📁 Path: {{blueprint.get_path_name()}}")
        print("SUCCESS")
    else:
        print("❌ Failed to create Blueprint")
        print("FAILED")
        
except Exception as e:
    print(f"❌ Error: {{str(e)}}")
    print("FAILED")
'''
        
        return self._execute_command(command)
    
    def create_blueprint_with_mesh(self, blueprint_name, mesh_path=None, package_path="/Game/"):
        """Create a Blueprint with a Static Mesh Component"""
        mesh_setup = ""
        if mesh_path:
            mesh_setup = f'''
    # Add Static Mesh Component
    mesh_component = blueprint.get_blueprint_generated_class().get_default_object().add_component(unreal.StaticMeshComponent, "StaticMeshComponent")
    if mesh_component:
        mesh_asset = unreal.EditorAssetLibrary.load_asset("{mesh_path}")
        if mesh_asset:
            mesh_component.set_static_mesh(mesh_asset)
            print("✅ Added Static Mesh Component")
'''
        
        command = f'''
try:
    import unreal
    
    # Create a Blueprint Factory
    factory = unreal.BlueprintFactory()
    factory.set_editor_property('parent_class', unreal.Actor)
    
    # Create the asset
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    blueprint = asset_tools.create_asset(
        asset_name='{blueprint_name}',
        package_path='{package_path}',
        asset_class=unreal.Blueprint,
        factory=factory
    )
    
    if blueprint:
        print(f"✅ Created Blueprint: {{blueprint.get_name()}}")
        print(f"📁 Path: {{blueprint.get_path_name()}}")
        {mesh_setup}
        print("SUCCESS")
    else:
        print("❌ Failed to create Blueprint")
        print("FAILED")
        
except Exception as e:
    print(f"❌ Error: {{str(e)}}")
    print("FAILED")
'''
        
        return self._execute_command(command)
    
    def create_blueprint_from_template(self, blueprint_name, template_type="Basic", package_path="/Game/"):
        """Create a Blueprint from a template"""
        parent_classes = {
            "Basic": "unreal.Actor",
            "Pawn": "unreal.Pawn", 
            "Character": "unreal.Character",
            "GameMode": "unreal.GameModeBase",
            "PlayerController": "unreal.PlayerController",
            "Widget": "unreal.UserWidget"
        }
        
        parent_class = parent_classes.get(template_type, "unreal.Actor")
        
        command = f'''
try:
    import unreal
    
    # Create appropriate factory based on template type
    if "{template_type}" == "Widget":
        factory = unreal.WidgetBlueprintFactory()
    else:
        factory = unreal.BlueprintFactory()
        factory.set_editor_property('parent_class', {parent_class})
    
    # Create the asset
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    
    if "{template_type}" == "Widget":
        blueprint = asset_tools.create_asset(
            asset_name='{blueprint_name}',
            package_path='{package_path}',
            asset_class=unreal.WidgetBlueprint,
            factory=factory
        )
    else:
        blueprint = asset_tools.create_asset(
            asset_name='{blueprint_name}',
            package_path='{package_path}',
            asset_class=unreal.Blueprint,
            factory=factory
        )
    
    if blueprint:
        print(f"✅ Created {{'{template_type}'}} Blueprint: {{blueprint.get_name()}}")
        print(f"📁 Path: {{blueprint.get_path_name()}}")
        print("SUCCESS")
    else:
        print("❌ Failed to create Blueprint")
        print("FAILED")
        
except Exception as e:
    print(f"❌ Error: {{str(e)}}")
    print("FAILED")
'''
        
        return self._execute_command(command)
    
    def _execute_command(self, command):
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
                            "message": "Blueprint created successfully",
                            "output": output_text
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Blueprint creation failed",
                            "output": output_text
                        }
                else:
                    return {
                        "success": False,
                        "message": "Command execution failed",
                        "output": str(result.output)
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "message": f"Remote execution error: {str(e)}",
                "output": ""
            }

# Convenience functions for AndrioV2
def create_blueprint_actor(blueprint_name, package_path="/Game/", parent_class="Actor"):
    """Create a basic Blueprint Actor"""
    tools = BlueprintCreationTools()
    return tools.create_blueprint_actor(blueprint_name, package_path, parent_class)

def create_blueprint_with_mesh(blueprint_name, mesh_path=None, package_path="/Game/"):
    """Create a Blueprint with a Static Mesh Component"""
    tools = BlueprintCreationTools()
    return tools.create_blueprint_with_mesh(blueprint_name, mesh_path, package_path)

def create_blueprint_from_template(blueprint_name, template_type="Basic", package_path="/Game/"):
    """Create a Blueprint from a template"""
    tools = BlueprintCreationTools()
    return tools.create_blueprint_from_template(blueprint_name, template_type, package_path) 