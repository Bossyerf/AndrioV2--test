"""
Auto-Generate UE5 Console Tools
Parses ConsoleHelp.html and creates working tools for all console commands
Tests them systematically with delays to prevent engine overload
"""

import re
import json
import time
from pathlib import Path
from upyrc import upyre
import os

class ConsoleToolGenerator:
    def __init__(self, html_file="ConsoleHelp.html", project_path=None):
        """Initialize the console tool generator"""
        self.html_file = html_file
        self.project_path = project_path or r"D:\Andrios Output\UnrealProjects\blueprintexperiment\blueprintexperiment.uproject"
        self.config = None
        self.commands = []
        self.generated_tools = {}
        self.test_results = {}
        self._setup_config()
    
    def _setup_config(self):
        """Setup UE5 remote execution configuration"""
        try:
            if os.path.exists(self.project_path):
                self.config = upyre.RemoteExecutionConfig.from_uproject_path(self.project_path)
            else:
                self.config = upyre.RemoteExecutionConfig(
                    multicast_group=("239.0.0.1", 6766),
                    multicast_bind_address="0.0.0.0"
                )
        except Exception as e:
            self.config = upyre.RemoteExecutionConfig(
                multicast_group=("239.0.0.1", 6766),
                multicast_bind_address="0.0.0.0"
            )
    
    def parse_console_help(self):
        """Parse ConsoleHelp.html to extract all console commands"""
        print("🔍 Parsing ConsoleHelp.html for console commands...")
        
        if not Path(self.html_file).exists():
            print(f"❌ Error: {self.html_file} not found!")
            return False
        
        with open(self.html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract command entries using regex
        # Looking for patterns like: {name: "CommandName", help:"Description", type:"Type"}
        command_pattern = r'\{name:\s*"([^"]+)",\s*help:\s*"([^"]*)",\s*type:\s*"(Cmd|Exec|Var)"\}'
        matches = re.findall(command_pattern, html_content)
        
        for name, help_text, cmd_type in matches:
            # Skip variables for now, focus on commands
            if cmd_type in ['Cmd', 'Exec']:
                self.commands.append({
                    'name': name,
                    'type': cmd_type,
                    'help': help_text,
                    'safe': self._is_safe_command(name, help_text)
                })
        
        print(f"✅ Found {len(self.commands)} console commands")
        print(f"📊 Safe commands: {len([c for c in self.commands if c['safe']])}")
        print(f"📊 Potentially risky: {len([c for c in self.commands if not c['safe']])}")
        
        return True
    
    def _is_safe_command(self, name, help_text):
        """Determine if a command is safe to test automatically"""
        # Commands that are definitely safe (read-only, stats, info)
        safe_patterns = [
            r'^stat\.',
            r'^show',
            r'^dump',
            r'^list',
            r'^get',
            r'^print',
            r'^log',
            r'stats?$',
            r'info$',
            r'help$',
            r'version$'
        ]
        
        # Commands that might be risky or cause crashes
        risky_patterns = [
            r'quit',
            r'exit',
            r'shutdown',
            r'restart',
            r'delete',
            r'remove',
            r'destroy',
            r'kill',
            r'crash',
            r'force',
            r'reset',
            r'clear',
            r'flush',
            r'reload',
            r'compile',
            r'build',
            r'cook',
            r'package',
            r'^AddWork$',  # CRITICAL: Causes division by zero crash in UE5
            r'work',       # Related work commands might be risky
            r'thread',     # Threading commands can be dangerous
            r'memory',     # Memory manipulation commands
            r'gc\.',       # Garbage collection commands
            r'malloc',     # Memory allocation commands
            r'free',       # Memory deallocation commands
            r'alloc',      # Allocation commands
            r'pool',       # Memory pool commands
        ]
        
        name_lower = name.lower()
        help_lower = help_text.lower()
        
        # Check if it's explicitly safe
        for pattern in safe_patterns:
            if re.search(pattern, name_lower):
                return True
        
        # Check if it's potentially risky
        for pattern in risky_patterns:
            if re.search(pattern, name_lower) or re.search(pattern, help_lower):
                return False
        
        # Default to safe for unknown commands (can be overridden)
        return True
    
    def generate_tool_function(self, command):
        """Generate a Python function for a console command"""
        func_name = self._sanitize_function_name(command['name'])
        
        # Create the function code
        function_code = f'''def {func_name}():
    """
    {command['help'][:100]}...
    Command: {command['name']}
    Type: {command['type']}
    """
    command = \'\'\'import unreal; unreal.SystemLibrary.execute_console_command(None, "{command['name']}"); print("✅ {command['name']} executed"); print("SUCCESS")\'\'\'
    return _execute_ue5_command(command, "{command['name']}")
'''
        
        return func_name, function_code
    
    def _sanitize_function_name(self, command_name):
        """Convert command name to valid Python function name"""
        # Replace dots, spaces, and special chars with underscores
        func_name = re.sub(r'[^a-zA-Z0-9_]', '_', command_name)
        # Remove consecutive underscores
        func_name = re.sub(r'_+', '_', func_name)
        # Remove leading/trailing underscores
        func_name = func_name.strip('_')
        # Ensure it doesn't start with a number
        if func_name and func_name[0].isdigit():
            func_name = 'cmd_' + func_name
        # Ensure it's not empty
        if not func_name:
            func_name = 'unknown_command'
        
        return func_name.lower()
    
    def generate_all_tools(self, safe_only=True):
        """Generate tool functions for all commands"""
        print(f"🔧 Generating tools for {'safe' if safe_only else 'all'} commands...")
        
        commands_to_process = [c for c in self.commands if not safe_only or c['safe']]
        
        # Header for the generated file
        file_content = '''"""
Auto-Generated UE5 Console Tools
Generated from ConsoleHelp.html
"""

from upyrc import upyre
import os

# Global config for remote execution
_config = None

def _setup_config():
    """Setup UE5 remote execution configuration"""
    global _config
    if _config is None:
        project_path = r"D:\\Andrios Output\\UnrealProjects\\blueprintexperiment\\blueprintexperiment.uproject"
        try:
            if os.path.exists(project_path):
                _config = upyre.RemoteExecutionConfig.from_uproject_path(project_path)
            else:
                _config = upyre.RemoteExecutionConfig(
                    multicast_group=("239.0.0.1", 6766),
                    multicast_bind_address="0.0.0.0"
                )
        except Exception as e:
            _config = upyre.RemoteExecutionConfig(
                multicast_group=("239.0.0.1", 6766),
                multicast_bind_address="0.0.0.0"
            )

def _execute_ue5_command(command, command_name):
    """Execute a command via UE5 remote execution"""
    _setup_config()
    try:
        with upyre.PythonRemoteConnection(_config) as conn:
            result = conn.execute_python_command(
                command,
                exec_type=upyre.ExecTypes.EXECUTE_STATEMENT,
                raise_exc=False
            )
            
            if result.success:
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

# Generated tool functions:

'''
        
        # Generate functions for each command
        function_names = []
        for command in commands_to_process:
            func_name, func_code = self.generate_tool_function(command)
            file_content += func_code + '\n'
            function_names.append((func_name, command))
            self.generated_tools[command['name']] = func_name
        
        # Add a master list at the end
        file_content += f'''
# Master list of all generated functions
ALL_CONSOLE_TOOLS = [
'''
        
        for func_name, command in function_names:
            file_content += f'    ("{command["name"]}", {func_name}, "{command["help"][:50]}..."),\n'
        
        file_content += ']\n'
        
        # Write the generated file
        output_file = "generated_console_tools.py"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        print(f"✅ Generated {len(function_names)} tools in {output_file}")
        return output_file, function_names
    
    def test_tools_systematically(self, function_names, delay_seconds=2, max_tests=50):
        """Test generated tools one by one with delays"""
        print(f"🧪 Testing {min(len(function_names), max_tests)} tools systematically...")
        print(f"⏱️  Delay between tests: {delay_seconds} seconds")
        print("=" * 60)
        
        # Import the generated module
        import importlib.util
        spec = importlib.util.spec_from_file_location("generated_console_tools", "generated_console_tools.py")
        generated_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_module)
        
        successful = 0
        failed = 0
        
        for i, (func_name, command) in enumerate(function_names[:max_tests]):
            print(f"\n🔍 Test {i+1}/{min(len(function_names), max_tests)}: {command['name']}")
            print(f"📝 Description: {command['help'][:80]}...")
            print("-" * 40)
            
            try:
                # Get the function from the generated module
                tool_func = getattr(generated_module, func_name)
                
                # Execute the tool
                result = tool_func()
                
                if result['success']:
                    print(f"✅ SUCCESS: {result['message']}")
                    successful += 1
                    self.test_results[command['name']] = 'SUCCESS'
                else:
                    print(f"❌ FAILED: {result['message']}")
                    failed += 1
                    self.test_results[command['name']] = 'FAILED'
                
            except Exception as e:
                print(f"💥 ERROR: {str(e)}")
                failed += 1
                self.test_results[command['name']] = f'ERROR: {str(e)}'
            
            # Delay between tests to prevent engine overload
            if i < len(function_names) - 1:  # Don't delay after the last test
                print(f"⏳ Waiting {delay_seconds} seconds...")
                time.sleep(delay_seconds)
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 SYSTEMATIC TESTING COMPLETE")
        print("=" * 60)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        total_tests = successful + failed
        if total_tests > 0:
            print(f"📈 Success Rate: {(successful/total_tests*100):.1f}%")
        else:
            print("📈 Success Rate: No tests performed")
        
        return successful, failed

def main():
    """Main execution function"""
    print("🚀 UE5 Console Tool Auto-Generator")
    print("=" * 50)
    
    generator = ConsoleToolGenerator()
    
    # Step 1: Parse the HTML file
    if not generator.parse_console_help():
        return
    
    # Step 2: Generate tools (safe commands only by default)
    output_file, function_names = generator.generate_all_tools(safe_only=True)
    
    # Step 3: Test tools systematically
    print(f"\n🎯 Ready to test {len(function_names)} generated tools")
    user_input = input("Press Enter to start systematic testing (or 'q' to quit): ")
    
    if user_input.lower() != 'q':
        successful, failed = generator.test_tools_systematically(
            function_names, 
            delay_seconds=2,  # 2 second delay between tests
            max_tests=50      # Test first 50 tools
        )
        
        # Save results
        results_file = "console_tool_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(generator.test_results, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
        print(f"🔧 Generated tools saved to {output_file}")

if __name__ == "__main__":
    main() 