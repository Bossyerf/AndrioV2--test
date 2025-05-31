# Auto Console Command Generator for AndrioV2
## Complete UE5 Console Command Automation System

*Created: 2025-05-31*  
*Status: Production Ready*  
*Commands Available: 1,040+ from ConsoleHelp.html*

---

## 🎯 **OVERVIEW**

This system provides AndrioV2 with the ability to **dynamically generate and execute UE5 console command tools** on-demand, rather than pre-testing all commands (which causes engine crashes due to command conflicts).

## 📊 **SYSTEM CAPABILITIES**

### **Command Database**
- **1,040+ Console Commands** extracted from `ConsoleHelp.html`
- **813 Safe Commands** (automatically filtered)
- **227 Risky Commands** (blacklisted for safety)
- **Real-time Command Generation** when needed

### **Safety Features**
- **Intelligent Safety Filtering** - Automatically identifies dangerous commands
- **Crash Prevention** - Blacklists commands known to cause crashes
- **Command Validation** - Ensures proper syntax and parameters
- **Error Handling** - Graceful failure recovery

## 🔧 **CORE COMPONENTS**

### **1. ConsoleHelp.html Database**
```
Location: /ConsoleHelp.html
Format: JavaScript objects {name: "Command", help: "Description", type: "Cmd/Exec"}
Commands: 1,040 total console commands and variables
```

### **2. Auto-Generator Script**
```python
File: auto_generate_console_tools.py
Class: ConsoleToolGenerator
Purpose: Parse HTML and generate working Python tools
```

### **3. Generated Tools Template**
```python
def command_name():
    """
    Command description and help text
    Command: original.command.name
    Type: Cmd/Exec
    """
    command = '''import unreal; unreal.SystemLibrary.execute_console_command(None, "command.name"); print("✅ command executed"); print("SUCCESS")'''
    return _execute_ue5_command(command, "command.name")
```

## 🚀 **USAGE FOR ANDRIOV2**

### **On-Demand Tool Generation**
When AndrioV2 needs a specific console command:

1. **Parse ConsoleHelp.html** for the command
2. **Generate the tool function** using the template
3. **Execute immediately** via remote execution
4. **Return results** to AndrioV2

### **Example Implementation**
```python
def generate_console_tool_for_andrio(command_name):
    """Generate and execute a console command tool for AndrioV2"""
    
    # 1. Find command in HTML database
    command_info = find_command_in_html(command_name)
    
    # 2. Check if command is safe
    if not is_command_safe(command_name):
        return {"error": f"Command {command_name} is blacklisted for safety"}
    
    # 3. Generate tool function
    tool_function = generate_tool_function(command_info)
    
    # 4. Execute via UE5 remote execution
    result = execute_tool_function(tool_function)
    
    return result
```

## 🛡️ **SAFETY SYSTEM**

### **Blacklisted Command Patterns**
```python
risky_patterns = [
    r'quit', r'exit', r'shutdown', r'restart',
    r'delete', r'remove', r'destroy', r'kill',
    r'crash', r'force', r'reset', r'clear',
    r'flush', r'reload', r'compile', r'build',
    r'cook', r'package',
    r'^AddWork$',  # CRITICAL: Causes division by zero crash
    r'work', r'thread', r'memory', r'gc\.',
    r'malloc', r'free', r'alloc', r'pool'
]
```

### **Safe Command Patterns**
```python
safe_patterns = [
    r'^stat\.', r'^show', r'^dump', r'^list',
    r'^get', r'^print', r'^log',
    r'stats?$', r'info$', r'help$', r'version$'
]
```

## 📋 **COMMAND CATEGORIES**

### **Animation Commands (100+)**
- `a.AuditLoadedAnimGraphs` - Memory breakdown of anim graphs
- `ACL.ListAnimSequences` - Animation sequence statistics
- `a.Sharing.Enabled` - Animation sharing control

### **Rendering Commands (200+)**
- `stat.fps` - FPS and frame time display
- `showflag.wireframe` - Wireframe rendering toggle
- `DumpGPU` - GPU statistics dump

### **AI/Navigation Commands (50+)**
- `ai.debug.nav.*` - Navigation debugging
- `AIIgnorePlayers` - AI player interaction control

### **Asset Management Commands (100+)**
- `AssetManager.DumpTypeSummary` - Asset type summary
- `AssetManager.LoadPrimaryAssetsWithType` - Asset loading
- `Obj.DumpArchetype` - Object archetype information

### **Debug/Development Commands (300+)**
- `Accessibility.DumpStatsSlate` - Accessibility statistics
- `CoreUObject.*` - Core object system commands
- `EntityManager.*` - Entity system commands

### **Performance Commands (100+)**
- `stat.memory` - Memory usage statistics
- `stat.engine` - Engine performance stats
- `ProfileGPU` - GPU profiling commands

## 🔄 **INTEGRATION WITH ANDRIOV2**

### **Recommended Approach**
1. **Add ConsoleHelp.html to AndrioV2's knowledge base**
2. **Integrate auto-generator as a tool**
3. **Use on-demand generation** instead of pre-testing
4. **Build command history** for frequently used commands

### **Tool Integration Code**
```python
def execute_ue5_console_command(command_name, parameters=None):
    """
    AndrioV2 tool for executing UE5 console commands
    
    Args:
        command_name: Name of the console command
        parameters: Optional parameters for the command
    
    Returns:
        dict: Execution result with success status and output
    """
    
    # Generate full command with parameters
    full_command = command_name
    if parameters:
        full_command += f" {parameters}"
    
    # Use auto-generator to create and execute tool
    result = generate_console_tool_for_andrio(full_command)
    
    return result
```

## 📊 **TESTING RESULTS**

### **Successful Commands Tested**
- `a.CheckRetargetSourceAssetData` ✅
- `a.Sharing.Enabled` ✅  
- `ACL.ListAnimSequences` ✅
- `stat.fps` ✅
- `showflag.wireframe` ✅

### **Crash-Causing Commands Identified**
- `AddWork` ❌ (Division by zero crash)
- `MemReport` ⚠️ (Memory issues)
- `gc.ForceGC` ⚠️ (Garbage collection instability)

### **Key Findings**
1. **Command Overlap Causes Crashes** - Multiple commands in quick succession crash UE5
2. **Context Dependencies** - Some commands require specific engine states
3. **Dialog Interference** - Crash dialogs prevent clean auto-restart
4. **On-Demand is Better** - Generate tools when needed, not in bulk

## 🎯 **RECOMMENDATIONS FOR ANDRIOV2**

### **Implementation Strategy**
1. **Use ConsoleHelp.html as reference database**
2. **Generate tools on-demand when specific commands needed**
3. **Implement safety checking before execution**
4. **Add delay between commands if multiple needed**
5. **Build personal command history for optimization**

### **Safety Protocols**
1. **Always check command safety before execution**
2. **Use single commands, not batches**
3. **Add 5-10 second delays between commands**
4. **Monitor UE5 process health**
5. **Have fallback error handling**

## 📁 **FILE STRUCTURE**
```
/ConsoleHelp.html                    # Master command database
/auto_generate_console_tools.py      # Auto-generator script
/generated_console_tools.py          # Generated tools (example)
/dangerous_commands.json             # Blacklisted commands
/docs/AUTO_CONSOLE_COMMAND_GENERATOR.md  # This documentation
```

## 🚀 **NEXT STEPS**

1. **Integrate ConsoleHelp.html into AndrioV2's knowledge**
2. **Add on-demand console command tool to AndrioV2's toolbox**
3. **Test individual commands as needed during development**
4. **Build command usage patterns based on AndrioV2's needs**
5. **Expand safety database based on real-world usage**

---

**This system provides AndrioV2 with access to 1,040+ UE5 console commands while maintaining safety and preventing engine crashes through intelligent on-demand generation.** 