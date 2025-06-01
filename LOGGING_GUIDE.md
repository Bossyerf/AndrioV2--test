# 📋 Andrio V2 Enhanced Logging Guide

## 🎯 **COMPLETE VISIBILITY INTO ANDRIO'S OPERATIONS**

Now you can see **EVERYTHING** that happens when Andrio runs! The enhanced logging system provides detailed insights into:

- 🔄 **Learning Cycles**: Every autonomous learning phase
- 🔧 **Tool Executions**: Every tool command with parameters and results  
- 📍 **Phase Transitions**: When Andrio advances between learning phases
- 👤 **User Interactions**: All commands and responses
- 🧠 **AI Reasoning**: Thinking processes and decision making
- 📊 **Progress Tracking**: Mastery updates and goal achievements

---

## 🚀 **HOW TO MONITOR ANDRIO'S ACTIVITY**

### **Method 1: Real-Time Log Viewer (Recommended)**

```bash
# Start the log viewer in a separate terminal
python view_andrio_logs.py

# Or with specific options
python view_andrio_logs.py --filter "LEARNING CYCLE"  # Filter for learning cycles
python view_andrio_logs.py --filter "TOOL EXECUTION"  # Filter for tool usage
python view_andrio_logs.py --no-color                 # Disable colors
python view_andrio_logs.py --stats                    # Show log statistics
```

### **Method 2: Direct Log File Monitoring**

```bash
# View the log file directly
tail -f andrio_v2.log

# Or on Windows PowerShell
Get-Content andrio_v2.log -Wait -Tail 10
```

### **Method 3: Log File Analysis**

```bash
# Show log statistics
python view_andrio_logs.py --stats

# Search for specific events
grep "PHASE ADVANCEMENT" andrio_v2.log
grep "TOOL EXECUTION" andrio_v2.log
grep "ERROR" andrio_v2.log
```

---

## 📊 **WHAT YOU'LL SEE IN THE LOGS**

### **🔄 Autonomous Learning Cycles**
```
🔄 === LEARNING CYCLE 1 START ===
📊 Current mastery: 0.0%
🎯 Target mastery: 60.0%
📍 Phase: installation_architecture
⏰ Cycle start time: 14:30:25
🎯 PHASE OBJECTIVE: Study UE5 installation structure and components
🏗️ EXECUTING: Installation Architecture Phase
```

### **🔧 Tool Executions**
```
==================================================
🔧 TOOL EXECUTION START
📋 TOOL NAME: list_files
📋 TOOL PARAMS: ['E:\\UE_5.5']
⏰ EXECUTION TIME: 14:30:26
------------------------------
🎯 TOOL NEEDS NO PARAMETERS: list_files
📋 FINAL PARAMETERS: ['E:\\UE_5.5']
🚀 EXECUTING WITH PARAMS: ['E:\\UE_5.5']
------------------------------
✅ TOOL 'list_files' EXECUTED SUCCESSFULLY
📄 TOOL RESULT TYPE: <class 'str'>
📄 TOOL RESULT LENGTH: 1247 characters
📄 TOOL RESULT (first 300 chars): ✅ Files in E:\UE_5.5:
📁 Directories:
   - Engine
   - FeaturePacks
   - Samples
   - Templates
📄 Files:
   - UE5.exe
   - UnrealEditor.exe
==================================================
```

### **📍 Phase Transitions**
```
🎉 PHASE ADVANCEMENT DETECTED!
   📍 OLD PHASE: installation_architecture
   📍 NEW PHASE: project_structure
   📈 Mastery at advancement: 10.2%
```

### **👤 User Interactions**
```
============================================================
👤 USER INPUT: autonomous
⏰ TIME: 14:30:15
📍 CURRENT PHASE: installation_architecture
------------------------------
🚀 USER REQUESTED AUTONOMOUS LEARNING
```

---

## 🎨 **COLOR-CODED LOG LEVELS**

The log viewer uses colors to make it easy to spot different types of activities:

- 🔴 **Red**: Errors and failures
- 🟡 **Yellow**: Warnings and issues  
- 🟢 **Green**: Successes and completions
- 🔵 **Blue**: Phase changes and objectives
- 🟣 **Magenta**: Learning cycles and progress
- 🔷 **Cyan**: Tool executions and operations
- ⚪ **White**: User inputs and interactions

---

## 🔍 **FILTERING AND SEARCHING**

### **Filter by Activity Type**
```bash
# Monitor only learning cycles
python view_andrio_logs.py --filter "LEARNING CYCLE"

# Monitor only tool executions  
python view_andrio_logs.py --filter "TOOL EXECUTION"

# Monitor only errors
python view_andrio_logs.py --filter "ERROR"

# Monitor only phase changes
python view_andrio_logs.py --filter "PHASE"

# Monitor only user interactions
python view_andrio_logs.py --filter "USER INPUT"
```

### **Search Historical Logs**
```bash
# Find all autonomous learning sessions
grep "AUTONOMOUS LEARNING" andrio_v2.log

# Find all successful tool executions
grep "EXECUTED SUCCESSFULLY" andrio_v2.log

# Find all phase advancements
grep "PHASE ADVANCEMENT" andrio_v2.log

# Find all mastery updates
grep "Updated mastery" andrio_v2.log
```

---

## 📈 **MONITORING LEARNING PROGRESS**

### **Key Metrics to Watch**
1. **Learning Cycles**: How many cycles Andrio completes
2. **Phase Progression**: Movement through learning phases
3. **Mastery Growth**: Percentage increases toward 60% target
4. **Tool Usage**: Which tools are being used most
5. **Success Rate**: Ratio of successful vs failed operations

### **Progress Indicators**
```
📈 Updated mastery: 15.3%
🎯 Progress to target: 25.5%
🔄 === LEARNING CYCLE 5 END ===
```

---

## 🛠️ **TROUBLESHOOTING**

### **No Logs Appearing**
1. Make sure Andrio V2 is actually running
2. Check that `andrio_v2.log` exists in the current directory
3. Verify logging is enabled (it should be by default now)

### **Log File Too Large**
```bash
# Clear the log file
> andrio_v2.log

# Or rotate logs
mv andrio_v2.log andrio_v2_backup.log
```

### **Missing Color Support**
```bash
# Disable colors if terminal doesn't support them
python view_andrio_logs.py --no-color
```

---

## 🎯 **USAGE EXAMPLES**

### **Scenario 1: Monitor Autonomous Learning**
```bash
# Terminal 1: Start Andrio
python andrio_v2.py

# Terminal 2: Monitor learning cycles
python view_andrio_logs.py --filter "LEARNING CYCLE"
```

### **Scenario 2: Debug Tool Issues**
```bash
# Monitor tool executions and errors
python view_andrio_logs.py --filter "TOOL\|ERROR"
```

### **Scenario 3: Track Progress**
```bash
# Monitor mastery and phase changes
python view_andrio_logs.py --filter "mastery\|PHASE"
```

---

## 📊 **LOG STATISTICS**

Get insights into Andrio's activity:

```bash
python view_andrio_logs.py --stats
```

This shows:
- Total log entries
- Error/warning counts
- Tool execution frequency
- Learning cycle statistics
- File size and modification time
- Recent activity summary

---

## 🎉 **RESULT**

You now have **COMPLETE VISIBILITY** into everything Andrio does:

✅ **Real-time monitoring** of all activities
✅ **Color-coded** log levels for easy scanning  
✅ **Filtering** to focus on specific activities
✅ **Historical analysis** of learning progress
✅ **Troubleshooting** capabilities for issues
✅ **Progress tracking** toward 60% mastery goal

**Run the log viewer alongside Andrio to see everything happening in real-time!** 🚀 