#!/usr/bin/env python3
"""
Andrio V2 Log Viewer
===================

Real-time log viewer for monitoring Andrio's learning activities.
Shows all learning phases, tool executions, and progress updates.

Usage:
    python view_andrio_logs.py
    
Features:
- Real-time log monitoring
- Color-coded log levels
- Filter by log types
- Search functionality
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

def get_log_color(line):
    """Get color code for log line based on content"""
    if "ERROR" in line or "❌" in line:
        return "\033[91m"  # Red
    elif "WARNING" in line or "⚠️" in line:
        return "\033[93m"  # Yellow
    elif "SUCCESS" in line or "✅" in line:
        return "\033[92m"  # Green
    elif "TOOL EXECUTION" in line or "🔧" in line:
        return "\033[96m"  # Cyan
    elif "LEARNING CYCLE" in line or "🔄" in line:
        return "\033[95m"  # Magenta
    elif "PHASE" in line or "📍" in line:
        return "\033[94m"  # Blue
    elif "USER INPUT" in line or "👤" in line:
        return "\033[97m"  # White
    else:
        return "\033[0m"   # Default

def reset_color():
    """Reset terminal color"""
    return "\033[0m"

def should_show_line(line, activity_filter=None):
    """Check if line should be shown based on filter"""
    if activity_filter is None:
        return True
    return activity_filter.lower() in line.lower()

def colorize_log_line(line):
    """Apply color to log line"""
    color = get_log_color(line)
    return f"{color}{line}{reset_color()}"

def tail_log_file(log_file, follow=True, filter_text=None, show_colors=True):
    """Tail the log file and display with colors"""
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("💡 Make sure Andrio is running to generate logs")
        return
    
    print(f"📋 Monitoring Andrio V2 logs: {log_file}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    if filter_text:
        print(f"🔍 Filtering for: {filter_text}")
    print("=" * 80)
    
    line_count = 0
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            if follow:
                # Follow mode - tail the file
                f.seek(0, 2)  # Go to end of file
                while True:
                    line = f.readline()
                    if line:
                        if should_show_line(line, filter_text):
                            if show_colors:
                                print(colorize_log_line(line.strip()))
                            else:
                                print(line.strip())
                            line_count += 1
                            sys.stdout.flush()
                    else:
                        time.sleep(0.1)
            else:
                # Static mode - read all lines
                lines = f.readlines()
                for line in lines:
                    if should_show_line(line, filter_text):
                        if show_colors:
                            print(colorize_log_line(line.strip()))
                        else:
                            print(line.strip())
                        line_count += 1
                    
    except KeyboardInterrupt:
        print(f"\n👋 Log monitoring stopped. Displayed {line_count} lines.")
    except Exception as e:
        print(f"❌ Error reading log file: {e}")

def show_log_stats(log_file):
    """Show statistics about the log file"""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Count different types of log entries
        stats = {
            "Total Lines": total_lines,
            "Errors": sum(1 for line in lines if "ERROR" in line or "❌" in line),
            "Warnings": sum(1 for line in lines if "WARNING" in line or "⚠️" in line),
            "Tool Executions": sum(1 for line in lines if "TOOL EXECUTION" in line),
            "Learning Cycles": sum(1 for line in lines if "LEARNING CYCLE" in line),
            "User Inputs": sum(1 for line in lines if "USER INPUT" in line),
            "Phase Changes": sum(1 for line in lines if "PHASE ADVANCEMENT" in line),
        }
        
        print(f"📊 Log File Statistics: {log_file}")
        print("=" * 50)
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Show file size and modification time
        file_stat = os.stat(log_file)
        file_size = file_stat.st_size
        mod_time = datetime.fromtimestamp(file_stat.st_mtime)
        
        print(f"\n📁 File Info:")
        print(f"   Size: {file_size:,} bytes")
        print(f"   Last Modified: {mod_time.strftime('%Y-%m-%d %I:%M:%S %p')}")
        print(f"   Lines: {total_lines}")
        
        # Show recent activity (last 5 lines)
        if lines:
            print(f"\n📋 Recent Activity (last 5 lines):")
            for line in lines[-5:]:
                print(f"   {line.strip()}")
        
    except Exception as e:
        print(f"❌ Error analyzing log file: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Andrio V2 Log Viewer")
    parser.add_argument("--log-file", "-f", default="andrio_v2.log", 
                       help="Path to log file (default: andrio_v2.log)")
    parser.add_argument("--follow", "-F", action="store_true", default=True,
                       help="Follow log file for real-time monitoring (default: True)")
    parser.add_argument("--no-follow", action="store_true",
                       help="Don't follow log file, just show existing content")
    parser.add_argument("--filter", "-g", 
                       help="Filter lines containing this text")
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output")
    parser.add_argument("--stats", "-s", action="store_true",
                       help="Show log file statistics and exit")
    
    args = parser.parse_args()
    
    # Handle follow logic
    follow = args.follow and not args.no_follow
    
    log_file = args.log_file
    
    if args.stats:
        show_log_stats(log_file)
        return
    
    print("🤖 Andrio V2 Log Viewer")
    print("=" * 50)
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("\n💡 Tips:")
        print("   1. Make sure Andrio V2 is running")
        print("   2. Check if the log file path is correct")
        print("   3. Try running Andrio first to generate logs")
        return
    
    print("📋 Controls:")
    print("   Ctrl+C: Stop monitoring")
    if follow:
        print("   Real-time: Following new log entries")
    else:
        print("   Static: Showing existing log content")
    
    if args.filter:
        print(f"   Filter: {args.filter}")
    
    print()
    
    tail_log_file(
        log_file=log_file,
        follow=follow,
        filter_text=args.filter,
        show_colors=not args.no_color
    )

if __name__ == "__main__":
    main() 