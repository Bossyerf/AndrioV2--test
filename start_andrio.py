#!/usr/bin/env python3
"""
🚀 AndrioV2 Startup Script
Simple launcher for the agentic Unreal Engine AI assistant.

Use ``--chat`` to start a lightweight chat interface that keeps context across
turns and logs conversations for future learning.
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from andrio_config import get_config

def check_requirements():
    """Check if required dependencies are installed"""
    required_packages = [
        'chromadb', 'sentence_transformers', 'numpy', 
        'aiohttp', 'aiofiles', 'networkx', 'spacy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running"""
    import aiohttp
    
    async def test_ollama():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    return asyncio.run(test_ollama())

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="Launch AndrioV2")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start the simple chat interface instead of the full agent",
    )
    args = parser.parse_args()

    print("🤖 AndrioV2 Startup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print("✅ Python version OK")
    
    # Check requirements
    print("📦 Checking dependencies...")
    if not check_requirements():
        return
    
    print("✅ Dependencies OK")
    
    # Check Ollama
    print("🤖 Checking Ollama connection...")
    if not check_ollama():
        print("❌ Ollama not running or not accessible")
        print("💡 Start Ollama first: ollama serve")
        return
    
    print("✅ Ollama connection OK")
    
    # Check UE paths from configuration or environment
    config = get_config()
    ue_source_path = Path(config.get("UE_SOURCE_DIR", "")).expanduser()
    ue_install_path = Path(config.get("UE_INSTALL_DIR", "")).expanduser()
    
    print("📁 Checking UE paths...")
    if not ue_source_path.exists():
        print(f"⚠️  UE source path not found: {ue_source_path}")
        print("   Andrio will work but won't be able to study source code")
    else:
        print("✅ UE source path found")
    
    if not ue_install_path.exists():
        print(f"⚠️  UE installation path not found: {ue_install_path}")
        print("   Andrio will work but won't be able to study installation")
    else:
        print("✅ UE installation path found")
    
    # Launch chosen mode
    if args.chat:
        print("\n💬 Starting simple chat mode...")
        try:
            from simple_chat import chat_main
            asyncio.run(chat_main())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error launching chat mode: {e}")
    else:
        print("\n🚀 Launching AndrioV2...")
        try:
            from andrio_v2 import main as andrio_main
            asyncio.run(andrio_main())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error launching AndrioV2: {e}")

if __name__ == "__main__":
    main()
