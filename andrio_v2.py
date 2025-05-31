"""
AndrioV2 - Advanced Agentic Unreal Engine AI Assistant
=====================================================

A sophisticated AI assistant designed to autonomously learn and master Unreal Engine
through systematic study, experimentation, and hands-on practice.

Features:
- Autonomous UE source code study
- Bidirectional RAG learning system  
- Agentic task planning and execution
- Systematic experimentation with UE
- Progress tracking toward mastery goals
- Native tool integration for UE development

Author: Advanced AI Development Team
Version: 2.0
"""

import asyncio
import json
import logging
import os
import random
import re
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

import chromadb
import numpy as np
import ollama
import requests
from sentence_transformers import SentenceTransformer
import aiohttp
import aiofiles
import networkx as nx
import torch

# Import the toolbox
from andrio_toolbox import AndrioToolbox

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class LearningPhase(Enum):
    """Andrio's learning phases"""
    HANDS_ON_LEARNING = "hands_on_learning"
    STUDYING_SOURCE = "studying_source"
    STUDYING_INSTALLATIONS = "studying_installations"
    EXPERIMENTING = "experimenting"
    INTERACTIVE = "interactive"

@dataclass
class Task:
    """Agentic task representation"""
    id: str
    description: str
    goal: str
    steps: List[str]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = None
    learning_extracted: List[str] = None
    next_tasks: List[str] = None

@dataclass
class LearningGoal:
    """Learning objective for Andrio"""
    id: str
    domain: str  # e.g., "blueprints", "cpp_integration", "rendering"
    description: str
    target_mastery: float  # 0.0 to 1.0
    current_mastery: float = 0.0
    experiments_completed: int = 0
    key_concepts: List[str] = None
    created_at: datetime = None

class AndrioV2:
    """
    🤖 Agentic Unreal Engine AI Assistant
    
    Combines the power of unified bidirectional RAG with agentic capabilities:
    - Autonomous learning through UE source code study
    - Systematic experimentation with UE
    - Task planning and execution
    - Bidirectional knowledge enhancement
    - Goal-oriented behavior toward 60% UE mastery
    """
    
    def __init__(
        self,
        model_name: str = "andriov2-qwen3",
        embedding_model: str = "all-MiniLM-L6-v2",
        ue_source_path: str = "D:\\UeSource-study",
        ue_installation_path: str = "E:\\UE_5.5",
        target_mastery: float = 0.6
    ):
        # Core settings
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"

        # CRITICAL DISTINCTION:
        # ue_source_path = ACTUAL UE SOURCE CODE for studying (D:\\UeSource-study)
        # ue_installation_path = UE INSTALLATION for running engine (E:\\UE_5.5)
        self.ue_source_path = Path(ue_source_path)  # REAL SOURCE CODE
        self.ue_installation_path = Path(ue_installation_path)  # ENGINE INSTALLATION
        self.target_mastery = target_mastery

        # Learning state
        self.current_phase = LearningPhase.HANDS_ON_LEARNING
        self.learning_goals = {}
        self.completed_tasks = {}  # Change from [] to {} to make it a dictionary
        self.current_task = None
        self.overall_mastery = 0.0
        
        # File tracking to prevent repetitive studying
        self.studied_files = set()  # Track files we've already studied
        self.study_session_count = 0  # Track study sessions
        self.last_study_patterns = []  # Track recent study patterns

        # Thinking mode settings (Ollama 0.9.0)
        self.thinking_mode_compatible = self._check_thinking_compatibility()
        self.thinking_mode_enabled = True  # Enable by default if compatible

        # System prompt for LLM interactions
        self.system_prompt = """You are Andrio V2, an advanced agentic AI assistant specialized in Unreal Engine development and learning.

Your core capabilities:
- Autonomous UE learning through hands-on experimentation
- Bidirectional RAG knowledge system
- Agentic task planning and execution
- Real-time UE source code analysis
- Progressive skill development

You have access to 16 native tools for UE development:
- File operations (8 tools)
- Epic Launcher management (3 tools) 
- Unreal Engine project tools (5 tools)

Always prioritize hands-on learning over theoretical study. Use tools to create real UE projects and experiments."""

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Check for GPU availability
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to('cuda')
            logger.info("Using GPU for embeddings")
        else:
            logger.info("Using CPU for embeddings")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./andrio_knowledge_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="andrio_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize knowledge graph and entities (needed for RAG processing)
        self.knowledge_graph = nx.MultiDiGraph()
        self.entities = {}
        self.relationships = {}
        
        # Thread pool for CPU-bound tasks
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Bidirectional learning settings
        self.enable_learning = True
        self.feedback_patterns = {
            "STORE_DOCUMENT": r"STORE_DOCUMENT:\s*(.+?)(?:\n|$)",
            "CREATE_RELATIONSHIP": r"CREATE_RELATIONSHIP:\s*(.+?)(?:\n|$)",
            "LEARN": r"LEARN:\s*(.+?)(?:\n|$)",
            "PLAN_TASK": r"PLAN_TASK:\s*(.+?)(?:\n|$)",
            "REFLECT": r"REFLECT:\s*(.+?)(?:\n|$)"
        }
        
        # Task management
        self.active_tasks = {}  # Track active tasks by ID

        # Initialize toolbox
        try:
            from andrio_toolbox import AndrioToolbox
            self.toolbox = AndrioToolbox()
            self.available_tools = self.toolbox.tools
            self.tool_descriptions = self.toolbox.get_tool_descriptions()
            logger.info(f"Available tools: {len(self.available_tools)}")
        except Exception as e:
            logger.error(f"Failed to initialize toolbox: {e}")
            self.toolbox = None
            self.available_tools = {}
            self.tool_descriptions = {}

        # Initialize learning goals
        self._initialize_learning_goals()
        
        logger.info("AndrioV2 initialized successfully")
        logger.info(f"Current phase: {self.current_phase.value}")
        logger.info(f"Target mastery: {self.target_mastery * 100}%")
        logger.info(f"REAL UE SOURCE CODE: {self.ue_source_path}")
        logger.info(f"UE INSTALLATION: {self.ue_installation_path}")
        logger.info(f"Available tools: {len(self.available_tools)}")
    
    def _initialize_learning_goals(self):
        """Initialize learning goals for different UE domains"""
        domains = [
            ("blueprints", "Visual scripting system mastery"),
            ("cpp_integration", "C++ programming with UE"),
            ("rendering", "Lumen, Nanite, and rendering pipeline"),
            ("physics", "Chaos physics system"),
            ("animation", "Animation blueprints and systems"),
            ("ai_behavior", "AI, behavior trees, and perception"),
            ("networking", "Multiplayer and replication"),
            ("performance", "Optimization and profiling"),
            ("tools", "Editor tools and plugins"),
            ("assets", "Asset pipeline and content creation")
        ]
        
        for domain, description in domains:
            goal = LearningGoal(
                id=f"goal_{domain}",
                domain=domain,
                description=description,
                target_mastery=self.target_mastery,
                key_concepts=[],
                created_at=datetime.now()
            )
            self.learning_goals[domain] = goal
        
        logger.info(f"Initialized {len(self.learning_goals)} learning goals")

    def _check_thinking_compatibility(self) -> bool:
        """Check if the current model supports thinking mode (Ollama 0.9.0 feature)"""
        compatible_models = ["deepseek-r1", "qwen3", "qwen2.5"]
        model_lower = self.model_name.lower()
        
        for compatible in compatible_models:
            if compatible in model_lower:
                logger.info(f"🧠 Thinking mode compatible model detected: {self.model_name}")
                return True
        
        logger.info(f"ℹ️  Model {self.model_name} may not support thinking mode")
        return False

    def toggle_thinking_mode(self) -> str:
        """Toggle thinking mode on/off"""
        if not self.thinking_mode_compatible:
            return "⚠️  Thinking mode not supported by current model. Use DeepSeek-R1 or Qwen3 for thinking mode."
        
        self.thinking_mode_enabled = not self.thinking_mode_enabled
        status = "enabled" if self.thinking_mode_enabled else "disabled"
        return f"🧠 Thinking mode {status}"

    def get_thinking_status(self) -> str:
        """Get current thinking mode status"""
        if not self.thinking_mode_compatible:
            return "❌ Not supported by current model"
        return "✅ Enabled" if self.thinking_mode_enabled else "❌ Disabled"

    # ==================== AGENTIC TASK SYSTEM ====================

    async def plan_and_execute_task(self, goal_description: str) -> Dict[str, Any]:
        """Agentic task planning and execution"""
        start_time = time.time()

        # 1. Plan the task using AI
        task_plan = await self._plan_task_with_ai(goal_description)

        if not task_plan:
            return {"success": False, "error": "Failed to plan task"}

        # 2. Create task object
        task = Task(
            id=f"task_{int(time.time())}_{hash(goal_description)}",
            description=goal_description,
            goal=task_plan.get("goal", goal_description),
            steps=task_plan.get("steps", []),
            status=TaskStatus.PLANNED,
            created_at=datetime.now(),
            results={},
            learning_extracted=[],
            next_tasks=[]
        )

        self.active_tasks[task.id] = task
        logger.info(f"📋 Planned task: {task.description}")
        logger.info(f"🎯 Goal: {task.goal}")
        logger.info(f"📝 Steps: {len(task.steps)}")

        # 3. Execute the task
        execution_result = await self._execute_task(task)

        # 4. Reflect on results and plan next steps
        reflection_result = await self._reflect_on_task(task)

        # 5. Update learning goals based on results
        await self._update_learning_goals(task)

        return {
            "success": execution_result["success"],
            "task_id": task.id,
            "execution_result": execution_result,
            "reflection": reflection_result,
            "total_time": time.time() - start_time,
            "learning_extracted": task.learning_extracted,
            "next_tasks": task.next_tasks
        }

    async def _plan_task_with_ai(self, goal_description: str) -> Optional[Dict[str, Any]]:
        """Use AI to plan a task systematically"""
        try:
            # Get current context about learning state
            context = self._get_learning_context()

            prompt = f"""As Andrio V2, plan how to accomplish this goal systematically:

GOAL: {goal_description}

CURRENT CONTEXT:
{context}

Plan this task by breaking it down into specific, actionable steps. Consider:
1. What UE knowledge/skills are needed
2. What files/systems to study
3. What experiments to conduct
4. How to measure success

Respond ONLY with valid JSON (no extra text before or after):
{{
    "goal": "Clear, specific goal statement",
    "domain": "UE domain this relates to (blueprints, cpp, rendering, etc.)",
    "steps": [
        "Step 1: Specific action to take",
        "Step 2: Next specific action",
        "Step 3: Additional action"
    ],
    "success_criteria": "How to know if task succeeded",
    "estimated_time": "Estimated time to complete",
    "required_knowledge": ["concept1", "concept2"],
    "tools_needed": ["tool1", "tool2"]
}}"""

            logger.info("🧠 Querying AI for task planning...")
            response = await self._query_llm(prompt)
            logger.info(f"📝 AI Response length: {len(response)} characters")

            if not response or len(response.strip()) < 10:
                logger.error("❌ AI response is empty or too short")
                return self._create_fallback_plan(goal_description)

            # Log the raw response for debugging
            logger.info(f"🔍 Raw AI response: {response[:200]}...")

            # Try to parse JSON response
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]

                logger.info(f"🔧 Extracted JSON: {json_str[:100]}...")

                try:
                    parsed_plan = json.loads(json_str)
                    logger.info("✅ Successfully parsed AI plan")
                    return parsed_plan
                except json.JSONDecodeError as je:
                    logger.error(f"❌ JSON parsing failed: {je}")
                    logger.error(f"🔍 Problematic JSON: {json_str}")
                    return self._create_fallback_plan(goal_description)
            else:
                logger.error("❌ No JSON structure found in AI response")
                return self._create_fallback_plan(goal_description)

        except Exception as e:
            logger.error(f"❌ Failed to plan task: {e}")
            return self._create_fallback_plan(goal_description)

    def _create_fallback_plan(self, goal_description: str) -> Dict[str, Any]:
        """Create a simple fallback plan when AI planning fails"""
        logger.info("🔄 Creating fallback plan...")

        # Determine domain based on keywords
        domain = self._determine_domain_from_description(goal_description)

        # Create basic steps based on current phase
        if self.current_phase == LearningPhase.HANDS_ON_LEARNING:
            steps = [
                f"Try creating a new UE project using {self.ue_installation_path} engine",
                "Experiment with basic UE operations (open project, navigate interface)",
                "Attempt simple tasks like placing objects or creating blueprints",
                "If stuck on any operation, then study relevant source code for solutions"
            ]
        elif self.current_phase == LearningPhase.STUDYING_SOURCE:
            steps = [
                f"Study REAL UE source files in {self.ue_source_path} (actual engine source code)",
                "Read and analyze selected .h and .cpp source files",
                "Extract key UE classes, functions, and patterns from source code",
                "Document learning insights from actual UE source code"
            ]
        elif self.current_phase == LearningPhase.STUDYING_INSTALLATIONS:
            steps = [
                f"Explore UE installation directory at {self.ue_installation_path} (engine binaries)",
                "Identify important engine tools and components in installation",
                "Analyze configuration and setup files in installation",
                "Document installation structure insights"
            ]
        else:
            steps = [
                "Research the topic using available knowledge",
                "Plan specific actions to take",
                "Execute the planned actions",
                "Evaluate results and extract learning"
            ]

        return {
            "goal": goal_description,
            "domain": domain,
            "steps": steps,
            "success_criteria": "Task completed with learning extracted",
            "estimated_time": "5-10 minutes",
            "required_knowledge": ["basic_ue_concepts"],
            "tools_needed": ["file_analysis", "documentation_study"]
        }

    def _determine_domain_from_description(self, description: str) -> str:
        """Determine UE domain from task description"""
        description_lower = description.lower()

        domain_keywords = {
            "blueprints": ["blueprint", "visual script", "node", "graph"],
            "cpp_integration": ["c++", "cpp", "code", "programming", "class", "function"],
            "rendering": ["render", "lumen", "nanite", "material", "shader", "light"],
            "physics": ["physics", "chaos", "collision", "simulation", "rigid body"],
            "animation": ["animation", "anim", "skeleton", "bone", "montage"],
            "ai_behavior": ["ai", "behavior", "tree", "perception", "blackboard", "pawn"],
            "networking": ["network", "multiplayer", "replication", "server", "client"],
            "performance": ["performance", "optimization", "profiling", "fps", "memory"],
            "tools": ["tool", "editor", "plugin", "widget", "slate"],
            "assets": ["asset", "content", "import", "pipeline", "mesh", "texture"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return domain

        return "cpp_integration"  # Default domain

    def _extract_task_from_response(self, ai_response: str) -> Optional[Dict[str, Any]]:
        """Extract task data from AI response for hands-on learning phase"""
        try:
            if not ai_response or len(ai_response.strip()) < 10:
                logger.error("❌ AI response is empty or too short")
                return None

            # Log the raw response for debugging
            logger.info(f"🔍 RAW AI RESPONSE LENGTH: {len(ai_response)} characters")
            logger.info(f"🔍 RAW AI RESPONSE (first 500 chars): {ai_response[:500]}...")
            
            # Show thinking process if present
            if "<think>" in ai_response and "</think>" in ai_response:
                thinking_start = ai_response.find("<think>") + 7
                thinking_end = ai_response.find("</think>")
                thinking_content = ai_response[thinking_start:thinking_end]
                logger.info(f"🧠 AI THINKING PROCESS: {thinking_content[:300]}...")

            # Try to parse JSON response first
            if '{' in ai_response and '}' in ai_response:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                json_str = ai_response[json_start:json_end]

                logger.info(f"🔧 ATTEMPTING JSON PARSE...")
                logger.info(f"🔧 JSON STRING (first 200 chars): {json_str[:200]}...")

                try:
                    parsed_task = json.loads(json_str)
                    logger.info("✅ SUCCESSFULLY PARSED AI TASK FROM JSON")
                    logger.info(f"📋 PARSED GOAL: {parsed_task.get('goal', 'No goal found')}")
                    logger.info(f"📋 PARSED STEPS: {len(parsed_task.get('steps', []))} steps")
                    
                    # Ensure required fields exist
                    if "goal" in parsed_task and "steps" in parsed_task:
                        return parsed_task
                    else:
                        logger.warning("⚠️ JSON missing required fields (goal, steps)")
                        
                except json.JSONDecodeError as je:
                    logger.warning(f"⚠️ JSON parsing failed: {je}")

            # Fallback: Extract task information from text
            logger.info("🔄 ATTEMPTING TEXT-BASED TASK EXTRACTION...")
            
            # Extract goal/objective
            goal_patterns = [
                r"(?:goal|objective|aim|purpose):\s*(.+?)(?:\n|$)",
                r"(?:i will|let me|plan to)\s+(.+?)(?:\n|\.)",
                r"(?:task|mission):\s*(.+?)(?:\n|$)"
            ]
            
            goal = "Learn UE through hands-on experimentation"  # Default
            for pattern in goal_patterns:
                match = re.search(pattern, ai_response, re.IGNORECASE)
                if match:
                    goal = match.group(1).strip()
                    logger.info(f"🎯 EXTRACTED GOAL: {goal}")
                    break

            # Extract steps
            steps = []
            
            # Look for numbered lists
            step_patterns = [
                r"(\d+[\.\)]\s*.+?)(?=\d+[\.\)]|\n\n|$)",
                r"(?:step\s*\d+|•|\-)\s*(.+?)(?=(?:step\s*\d+|•|\-|\n\n|$))",
                r"(?:first|then|next|finally),?\s*(.+?)(?=(?:first|then|next|finally|\n\n|$))"
            ]
            
            for pattern in step_patterns:
                matches = re.findall(pattern, ai_response, re.IGNORECASE | re.DOTALL)
                if matches:
                    steps.extend([match.strip() for match in matches if len(match.strip()) > 10])
                    logger.info(f"📋 EXTRACTED {len(matches)} STEPS FROM PATTERN")
                    break
            
            # If no structured steps found, extract sentences that look like actions
            if not steps:
                logger.info("🔄 LOOKING FOR ACTION SENTENCES...")
                action_patterns = [
                    r"(?:try|attempt|create|launch|experiment|test|build|make)\s+.+?[\.!]",
                    r"(?:open|start|run|execute|perform)\s+.+?[\.!]"
                ]
                
                for pattern in action_patterns:
                    matches = re.findall(pattern, ai_response, re.IGNORECASE)
                    steps.extend([match.strip() for match in matches])
                    
                logger.info(f"📋 EXTRACTED {len(steps)} ACTION SENTENCES")
            
            # Ensure we have at least some basic steps
            if not steps:
                logger.warning("⚠️ NO STEPS FOUND, USING DEFAULTS")
                steps = [
                    "Try launching UE editor and creating a new project",
                    "Experiment with basic editor operations and interface",
                    "Attempt simple tasks like placing objects or creating blueprints"
                ]

            # Determine domain
            domain = self._determine_domain_from_description(goal + " " + " ".join(steps))

            extracted_task = {
                "goal": goal,
                "steps": steps[:6],  # Limit to 6 steps max
                "domain": domain
            }
            
            logger.info(f"✅ EXTRACTED TASK FROM TEXT:")
            logger.info(f"🎯 GOAL: {goal}")
            logger.info(f"📋 STEPS: {len(steps)} steps")
            for i, step in enumerate(steps[:3], 1):
                logger.info(f"   {i}. {step[:100]}...")
            
            return extracted_task

        except Exception as e:
            logger.error(f"❌ FAILED TO EXTRACT TASK FROM RESPONSE: {e}")
            logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            return None

    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a planned task step by step"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        logger.info(f"🚀 Executing task: {task.description}")

        step_results = []

        try:
            for i, step in enumerate(task.steps, 1):
                logger.info(f"📝 Step {i}/{len(task.steps)}: {step}")

                # Execute the step
                step_result = await self._execute_task_step(step, task)
                step_results.append(step_result)

                # Store results
                task.results[f"step_{i}"] = step_result

                if not step_result.get("success", False):
                    logger.warning(f"⚠️ Step {i} failed: {step_result.get('error', 'Unknown error')}")
                    # Continue with other steps rather than failing completely

                # Brief pause between steps
                await asyncio.sleep(1)

            # Determine overall success
            successful_steps = sum(1 for result in step_results if result.get("success", False))
            success_rate = successful_steps / len(step_results) if step_results else 0

            if success_rate >= 0.5:  # At least 50% of steps succeeded
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                success = True
            else:
                task.status = TaskStatus.FAILED
                success = False

            # Move to completed tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            self.completed_tasks[task.id] = task

            return {
                "success": success,
                "steps_completed": successful_steps,
                "total_steps": len(step_results),
                "success_rate": success_rate,
                "step_results": step_results
            }

        except Exception as e:
            task.status = TaskStatus.FAILED
            logger.error(f"Task execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_task_step(self, step: str, task: Task) -> Dict[str, Any]:
        """Execute a single task step"""
        try:
            logger.info(f"🔍 ANALYZING STEP: {step}")
            
            # PRIORITY 1: ALWAYS try to execute as a tool command first
            tool_result = await self._try_execute_tool_step(step)
            if tool_result["success"]:
                logger.info(f"✅ TOOL EXECUTED: {tool_result.get('tool_used', 'unknown')}")
                logger.info(f"📋 TOOL RESULT: {tool_result.get('tool_result', '')[:200]}...")
                return tool_result
            else:
                logger.info(f"⚠️ NOT A TOOL COMMAND: {tool_result.get('reason', 'No tool pattern matched')}")
            
            # PRIORITY 2: Only if NOT a tool, then route by keywords
            step_lower = step.lower()
            
            # Force tool execution for obvious tool commands that weren't caught
            if any(tool_name in step_lower for tool_name in self.available_tools.keys()):
                logger.info(f"🎯 FORCING TOOL EXECUTION - Found tool name in step")
                # Try to extract and execute the tool
                for tool_name in self.available_tools.keys():
                    if tool_name in step_lower:
                        logger.info(f"🚀 FORCING EXECUTION OF: {tool_name}")
                        
                        # Extract parameters if possible
                        params = []
                        if tool_name == "create_unreal_project":
                            # Look for project name
                            import re
                            project_match = re.search(r'create_unreal_project\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?', step, re.IGNORECASE)
                            if project_match:
                                params = [project_match.group(1)]
                                if project_match.group(2):
                                    params.append(project_match.group(2))
                                else:
                                    params.append("ThirdPersonBP")
                            else:
                                params = ["MyHandsOnProject", "ThirdPersonBP"]
                        
                        tool_result = await self._execute_tool(tool_name, params)
                        return {
                            "success": True,
                            "step": step,
                            "tool_used": tool_name,
                            "tool_result": tool_result,
                            "type": "forced_tool_execution",
                            "params": params
                        }
            
            # If still not a tool, route by step type
            if "study" in step_lower or "analyze" in step_lower:
                logger.info(f"📚 ROUTING TO: Study step")
                return await self._execute_study_step(step, task)
            elif "experiment" in step_lower or "test" in step_lower:
                logger.info(f"🧪 ROUTING TO: Experiment step")
                return await self._execute_experiment_step(step, task)
            elif "create" in step_lower or "build" in step_lower:
                logger.info(f"🔨 ROUTING TO: Creation step (fallback)")
                return await self._execute_creation_step(step, task)
            else:
                logger.info(f"🔧 ROUTING TO: General step")
                return await self._execute_general_step(step, task)

        except Exception as e:
            logger.error(f"❌ STEP EXECUTION FAILED: {e}")
            return {"success": False, "error": str(e)}

    def _get_learning_context(self) -> str:
        """Get current learning context for AI planning"""
        context_parts = []

        # Current phase
        context_parts.append(f"Current Phase: {self.current_phase.value}")
        context_parts.append(f"Overall Mastery: {self.overall_mastery:.1%}")

        # Learning goals progress
        context_parts.append("\nLearning Goals Progress:")
        for domain, goal in self.learning_goals.items():
            context_parts.append(f"  {domain}: {goal.current_mastery:.1%} (target: {goal.target_mastery:.1%})")

        # Recent tasks
        recent_tasks = list(self.completed_tasks.values())[-3:]  # Last 3 tasks
        if recent_tasks:
            context_parts.append("\nRecent Tasks:")
            for task in recent_tasks:
                status_emoji = "✅" if task.status == TaskStatus.COMPLETED else "❌"
                context_parts.append(f"  {status_emoji} {task.description}")

        # Available paths (escape backslashes for JSON compatibility)
        context_parts.append(f"\nREAL UE SOURCE CODE (for studying): {str(self.ue_source_path).replace(chr(92), '/')}")
        context_parts.append(f"UE INSTALLATION (engine binaries): {str(self.ue_installation_path).replace(chr(92), '/')}")

        return "\n".join(context_parts)

    # ==================== BIDIRECTIONAL RAG SYSTEM ====================

    async def process_and_learn(self, content: str, source: str) -> Dict[str, Any]:
        """Process content and learn from it using bidirectional RAG"""
        start_time = time.time()

        try:
            # 1. Chunk the content
            chunks = self._chunk_text(content)
            logger.info(f"📦 Created {len(chunks)} chunks from {source}")

            # 2. Extract entities and relationships
            all_entities = []
            all_relationships = []

            for chunk in chunks:
                entities, relationships = self._extract_entities_and_relationships(chunk)
                all_entities.extend(entities)
                all_relationships.extend(relationships)

            logger.info(f"🏷️ Extracted {len(all_entities)} entities, {len(all_relationships)} relationships")

            # 3. Store in knowledge graph (handle both dict and dataclass entities)
            for entity in all_entities:
                if isinstance(entity, dict):
                    entity_id = entity.get("id", f"entity_{hash(str(entity))}_{int(time.time())}")
                    self.entities[entity_id] = entity
                    self.knowledge_graph.add_node(entity_id, **entity)
                else:
                    self.entities[entity.id] = entity
                    self.knowledge_graph.add_node(entity.id, **asdict(entity))

            for relationship in all_relationships:
                if isinstance(relationship, dict):
                    rel_id = relationship.get("id", f"rel_{hash(str(relationship))}_{int(time.time())}")
                    self.relationships[rel_id] = relationship
                    self.knowledge_graph.add_edge(
                        relationship.get("source_entity", "unknown"),
                        relationship.get("target_entity", "unknown"),
                        key=rel_id,
                        **relationship
                    )
                else:
                    self.relationships[relationship.id] = relationship
                    self.knowledge_graph.add_edge(
                        relationship.source_entity,
                        relationship.target_entity,
                        key=relationship.id,
                        **asdict(relationship)
                    )

            # 4. Create documents and store in vector DB
            documents = []
            for i, chunk in enumerate(chunks):
                doc_id = f"{source}_{i}_{int(time.time())}"

                # Generate embedding
                embedding = self.embedding_model.encode(chunk)

                # Store in ChromaDB
                self.collection.add(
                    documents=[chunk],
                    metadatas=[{
                        "source": source,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat(),
                        "text_length": len(chunk)
                    }],
                    ids=[doc_id],
                    embeddings=[embedding.tolist()]
                )

                documents.append(doc_id)

            # 5. Have AI analyze and provide structured feedback
            analysis_result = await self._analyze_content_with_ai(content, source)

            # 6. Process AI feedback for bidirectional learning
            feedback_processed = False
            if analysis_result and self.enable_learning:
                feedback_processed = await self._process_ai_feedback(analysis_result, source)

            return {
                "success": True,
                "documents_created": len(documents),
                "entities_extracted": len(all_entities),
                "relationships_created": len(all_relationships),
                "processing_time": time.time() - start_time,
                "source": source,
                "feedback_processed": feedback_processed,
                "ai_analysis": analysis_result
            }

        except Exception as e:
            logger.error(f"Error processing content: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    async def query_knowledge(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the knowledge base with bidirectional learning"""
        start_time = time.time()

        try:
            # 1. Vector similarity search
            query_embedding = self.embedding_model.encode(query)

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            # 2. Prepare context from results
            context_parts = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                similarity = 1 - distance
                context_parts.append(f"[{similarity:.3f}] {doc}")

            context = "\n\n".join(context_parts)

            # 3. Query AI with context
            ai_response = await self._query_llm_with_context(query, context)

            # 4. Process AI feedback for learning
            feedback_processed = False
            if ai_response and self.enable_learning:
                feedback_processed = await self._process_ai_feedback(ai_response, f"Query: {query}")

            return {
                "query": query,
                "ai_response": ai_response,
                "context_used": context,
                "results_found": len(results["documents"][0]),
                "feedback_processed": feedback_processed,
                "total_time": time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Error querying knowledge: {e}")
            return {"success": False, "error": str(e)}

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Intelligent text chunking with overlap"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundaries
            if end < len(text):
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def _extract_entities_and_relationships(self, text: str) -> Tuple[List[Any], List[Any]]:
        """Extract entities and relationships from text (simplified for now)"""
        entities = []
        relationships = []

        # UE-specific patterns
        ue_patterns = {
            "UE_CLASS": r'\b[AU][A-Z][a-zA-Z]*(?:Component|Controller|Manager|System|Engine|Actor|Pawn|Character|GameMode|PlayerController|HUD|Widget)\b',
            "UE_FUNCTION": r'\b(?:Begin|End|Tick|Update|Initialize|Destroy|Spawn|Create|Get|Set)[A-Z][a-zA-Z]*\b',
            "UE_SYSTEM": r'\b(?:Lumen|Nanite|Chaos|MetaHuman|Blueprint|Material|Landscape|Foliage|Animation|Physics|Rendering|Audio|Networking|AI|Behavior Tree|Blackboard)\b',
            "FILE_TYPE": r'\b\w+\.(?:uasset|umap|cpp|h|hpp|cs|py|js|ts|md|txt)\b'
        }

        for entity_type, pattern in ue_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = {
                    "id": f"entity_{hash(match.group())}_{int(time.time())}",
                    "name": match.group().strip(),
                    "type": entity_type,
                    "confidence": 0.8,
                    "created_at": datetime.now()
                }
                entities.append(entity)

        return entities, relationships

    # ==================== AI COMMUNICATION ====================

    async def _query_llm(self, prompt: str, enable_thinking: bool = None) -> str:
        """Query the LLM with a prompt, optionally with thinking mode"""
        try:
            # Use instance thinking mode setting if not explicitly specified
            if enable_thinking is None:
                enable_thinking = self.thinking_mode_enabled and self.thinking_mode_compatible
            
            payload = {
                "model": self.model_name,
                "prompt": f"{self.system_prompt}\n\n{prompt}",
                "stream": False
            }
            
            # Enable thinking mode for compatible models
            if enable_thinking and self.thinking_mode_compatible:
                payload["think"] = True

            async with aiohttp.ClientSession() as session:
                async with session.post(self.ollama_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Handle thinking mode response
                        if enable_thinking and "thinking" in result:
                            thinking = result.get("thinking", "")
                            response_content = result.get("response", "")
                            
                            # Log thinking process for debugging
                            if thinking:
                                logger.info(f"🧠 AI Thinking Process: {thinking[:200]}...")
                            
                            return response_content
                        else:
                            return result.get("response", "")
                    else:
                        logger.error(f"LLM query failed: HTTP {response.status}")
                        return f"Error: HTTP {response.status}"

        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return f"Error: {e}"

    async def _query_llm_with_thinking(self, prompt: str) -> Dict[str, str]:
        """Query LLM with thinking mode enabled, returning both thinking and response"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": f"{self.system_prompt}\n\n{prompt}",
                "stream": False,
                "think": True
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.ollama_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "thinking": result.get("thinking", ""),
                            "response": result.get("response", ""),
                            "success": True
                        }
                    else:
                        logger.error(f"LLM thinking query failed: HTTP {response.status}")
                        return {
                            "thinking": "",
                            "response": f"Error: HTTP {response.status}",
                            "success": False
                        }

        except Exception as e:
            logger.error(f"Error querying LLM with thinking: {e}")
            return {
                "thinking": "",
                "response": f"Error: {e}",
                "success": False
            }

    async def _query_llm_with_context(self, query: str, context: str, enable_thinking: bool = None) -> str:
        """Query LLM with context for bidirectional learning"""
        # Use instance thinking mode setting if not explicitly specified
        if enable_thinking is None:
            enable_thinking = self.thinking_mode_enabled and self.thinking_mode_compatible
            
        prompt = f"""Based on the following context, answer the query and provide structured feedback to enhance the knowledge base.

Context:
{context}

Query: {query}

Instructions:
- Provide a comprehensive answer based on the context
- If you discover new information that should be stored, use: STORE_DOCUMENT: [new information]
- If you want to create a relationship between entities, use: CREATE_RELATIONSHIP: [entity1] -> [relationship_type] -> [entity2]
- If you learn something new from this interaction, use: LEARN: [new_knowledge]
- If you identify a task that should be planned, use: PLAN_TASK: [task description]
- If you want to reflect on results, use: REFLECT: [analysis and insights]

Answer:"""

        return await self._query_llm(prompt, enable_thinking=enable_thinking)

    async def _analyze_content_with_ai(self, content: str, source: str, enable_thinking: bool = False) -> str:
        """Have AI analyze content and provide structured feedback"""
        prompt = f"""Analyze the following UE-related content and provide structured feedback for knowledge enhancement:

Source: {source}
Content:
{content[:2000]}{"..." if len(content) > 2000 else ""}

Instructions:
- Analyze the code/content and understand its purpose and functionality
- Identify important UE concepts, classes, functions, and systems
- Determine relationships between entities
- Provide structured feedback using these formats:
  * STORE_DOCUMENT: [important UE knowledge worth preserving]
  * CREATE_RELATIONSHIP: [entity1] -> [relationship_type] -> [entity2]
  * LEARN: [new UE knowledge gained from this content]

Analysis:"""

        return await self._query_llm(prompt, enable_thinking=enable_thinking)

    async def _process_ai_feedback(self, ai_response: str, source: str) -> bool:
        """Process structured feedback from AI"""
        feedback_processed = False

        try:
            for action, pattern in self.feedback_patterns.items():
                matches = re.findall(pattern, ai_response, re.MULTILINE | re.IGNORECASE)

                for match in matches:
                    if action == "STORE_DOCUMENT":
                        success = await self._store_feedback_document(match.strip(), source)
                    elif action == "CREATE_RELATIONSHIP":
                        success = await self._create_feedback_relationship(match.strip())
                    elif action == "LEARN":
                        success = await self._learn_from_feedback(match.strip(), source)
                    elif action == "PLAN_TASK":
                        success = await self._plan_task_from_feedback(match.strip())
                    elif action == "REFLECT":
                        success = await self._reflect_from_feedback(match.strip(), source)
                    else:
                        success = False

                    if success:
                        feedback_processed = True
                        logger.info(f"Processed AI feedback: {action}")

        except Exception as e:
            logger.error(f"Failed to process AI feedback: {e}")

        return feedback_processed

    async def _store_feedback_document(self, content: str, source: str) -> bool:
        """Store new document from AI feedback"""
        try:
            doc_id = f"feedback_{int(time.time())}_{hash(content)}"
            embedding = self.embedding_model.encode(content)

            self.collection.add(
                documents=[content],
                metadatas=[{
                    "source": f"ai_feedback_{source}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "ai_generated"
                }],
                ids=[doc_id],
                embeddings=[embedding.tolist()]
            )

            logger.info(f"Stored feedback document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store feedback document: {e}")
            return False

    async def _create_feedback_relationship(self, relationship_text: str) -> bool:
        """Create relationship from AI feedback"""
        logger.info(f"Relationship creation requested: {relationship_text}")
        # TODO: Parse and create actual relationships
        return True

    async def _learn_from_feedback(self, learning_text: str, source: str) -> bool:
        """Learn from AI feedback"""
        logger.info(f"Learning from feedback: {learning_text}")
        # TODO: Update learning goals based on feedback
        return True

    async def _plan_task_from_feedback(self, task_description: str) -> bool:
        """Plan a new task from AI feedback"""
        logger.info(f"Task planning requested: {task_description}")
        # TODO: Add to task queue
        return True

    async def _reflect_from_feedback(self, reflection_text: str, source: str) -> bool:
        """Process reflection from AI feedback"""
        logger.info(f"Reflection: {reflection_text}")
        # TODO: Update strategies based on reflection
        return True

    # ==================== UE TOOLS & STEP EXECUTION ====================

    async def _execute_study_step(self, step: str, task: Task) -> Dict[str, Any]:
        """Execute a study/analysis step"""
        try:
            if "source code" in step.lower():
                return await self._study_ue_source_code(step)
            elif "installation" in step.lower():
                return await self._study_ue_installation(step)
            else:
                return await self._study_general_content(step)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_experiment_step(self, step: str, task: Task) -> Dict[str, Any]:
        """Execute a real experiment/test step by studying UE systems"""
        try:
            logger.info(f"🧪 Executing real experiment: {step}")

            # Real experiment: Study UE systems to understand how they work
            if "shader" in step.lower() or "material" in step.lower():
                return await self._study_ue_rendering_system(step)
            elif "physics" in step.lower() or "collision" in step.lower():
                return await self._study_ue_physics_system(step)
            elif "input" in step.lower() or "control" in step.lower():
                return await self._study_ue_input_system(step)
            else:
                # General UE system study
                return await self._study_ue_general_system(step)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_creation_step(self, step: str, task: Task) -> Dict[str, Any]:
        """Execute a real creation/building step by analyzing UE architecture"""
        try:
            logger.info(f"🔨 Executing real creation analysis: {step}")

            # Real creation: Analyze how UE creates and manages objects
            if "actor" in step.lower() or "object" in step.lower():
                return await self._study_ue_object_creation(step)
            elif "component" in step.lower():
                return await self._study_ue_component_system(step)
            elif "world" in step.lower() or "level" in step.lower():
                return await self._study_ue_world_system(step)
            else:
                # General UE creation study
                return await self._study_ue_creation_patterns(step)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_general_step(self, step: str, task: Task) -> Dict[str, Any]:
        """Execute a general task step"""
        try:
            logger.info(f"🔧 Executing general step: {step}")
            
            # Check if this step involves using a tool
            tool_result = await self._try_execute_tool_step(step)
            if tool_result["success"]:
                return tool_result
            
            # If not a tool step, treat as general learning/analysis
            result = await self._study_general_content(step)
            
            # Store results in task
            if task.results is None:
                task.results = {}
            task.results[f"general_step_{len(task.results)}"] = result
            
            return {
                "success": True,
                "step": step,
                "result": result,
                "type": "general"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to execute general step: {e}")
            return {
                "success": False,
                "step": step,
                "error": str(e),
                "type": "general"
            }

    # ==================== TOOL EXECUTION SYSTEM ====================

    async def _try_execute_tool_step(self, step: str) -> Dict[str, Any]:
        """Try to execute a step as a tool command"""
        try:
            logger.info(f"🔍 TOOL ANALYSIS: Checking step for tool patterns")
            logger.info(f"📝 STEP TEXT: {step}")
            
            # First, check for explicit tool commands in backticks or after "Command:"
            explicit_command_patterns = [
                r'`([^`]+)`',  # Commands in backticks
                r'Command:\s*`?([^`\n]+)`?',  # Commands after "Command:"
                r'Tool:\s*`?([^`\n]+)`?',  # Commands after "Tool:"
                r'Execute:\s*`?([^`\n]+)`?'  # Commands after "Execute:"
            ]
            
            for pattern in explicit_command_patterns:
                match = re.search(pattern, step, re.IGNORECASE)
                if match:
                    command = match.group(1).strip()
                    logger.info(f"🎯 FOUND EXPLICIT COMMAND: {command}")
                    
                    # Check if it's a direct tool name
                    if command in self.available_tools:
                        tool_result = await self._execute_tool(command, [])
                        return {
                            "success": True,
                            "step": step,
                            "tool_used": command,
                            "tool_result": tool_result,
                            "type": "explicit_tool_execution"
                        }
                    
                    # Try to parse as "tool_name param1 param2"
                    parts = command.split()
                    if parts and parts[0] in self.available_tools:
                        tool_name = parts[0]
                        params = parts[1:] if len(parts) > 1 else []
                        logger.info(f"🔧 EXECUTING TOOL: {tool_name} with params: {params}")
                        tool_result = await self._execute_tool(tool_name, params)
                        return {
                            "success": True,
                            "step": step,
                            "tool_used": tool_name,
                            "tool_result": tool_result,
                            "type": "explicit_tool_execution"
                        }
            
            # Enhanced pattern matching for natural language
            tool_patterns = {
                # Epic Launcher patterns
                r"(?:close|stop|shut.*down).*epic": "close_epic_games_launcher",
                r"(?:launch|start|open).*epic": "launch_epic_games_launcher", 
                r"(?:check|status).*epic": "check_epic_launcher_status",
                
                # Project patterns - ENHANCED
                r"create.*project.*?([A-Za-z0-9_]+)": "create_unreal_project",
                r"(?:make|build|new).*project.*?([A-Za-z0-9_]+)": "create_unreal_project",
                r"create.*unreal.*project.*?([A-Za-z0-9_]+)": "create_unreal_project",
                r"(?:execute|use|run).*create_unreal_project\s+([A-Za-z0-9_]+)": "create_unreal_project",
                
                r"open.*project.*?([A-Za-z0-9_./\\]+\.uproject)": "open_unreal_project",
                r"list.*templates": "list_unreal_templates",
                
                # File operations
                r"show.*(?:output|andrio)": "show_andrio_output",
                r"create.*workspace.*?([A-Za-z0-9_]+)": "create_andrio_workspace",
                r"list.*drives": "list_drives",
                r"list.*files": "list_files",
                r"get.*ue.*info": "get_unreal_engine_info",
                
                # Workspace patterns
                r"create.*workspace.*?[\"']([^\"']+)[\"']": "create_andrio_workspace",
                r"workspace.*?[\"']([^\"']+)[\"']": "create_andrio_workspace"
            }
            
            for pattern, tool_name in tool_patterns.items():
                match = re.search(pattern, step.lower())
                if match and tool_name in self.available_tools:
                    logger.info(f"🔧 DETECTED TOOL USAGE: {tool_name}")
                    logger.info(f"🎯 MATCHED PATTERN: {pattern}")
                    
                    # Extract parameters from the match
                    params = []
                    if match.groups():
                        params = [group.strip().strip('"\'') for group in match.groups() if group and group.strip()]
                        logger.info(f"📋 EXTRACTED PARAMS: {params}")
                    
                    # Special handling for specific tools
                    if tool_name == "create_unreal_project":
                        if len(params) == 1:
                            # Add default template if not specified
                            params.append("ThirdPersonBP")
                            logger.info(f"🎯 ADDED DEFAULT TEMPLATE: {params}")
                        elif len(params) == 0:
                            # No project name found, use default
                            params = ["MyHandsOnProject", "ThirdPersonBP"]
                            logger.info(f"🎯 USING DEFAULT PROJECT: {params}")
                    elif tool_name == "list_files" and not params:
                        # Default to current directory
                        params = ["."]
                    
                    # Execute the tool
                    logger.info(f"🚀 EXECUTING TOOL: {tool_name} with params: {params}")
                    tool_result = await self._execute_tool(tool_name, params)
                    
                    return {
                        "success": True,
                        "step": step,
                        "tool_used": tool_name,
                        "tool_result": tool_result,
                        "type": "pattern_tool_execution",
                        "params": params,
                        "matched_pattern": pattern
                    }
            
            # No tool pattern matched
            logger.info(f"❌ NO TOOL PATTERN MATCHED")
            logger.info(f"📋 Available tools: {list(self.available_tools.keys())}")
            return {"success": False, "reason": "No tool pattern matched", "step": step}
            
        except Exception as e:
            logger.error(f"❌ ERROR IN TOOL STEP EXECUTION: {e}")
            return {"success": False, "error": str(e), "step": step}

    async def _execute_tool(self, tool_name: str, params: List[str] = None) -> str:
        """Execute a specific tool with parameters"""
        try:
            logger.info(f"🔧 TOOL EXECUTION START: {tool_name}")
            logger.info(f"📋 TOOL PARAMS: {params}")
            
            if tool_name not in self.available_tools:
                error_msg = f"❌ Tool '{tool_name}' not available"
                logger.error(error_msg)
                return error_msg
            
            tool_func = self.available_tools[tool_name]
            logger.info(f"🎯 TOOL FUNCTION FOUND: {tool_func}")
            
            # Execute tool with parameters
            if params:
                logger.info(f"🚀 EXECUTING WITH PARAMS: {params}")
                if len(params) == 1:
                    result = tool_func(params[0])
                elif len(params) == 2:
                    result = tool_func(params[0], params[1])
                elif len(params) == 3:
                    result = tool_func(params[0], params[1], params[2])
                else:
                    result = tool_func(*params)
            else:
                logger.info(f"🚀 EXECUTING WITHOUT PARAMS")
                result = tool_func()
            
            logger.info(f"✅ TOOL '{tool_name}' EXECUTED SUCCESSFULLY")
            logger.info(f"📄 TOOL RESULT (first 300 chars): {str(result)[:300]}...")
            
            # Special logging for project creation
            if tool_name == "create_unreal_project":
                if "✅" in str(result):
                    logger.info(f"🎉 PROJECT CREATION SUCCESS!")
                else:
                    logger.warning(f"⚠️ PROJECT CREATION MAY HAVE FAILED")
                    
            return result
            
        except Exception as e:
            error_msg = f"❌ Error executing tool '{tool_name}': {e}"
            logger.error(error_msg)
            logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            return error_msg

    def execute_tool_command(self, command: str) -> str:
        """Execute a tool command directly (for interactive use)"""
        try:
            # Parse command format: "tool_name param1 param2"
            parts = command.strip().split()
            if not parts:
                return "❌ Empty command"
            
            tool_name = parts[0]
            params = parts[1:] if len(parts) > 1 else []
            
            if tool_name not in self.available_tools:
                available = ", ".join(self.available_tools.keys())
                return f"❌ Tool '{tool_name}' not found. Available: {available}"
            
            # Execute synchronously for interactive use
            tool_func = self.available_tools[tool_name]
            
            if params:
                if len(params) == 1:
                    result = tool_func(params[0])
                elif len(params) == 2:
                    result = tool_func(params[0], params[1])
                elif len(params) == 3:
                    result = tool_func(params[0], params[1], params[2])
                else:
                    result = tool_func(*params)
            else:
                result = tool_func()
            
            return result
            
        except Exception as e:
            return f"❌ Error executing command: {e}"

    def list_available_tools(self) -> str:
        """List all available tools with descriptions"""
        if not self.available_tools:
            return "❌ No tools available"
        
        result = ["🧰 Available Tools:\n"]
        
        for tool_name, description in self.tool_descriptions.items():
            result.append(f"  🔧 {tool_name}: {description}")
        
        return "\n".join(result)

    # ==================== ENHANCED TASK EXECUTION ====================

    async def _study_ue_source_code(self, step: str) -> Dict[str, Any]:
        """Study UE source code systematically with enhanced analysis"""
        try:
            if not self.ue_source_path.exists():
                return {"success": False, "error": f"UE source path not found: {self.ue_source_path}"}

            # 1. Analyze UE project structure first
            logger.info("🔍 Analyzing UE project structure...")
            structure_analysis = await self._analyze_ue_project_structure(self.ue_source_path)

            # 2. Find UE classes and functions
            logger.info("🏗️ Finding UE classes and functions...")
            classes_analysis = await self._find_ue_classes_and_functions(self.ue_source_path)

            # 3. Study specific files with enhanced processing
            files_to_study = []
            for ext in ['.cpp', '.h', '.hpp']:
                files_to_study.extend(list(self.ue_source_path.rglob(f'*{ext}'))[:5])  # Limit to 5 files per type

            if not files_to_study:
                return {"success": False, "error": "No source files found to study"}

            studied_files = []
            total_content_length = 0
            ue_concepts_found = []

            for file_path in files_to_study[:3]:  # Study first 3 files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 100:  # Only study non-empty files
                        # Create enhanced learning summary
                        learning_summary = await self._create_ue_learning_summary(content, str(file_path))

                        # Process with bidirectional RAG
                        result = await self.process_and_learn(learning_summary, str(file_path))
                        if result["success"]:
                            studied_files.append(str(file_path))
                            total_content_length += len(content)
                            logger.info(f"📚 Studied: {file_path.name}")

                            # Extract UE concepts for reporting
                            if "Key Concepts Found:" in learning_summary:
                                concepts_section = learning_summary.split("Key Concepts Found:")[1].split("Content Length:")[0]
                                ue_concepts_found.extend([line.strip() for line in concepts_section.split('\n') if line.strip().startswith('-')])

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            # 4. Create comprehensive study report
            study_report = {
                "success": len(studied_files) > 0,
                "files_studied": studied_files,
                "total_files": len(studied_files),
                "total_content_length": total_content_length,
                "ue_concepts_found": ue_concepts_found[:10],  # Limit to 10 concepts
                "structure_analysis": structure_analysis.get("analysis", {}) if structure_analysis.get("success") else {},
                "classes_found": len(classes_analysis.get("results", {}).get("classes", [])) if classes_analysis.get("success") else 0,
                "functions_found": len(classes_analysis.get("results", {}).get("functions", [])) if classes_analysis.get("success") else 0,
                "message": f"Enhanced study of {len(studied_files)} UE source files with {len(ue_concepts_found)} concepts identified"
            }

            return study_report

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _study_ue_installation(self, step: str) -> Dict[str, Any]:
        """Study UE installation structure"""
        try:
            if not self.ue_installation_path.exists():
                return {"success": False, "error": f"UE installation path not found: {self.ue_installation_path}"}

            # Analyze installation structure
            important_dirs = []
            important_files = []

            for item in self.ue_installation_path.iterdir():
                if item.is_dir():
                    important_dirs.append(item.name)
                elif item.suffix in ['.exe', '.dll', '.txt', '.md']:
                    important_files.append(item.name)

            # Create summary content to learn from
            summary_content = f"""UE Installation Analysis: {self.ue_installation_path}

Important Directories:
{chr(10).join(f"- {d}" for d in important_dirs[:10])}

Important Files:
{chr(10).join(f"- {f}" for f in important_files[:10])}

This represents the structure of an Unreal Engine installation, showing the organization of engine components, tools, and resources."""

            # Process this knowledge
            result = await self.process_and_learn(summary_content, "UE_Installation_Analysis")

            return {
                "success": True,
                "directories_found": len(important_dirs),
                "files_found": len(important_files),
                "analysis_processed": result["success"],
                "message": f"Analyzed UE installation structure: {len(important_dirs)} dirs, {len(important_files)} files"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _study_general_content(self, step: str) -> Dict[str, Any]:
        """Study general content or concepts"""
        try:
            # Use AI to determine what to study
            prompt = f"""As Andrio V2, determine what content to study for this step: {step}

Based on my current learning goals and UE knowledge, what specific content should I focus on?
Provide specific recommendations for files, concepts, or areas to study."""

            response = await self._query_llm(prompt)

            return {
                "success": True,
                "message": f"General study completed: {step}",
                "ai_guidance": response
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== ENHANCED UE TOOLS ====================

    async def _analyze_ue_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Analyze UE project structure and extract key information"""
        try:
            analysis = {
                "project_files": [],
                "source_directories": [],
                "content_directories": [],
                "config_files": [],
                "build_files": [],
                "total_files": 0,
                "ue_version": "Unknown"
            }

            if not project_path.exists():
                return {"success": False, "error": f"Project path not found: {project_path}"}

            # Find .uproject file
            uproject_files = list(project_path.glob("*.uproject"))
            if uproject_files:
                analysis["project_files"] = [str(f) for f in uproject_files]

                # Try to extract UE version from .uproject file
                try:
                    with open(uproject_files[0], 'r') as f:
                        project_data = json.load(f)
                        analysis["ue_version"] = project_data.get("EngineAssociation", "Unknown")
                except:
                    pass

            # Analyze directory structure
            for item in project_path.rglob("*"):
                if item.is_file():
                    analysis["total_files"] += 1

                    # Categorize files
                    if item.suffix in ['.cpp', '.h', '.hpp']:
                        analysis["source_directories"].append(str(item.parent))
                    elif item.suffix in ['.uasset', '.umap']:
                        analysis["content_directories"].append(str(item.parent))
                    elif item.name.endswith('.ini'):
                        analysis["config_files"].append(str(item))
                    elif item.name in ['Build.cs', 'Target.cs'] or item.suffix == '.cs':
                        analysis["build_files"].append(str(item))

            # Remove duplicates and limit results
            analysis["source_directories"] = list(set(analysis["source_directories"]))[:10]
            analysis["content_directories"] = list(set(analysis["content_directories"]))[:10]

            return {"success": True, "analysis": analysis}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _find_ue_classes_and_functions(self, source_path: Path, pattern: str = None) -> Dict[str, Any]:
        """Find UE classes and functions in source code"""
        try:
            results = {
                "classes": [],
                "functions": [],
                "macros": [],
                "includes": [],
                "files_analyzed": 0
            }

            # UE-specific patterns
            class_patterns = [
                r'class\s+[A-Z_]*API\s+([AU][A-Z][a-zA-Z0-9_]*)',  # UE classes with API
                r'UCLASS\(\s*[^)]*\s*\)\s*class\s+[A-Z_]*API\s+([AU][A-Z][a-zA-Z0-9_]*)',  # UCLASS
                r'USTRUCT\(\s*[^)]*\s*\)\s*struct\s+[A-Z_]*API\s+([FT][A-Z][a-zA-Z0-9_]*)'  # USTRUCT
            ]

            function_patterns = [
                r'UFUNCTION\(\s*[^)]*\s*\)\s*[a-zA-Z_][a-zA-Z0-9_\s\*&]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # UFUNCTION
                r'virtual\s+[a-zA-Z_][a-zA-Z0-9_\s\*&]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*override',  # Virtual overrides
                r'static\s+[a-zA-Z_][a-zA-Z0-9_\s\*&]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('  # Static functions
            ]

            macro_patterns = [
                r'(UPROPERTY|UFUNCTION|UCLASS|USTRUCT|UENUM|UMETA)\s*\([^)]*\)',
                r'(GENERATED_BODY|GENERATED_UCLASS_BODY)\s*\(\s*\)'
            ]

            include_patterns = [
                r'#include\s+[<"]([^>"]+)[>"]'
            ]

            # Search through source files
            for file_path in source_path.rglob("*.h"):
                if results["files_analyzed"] >= 20:  # Limit for performance
                    break

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    results["files_analyzed"] += 1

                    # Find classes
                    for pattern in class_patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            results["classes"].append({
                                "name": match.group(1),
                                "file": str(file_path),
                                "type": "UE_CLASS"
                            })

                    # Find functions
                    for pattern in function_patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            results["functions"].append({
                                "name": match.group(1),
                                "file": str(file_path),
                                "type": "UE_FUNCTION"
                            })

                    # Find macros
                    for pattern in macro_patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            results["macros"].append({
                                "name": match.group(1),
                                "file": str(file_path),
                                "full_match": match.group(0)
                            })

                    # Find includes
                    for pattern in include_patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            if "Engine/" in match.group(1) or "UObject/" in match.group(1):
                                results["includes"].append({
                                    "include": match.group(1),
                                    "file": str(file_path)
                                })

                except Exception as e:
                    continue

            # Limit results
            results["classes"] = results["classes"][:50]
            results["functions"] = results["functions"][:50]
            results["macros"] = results["macros"][:30]
            results["includes"] = results["includes"][:30]

            return {"success": True, "results": results}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_ue_learning_summary(self, content: str, source: str) -> str:
        """Create a learning summary from UE content"""
        try:
            # Extract key UE concepts
            ue_concepts = []

            # Look for UE-specific patterns
            patterns = {
                "Classes": r'\b[AU][A-Z][a-zA-Z0-9]*\b',
                "Functions": r'\b(?:Begin|End|Tick|Update|Initialize|Destroy|Spawn|Create|Get|Set)[A-Z][a-zA-Z]*\b',
                "Systems": r'\b(?:Lumen|Nanite|Chaos|MetaHuman|Blueprint|Material|Landscape|Foliage|Animation|Physics|Rendering|Audio|Networking|AI|Behavior Tree|Blackboard)\b',
                "Macros": r'\b(?:UPROPERTY|UFUNCTION|UCLASS|USTRUCT|UENUM|GENERATED_BODY)\b'
            }

            for concept_type, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    unique_matches = list(set(matches))[:5]  # Limit to 5 per type
                    ue_concepts.append(f"{concept_type}: {', '.join(unique_matches)}")

            # Create summary
            summary = f"""UE Learning Summary from {source}:

Key Concepts Found:
{chr(10).join(f"- {concept}" for concept in ue_concepts)}

Content Length: {len(content)} characters
Source: {source}

This content provides insights into Unreal Engine architecture, classes, and systems."""

            return summary

        except Exception as e:
            return f"Error creating summary: {e}"

    # ==================== REAL UE SYSTEM STUDY METHODS ====================

    async def _study_ue_rendering_system(self, step: str) -> Dict[str, Any]:
        """Study UE rendering system by analyzing real source files"""
        try:
            logger.info("🎨 Studying UE rendering system...")

            # Look for rendering-related files
            rendering_files = []
            search_patterns = ["*Render*.h", "*Material*.h", "*Shader*.h", "*Light*.h"]

            for pattern in search_patterns:
                files = list(self.ue_source_path.rglob(pattern))
                rendering_files.extend(files[:3])  # Limit to 3 files per pattern

            if not rendering_files:
                return {"success": False, "error": "No rendering files found"}

            studied_concepts = []
            total_content = 0

            for file_path in rendering_files[:5]:  # Study first 5 files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 200:
                        # Extract rendering concepts
                        concepts = self._extract_rendering_concepts(content, str(file_path))
                        studied_concepts.extend(concepts)

                        # Process with RAG
                        summary = f"UE Rendering Study from {file_path.name}:\n{content[:1000]}"
                        await self.process_and_learn(summary, f"rendering_{file_path.name}")
                        total_content += len(content)

                        logger.info(f"📖 Studied rendering file: {file_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            return {
                "success": True,
                "system": "rendering",
                "files_studied": len([f for f in rendering_files[:5] if f.exists()]),
                "concepts_learned": studied_concepts[:10],
                "total_content_length": total_content,
                "message": f"Studied UE rendering system - {len(studied_concepts)} concepts learned"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _study_ue_physics_system(self, step: str) -> Dict[str, Any]:
        """Study UE physics system by analyzing real source files"""
        try:
            logger.info("⚡ Studying UE physics system...")

            # Look for physics-related files
            physics_files = []
            search_patterns = ["*Physics*.h", "*Collision*.h", "*Chaos*.h", "*Body*.h"]

            for pattern in search_patterns:
                files = list(self.ue_source_path.rglob(pattern))
                physics_files.extend(files[:3])

            if not physics_files:
                return {"success": False, "error": "No physics files found"}

            studied_concepts = []
            total_content = 0

            for file_path in physics_files[:5]:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 200:
                        # Extract physics concepts
                        concepts = self._extract_physics_concepts(content, str(file_path))
                        studied_concepts.extend(concepts)

                        # Process with RAG
                        summary = f"UE Physics Study from {file_path.name}:\n{content[:1000]}"
                        await self.process_and_learn(summary, f"physics_{file_path.name}")
                        total_content += len(content)

                        logger.info(f"📖 Studied physics file: {file_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            return {
                "success": True,
                "system": "physics",
                "files_studied": len([f for f in physics_files[:5] if f.exists()]),
                "concepts_learned": studied_concepts[:10],
                "total_content_length": total_content,
                "message": f"Studied UE physics system - {len(studied_concepts)} concepts learned"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _study_ue_input_system(self, step: str) -> Dict[str, Any]:
        """Study UE input system by analyzing real source files"""
        try:
            logger.info("🎮 Studying UE input system...")

            # Look for input-related files
            input_files = []
            search_patterns = ["*Input*.h", "*Controller*.h", "*Player*.h"]

            for pattern in search_patterns:
                files = list(self.ue_source_path.rglob(pattern))
                input_files.extend(files[:3])

            if not input_files:
                return {"success": False, "error": "No input files found"}

            studied_concepts = []
            total_content = 0

            for file_path in input_files[:5]:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 200:
                        # Extract input concepts
                        concepts = self._extract_input_concepts(content, str(file_path))
                        studied_concepts.extend(concepts)

                        # Process with RAG
                        summary = f"UE Input Study from {file_path.name}:\n{content[:1000]}"
                        await self.process_and_learn(summary, f"input_{file_path.name}")
                        total_content += len(content)

                        logger.info(f"📖 Studied input file: {file_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            return {
                "success": True,
                "system": "input",
                "files_studied": len([f for f in input_files[:5] if f.exists()]),
                "concepts_learned": studied_concepts[:10],
                "total_content_length": total_content,
                "message": f"Studied UE input system - {len(studied_concepts)} concepts learned"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_rendering_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract rendering concepts from UE source code"""
        concepts = []

        # Rendering-specific patterns
        patterns = {
            "Materials": r'\b(?:UMaterial|FMaterial|MaterialInstance|MaterialExpression)[A-Za-z0-9_]*\b',
            "Shaders": r'\b(?:Shader|Vertex|Pixel|Compute|Hull|Domain)[A-Za-z0-9_]*\b',
            "Rendering": r'\b(?:Render|Draw|GPU|Texture|Buffer|Pipeline)[A-Za-z0-9_]*\b',
            "Lighting": r'\b(?:Light|Shadow|Illumination|Lumen|GI)[A-Za-z0-9_]*\b'
        }

        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                unique_matches = list(set(matches))[:3]
                concepts.extend([f"{concept_type}: {match}" for match in unique_matches])

        return concepts

    def _extract_physics_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract physics concepts from UE source code"""
        concepts = []

        # Physics-specific patterns
        patterns = {
            "Physics": r'\b(?:Physics|Rigid|Body|Constraint|Force)[A-Za-z0-9_]*\b',
            "Collision": r'\b(?:Collision|Overlap|Hit|Trace|Shape)[A-Za-z0-9_]*\b',
            "Chaos": r'\b(?:Chaos|Particle|Solver|Evolution)[A-Za-z0-9_]*\b',
            "Simulation": r'\b(?:Simulate|Update|Tick|Step)[A-Za-z0-9_]*\b'
        }

        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                unique_matches = list(set(matches))[:3]
                concepts.extend([f"{concept_type}: {match}" for match in unique_matches])

        return concepts

    def _extract_input_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract input concepts from UE source code"""
        concepts = []

        # Input-specific patterns
        patterns = {
            "Input": r'\b(?:Input|Key|Button|Axis|Action)[A-Za-z0-9_]*\b',
            "Controllers": r'\b(?:Controller|Player|Pawn|Character)[A-Za-z0-9_]*\b',
            "Events": r'\b(?:Event|Pressed|Released|Triggered)[A-Za-z0-9_]*\b',
            "Devices": r'\b(?:Mouse|Keyboard|Gamepad|Touch)[A-Za-z0-9_]*\b'
        }

        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                unique_matches = list(set(matches))[:3]
                concepts.extend([f"{concept_type}: {match}" for match in unique_matches])

        return concepts

    async def _study_ue_general_system(self, step: str) -> Dict[str, Any]:
        """Study general UE system by analyzing source files"""
        try:
            logger.info("🔧 Studying general UE system...")

            # Look for core UE files
            core_files = []
            search_patterns = ["*Engine*.h", "*Object*.h", "*Actor*.h", "*Component*.h"]

            for pattern in search_patterns:
                files = list(self.ue_source_path.rglob(pattern))
                core_files.extend(files[:2])

            if not core_files:
                return {"success": False, "error": "No core UE files found"}

            studied_concepts = []
            total_content = 0

            for file_path in core_files[:4]:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 200:
                        # Extract general UE concepts
                        concepts = self._extract_general_ue_concepts(content, str(file_path))
                        studied_concepts.extend(concepts)

                        # Process with RAG
                        summary = f"UE Core Study from {file_path.name}:\n{content[:1000]}"
                        await self.process_and_learn(summary, f"core_{file_path.name}")
                        total_content += len(content)

                        logger.info(f"📖 Studied core file: {file_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            return {
                "success": True,
                "system": "general",
                "files_studied": len([f for f in core_files[:4] if f.exists()]),
                "concepts_learned": studied_concepts[:10],
                "total_content_length": total_content,
                "message": f"Studied general UE system - {len(studied_concepts)} concepts learned"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_general_ue_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract general UE concepts from source code"""
        concepts = []

        # General UE patterns
        patterns = {
            "Core Classes": r'\b(?:UObject|AActor|UComponent|UEngine|UWorld)[A-Za-z0-9_]*\b',
            "Functions": r'\b(?:Begin|End|Tick|Update|Initialize|Destroy)[A-Za-z0-9_]*\b',
            "Properties": r'\b(?:UPROPERTY|UFUNCTION|UCLASS|USTRUCT)\b',
            "Memory": r'\b(?:GC|Garbage|Memory|Allocate|Delete)[A-Za-z0-9_]*\b'
        }

        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                unique_matches = list(set(matches))[:3]
                concepts.extend([f"{concept_type}: {match}" for match in unique_matches])

        return concepts

    # ==================== MISSING UE CREATION STUDY METHODS ====================

    async def _study_ue_object_creation(self, step: str) -> Dict[str, Any]:
        """Study UE object creation patterns by analyzing source files"""
        try:
            logger.info("🏗️ Studying UE object creation...")

            # Look for object creation related files
            creation_files = []
            search_patterns = ["*Object*.h", "*Actor*.h", "*Spawn*.h", "*Create*.h"]

            for pattern in search_patterns:
                files = list(self.ue_source_path.rglob(pattern))
                creation_files.extend(files[:2])

            if not creation_files:
                return {"success": False, "error": "No object creation files found"}

            studied_concepts = []
            total_content = 0

            for file_path in creation_files[:4]:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 200:
                        # Extract creation concepts
                        concepts = self._extract_creation_concepts(content, str(file_path))
                        studied_concepts.extend(concepts)

                        # Process with RAG
                        summary = f"UE Object Creation Study from {file_path.name}:\n{content[:1000]}"
                        await self.process_and_learn(summary, f"creation_{file_path.name}")
                        total_content += len(content)

                        logger.info(f"📖 Studied creation file: {file_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            return {
                "success": True,
                "system": "object_creation",
                "files_studied": len([f for f in creation_files[:4] if f.exists()]),
                "concepts_learned": studied_concepts[:10],
                "total_content_length": total_content,
                "message": f"Studied UE object creation - {len(studied_concepts)} concepts learned"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _study_ue_component_system(self, step: str) -> Dict[str, Any]:
        """Study UE component system by analyzing source files"""
        try:
            logger.info("🔧 Studying UE component system...")

            # Look for component related files
            component_files = []
            search_patterns = ["*Component*.h", "*Scene*.h", "*Primitive*.h"]

            for pattern in search_patterns:
                files = list(self.ue_source_path.rglob(pattern))
                component_files.extend(files[:2])

            if not component_files:
                return {"success": False, "error": "No component files found"}

            studied_concepts = []
            total_content = 0

            for file_path in component_files[:4]:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 200:
                        # Extract component concepts
                        concepts = self._extract_component_concepts(content, str(file_path))
                        studied_concepts.extend(concepts)

                        # Process with RAG
                        summary = f"UE Component Study from {file_path.name}:\n{content[:1000]}"
                        await self.process_and_learn(summary, f"component_{file_path.name}")
                        total_content += len(content)

                        logger.info(f"📖 Studied component file: {file_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            return {
                "success": True,
                "system": "component_system",
                "files_studied": len([f for f in component_files[:4] if f.exists()]),
                "concepts_learned": studied_concepts[:10],
                "total_content_length": total_content,
                "message": f"Studied UE component system - {len(studied_concepts)} concepts learned"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _study_ue_world_system(self, step: str) -> Dict[str, Any]:
        """Study UE world system by analyzing source files"""
        try:
            logger.info("🌍 Studying UE world system...")

            # Look for world related files
            world_files = []
            search_patterns = ["*World*.h", "*Level*.h", "*Game*.h"]

            for pattern in search_patterns:
                files = list(self.ue_source_path.rglob(pattern))
                world_files.extend(files[:2])

            if not world_files:
                return {"success": False, "error": "No world files found"}

            studied_concepts = []
            total_content = 0

            for file_path in world_files[:4]:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 200:
                        # Extract world concepts
                        concepts = self._extract_world_concepts(content, str(file_path))
                        studied_concepts.extend(concepts)

                        # Process with RAG
                        summary = f"UE World Study from {file_path.name}:\n{content[:1000]}"
                        await self.process_and_learn(summary, f"world_{file_path.name}")
                        total_content += len(content)

                        logger.info(f"📖 Studied world file: {file_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            return {
                "success": True,
                "system": "world_system",
                "files_studied": len([f for f in world_files[:4] if f.exists()]),
                "concepts_learned": studied_concepts[:10],
                "total_content_length": total_content,
                "message": f"Studied UE world system - {len(studied_concepts)} concepts learned"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _study_ue_creation_patterns(self, step: str) -> Dict[str, Any]:
        """Study general UE creation patterns by analyzing source files"""
        try:
            logger.info("🎨 Studying UE creation patterns...")
            
            # Increment study session count
            self.study_session_count += 1

            # Look for creation pattern files, but avoid recently studied ones
            pattern_files = []
            search_patterns = ["*Factory*.h", "*Builder*.h", "*Manager*.h", "*System*.h", "*Creator*.h", "*Generator*.h"]

            for pattern in search_patterns:
                files = list(self.ue_source_path.rglob(pattern))
                # Filter out already studied files
                new_files = [f for f in files if str(f) not in self.studied_files]
                pattern_files.extend(new_files[:2])  # Take 2 new files per pattern

            # If we've studied most files, reset and allow re-studying with different focus
            if not pattern_files and len(self.studied_files) > 20:
                logger.info("🔄 Resetting studied files to allow re-analysis with new perspective")
                self.studied_files.clear()
                # Try again with fresh perspective
                for pattern in search_patterns:
                    files = list(self.ue_source_path.rglob(pattern))
                    pattern_files.extend(files[:1])  # Take 1 file per pattern

            if not pattern_files:
                return {"success": False, "error": "No new creation pattern files found to study"}

            studied_concepts = []
            total_content = 0
            files_processed = []

            # Study up to 3 files to avoid overwhelming
            for file_path in pattern_files[:3]:
                try:
                    # Mark as studied
                    self.studied_files.add(str(file_path))
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if len(content) > 200:
                        # Extract pattern concepts
                        concepts = self._extract_pattern_concepts(content, str(file_path))
                        studied_concepts.extend(concepts)

                        # Create focused summary based on study session
                        focus_area = self._get_study_focus(self.study_session_count)
                        summary = f"UE Creation Patterns Study (Session {self.study_session_count}, Focus: {focus_area}) from {file_path.name}:\n"
                        summary += f"File: {file_path}\n"
                        summary += f"Key Concepts: {', '.join(concepts[:5])}\n"
                        summary += f"Content Preview: {content[:800]}"
                        
                        await self.process_and_learn(summary, f"patterns_{file_path.name}_session_{self.study_session_count}")
                        total_content += len(content)
                        files_processed.append(file_path.name)

                        logger.info(f"📖 Studied pattern file: {file_path.name} (Session {self.study_session_count})")

                except Exception as e:
                    logger.warning(f"Failed to study {file_path}: {e}")
                    continue

            # Track study patterns to avoid repetition
            current_pattern = f"creation_patterns_session_{self.study_session_count}"
            self.last_study_patterns.append(current_pattern)
            if len(self.last_study_patterns) > 10:
                self.last_study_patterns.pop(0)  # Keep only last 10 patterns

            return {
                "success": True,
                "system": "creation_patterns",
                "session": self.study_session_count,
                "files_studied": len(files_processed),
                "files_processed": files_processed,
                "concepts_learned": studied_concepts[:10],
                "total_content_length": total_content,
                "total_files_studied": len(self.studied_files),
                "message": f"Session {self.study_session_count}: Studied {len(files_processed)} UE creation pattern files - {len(studied_concepts)} concepts learned"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_study_focus(self, session_count: int) -> str:
        """Get study focus based on session count to vary learning approach"""
        focuses = [
            "Object Creation",
            "Memory Management", 
            "Factory Patterns",
            "Builder Patterns",
            "System Architecture",
            "Component Design",
            "Resource Management",
            "Performance Optimization"
        ]
        return focuses[session_count % len(focuses)]

    # ==================== CONCEPT EXTRACTION HELPERS ====================

    def _extract_creation_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract creation concepts from UE source code"""
        concepts = []

        patterns = {
            "Creation": r'\b(?:Create|New|Spawn|Instantiate|Construct)[A-Za-z0-9_]*\b',
            "Objects": r'\b(?:UObject|AActor|UComponent|FObject)[A-Za-z0-9_]*\b',
            "Factories": r'\b(?:Factory|Builder|Creator|Generator)[A-Za-z0-9_]*\b',
            "Memory": r'\b(?:Allocate|Memory|Pool|GC)[A-Za-z0-9_]*\b'
        }

        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                unique_matches = list(set(matches))[:3]
                concepts.extend([f"{concept_type}: {match}" for match in unique_matches])

        return concepts

    def _extract_component_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract component concepts from UE source code"""
        concepts = []

        patterns = {
            "Components": r'\b(?:UComponent|USceneComponent|UPrimitiveComponent)[A-Za-z0-9_]*\b',
            "Hierarchy": r'\b(?:Parent|Child|Attach|Detach|Root)[A-Za-z0-9_]*\b',
            "Transform": r'\b(?:Transform|Location|Rotation|Scale)[A-Za-z0-9_]*\b',
            "Updates": r'\b(?:Tick|Update|Begin|End)[A-Za-z0-9_]*\b'
        }

        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                unique_matches = list(set(matches))[:3]
                concepts.extend([f"{concept_type}: {match}" for match in unique_matches])

        return concepts

    def _extract_world_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract world concepts from UE source code"""
        concepts = []

        patterns = {
            "World": r'\b(?:UWorld|AWorldSettings|ULevel)[A-Za-z0-9_]*\b',
            "Levels": r'\b(?:Level|Map|Persistent|Streaming)[A-Za-z0-9_]*\b',
            "Game": r'\b(?:GameMode|GameState|PlayerController)[A-Za-z0-9_]*\b',
            "Actors": r'\b(?:AActor|APawn|ACharacter)[A-Za-z0-9_]*\b'
        }

        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                unique_matches = list(set(matches))[:3]
                concepts.extend([f"{concept_type}: {match}" for match in unique_matches])

        return concepts

    def _extract_pattern_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract pattern concepts from UE source code"""
        concepts = []

        patterns = {
            "Patterns": r'\b(?:Factory|Singleton|Observer|Strategy)[A-Za-z0-9_]*\b',
            "Systems": r'\b(?:System|Manager|Controller|Handler)[A-Za-z0-9_]*\b',
            "Architecture": r'\b(?:Module|Plugin|Subsystem|Interface)[A-Za-z0-9_]*\b',
            "Design": r'\b(?:Abstract|Virtual|Override|Template)[A-Za-z0-9_]*\b'
        }

        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                unique_matches = list(set(matches))[:3]
                concepts.extend([f"{concept_type}: {match}" for match in unique_matches])

        return concepts

    # ==================== LEARNING MANAGEMENT ====================

    async def _reflect_on_task(self, task: Task) -> Dict[str, Any]:
        """Reflect on task results and extract learning"""
        try:
            # Prepare reflection prompt
            results_summary = json.dumps(task.results, indent=2)

            prompt = f"""As Andrio V2, reflect on the completed task and extract learning:

Task: {task.description}
Goal: {task.goal}
Status: {task.status.value}
Results Summary:
{results_summary}

Reflection Instructions:
1. Analyze what was accomplished and what wasn't
2. Identify key learnings about UE concepts or development
3. Determine what should be studied next
4. Assess progress toward learning goals
5. Suggest improvements for future tasks

Provide structured feedback:
- LEARN: [key insights gained]
- REFLECT: [analysis of approach and results]
- PLAN_TASK: [next logical task to pursue]"""

            reflection = await self._query_llm(prompt)

            # Process the reflection feedback
            feedback_processed = await self._process_ai_feedback(reflection, f"Task_Reflection_{task.id}")

            return {
                "success": True,
                "reflection": reflection,
                "feedback_processed": feedback_processed
            }

        except Exception as e:
            logger.error(f"Failed to reflect on task: {e}")
            return {"success": False, "error": str(e)}

    async def _update_learning_goals(self, task: Task) -> bool:
        """Update learning goals based on task completion"""
        try:
            # Simple progress update based on task success
            if task.status == TaskStatus.COMPLETED:
                # Determine which domain this task relates to
                domain = self._determine_task_domain(task)

                if domain and domain in self.learning_goals:
                    goal = self.learning_goals[domain]

                    # Increase mastery slightly
                    mastery_increase = 0.01  # 1% per completed task
                    goal.current_mastery = min(goal.target_mastery, goal.current_mastery + mastery_increase)
                    goal.experiments_completed += 1

                    logger.info(f"📈 Updated {domain} mastery: {goal.current_mastery:.1%}")

                # Update overall mastery
                self._update_overall_mastery()

            return True

        except Exception as e:
            logger.error(f"Failed to update learning goals: {e}")
            return False

    def _determine_task_domain(self, task: Task) -> Optional[str]:
        """Determine which learning domain a task relates to"""
        description = task.description.lower()

        domain_keywords = {
            "blueprints": ["blueprint", "visual script", "node"],
            "cpp_integration": ["c++", "cpp", "code", "programming"],
            "rendering": ["render", "lumen", "nanite", "material", "shader"],
            "physics": ["physics", "chaos", "collision", "simulation"],
            "animation": ["animation", "anim", "skeleton", "bone"],
            "ai_behavior": ["ai", "behavior", "tree", "perception", "blackboard"],
            "networking": ["network", "multiplayer", "replication"],
            "performance": ["performance", "optimization", "profiling"],
            "tools": ["tool", "editor", "plugin"],
            "assets": ["asset", "content", "import", "pipeline"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in description for keyword in keywords):
                return domain

        return None

    def _update_overall_mastery(self):
        """Update overall mastery based on individual domain progress"""
        total_mastery = sum(goal.current_mastery for goal in self.learning_goals.values())
        self.overall_mastery = total_mastery / len(self.learning_goals)

        logger.info(f"🎯 Overall mastery: {self.overall_mastery:.1%}")

    # ==================== AUTONOMOUS LEARNING PHASES ====================

    async def start_autonomous_learning(self) -> Dict[str, Any]:
        """Start CONTINUOUS autonomous learning until 60% mastery"""
        logger.info("🚀 Starting Andrio V2 CONTINUOUS autonomous learning")
        logger.info(f"📍 Current phase: {self.current_phase.value}")
        logger.info(f"🎯 Target mastery: {self.target_mastery:.1%}")
        logger.info("🔄 Will continue learning until 60% mastery is achieved!")

        learning_cycles = 0
        total_start_time = time.time()

        try:
            # CONTINUOUS LEARNING LOOP - Never stop until 60% mastery
            while self.overall_mastery < self.target_mastery:
                learning_cycles += 1
                cycle_start_time = time.time()

                logger.info(f"\n🔄 === LEARNING CYCLE {learning_cycles} ===")
                logger.info(f"📊 Current mastery: {self.overall_mastery:.1%}")
                logger.info(f"🎯 Target mastery: {self.target_mastery:.1%}")
                logger.info(f"📍 Phase: {self.current_phase.value}")

                # Execute learning cycle based on current phase
                if self.current_phase == LearningPhase.HANDS_ON_LEARNING:
                    cycle_result = await self._phase_hands_on_learning()
                elif self.current_phase == LearningPhase.STUDYING_SOURCE:
                    cycle_result = await self._phase_study_source_code()
                elif self.current_phase == LearningPhase.STUDYING_INSTALLATIONS:
                    cycle_result = await self._phase_study_installations()
                elif self.current_phase == LearningPhase.EXPERIMENTING:
                    cycle_result = await self._phase_experimentation()
                else:
                    cycle_result = {"success": False, "error": f"Unknown phase: {self.current_phase}"}

                # Check if ready to advance to next phase
                await self._check_phase_advancement()

                cycle_time = time.time() - cycle_start_time
                logger.info(f"⏱️ Cycle {learning_cycles} completed in {cycle_time:.2f}s")
                logger.info(f"📈 Updated mastery: {self.overall_mastery:.1%}")

                # Brief pause between cycles to prevent overwhelming
                await asyncio.sleep(2)

            total_time = time.time() - total_start_time

            if self.overall_mastery >= self.target_mastery:
                logger.info("🎉 TARGET MASTERY ACHIEVED!")
                logger.info(f"✅ Reached {self.overall_mastery:.1%} mastery in {learning_cycles} cycles")
                logger.info(f"⏱️ Total learning time: {total_time:.2f}s")
                return {
                    "success": True,
                    "mastery_achieved": True,
                    "final_mastery": self.overall_mastery,
                    "learning_cycles": learning_cycles,
                    "total_time": total_time
                }
            else:
                return {
                    "success": True,
                    "mastery_achieved": False,
                    "current_mastery": self.overall_mastery,
                    "learning_cycles": learning_cycles,
                    "total_time": total_time,
                    "message": "Learning paused - restart to continue toward 60% mastery"
                }

        except Exception as e:
            logger.error(f"Autonomous learning failed: {e}")
            return {"success": False, "error": str(e)}

    async def _phase_hands_on_learning(self) -> Dict[str, Any]:
        """Phase 1: Hands-on UE learning - try operations first, study source when stuck"""
        logger.info("🎮 Phase 1: Hands-on UE learning")

        try:
            # Let AI decide what hands-on UE operations to try using thinking mode
            context = self._get_learning_context()

            prompt = f"""
            {context}

            You are in HANDS-ON LEARNING phase. Your goal is to learn UE by actually USING it.

            Plan hands-on UE operations to try (create projects, use editor, experiment with features).
            Only study source code if you get stuck and need to understand how something works.

            Focus on practical UE usage and experimentation. Use your available tools to create actionable steps.

            AVAILABLE TOOLS:
            {self.toolbox.get_tools_summary() if self.toolbox else "No tools available"}

            Create a task plan with specific tool commands for hands-on UE learning.
            """

            # Use thinking mode if available for better task planning
            if self.thinking_mode_enabled and self.thinking_mode_compatible:
                logger.info("🧠 Using thinking mode for hands-on learning task planning")
                task_data = await self._plan_task_with_thinking(
                    "Plan hands-on UE learning operations using available tools"
                )
            else:
                ai_response = await self._query_llm(prompt)
                task_data = self._extract_task_from_response(ai_response)

            if not task_data:
                # Fallback to basic hands-on operations
                task_data = {
                    "goal": "Learn UE through hands-on experimentation",
                "steps": [
                        "Check Epic Launcher status using check_epic_launcher_status",
                        "Launch Epic Games Launcher if needed using launch_epic_games_launcher",
                        "List available UE templates using list_unreal_templates",
                        "Create new project using create_unreal_project HandsOnLearning ThirdPersonBP",
                        "Open the created project using open_unreal_project"
                    ],
                    "domain": "hands_on_operations"
                }

            # Execute the hands-on learning task
            task = Task(
                id=f"hands_on_{int(time.time())}",
                description=task_data["goal"],
                goal=task_data["goal"],
                steps=task_data["steps"],
                status=TaskStatus.PLANNED,
                created_at=datetime.now(),
                results={},
                learning_extracted=[],
                next_tasks=[]
            )

            result = await self._execute_task(task)

            # Reflect on the task with thinking mode for deeper insights
            if result.get("success", False):
                reflection = await self._reflect_on_task_with_thinking(task)
                result["reflection"] = reflection
                
                # Smaller increment for hands-on learning to require more cycles
                self.overall_mastery += 0.02  # Reduced from 0.05 to 0.02
                logger.info(f"📈 Hands-on learning cycle completed. Mastery: {self.overall_mastery:.1%}")
            else:
                logger.warning(f"⚠️ Hands-on learning cycle failed, but continuing...")
                # Still give small progress for attempting
                self.overall_mastery += 0.01
                logger.info(f"📈 Attempted hands-on learning. Mastery: {self.overall_mastery:.1%}")

            return result

        except Exception as e:
            logger.error(f"Hands-on learning phase failed: {e}")
            return {"success": False, "error": str(e)}

    async def _phase_study_source_code(self) -> Dict[str, Any]:
        """Phase 2: Study UE source code systematically"""
        logger.info("📚 Phase 2: Studying UE source code")

        # Use thinking mode for strategic source code study planning
        task_description = "Study UE source code to build foundational knowledge"
        task_data = await self._plan_task_with_thinking(task_description)
        
        if task_data:
            result = await self.plan_and_execute_task(task_description)
        else:
            result = await self.plan_and_execute_task(task_description)

        return result

    async def _phase_study_installations(self) -> Dict[str, Any]:
        """Phase 2: Study UE installations and structure"""
        logger.info("🏗️ Phase 2: Studying UE installations")

        # Use thinking mode for installation analysis planning
        task_description = "Analyze UE installation structure and understand engine organization"
        task_data = await self._plan_task_with_thinking(task_description)
        
        if task_data:
            result = await self.plan_and_execute_task(task_description)
        else:
            result = await self.plan_and_execute_task(task_description)

        return result

    async def _phase_experimentation(self) -> Dict[str, Any]:
        """Phase 3: Experimentation with UE features"""
        logger.info("🧪 Phase 3: UE experimentation")

        # Use thinking mode for experimental learning planning
        task_description = "Experiment with UE features to gain practical experience"
        task_data = await self._plan_task_with_thinking(task_description)
        
        if task_data:
            result = await self.plan_and_execute_task(task_description)
        else:
            result = await self.plan_and_execute_task(task_description)

        return result

    async def _check_phase_advancement(self):
        """Check if ready to advance to next learning phase"""
        if self.current_phase == LearningPhase.STUDYING_SOURCE and self.overall_mastery >= 0.1:
            self.current_phase = LearningPhase.STUDYING_INSTALLATIONS
            logger.info("🎉 Advanced to Phase 2: Studying Installations")
        elif self.current_phase == LearningPhase.STUDYING_INSTALLATIONS and self.overall_mastery >= 0.2:
            self.current_phase = LearningPhase.EXPERIMENTING
            logger.info("🎉 Advanced to Phase 3: Experimentation")
        elif self.current_phase == LearningPhase.EXPERIMENTING and self.overall_mastery >= self.target_mastery:
            self.current_phase = LearningPhase.INTERACTIVE
            logger.info("🎉 Advanced to Phase 4: Interactive Mode - Target mastery achieved!")

    def get_status(self) -> Dict[str, Any]:
        """Get current learning status"""
        return {
            "current_phase": self.current_phase.value,
            "overall_mastery": self.overall_mastery,
            "target_mastery": self.target_mastery,
            "learning_goals": {domain: {
                "current_mastery": goal.current_mastery,
                "target_mastery": goal.target_mastery,
                "experiments_completed": goal.experiments_completed
            } for domain, goal in self.learning_goals.items()},
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "knowledge_base_size": self.collection.count(),
            "entities_in_graph": len(self.entities)
        }

    async def _plan_task_with_thinking(self, goal_description: str) -> Optional[Dict[str, Any]]:
        """Plan a task using thinking mode to understand the reasoning process"""
        try:
            logger.info(f"🧠 Planning task with thinking mode: {goal_description}")
            
            context = self._get_learning_context()
            
            prompt = f"""Plan a hands-on UE learning task for the following goal. Use your available tools to create actionable steps.

GOAL: {goal_description}

CURRENT CONTEXT:
{context}

AVAILABLE TOOLS:
{self.toolbox.get_tools_summary()}

INSTRUCTIONS:
1. Create a detailed task plan with specific tool commands
2. Focus on hands-on experimentation and real UE operations
3. Use exact tool syntax (e.g., "create_unreal_project MyProject ThirdPersonBP")
4. Structure as JSON with goal, steps, and expected_outcomes

TASK PLAN:"""

            # Use thinking mode to see the planning reasoning
            result = await self._query_llm_with_thinking(prompt)
            
            if result["success"] and result["thinking"]:
                logger.info(f"🧠 Task Planning Reasoning: {result['thinking'][:300]}...")
            
            return self._extract_task_from_response(result["response"])
            
        except Exception as e:
            logger.error(f"Error in thinking-enabled task planning: {e}")
            return None

    async def _reflect_on_task_with_thinking(self, task: Task) -> Dict[str, Any]:
        """Reflect on task completion with thinking mode for deeper insights"""
        try:
            logger.info(f"🧠 Reflecting on task with thinking mode: {task.description}")
            
            prompt = f"""Reflect on the completed task and analyze the learning outcomes with deep reasoning.

TASK DETAILS:
- Goal: {task.goal}
- Description: {task.description}
- Status: {task.status.value}
- Steps Completed: {len(task.steps)}
- Results: {task.results}

REFLECTION INSTRUCTIONS:
1. Analyze what was learned from this task
2. Identify successful strategies and areas for improvement
3. Determine how this advances toward 60% UE mastery
4. Suggest next logical learning steps
5. Update learning strategy if needed

DEEP REFLECTION:"""

            # Use thinking mode for comprehensive reflection
            result = await self._query_llm_with_thinking(prompt)
            
            reflection_data = {
                "task_id": task.id,
                "thinking_process": result.get("thinking", ""),
                "reflection_content": result.get("response", ""),
                "timestamp": datetime.now().isoformat(),
                "success": result.get("success", False)
            }
            
            if result["success"] and result["thinking"]:
                logger.info(f"🧠 Task Reflection Reasoning: {result['thinking'][:300]}...")
                
                # Store the thinking process for learning analysis
                await self._store_thinking_process(task.id, result["thinking"], "task_reflection")
            
            return reflection_data
            
        except Exception as e:
            logger.error(f"Error in thinking-enabled reflection: {e}")
            return {"success": False, "error": str(e)}

    async def _store_thinking_process(self, context_id: str, thinking: str, process_type: str) -> bool:
        """Store AI thinking processes for learning analysis"""
        try:
            doc_id = f"thinking_{process_type}_{context_id}_{int(time.time())}"
            embedding = self.embedding_model.encode(thinking)

            self.collection.add(
                documents=[thinking],
                metadatas=[{
                    "source": f"ai_thinking_{process_type}",
                    "context_id": context_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "thinking_process"
                }],
                ids=[doc_id],
                embeddings=[embedding.tolist()]
            )

            logger.info(f"Stored thinking process: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store thinking process: {e}")
            return False

    async def _analyze_learning_progress_with_thinking(self) -> Dict[str, Any]:
        """Analyze overall learning progress using thinking mode for strategic insights"""
        try:
            logger.info("🧠 Analyzing learning progress with thinking mode")
            
            # Gather comprehensive learning data
            status = self.get_status()
            completed_tasks = len(self.completed_tasks)
            active_tasks = len(self.active_tasks)
            
            prompt = f"""Analyze AndrioV2's learning progress and provide strategic insights for achieving 60% UE mastery.

CURRENT LEARNING STATUS:
- Overall Mastery: {self.overall_mastery:.1%}
- Target Mastery: {self.target_mastery:.1%}
- Current Phase: {self.current_phase.value}
- Completed Tasks: {completed_tasks}
- Active Tasks: {active_tasks}
- Learning Goals: {len(self.learning_goals)}

LEARNING GOALS PROGRESS:
{self._format_learning_goals_for_analysis()}

ANALYSIS INSTRUCTIONS:
1. Evaluate current learning trajectory toward 60% mastery
2. Identify knowledge gaps and priority learning areas
3. Assess effectiveness of hands-on vs study approaches
4. Recommend strategic adjustments to learning plan
5. Suggest specific next actions for maximum learning impact

STRATEGIC LEARNING ANALYSIS:"""

            # Use thinking mode for strategic analysis
            result = await self._query_llm_with_thinking(prompt)
            
            analysis_data = {
                "thinking_process": result.get("thinking", ""),
                "strategic_analysis": result.get("response", ""),
                "timestamp": datetime.now().isoformat(),
                "current_mastery": self.overall_mastery,
                "target_mastery": self.target_mastery,
                "success": result.get("success", False)
            }
            
            if result["success"] and result["thinking"]:
                logger.info(f"🧠 Strategic Learning Analysis: {result['thinking'][:300]}...")
                
                # Store strategic thinking for future reference
                await self._store_thinking_process("learning_progress", result["thinking"], "strategic_analysis")
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error in thinking-enabled learning analysis: {e}")
            return {"success": False, "error": str(e)}

    def _format_learning_goals_for_analysis(self) -> str:
        """Format learning goals data for AI analysis"""
        if not self.learning_goals:
            return "No learning goals defined yet."
        
        formatted = []
        for goal_id, goal in self.learning_goals.items():
            formatted.append(f"- {goal.domain}: {goal.current_mastery:.1%} / {goal.target_mastery:.1%} ({goal.experiments_completed} experiments)")
        
        return "\n".join(formatted)


# ==================== MAIN INTERFACE ====================

async def main():
    """Main interface for AndrioV2"""
    print("🤖 AndrioV2 - Agentic Unreal Engine AI Assistant")
    print("=" * 60)
    print("✨ Features:")
    print("   📚 Autonomous UE source code study")
    print("   🧠 Bidirectional RAG learning system")
    print("   🎯 Agentic task planning and execution")
    print("   🔄 Systematic experimentation with UE")
    print("   📈 Progress tracking toward 60% mastery")
    print("=" * 60)

    # Check available models
    print("\n🤖 Checking available Ollama models...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    print(f"📋 Available models: {', '.join(available_models)}")

                    # Check for preferred models (prioritize andriov2)
                    preferred_models = ["andriov2", "deepseek-r1:14b", "andrio:latest", "qwen3:14b"]
                    selected_model = None

                    for model in preferred_models:
                        if any(available.startswith(model.split(':')[0]) for available in available_models):
                            selected_model = next(available for available in available_models
                                                if available.startswith(model.split(':')[0]))
                            break

                    if not selected_model:
                        selected_model = available_models[0] if available_models else "andriov2"

                    print(f"✅ Selected model: {selected_model}")
                else:
                    print("⚠️  Could not connect to Ollama, using default model")
                    selected_model = "andriov2"
    except Exception as e:
        print(f"⚠️  Error checking models: {e}, using default")
        selected_model = "andriov2"

    # Initialize AndrioV2
    print(f"\n🚀 Initializing AndrioV2 with {selected_model}...")
    try:
        andrio = AndrioV2(model_name=selected_model)
        print("✅ AndrioV2 initialized successfully!")

        # Show current status
        status = andrio.get_status()
        print(f"\n📊 Current Status:")
        print(f"   📍 Phase: {status['current_phase']}")
        print(f"   🎯 Overall Mastery: {status['overall_mastery']:.1%}")
        print(f"   📚 Knowledge Base: {status['knowledge_base_size']} documents")
        print(f"   🕸️  Knowledge Graph: {status['entities_in_graph']} entities")

    except Exception as e:
        print(f"❌ Failed to initialize AndrioV2: {e}")
        return

    # Main interaction loop
    print(f"\n🎮 AndrioV2 Interactive Mode")
    print("Commands:")
    print("  'autonomous' - Start autonomous learning")
    print("  'task <description>' - Plan and execute a specific task")
    print("  'query <question>' - Query the knowledge base")
    print("  'status' - Show current learning status")
    print("  'study <path>' - Study specific content")
    print("  'analyze' - Analyze UE source code structure")
    print("  'tools' - List available tools")
    print("  'tool <command>' - Execute a tool command")
    print("  'thinking' - Toggle thinking mode (Ollama 0.9.0)")
    print("  'progress' - Analyze learning progress with thinking")
    print("  'exit' - Exit system")
    print("-" * 60)
    
    # Show thinking mode status
    if andrio.thinking_mode_compatible:
        thinking_status = andrio.get_thinking_status()
        print(f"🧠 Thinking Mode: {thinking_status}")
        print("-" * 60)

    while True:
        try:
            user_input = input(f"\n🤖 Andrio ({andrio.current_phase.value}): ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye! Andrio's learning progress has been saved.")
                break

            elif user_input.lower() == 'autonomous':
                print("🚀 Starting autonomous learning...")
                result = await andrio.start_autonomous_learning()

                if result["success"]:
                    print(f"✅ Autonomous learning cycle completed!")
                    print(f"📊 Task: {result.get('task_id', 'N/A')}")
                    print(f"⏱️  Time: {result.get('total_time', 0):.2f}s")

                    # Show updated status
                    status = andrio.get_status()
                    print(f"📈 Updated mastery: {status['overall_mastery']:.1%}")
                else:
                    print(f"❌ Autonomous learning failed: {result.get('error', 'Unknown error')}")

            elif user_input.lower().startswith('task '):
                task_description = user_input[5:].strip()
                if task_description:
                    print(f"📋 Planning and executing task: {task_description}")
                    result = await andrio.plan_and_execute_task(task_description)

                    if result["success"]:
                        print(f"✅ Task completed successfully!")
                        print(f"📊 Steps: {result['execution_result']['steps_completed']}/{result['execution_result']['total_steps']}")
                        print(f"⏱️  Time: {result['total_time']:.2f}s")
                    else:
                        print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Please provide a task description")

            elif user_input.lower().startswith('query '):
                query = user_input[6:].strip()
                if query:
                    print(f"🔍 Querying knowledge base: {query}")
                    result = await andrio.query_knowledge(query)

                    if result.get("ai_response"):
                        print(f"\n📝 Response:")
                        print("-" * 50)
                        print(result["ai_response"])
                        print("-" * 50)
                        print(f"📊 Results found: {result['results_found']}")
                        print(f"⏱️  Time: {result['total_time']:.2f}s")
                    else:
                        print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Please provide a query")

            elif user_input.lower() == 'status':
                status = andrio.get_status()
                print(f"\n📊 AndrioV2 Learning Status:")
                print(f"   📍 Current Phase: {status['current_phase']}")
                print(f"   🎯 Overall Mastery: {status['overall_mastery']:.1%} (target: {status['target_mastery']:.1%})")
                print(f"   📚 Knowledge Base: {status['knowledge_base_size']} documents")
                print(f"   🕸️  Knowledge Graph: {status['entities_in_graph']} entities")
                print(f"   📋 Active Tasks: {status['active_tasks']}")
                print(f"   ✅ Completed Tasks: {status['completed_tasks']}")

                print(f"\n📈 Domain Progress:")
                for domain, goal_status in status['learning_goals'].items():
                    mastery = goal_status['current_mastery']
                    target = goal_status['target_mastery']
                    experiments = goal_status['experiments_completed']
                    print(f"   {domain}: {mastery:.1%}/{target:.1%} ({experiments} experiments)")

            elif user_input.lower().startswith('study '):
                path = user_input[6:].strip()
                if path and os.path.exists(path):
                    print(f"📚 Studying content: {path}")
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        result = await andrio.process_and_learn(content, path)

                        if result["success"]:
                            print(f"✅ Content processed successfully!")
                            print(f"📊 Documents: {result['documents_created']}")
                            print(f"🏷️  Entities: {result['entities_extracted']}")
                            print(f"⏱️  Time: {result['processing_time']:.2f}s")
                        else:
                            print(f"❌ Failed to process content: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        print(f"❌ Error reading file: {e}")
                else:
                    print("❌ Please provide a valid file path")

            elif user_input.lower() == 'analyze':
                print("🔍 Analyzing UE source code structure...")
                try:
                    # Analyze UE source structure
                    structure_result = await andrio._analyze_ue_project_structure(andrio.ue_source_path)
                    classes_result = await andrio._find_ue_classes_and_functions(andrio.ue_source_path)

                    if structure_result.get("success"):
                        analysis = structure_result["analysis"]
                        print(f"\n📊 UE Source Analysis:")
                        print(f"   📁 Total Files: {analysis.get('total_files', 0)}")
                        print(f"   📂 Source Directories: {len(analysis.get('source_directories', []))}")
                        print(f"   📦 Content Directories: {len(analysis.get('content_directories', []))}")
                        print(f"   ⚙️  Config Files: {len(analysis.get('config_files', []))}")
                        print(f"   🔨 Build Files: {len(analysis.get('build_files', []))}")

                    if classes_result.get("success"):
                        results = classes_result["results"]
                        print(f"\n🏗️ UE Code Analysis:")
                        print(f"   📝 Files Analyzed: {results.get('files_analyzed', 0)}")
                        print(f"   🏛️  Classes Found: {len(results.get('classes', []))}")
                        print(f"   ⚙️  Functions Found: {len(results.get('functions', []))}")
                        print(f"   🔧 Macros Found: {len(results.get('macros', []))}")
                        print(f"   📚 Engine Includes: {len(results.get('includes', []))}")

                        # Show some examples
                        if results.get('classes'):
                            print(f"\n📋 Example Classes:")
                            for cls in results['classes'][:5]:
                                print(f"   - {cls['name']} ({cls['type']})")

                        if results.get('functions'):
                            print(f"\n📋 Example Functions:")
                            for func in results['functions'][:5]:
                                print(f"   - {func['name']} ({func['type']})")

                    print(f"\n✅ Analysis complete!")

                except Exception as e:
                    print(f"❌ Analysis failed: {e}")

            elif user_input.lower() == 'tools':
                print("🧰 Available Tools:")
                print(andrio.list_available_tools())

            elif user_input.lower().startswith('tool '):
                command = user_input[5:].strip()
                if command:
                    print(f"🔧 Executing tool command: {command}")
                    result = andrio.execute_tool_command(command)
                    print(f"📝 Tool output: {result}")
                else:
                    print("❌ Please provide a tool command")

            elif user_input.lower() == 'thinking':
                print("🧠 Toggling thinking mode...")
                result = andrio.toggle_thinking_mode()
                print(result)

            elif user_input.lower() == 'progress':
                print("🧠 Analyzing learning progress with thinking mode...")
                result = await andrio._analyze_learning_progress_with_thinking()
                
                if result.get("success"):
                    print(f"\n📊 Learning Progress Analysis:")
                    print(f"🎯 Overall Mastery: {result['current_mastery']:.1%}")
                    print(f"📈 Target Mastery: {result['target_mastery']:.1%}")
                    progress = (result['current_mastery'] / result['target_mastery']) * 100
                    print(f"📈 Progress towards target: {progress:.1f}%")
                    
                    if result.get("thinking_process"):
                        print(f"\n🧠 AI Strategic Thinking:")
                        print("-" * 50)
                        print(result["thinking_process"][:500] + "..." if len(result["thinking_process"]) > 500 else result["thinking_process"])
                        print("-" * 50)
                    
                    if result.get("strategic_analysis"):
                        print(f"\n📝 Strategic Analysis:")
                        print("-" * 50)
                        print(result["strategic_analysis"])
                        print("-" * 50)
                else:
                    print(f"❌ Progress analysis failed: {result.get('error', 'Unknown error')}")

            elif not user_input:
                continue
            else:
                print("❌ Unknown command. Type 'exit' to quit or use one of the available commands.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye! Andrio's learning progress has been saved.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
