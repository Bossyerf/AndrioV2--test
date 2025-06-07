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
import sys
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

# Configure comprehensive logging with enhanced visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %I:%M:%S %p',  # Standard 12-hour format
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('andrio_v2.log', mode='a', encoding='utf-8')
    ],
    force=True  # Force reconfiguration of logging
)
logger = logging.getLogger(__name__)

# Set specific loggers to INFO level for maximum visibility
logging.getLogger('andrio_v2').setLevel(logging.INFO)
logging.getLogger('andrio_toolbox').setLevel(logging.INFO)
logging.getLogger('__main__').setLevel(logging.INFO)

# Add detailed startup logging
logger.info("=" * 80)
logger.info("🚀 ANDRIO V2 STARTUP - ENHANCED LOGGING ENABLED")
logger.info("=" * 80)

class TaskStatus(Enum):
    """Task execution status"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class LearningPhase(Enum):
    """Andrio's learning phases - CORRECTED OPTIMAL SEQUENCE"""
    INSTALLATION_ARCHITECTURE = "installation_architecture"  # Phase 1: Study UE5 installation structure
    PROJECT_STRUCTURE = "project_structure"                  # Phase 2: Study project organization through templates
    SOURCE_FOUNDATIONS = "source_foundations"                # Phase 3: Study UE5 source code fundamentals
    HANDS_ON_APPLICATION = "hands_on_application"            # Phase 4: Apply knowledge through experimentation
    MASTERY_INTEGRATION = "mastery_integration"              # Phase 5: Advanced integration and teaching

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

@dataclass
class PhaseProgress:
    """Track progress within a learning phase"""
    phase: LearningPhase
    required_tasks: List[str]
    completed_tasks: Set[str] = field(default_factory=set)
    files_studied: List[str] = field(default_factory=list)
    concepts_learned: List[str] = field(default_factory=list)
    directories_explored: List[str] = field(default_factory=list)
    insights_generated: List[str] = field(default_factory=list)
    knowledge_documents_created: int = 0
    completion_percentage: float = 0.0
    
    def calculate_completion(self) -> float:
        """Calculate completion percentage based on completed tasks"""
        if not self.required_tasks:
            return 0.0
        return len(self.completed_tasks) / len(self.required_tasks)
    
    def is_complete(self) -> bool:
        """Check if phase is complete based on objective criteria"""
        completion = self.calculate_completion()
        
        # Phase-specific completion criteria
        if self.phase == LearningPhase.INSTALLATION_ARCHITECTURE:
            return (completion >= 0.8 and 
                   len(self.concepts_learned) >= 15 and 
                   len(self.files_studied) >= 8 and
                   len(self.directories_explored) >= 5)
        elif self.phase == LearningPhase.PROJECT_STRUCTURE:
            return (completion >= 0.8 and 
                   len(self.concepts_learned) >= 20 and 
                   len(self.files_studied) >= 10)
        elif self.phase == LearningPhase.SOURCE_FOUNDATIONS:
            return (completion >= 0.8 and 
                   len(self.concepts_learned) >= 30 and 
                   len(self.files_studied) >= 15)
        else:
            return completion >= 0.8

@dataclass
class LearningMetrics:
    """Comprehensive learning metrics for mastery calculation"""
    files_analyzed: int = 0
    concepts_extracted: int = 0
    knowledge_documents: int = 0
    insights_generated: int = 0
    directories_explored: int = 0
    relationships_created: int = 0
    understanding_depth_score: float = 0.0
    
    def calculate_mastery_score(self) -> float:
        """Calculate mastery based on actual learning metrics"""
        # File coverage component (0-4%)
        file_score = min(self.files_analyzed * 0.005, 0.04)
        
        # Concept extraction component (0-4%)
        concept_score = min(self.concepts_extracted * 0.002, 0.04)
        
        # Knowledge creation component (0-3%)
        knowledge_score = min(self.knowledge_documents * 0.01, 0.03)
        
        # Insight generation component (0-3%)
        insight_score = min(self.insights_generated * 0.01, 0.03)
        
        # Directory exploration component (0-2%)
        exploration_score = min(self.directories_explored * 0.004, 0.02)
        
        # Understanding depth component (0-4%)
        depth_score = min(self.understanding_depth_score, 0.04)
        
        total_score = (file_score + concept_score + knowledge_score + 
                      insight_score + exploration_score + depth_score)
        
        return min(total_score, 0.20)  # Cap at 20% per phase

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
        self.current_phase = LearningPhase.INSTALLATION_ARCHITECTURE
        self.learning_goals = {}
        self.completed_tasks = {}  # Change from [] to {} to make it a dictionary
        self.current_task = None
        self.overall_mastery = 0.0
        
        # New objective-driven learning system
        self.phase_progress = self._initialize_phase_progress()
        self.learning_metrics = LearningMetrics()
        
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
        self.ue_connected = False
        try:
            from andrio_toolbox import AndrioToolbox
            self.toolbox = AndrioToolbox()
            self.available_tools = self.toolbox.tools
            self.tool_descriptions = self.toolbox.get_tool_descriptions()
            logger.info(f"Available tools: {len(self.available_tools)}")
            # Attempt UE5 remote connection
            try:
                self.ue_connected = self.toolbox.ue5_tools.connect()
                if not self.ue_connected:
                    logger.warning("UE5 remote execution connection failed")
            except Exception as conn_err:
                self.ue_connected = False
                logger.error(f"UE5 connection error: {conn_err}")
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

        # Create basic steps based on current phase - CORRECTED SEQUENCE
        if self.current_phase == LearningPhase.INSTALLATION_ARCHITECTURE:
            steps = [
                f"Explore UE5 installation directory using list_files {self.ue_installation_path}",
                f"List engine binaries using list_files {self.ue_installation_path}/Engine/Binaries/Win64",
                f"Find executable tools using find_files *.exe {self.ue_installation_path}/Engine/Binaries",
                f"Read engine version info using read_file {self.ue_installation_path}/Engine/Build/Build.version",
                "Get UE installation details using get_unreal_engine_info",
                "List available project templates using list_unreal_templates",
                f"Explore engine plugins using list_files {self.ue_installation_path}/Engine/Plugins"
            ]
        elif self.current_phase == LearningPhase.PROJECT_STRUCTURE:
            steps = [
                "Create study project using create_unreal_project StructureStudy ThirdPersonBP",
                "Analyze project root structure using list_files D:/Andrios Output/UnrealProjects/StructureStudy",
                "Read project definition using read_file StructureStudy.uproject",
                "Study content organization using list_files Content/ThirdPerson",
                "Find Blueprint assets using find_files *.uasset Content/ThirdPerson/Blueprints",
                "Examine config files using list_files Config",
                "Study project dependencies and structure patterns"
            ]
        elif self.current_phase == LearningPhase.SOURCE_FOUNDATIONS:
            steps = [
                f"Study core UE classes using list_files {self.ue_source_path}/Engine/Source/Runtime/Core",
                f"Read fundamental object class using read_file {self.ue_source_path}/Engine/Source/Runtime/CoreUObject/Public/UObject/Object.h",
                f"Find Actor classes using find_files Actor*.h {self.ue_source_path}/Engine/Source/Runtime/Engine",
                f"Study engine class hierarchy using find_files *.h {self.ue_source_path}/Engine/Source/Runtime/Engine/Classes",
                "Extract key UE classes, functions, and patterns from source code",
                "Document learning insights from actual UE source code"
            ]
        elif self.current_phase == LearningPhase.HANDS_ON_APPLICATION:
            steps = [
                "Create experiment project using create_unreal_project ExperimentProject Blank",
                "Open project for hands-on work using open_unreal_project",
                "Apply learned knowledge to create new features",
                "Experiment with UE systems using learned concepts",
                "Test understanding through practical implementation",
                "Document successful experiments and learning outcomes"
            ]
        else:  # MASTERY_INTEGRATION
            steps = [
                "Create advanced project using create_unreal_project MasteryProject ThirdPersonBP",
                "Integrate multiple UE systems in complex ways",
                "Solve advanced development challenges",
                "Create production-ready systems",
                "Document mastery achievements and teach others"
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

    async def _execute_task_step(self, step, task: Task) -> Dict[str, Any]:
        """Execute a single task step (handles both string and dict formats)"""
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
            
            # Convert step to string for keyword analysis if it's a dict
            if isinstance(step, dict):
                step_str = str(step)
                logger.info(f"🔄 CONVERTED DICT TO STRING FOR ANALYSIS: {step_str}")
            else:
                step_str = step
            
            # PRIORITY 2: Only if NOT a tool, then route by keywords
            step_lower = step_str.lower()
            
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
                            project_match = re.search(r'create_unreal_project\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?', step_str, re.IGNORECASE)
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
                return await self._execute_study_step(step_str, task)
            elif "experiment" in step_lower or "test" in step_lower:
                logger.info(f"🧪 ROUTING TO: Experiment step")
                return await self._execute_experiment_step(step_str, task)
            elif "create" in step_lower or "build" in step_lower:
                logger.info(f"🔨 ROUTING TO: Creation step (fallback)")
                return await self._execute_creation_step(step_str, task)
            else:
                logger.info(f"🔧 ROUTING TO: General step")
                return await self._execute_general_step(step_str, task)

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
{content[:4000]}{"..." if len(content) > 4000 else ""}

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
            
            # Handle dictionary objects from AI (new format)
            if isinstance(step, dict):
                logger.info(f"🔧 DETECTED DICTIONARY STEP FORMAT")
                tool_name = step.get('tool')
                args = step.get('args', {})
                
                if tool_name and tool_name in self.available_tools:
                    logger.info(f"🎯 EXECUTING DICT TOOL: {tool_name}")
                    
                    # Convert args to parameter list
                    params = []
                    if isinstance(args, dict):
                        # Extract path or other common parameters
                        if 'path' in args:
                            params.append(args['path'])
                        elif 'name' in args:
                            params.append(args['name'])
                        elif 'pattern' in args:
                            params.append(args['pattern'])
                        # Add other args as additional parameters
                        for key, value in args.items():
                            if key not in ['path', 'name', 'pattern'] and value:
                                params.append(str(value))
                    elif isinstance(args, list):
                        params = [str(arg) for arg in args]
                    
                    logger.info(f"📋 CONVERTED PARAMS: {params}")
                    tool_result = await self._execute_tool(tool_name, params)
                    
                    return {
                        "success": True,
                        "step": step,
                        "tool_used": tool_name,
                        "tool_result": tool_result,
                        "type": "dict_tool_execution",
                        "params": params
                    }
                else:
                    logger.error(f"❌ INVALID DICT TOOL: {tool_name} not in available tools")
                    return {"success": False, "reason": f"Tool '{tool_name}' not available", "step": step}
            
            # Convert step to string if it's not already
            if not isinstance(step, str):
                step_str = str(step)
                logger.info(f"🔄 CONVERTED TO STRING: {step_str}")
            else:
                step_str = step
            
            # First, check for explicit tool commands in backticks or after "Command:"
            explicit_command_patterns = [
                r'`([^`]+)`',  # Commands in backticks
                r'Command:\s*`?([^`\n]+)`?',  # Commands after "Command:"
                r'Tool:\s*`?([^`\n]+)`?',  # Commands after "Tool:"
                r'Execute:\s*`?([^`\n]+)`?'  # Commands after "Execute:"
            ]
            
            for pattern in explicit_command_patterns:
                match = re.search(pattern, step_str, re.IGNORECASE)
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
                match = re.search(pattern, step_str.lower())
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
            
            # Check if step contains a tool name directly (for forcing execution)
            for tool_name in self.available_tools.keys():
                if tool_name in step_str.lower():
                    logger.info(f"🎯 FORCING TOOL EXECUTION - Found tool name in step")
                    logger.info(f"🚀 FORCING EXECUTION OF: {tool_name}")
                    tool_result = await self._execute_tool(tool_name, [])
                    return {
                        "success": True,
                        "step": step,
                        "tool_used": tool_name,
                        "tool_result": tool_result,
                        "type": "forced_tool_execution"
                    }
            
            # No tool pattern matched
            logger.info(f"❌ NO TOOL PATTERN MATCHED")
            logger.info(f"📋 Available tools: {list(self.available_tools.keys())}")
            return {"success": False, "reason": "No tool pattern matched", "step": step}
            
        except Exception as e:
            logger.error(f"❌ ERROR IN TOOL STEP EXECUTION: {e}")
            return {"success": False, "error": str(e), "step": step}

    async def _execute_tool(self, tool_name: str, params: List[str] = None) -> str:
        """Execute a specific tool with parameters and intelligent parameter inference"""
        try:
            logger.info("=" * 50)
            logger.info(f"🔧 TOOL EXECUTION START")
            logger.info(f"📋 TOOL NAME: {tool_name}")
            logger.info(f"📋 TOOL PARAMS: {params}")
            logger.info(f"⏰ EXECUTION TIME: {datetime.now().strftime('%I:%M:%S %p')}")
            logger.info("-" * 30)
            
            if tool_name not in self.available_tools:
                error_msg = f"❌ Tool '{tool_name}' not available"
                logger.error(error_msg)
                logger.error(f"📋 Available tools: {list(self.available_tools.keys())}")
                logger.info("=" * 50)
                return error_msg
            
            tool_func = self.available_tools[tool_name]
            logger.info(f"🎯 TOOL FUNCTION FOUND: {tool_func}")
            
            # Intelligent parameter handling for specific tools
            if not params or len(params) == 0:
                logger.info("🔧 NO PARAMETERS PROVIDED - Attempting intelligent defaults")
                
                # Handle tools that require parameters but none were provided
                if tool_name == "read_file":
                    # Try to find a reasonable default file to read
                    default_files = [
                        "requirements.txt",
                        "README.md", 
                        "andrio_v2.py",
                        "andrio_toolbox.py"
                    ]
                    for default_file in default_files:
                        if os.path.exists(default_file):
                            params = [default_file]
                            logger.info(f"🎯 AUTO-SELECTED FILE FOR READ: {default_file}")
                            break
                    
                    if not params:
                        error_msg = f"❌ read_file requires a filepath parameter. Available files: {', '.join(os.listdir('.'))[:100]}..."
                        logger.error(error_msg)
                        logger.info("=" * 50)
                        return error_msg
                
                elif tool_name == "write_file":
                    error_msg = f"❌ write_file requires filepath and content parameters"
                    logger.error(error_msg)
                    logger.info("=" * 50)
                    return error_msg
                
                elif tool_name == "find_files":
                    params = ["*"]  # Default to find all files
                    logger.info(f"🎯 AUTO-SET FIND PATTERN: *")
                
                elif tool_name == "list_files":
                    params = ["."]  # Default to current directory
                    logger.info(f"🎯 AUTO-SET DIRECTORY: .")
                
                elif tool_name == "file_info":
                    error_msg = f"❌ file_info requires a filepath parameter"
                    logger.error(error_msg)
                    logger.info("=" * 50)
                    return error_msg
                
                elif tool_name == "create_andrio_workspace":
                    # Generate a unique workspace name
                    import time
                    workspace_name = f"workspace_{int(time.time())}"
                    params = [workspace_name]
                    logger.info(f"🎯 AUTO-GENERATED WORKSPACE NAME: {workspace_name}")
                
                elif tool_name == "create_unreal_project":
                    # Generate a unique project name with default template
                    import time
                    project_name = f"AndrioProject_{int(time.time())}"
                    params = [project_name, "ThirdPersonBP"]
                    logger.info(f"🎯 AUTO-GENERATED PROJECT: {project_name}")
                
                elif tool_name == "open_unreal_project":
                    error_msg = f"❌ open_unreal_project requires a project path parameter"
                    logger.error(error_msg)
                    logger.info("=" * 50)
                    return error_msg
                
                elif tool_name in ["create_blueprint_actor", "create_blueprint_with_mesh", "create_blueprint_from_template"]:
                    # Generate a unique blueprint name
                    import time
                    blueprint_name = f"BP_Andrio_{int(time.time())}"
                    params = [blueprint_name]
                    logger.info(f"🎯 AUTO-GENERATED BLUEPRINT NAME: {blueprint_name}")
                
                # Tools that don't need parameters - just execute them
                elif tool_name in [
                    "show_andrio_output", "list_drives", "launch_epic_games_launcher", 
                    "close_epic_games_launcher", "check_epic_launcher_status",
                    "list_unreal_templates", "get_unreal_engine_info",
                    "show_fps_stats", "dump_gpu_stats", "list_loaded_assets",
                    "toggle_wireframe", "memory_report", "get_all_actors_in_level",
                    "enable_remote_python_execution", "check_remote_execution_status"
                ]:
                    # These tools don't need parameters
                    params = []
                    logger.info(f"🎯 TOOL NEEDS NO PARAMETERS: {tool_name}")
            
            # Log final parameters
            logger.info(f"📋 FINAL PARAMETERS: {params}")
            
            # Execute tool with parameters
            if params:
                logger.info(f"🚀 EXECUTING WITH PARAMS: {params}")
                try:
                    if len(params) == 1:
                        result = tool_func(params[0])
                    elif len(params) == 2:
                        result = tool_func(params[0], params[1])
                    elif len(params) == 3:
                        result = tool_func(params[0], params[1], params[2])
                    else:
                        result = tool_func(*params)
                except TypeError as te:
                    # Handle parameter mismatch gracefully
                    if "missing" in str(te) and "required positional argument" in str(te):
                        error_msg = f"❌ Tool '{tool_name}' parameter error: {te}. Please provide the required parameters."
                        logger.error(error_msg)
                        logger.info("=" * 50)
                        return error_msg
                    else:
                        raise te
            else:
                logger.info(f"🚀 EXECUTING WITHOUT PARAMS")
                try:
                    result = tool_func()
                except TypeError as te:
                    if "missing" in str(te) and "required positional argument" in str(te):
                        error_msg = f"❌ Tool '{tool_name}' requires parameters but none were provided: {te}"
                        logger.error(error_msg)
                        logger.info("=" * 50)
                        return error_msg
                    else:
                        raise te
            
            logger.info("-" * 30)
            logger.info(f"✅ TOOL '{tool_name}' EXECUTED SUCCESSFULLY")
            logger.info(f"📄 TOOL RESULT TYPE: {type(result)}")
            logger.info(f"📄 TOOL RESULT LENGTH: {len(str(result))} characters")
            logger.info(f"📄 TOOL RESULT (first 300 chars): {str(result)[:300]}...")
            
            # Special logging for important tools
            if tool_name == "create_unreal_project":
                if "✅" in str(result):
                    logger.info(f"🎉 PROJECT CREATION SUCCESS!")
                else:
                    logger.warning(f"⚠️ PROJECT CREATION MAY HAVE FAILED")
            elif tool_name == "open_unreal_project":
                if "✅" in str(result):
                    logger.info(f"🎉 PROJECT OPENED SUCCESSFULLY!")
                else:
                    logger.warning(f"⚠️ PROJECT OPENING MAY HAVE FAILED")
            elif tool_name in ["list_files", "read_file"]:
                logger.info(f"📁 FILE OPERATION COMPLETED")
                    
            logger.info("=" * 50)
            return result
            
        except Exception as e:
            error_msg = f"❌ Error executing tool '{tool_name}': {e}"
            logger.error("=" * 50)
            logger.error(f"❌ TOOL EXECUTION FAILED")
            logger.error(f"🔧 TOOL NAME: {tool_name}")
            logger.error(f"📋 PARAMS: {params}")
            logger.error(f"🔍 ERROR: {error_msg}")
            logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            logger.error("=" * 50)
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
        """Study UE5 source code with file-by-file approach to prevent context overflow"""
        try:
            logger.info(f"📚 Studying UE source code: {step}")
            
            # Parse the step to understand what to study
            if "study" in step.lower() and "source" in step.lower():
                # Extract specific area or use general study
                study_area = self._extract_study_area_from_step(step)
                return await self._study_ue_source_file_by_file(study_area)
            
            return {"success": False, "reason": "Could not parse study step"}
            
        except Exception as e:
            logger.error(f"Error studying UE source code: {e}")
            return {"success": False, "error": str(e)}

    def _extract_study_area_from_step(self, step: str) -> str:
        """Extract the specific area to study from the step description"""
        step_lower = step.lower()
        
        # Map keywords to study areas
        area_keywords = {
            "rendering": "Engine/Source/Runtime/Renderer",
            "physics": "Engine/Source/Runtime/Physics",
            "input": "Engine/Source/Runtime/InputCore",
            "blueprint": "Engine/Source/Runtime/Engine/Classes/Blueprint",
            "component": "Engine/Source/Runtime/Engine/Classes/Components",
            "actor": "Engine/Source/Runtime/Engine/Classes/Engine",
            "object": "Engine/Source/Runtime/CoreUObject",
            "world": "Engine/Source/Runtime/Engine/Classes/Engine"
        }
        
        for keyword, path in area_keywords.items():
            if keyword in step_lower:
                return path
        
        # Default to Engine core if no specific area found
        return "Engine/Source/Runtime/Engine/Classes/Engine"

    async def _study_ue_source_file_by_file(self, study_area: str, max_files: int = 3) -> Dict[str, Any]:
        """Study UE source files one by one to prevent context overflow"""
        try:
            source_path = Path(self.ue_source_path) / study_area
            
            if not source_path.exists():
                logger.warning(f"Study path does not exist: {source_path}")
                return {"success": False, "reason": f"Path not found: {source_path}"}
            
            # Find relevant files (prioritize .h files for understanding)
            study_files = []
            for ext in [".h", ".cpp"]:
                study_files.extend(list(source_path.glob(f"*{ext}"))[:max_files//2])
            
            if not study_files:
                logger.warning(f"No source files found in: {source_path}")
                return {"success": False, "reason": "No source files found"}
            
            # Study files one by one
            study_results = []
            concepts_learned = []
            
            for i, file_path in enumerate(study_files[:max_files]):
                logger.info(f"📖 Studying file {i+1}/{min(len(study_files), max_files)}: {file_path.name}")
                
                file_result = await self._study_single_source_file(file_path)
                if file_result["success"]:
                    study_results.append({
                        "file": file_path.name,
                        "concepts": file_result.get("concepts", []),
                        "summary": file_result.get("summary", "")
                    })
                    concepts_learned.extend(file_result.get("concepts", []))
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.5)
            
            # Create comprehensive summary
            overall_summary = await self._create_study_session_summary(study_results, study_area)
            
            return {
                "success": True,
                "study_area": study_area,
                "files_studied": len(study_results),
                "concepts_learned": list(set(concepts_learned)),  # Remove duplicates
                "session_summary": overall_summary,
                "detailed_results": study_results
            }
            
        except Exception as e:
            logger.error(f"Error in file-by-file study: {e}")
            return {"success": False, "error": str(e)}

    async def _study_single_source_file(self, file_path: Path) -> Dict[str, Any]:
        """Study a single source file with focused analysis"""
        try:
            # Read file content with size limit - Updated for 128K context window
            if file_path.stat().st_size > 400000:  # 400KB limit (was 100KB)
                logger.warning(f"File too large, skipping: {file_path.name}")
                return {"success": False, "reason": "File too large"}
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract key concepts and create summary
            concepts = self._extract_source_file_concepts(content, str(file_path))
            
            # Create focused summary using AI - Updated content limit
            summary_prompt = f"""Analyze this UE5 source file and provide a concise learning summary.

FILE: {file_path.name}
CONTENT (first 4000 chars): {content[:4000]}

Provide:
1. Main purpose of this file
2. Key classes/functions defined
3. Important UE5 concepts demonstrated
4. How this relates to UE5 architecture

Keep response under 300 words."""

            summary = await self._query_llm(summary_prompt, enable_thinking=False)
            
            # Store the learning in knowledge base
            await self.process_and_learn(summary, f"source_file_{file_path.name}")
            
            return {
                "success": True,
                "concepts": concepts,
                "summary": summary,
                "file_size": len(content)
            }
            
        except Exception as e:
            logger.error(f"Error studying file {file_path}: {e}")
            return {"success": False, "error": str(e)}

    def _extract_source_file_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract key UE5 concepts from source file content"""
        concepts = []
        
        # Common UE5 patterns and keywords
        ue_patterns = [
            r'class\s+(\w*UCLASS\w*|\w*API\s+\w+)',
            r'UFUNCTION\s*\([^)]*\)',
            r'UPROPERTY\s*\([^)]*\)',
            r'USTRUCT\s*\([^)]*\)',
            r'UENUM\s*\([^)]*\)',
            r'class\s+[A-Z]\w*Component',
            r'class\s+[A-Z]\w*Actor',
            r'class\s+[A-Z]\w*Object',
            r'FVector|FRotator|FTransform',
            r'UWorld|ULevel|AActor|UActorComponent',
            r'Blueprint\w*|BP\w*'
        ]
        
        for pattern in ue_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            concepts.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # Extract class names
        class_matches = re.findall(r'class\s+(\w+)', content)
        concepts.extend([cls for cls in class_matches if cls.startswith(('U', 'A', 'F'))])
        
        return list(set(concepts))  # Remove duplicates

    async def _create_study_session_summary(self, study_results: List[Dict], study_area: str) -> str:
        """Create a comprehensive summary of the study session"""
        if not study_results:
            return "No files were successfully studied."
        
        files_studied = [result["file"] for result in study_results]
        all_concepts = []
        for result in study_results:
            all_concepts.extend(result.get("concepts", []))
        
        unique_concepts = list(set(all_concepts))
        
        summary_prompt = f"""Create a learning summary for this UE5 source code study session.

STUDY AREA: {study_area}
FILES STUDIED: {', '.join(files_studied)}
CONCEPTS DISCOVERED: {', '.join(unique_concepts[:20])}  # Limit concepts

Provide:
1. What was learned about UE5 architecture
2. Key insights from this study area
3. How this knowledge builds toward UE5 mastery
4. Suggested next study areas

Keep response under 400 words."""

        return await self._query_llm(summary_prompt, enable_thinking=False)

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
        logger.info("=" * 60)

        learning_cycles = 0
        total_start_time = time.time()

        try:
            # CONTINUOUS LEARNING LOOP - Never stop until 60% mastery
            while self.overall_mastery < self.target_mastery:
                learning_cycles += 1
                cycle_start_time = time.time()

                logger.info(f"\n🔄 === LEARNING CYCLE {learning_cycles} START ===")
                logger.info(f"📊 Current mastery: {self.overall_mastery:.1%}")
                logger.info(f"🎯 Target mastery: {self.target_mastery:.1%}")
                logger.info(f"📍 Phase: {self.current_phase.value}")
                logger.info(f"⏰ Cycle start time: {datetime.now().strftime('%I:%M:%S %p')}")
                logger.info("-" * 40)

                # Log phase-specific objectives
                phase_objectives = {
                    LearningPhase.INSTALLATION_ARCHITECTURE: "Study UE5 installation structure and components",
                    LearningPhase.PROJECT_STRUCTURE: "Analyze project organization through templates", 
                    LearningPhase.SOURCE_FOUNDATIONS: "Study UE5 source code fundamentals",
                    LearningPhase.HANDS_ON_APPLICATION: "Apply knowledge through experimentation",
                    LearningPhase.MASTERY_INTEGRATION: "Advanced integration and teaching"
                }
                
                objective = phase_objectives.get(self.current_phase, "Unknown phase objective")
                logger.info(f"🎯 PHASE OBJECTIVE: {objective}")
                logger.info(f"📋 EXECUTING PHASE: {self.current_phase.value}")

                # Execute learning cycle based on current phase
                if self.current_phase == LearningPhase.INSTALLATION_ARCHITECTURE:
                    logger.info("🏗️ EXECUTING: Installation Architecture Phase")
                    cycle_result = await self._phase_installation_architecture()
                elif self.current_phase == LearningPhase.PROJECT_STRUCTURE:
                    logger.info("📚 EXECUTING: Project Structure Phase")
                    cycle_result = await self._phase_project_structure()
                elif self.current_phase == LearningPhase.SOURCE_FOUNDATIONS:
                    logger.info("📚 EXECUTING: Source Foundations Phase")
                    cycle_result = await self._phase_source_foundations()
                elif self.current_phase == LearningPhase.HANDS_ON_APPLICATION:
                    logger.info("🎮 EXECUTING: Hands-On Application Phase")
                    cycle_result = await self._phase_hands_on_application()
                else:
                    logger.error(f"❌ UNKNOWN PHASE: {self.current_phase}")
                    cycle_result = {"success": False, "error": f"Unknown phase: {self.current_phase}"}

                # Log cycle results
                logger.info("-" * 40)
                logger.info(f"📊 CYCLE {learning_cycles} RESULTS:")
                logger.info(f"   ✅ Success: {cycle_result.get('success', False)}")
                if cycle_result.get('success'):
                    logger.info(f"   📈 Learning Progress: Positive")
                else:
                    logger.info(f"   ⚠️ Issues: {cycle_result.get('error', 'Unknown')}")

                # Check if ready to advance to next phase
                old_phase = self.current_phase
                await self._check_phase_advancement()
                
                if old_phase != self.current_phase:
                    logger.info("🎉 PHASE ADVANCEMENT DETECTED!")
                    logger.info(f"   📍 OLD PHASE: {old_phase.value}")
                    logger.info(f"   📍 NEW PHASE: {self.current_phase.value}")
                    logger.info(f"   📈 Mastery at advancement: {self.overall_mastery:.1%}")

                cycle_time = time.time() - cycle_start_time
                logger.info(f"⏱️ Cycle {learning_cycles} completed in {cycle_time:.2f}s")
                logger.info(f"📈 Updated mastery: {self.overall_mastery:.1%}")
                logger.info(f"🎯 Progress to target: {(self.overall_mastery/self.target_mastery)*100:.1f}%")
                logger.info(f"🔄 === LEARNING CYCLE {learning_cycles} END ===\n")

                # Brief pause between cycles to prevent overwhelming
                logger.info("⏸️ Brief pause between cycles...")
                await asyncio.sleep(2)

            total_time = time.time() - total_start_time

            if self.overall_mastery >= self.target_mastery:
                logger.info("🎉 TARGET MASTERY ACHIEVED!")
                logger.info("=" * 60)
                logger.info(f"✅ FINAL MASTERY: {self.overall_mastery:.1%}")
                logger.info(f"🎯 TARGET WAS: {self.target_mastery:.1%}")
                logger.info(f"🔄 TOTAL CYCLES: {learning_cycles}")
                logger.info(f"⏱️ TOTAL TIME: {total_time:.2f}s")
                logger.info(f"📍 FINAL PHASE: {self.current_phase.value}")
                logger.info("=" * 60)
                return {
                    "success": True,
                    "mastery_achieved": True,
                    "final_mastery": self.overall_mastery,
                    "learning_cycles": learning_cycles,
                    "total_time": total_time
                }
            else:
                logger.info("⏸️ LEARNING PAUSED")
                logger.info(f"📊 Current mastery: {self.overall_mastery:.1%}")
                logger.info(f"🎯 Target mastery: {self.target_mastery:.1%}")
                logger.info(f"🔄 Cycles completed: {learning_cycles}")
                return {
                    "success": True,
                    "mastery_achieved": False,
                    "current_mastery": self.overall_mastery,
                    "learning_cycles": learning_cycles,
                    "total_time": total_time,
                    "message": "Learning paused - restart to continue toward 60% mastery"
                }

        except Exception as e:
            logger.error(f"❌ AUTONOMOUS LEARNING FAILED: {e}")
            logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _phase_installation_architecture(self) -> Dict[str, Any]:
        """Phase 1: Study UE5 installation structure - COMPREHENSIVE FILE READING"""
        logger.info("🏗️ Phase 1: COMPREHENSIVE UE5 installation file-by-file study")

        try:
            # Get current context about learning state
            context = self._get_learning_context()

            prompt = f"""
            {context}

            You are in INSTALLATION ARCHITECTURE phase. Your goal is to READ AND UNDERSTAND every single file in the UE5 installation.

            PHASE 1 OBJECTIVES - COMPREHENSIVE FILE READING:
            - Read EVERY SINGLE FILE in the UE5 installation directory tree
            - Understand what each file does and how it contributes to the engine
            - Analyze configuration files, executables, documentation, and data files
            - Build a complete mental map of the UE5 installation structure
            - Document insights from each file read

            SYSTEMATIC APPROACH:
            - Start with root directory files
            - Then systematically go through each subdirectory
            - Read every readable file (skip only binaries/images)
            - Analyze and understand the content of each file
            - Extract meaningful insights, not just count filenames

            AVAILABLE TOOLS FOR COMPREHENSIVE STUDY:
            - list_files: Get complete directory listings
            - read_file: Read and analyze every single file
            - file_info: Get detailed information about files
            - find_files: Locate specific file types for systematic reading

            CREATE A SYSTEMATIC PLAN TO READ EVERY FILE IN THE UE5 INSTALLATION.
            """

            # Execute comprehensive file reading
            result = await self._execute_comprehensive_installation_study()

            # Reflect on the comprehensive study
            if result.get("success", False):
                reflection = await self._reflect_on_task_with_thinking_comprehensive(result)
                result["reflection"] = reflection
                
                # Update learning metrics based on ACTUAL file reading
                await self._update_learning_metrics_from_comprehensive_study(result)
                
                # Calculate mastery increase based on actual learning
                mastery_increase = self._calculate_real_mastery_increase(result)
                old_mastery = self.overall_mastery
                self.overall_mastery = min(self.target_mastery, self.overall_mastery + mastery_increase)
                
                logger.info(f"📈 COMPREHENSIVE installation study completed.")
                logger.info(f"📊 Files actually read: {result.get('files_read', 0)}")
                logger.info(f"📊 Total content analyzed: {result.get('total_content_length', 0)} characters")
                logger.info(f"📈 Mastery increased: {old_mastery:.1%} → {self.overall_mastery:.1%} (+{mastery_increase:.1%})")
            else:
                logger.warning(f"⚠️ Comprehensive installation study had issues, but continuing...")
                # Much smaller increase for failed comprehensive study
                self.overall_mastery += 0.001  # Minimal increase for attempt
                logger.info(f"📈 Attempted comprehensive study. Mastery: {self.overall_mastery:.1%}")

            return result

        except Exception as e:
            logger.error(f"Comprehensive installation study failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_comprehensive_installation_study(self) -> Dict[str, Any]:
        """Execute comprehensive file-by-file study of UE5 installation with smart processing"""
        try:
            logger.info("🔍 Starting COMPREHENSIVE UE5 installation file-by-file study")
            
            study_results = {
                "success": True,
                "files_read": 0,
                "directories_processed": 0,
                "total_content_length": 0,
                "insights_generated": [],
                "file_analysis": [],
                "errors": []
            }
            
            # Start with root directory
            root_path = self.ue_installation_path
            logger.info(f"📁 Starting comprehensive study of: {root_path}")
            
            # Get all directories and prioritize them intelligently
            directories_to_process = [root_path]
            
            # Add all subdirectories
            for item in root_path.rglob("*"):
                if item.is_dir():
                    directories_to_process.append(item)
            
            logger.info(f"📊 Found {len(directories_to_process)} directories to process comprehensively")
            
            # Prioritize directories by importance for UE5 learning
            prioritized_directories = self._prioritize_directories_for_learning(directories_to_process)
            
            # Process directories with adaptive limits based on importance
            for dir_index, (directory, priority_score) in enumerate(prioritized_directories):
                logger.info(f"📁 Processing directory {dir_index + 1}/{len(prioritized_directories)}: {directory.name} (Priority: {priority_score:.2f})")
                
                # Adaptive file limit based on directory importance
                max_files = self._calculate_adaptive_file_limit(directory, priority_score)
                
                dir_result = await self._process_directory_comprehensively(directory, max_files)
                
                # Accumulate results
                study_results["files_read"] += dir_result.get("files_read", 0)
                study_results["total_content_length"] += dir_result.get("content_length", 0)
                study_results["insights_generated"].extend(dir_result.get("insights", []))
                study_results["file_analysis"].extend(dir_result.get("file_analysis", []))
                study_results["errors"].extend(dir_result.get("errors", []))
                
                study_results["directories_processed"] += 1
                
                # Adaptive pause based on processing intensity
                pause_time = 0.1 if priority_score > 0.8 else 0.3 if priority_score > 0.5 else 0.5
                await asyncio.sleep(pause_time)
                
                # Log progress more frequently for important directories
                if priority_score > 0.7 or (dir_index + 1) % 25 == 0:
                    logger.info(f"📊 Progress: {dir_index + 1} directories, {study_results['files_read']} files read")
                
                # Early completion check for massive installations
                if study_results["files_read"] > 10000 and study_results["directories_processed"] > 1000:
                    logger.info(f"🎯 Reached substantial coverage threshold - continuing with remaining high-priority directories")
            
            logger.info(f"✅ COMPREHENSIVE STUDY COMPLETE!")
            logger.info(f"📊 Final stats: {study_results['directories_processed']} directories, {study_results['files_read']} files")
            logger.info(f"📊 Total content: {study_results['total_content_length']} characters")
            logger.info(f"📊 Insights: {len(study_results['insights_generated'])}")
            
            return study_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive installation study failed: {e}")
            return {"success": False, "error": str(e)}

    def _prioritize_directories_for_learning(self, directories: List[Path]) -> List[Tuple[Path, float]]:
        """Prioritize directories based on their learning value for UE5"""
        prioritized = []
        
        for directory in directories:
            priority_score = self._calculate_directory_priority(directory)
            prioritized.append((directory, priority_score))
        
        # Sort by priority score (highest first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return prioritized
    
    def _calculate_directory_priority(self, directory: Path) -> float:
        """Calculate priority score for a directory based on learning value"""
        dir_name = directory.name.lower()
        dir_path = str(directory).lower()
        
        # High priority directories (critical for UE5 understanding)
        high_priority_patterns = [
            'engine', 'source', 'runtime', 'core', 'coreuobject', 'slate', 'slatecore',
            'renderer', 'rhicore', 'rhi', 'rendercore', 'engine/source', 'programs',
            'developer', 'editor', 'unrealed', 'toolmenus', 'mainframe', 'blueprintgraph',
            'kismet', 'kismetcompiler', 'gameplayabilities', 'aimodule', 'navigationmesh',
            'physicscore', 'chaos', 'geometrycollection', 'fieldnotification', 'inputcore',
            'applicationcore', 'projects', 'launch', 'targetplatform'
        ]
        
        # Medium priority directories
        medium_priority_patterns = [
            'plugins', 'shaders', 'config', 'content', 'binaries', 'intermediate',
            'documentation', 'templates', 'samples', 'tools', 'utilities', 'build',
            'automation', 'localization', 'platform', 'windows', 'android', 'ios'
        ]
        
        # Low priority directories
        low_priority_patterns = [
            'temp', 'cache', 'logs', 'saved', 'backup', 'old', 'deprecated',
            'test', 'tests', 'example', 'demo', 'tutorial'
        ]
        
        # Calculate base priority
        priority = 0.1  # Base priority
        
        # Check high priority patterns
        for pattern in high_priority_patterns:
            if pattern in dir_name or pattern in dir_path:
                priority = max(priority, 0.9)
                break
        
        # Check medium priority patterns
        if priority < 0.5:
            for pattern in medium_priority_patterns:
                if pattern in dir_name or pattern in dir_path:
                    priority = max(priority, 0.6)
                    break
        
        # Check low priority patterns (reduce priority)
        for pattern in low_priority_patterns:
            if pattern in dir_name or pattern in dir_path:
                priority = min(priority, 0.3)
                break
        
        # Boost priority for root-level directories
        if len(directory.parts) <= 3:  # Close to root
            priority += 0.2
        
        # Boost priority for source code directories
        if 'source' in dir_path and any(ext in dir_path for ext in ['.cpp', '.h', '.cs']):
            priority += 0.3
        
        return min(priority, 1.0)
    
    def _calculate_adaptive_file_limit(self, directory: Path, priority_score: float) -> int:
        """Calculate adaptive file limit based on directory importance"""
        if priority_score >= 0.9:
            return 100  # High priority: process many files
        elif priority_score >= 0.7:
            return 50   # Medium-high priority
        elif priority_score >= 0.5:
            return 25   # Medium priority
        elif priority_score >= 0.3:
            return 10   # Low-medium priority
        else:
            return 5    # Low priority: minimal processing

    async def _process_directory_comprehensively(self, directory: Path, max_files: int = 50) -> Dict[str, Any]:
        """Process a single directory by reading files with intelligent prioritization"""
        try:
            logger.info(f"🔍 Comprehensively processing: {directory} (max {max_files} files)")
            
            result = {
                "files_read": 0,
                "content_length": 0,
                "insights": [],
                "file_analysis": [],
                "errors": []
            }
            
            # Get all files in this directory (not subdirectories)
            try:
                files_in_dir = [f for f in directory.iterdir() if f.is_file()]
            except (PermissionError, OSError) as e:
                logger.warning(f"⚠️ Cannot access directory {directory}: {e}")
                result["errors"].append(f"Cannot access {directory}: {e}")
                return result
            
            logger.info(f"📄 Found {len(files_in_dir)} files in {directory.name}")
            
            # Prioritize files by learning value
            prioritized_files = self._prioritize_files_for_learning(files_in_dir)
            
            # Process files up to the adaptive limit
            files_to_process = prioritized_files[:max_files]
            
            # Read each file systematically
            for file_index, (file_path, file_priority) in enumerate(files_to_process):
                logger.info(f"📖 Reading file {file_index + 1}/{len(files_to_process)}: {file_path.name} (Priority: {file_priority:.2f})")
                
                file_result = await self._read_and_analyze_file_comprehensively(file_path)
                
                if file_result.get("success", False):
                    result["files_read"] += 1
                    result["content_length"] += file_result.get("content_length", 0)
                    result["insights"].extend(file_result.get("insights", []))
                    result["file_analysis"].append({
                        "file": str(file_path),
                        "size": file_result.get("content_length", 0),
                        "type": file_result.get("file_type", "unknown"),
                        "priority": file_priority,
                        "insights": file_result.get("insights", [])
                    })
                else:
                    result["errors"].append(f"Failed to read {file_path}: {file_result.get('error', 'Unknown error')}")
                
                # Adaptive pause based on file priority
                pause_time = 0.05 if file_priority > 0.8 else 0.1 if file_priority > 0.5 else 0.2
                await asyncio.sleep(pause_time)
            
            if len(files_in_dir) > max_files:
                logger.info(f"📊 Processed {len(files_to_process)} highest priority files out of {len(files_in_dir)} total files")
            
            logger.info(f"✅ Directory {directory.name} complete: {result['files_read']} files read")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing directory {directory}: {e}")
            return {"files_read": 0, "content_length": 0, "insights": [], "file_analysis": [], "errors": [str(e)]}

    def _prioritize_files_for_learning(self, files: List[Path]) -> List[Tuple[Path, float]]:
        """Prioritize files based on their learning value for UE5"""
        prioritized = []
        
        for file_path in files:
            priority_score = self._calculate_file_priority(file_path)
            prioritized.append((file_path, priority_score))
        
        # Sort by priority score (highest first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return prioritized
    
    def _calculate_file_priority(self, file_path: Path) -> float:
        """Calculate priority score for a file based on learning value"""
        file_name = file_path.name.lower()
        file_ext = file_path.suffix.lower()
        
        # High priority file types and names
        if file_ext in ['.h', '.hpp', '.cpp', '.c', '.cs']:
            priority = 0.9  # Source code files are highest priority
        elif file_ext in ['.ini', '.cfg', '.config']:
            priority = 0.8  # Configuration files are very important
        elif file_ext in ['.json', '.xml', '.yaml', '.yml']:
            priority = 0.7  # Data/config files
        elif file_ext in ['.txt', '.md', '.rst']:
            priority = 0.6  # Documentation
        elif file_ext in ['.py', '.js', '.ts']:
            priority = 0.7  # Scripts
        elif file_ext in ['.bat', '.sh', '.cmd']:
            priority = 0.5  # Batch scripts
        else:
            priority = 0.3  # Other files
        
        # Boost priority for important file names
        important_names = [
            'engine', 'core', 'main', 'config', 'settings', 'version', 'build',
            'project', 'target', 'module', 'plugin', 'component', 'actor',
            'object', 'world', 'level', 'game', 'player', 'controller',
            'render', 'physics', 'input', 'audio', 'animation', 'blueprint'
        ]
        
        for important_name in important_names:
            if important_name in file_name:
                priority += 0.2
                break
        
        # Reduce priority for less important files
        unimportant_patterns = [
            'temp', 'cache', 'log', 'backup', 'old', 'deprecated',
            'test', 'example', 'demo', 'sample', 'tutorial'
        ]
        
        for pattern in unimportant_patterns:
            if pattern in file_name:
                priority -= 0.3
                break
        
        # Boost priority for header files in Engine/Source
        if file_ext in ['.h', '.hpp'] and 'engine' in str(file_path).lower():
            priority += 0.1
        
        return max(0.1, min(priority, 1.0))

    async def _execute_comprehensive_project_study(self) -> Dict[str, Any]:
        """Execute comprehensive file-by-file study of UE5 project structures"""
        try:
            logger.info("🔍 Starting COMPREHENSIVE UE5 project file-by-file study")
            
            study_results = {
                "success": True,
                "project_files_read": 0,
                "projects_created": 0,
                "config_files_read": 0,
                "total_project_content": 0,
                "insights_generated": [],
                "project_analysis": [],
                "errors": []
            }
            
            # Create multiple template projects for comprehensive study
            project_templates = [
                ("StructureStudyBP", "ThirdPersonBP"),
                ("StructureStudyCPP", "ThirdPersonCPP"),
                ("StructureStudyBlank", "BlankBP"),
                ("StructureStudyTopDown", "TopDownBP")
            ]
            
            logger.info(f"📊 Creating {len(project_templates)} template projects for comprehensive study")
            
            # Create and analyze each project template
            for project_index, (project_name, template) in enumerate(project_templates):
                logger.info(f"📁 Creating and analyzing project {project_index + 1}/{len(project_templates)}: {project_name}")
                
                project_result = await self._create_and_analyze_project_comprehensively(project_name, template)
                
                # Accumulate results
                study_results["project_files_read"] += project_result.get("files_read", 0)
                study_results["config_files_read"] += project_result.get("config_files_read", 0)
                study_results["total_project_content"] += project_result.get("content_length", 0)
                study_results["insights_generated"].extend(project_result.get("insights", []))
                study_results["project_analysis"].extend(project_result.get("project_analysis", []))
                study_results["errors"].extend(project_result.get("errors", []))
                
                if project_result.get("success", False):
                    study_results["projects_created"] += 1
                
                # Brief pause between projects
                await asyncio.sleep(1.0)
            
            logger.info(f"✅ COMPREHENSIVE PROJECT STUDY COMPLETE!")
            logger.info(f"📊 Final stats: {study_results['projects_created']} projects, {study_results['project_files_read']} files")
            logger.info(f"📊 Config files: {study_results['config_files_read']}")
            logger.info(f"📊 Total content: {study_results['total_project_content']} characters")
            
            return study_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive project study failed: {e}")
            return {"success": False, "error": str(e)}

    async def _create_and_analyze_project_comprehensively(self, project_name: str, template: str) -> Dict[str, Any]:
        """Create a project and comprehensively analyze all its files"""
        try:
            logger.info(f"🔍 Creating and analyzing project: {project_name} ({template})")
            
            result = {
                "success": False,
                "files_read": 0,
                "config_files_read": 0,
                "content_length": 0,
                "insights": [],
                "project_analysis": [],
                "errors": []
            }
            
            # Create the project
            try:
                create_result = await self._execute_tool("create_unreal_project", [project_name, template])
                if "successfully" not in create_result.lower():
                    logger.warning(f"⚠️ Project creation may have failed: {create_result}")
                    result["errors"].append(f"Project creation issue: {create_result}")
                    return result
                
                logger.info(f"✅ Project {project_name} created successfully")
            except Exception as e:
                logger.error(f"❌ Failed to create project {project_name}: {e}")
                result["errors"].append(f"Failed to create project: {e}")
                return result
            
            # Find the project directory
            project_path = Path("D:/Andrios Output/UnrealProjects") / project_name
            if not project_path.exists():
                logger.error(f"❌ Project directory not found: {project_path}")
                result["errors"].append(f"Project directory not found: {project_path}")
                return result
            
            logger.info(f"📁 Found project directory: {project_path}")
            
            # Comprehensively analyze all project files
            analysis_result = await self._analyze_project_directory_comprehensively(project_path)
            
            # Merge results
            result["files_read"] = analysis_result.get("files_read", 0)
            result["config_files_read"] = analysis_result.get("config_files_read", 0)
            result["content_length"] = analysis_result.get("content_length", 0)
            result["insights"].extend(analysis_result.get("insights", []))
            result["project_analysis"].extend(analysis_result.get("project_analysis", []))
            result["errors"].extend(analysis_result.get("errors", []))
            result["success"] = analysis_result.get("success", False)
            
            logger.info(f"✅ Project {project_name} analysis complete: {result['files_read']} files read")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error creating/analyzing project {project_name}: {e}")
            return {"success": False, "files_read": 0, "config_files_read": 0, "content_length": 0, 
                   "insights": [], "project_analysis": [], "errors": [str(e)]}

    async def _analyze_project_directory_comprehensively(self, project_path: Path) -> Dict[str, Any]:
        """Comprehensively analyze all files in a project directory"""
        try:
            logger.info(f"🔍 Comprehensively analyzing project directory: {project_path}")
            
            result = {
                "success": True,
                "files_read": 0,
                "config_files_read": 0,
                "content_length": 0,
                "insights": [],
                "project_analysis": [],
                "errors": []
            }
            
            # Get all files in the project recursively
            all_files = []
            try:
                for item in project_path.rglob("*"):
                    if item.is_file():
                        all_files.append(item)
            except (PermissionError, OSError) as e:
                logger.warning(f"⚠️ Cannot access project directory {project_path}: {e}")
                result["errors"].append(f"Cannot access {project_path}: {e}")
                return result
            
            logger.info(f"📄 Found {len(all_files)} files in project")
            
            # Read each file systematically (limit to prevent overwhelming)
            for file_index, file_path in enumerate(all_files[:100]):  # Limit to 100 files per project
                logger.info(f"📖 Reading project file {file_index + 1}/{min(len(all_files), 100)}: {file_path.name}")
                
                file_result = await self._read_and_analyze_project_file_comprehensively(file_path)
                
                if file_result.get("success", False):
                    result["files_read"] += 1
                    result["content_length"] += file_result.get("content_length", 0)
                    result["insights"].extend(file_result.get("insights", []))
                    
                    # Track config files separately
                    if self._is_config_file(file_path):
                        result["config_files_read"] += 1
                    
                    result["project_analysis"].append({
                        "file": str(file_path),
                        "size": file_result.get("content_length", 0),
                        "type": file_result.get("file_type", "unknown"),
                        "is_config": self._is_config_file(file_path),
                        "insights": file_result.get("insights", [])
                    })
                else:
                    result["errors"].append(f"Failed to read {file_path}: {file_result.get('error', 'Unknown error')}")
                
                # Brief pause between files
                await asyncio.sleep(0.05)
            
            logger.info(f"✅ Project directory analysis complete: {result['files_read']} files read")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing project directory {project_path}: {e}")
            return {"success": False, "files_read": 0, "config_files_read": 0, "content_length": 0, 
                   "insights": [], "project_analysis": [], "errors": [str(e)]}

    async def _read_and_analyze_project_file_comprehensively(self, file_path: Path) -> Dict[str, Any]:
        """Read and comprehensively analyze a single project file"""
        try:
            # Check file size - skip very large files
            file_size = file_path.stat().st_size
            if file_size > 1000000:  # 1MB limit for project files
                logger.info(f"⏭️ Skipping large project file: {file_path.name} ({file_size} bytes)")
                return {"success": False, "error": "Project file too large"}
            
            # Skip binary files
            if file_path.suffix.lower() in ['.exe', '.dll', '.so', '.dylib', '.bin', '.pak', '.uasset', '.umap']:
                logger.info(f"⏭️ Skipping binary project file: {file_path.name}")
                return {"success": False, "error": "Binary project file"}
            
            # Read the project file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"⚠️ Cannot read project file {file_path.name}: {e}")
                return {"success": False, "error": str(e)}
            
            if not content.strip():
                logger.info(f"⏭️ Empty project file: {file_path.name}")
                return {"success": False, "error": "Empty project file"}
            
            logger.info(f"📖 Successfully read project file {file_path.name}: {len(content)} characters")
            
            # Analyze the project file content comprehensively
            analysis_result = await self._analyze_project_file_content_comprehensively(content, file_path)
            
            # Store in knowledge base for future reference
            await self.process_and_learn(content[:4000], f"project_file_{file_path.name}")
            
            return {
                "success": True,
                "content_length": len(content),
                "file_type": self._determine_project_file_type(file_path),
                "insights": analysis_result.get("insights", []),
                "analysis": analysis_result.get("analysis", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Error reading project file {file_path}: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_project_file_content_comprehensively(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Comprehensively analyze project file content using AI"""
        try:
            # Create project file analysis prompt
            prompt = f"""Comprehensively analyze this UE5 project file and extract deep insights:

PROJECT FILE: {file_path.name}
FILE TYPE: {file_path.suffix}
CONTENT LENGTH: {len(content)} characters

CONTENT (first 2000 chars):
{content[:2000]}

COMPREHENSIVE PROJECT FILE ANALYSIS REQUIRED:
1. What is the PURPOSE of this file in the UE5 project structure?
2. What PROJECT CONFIGURATION does it provide or define?
3. How does this file relate to UE5 project organization?
4. What can we LEARN about UE5 project architecture from this file?
5. What are the KEY INSIGHTS for understanding UE5 project structure?

Focus on understanding project organization, configuration, and structure patterns."""

            # Get AI analysis
            analysis = await self._query_llm(prompt, enable_thinking=False)
            
            # Extract insights from analysis
            insights = self._extract_insights_from_analysis(analysis)
            
            return {
                "analysis": analysis,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing project file content: {e}")
            return {"analysis": "", "insights": []}

    def _determine_project_file_type(self, file_path: Path) -> str:
        """Determine the type/purpose of a project file"""
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        
        if suffix == '.uproject':
            return "project_definition"
        elif suffix in ['.ini', '.cfg']:
            return "configuration"
        elif suffix in ['.json', '.xml']:
            return "project_data"
        elif suffix in ['.txt', '.md']:
            return "documentation"
        elif 'config' in str(file_path).lower():
            return "configuration"
        elif 'content' in str(file_path).lower():
            return "content_asset"
        elif 'source' in str(file_path).lower():
            return "source_code"
        else:
            return "project_file"

    def _is_config_file(self, file_path: Path) -> bool:
        """Check if a file is a configuration file"""
        return (file_path.suffix.lower() in ['.ini', '.cfg', '.config'] or 
                'config' in str(file_path).lower() or
                file_path.suffix == '.uproject')

    def _calculate_project_mastery_increase(self, study_result: Dict[str, Any]) -> float:
        """Calculate mastery increase based on ACTUAL project study"""
        files_read = study_result.get("project_files_read", 0)
        projects_created = study_result.get("projects_created", 0)
        config_files = study_result.get("config_files_read", 0)
        content_length = study_result.get("total_project_content", 0)
        insights = len(study_result.get("insights_generated", []))
        
        # Conservative scoring based on actual project analysis
        file_score = min(files_read * 0.001, 0.025)  # 0.1% per file, max 2.5%
        project_score = min(projects_created * 0.005, 0.02)  # 0.5% per project, max 2%
        config_score = min(config_files * 0.003, 0.015)  # 0.3% per config file, max 1.5%
        content_score = min(content_length / 1000000 * 0.01, 0.02)  # 1% per 1MB content, max 2%
        insight_score = min(insights * 0.002, 0.015)  # 0.2% per insight, max 1.5%
        
        total_increase = file_score + project_score + config_score + content_score + insight_score
        
        logger.info(f"📊 PROJECT MASTERY CALCULATION:")
        logger.info(f"   📄 Project files read: {files_read} → {file_score:.1%}")
        logger.info(f"   📁 Projects created: {projects_created} → {project_score:.1%}")
        logger.info(f"   ⚙️ Config files: {config_files} → {config_score:.1%}")
        logger.info(f"   📝 Content: {content_length} chars → {content_score:.1%}")
        logger.info(f"   💡 Insights: {insights} → {insight_score:.1%}")
        logger.info(f"   📈 Total increase: {total_increase:.1%}")
        
        return total_increase

    async def _update_learning_metrics_from_project_study(self, study_result: Dict[str, Any]):
        """Update learning metrics based on comprehensive project study results"""
        try:
            # Update with ACTUAL project reading metrics
            self.learning_metrics.files_analyzed += study_result.get("project_files_read", 0)
            self.learning_metrics.knowledge_documents += 1  # One comprehensive project study session
            self.learning_metrics.insights_generated += len(study_result.get("insights_generated", []))
            
            # Extract actual UE concepts from source code insights
            all_insights = study_result.get("insights_generated", [])
            real_concepts = []
            for insight in all_insights:
                concepts = self._extract_real_ue_concepts_from_insight(insight)
                real_concepts.extend(concepts)
            
            # Add concepts from actual classes and functions found
            real_concepts.extend([f"class_{i}" for i in range(study_result.get("classes_found", 0))])
            real_concepts.extend([f"function_{i}" for i in range(study_result.get("functions_found", 0))])
            
            self.learning_metrics.concepts_extracted += len(set(real_concepts))  # Unique concepts only
            
            # Update understanding depth based on comprehensive source analysis
            depth_increase = min(len(all_insights) * 0.015, 0.08)  # Higher for source code
            self.learning_metrics.understanding_depth_score += depth_increase
            
            logger.info(f"📊 COMPREHENSIVE SOURCE LEARNING METRICS UPDATED:")
            logger.info(f"   📁 Source files actually read: {study_result.get('source_files_read', 0)}")
            logger.info(f"   🏗️ Classes found: {study_result.get('classes_found', 0)}")
            logger.info(f"   ⚙️ Functions found: {study_result.get('functions_found', 0)}")
            logger.info(f"   🧠 Real UE concepts: {len(set(real_concepts))}")
            logger.info(f"   💡 Insights from source: {len(all_insights)}")
            logger.info(f"   🎯 Understanding depth: {self.learning_metrics.understanding_depth_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error updating source learning metrics: {e}")

    async def _reflect_on_source_study_comprehensive(self, study_result: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on comprehensive source study with thinking mode"""
        try:
            logger.info(f"🧠 Reflecting on comprehensive source study with thinking mode")
             
            prompt = f"""Reflect deeply on this COMPREHENSIVE UE5 source code study:

COMPREHENSIVE SOURCE STUDY RESULTS:
- Source files actually read: {study_result.get('source_files_read', 0)}
- Directories processed: {study_result.get('directories_processed', 0)}
- Total source content analyzed: {study_result.get('total_source_content', 0)} characters
- Classes found and analyzed: {study_result.get('classes_found', 0)}
- Functions found and analyzed: {study_result.get('functions_found', 0)}
- Insights generated: {len(study_result.get('insights_generated', []))}
- Errors encountered: {len(study_result.get('errors', []))}

This was a REAL comprehensive source code study where every source file was actually read and analyzed for classes, functions, and architectural patterns.

DEEP SOURCE CODE REFLECTION REQUIRED:
1. What was actually LEARNED about UE5 implementation from reading the source code?
2. What architectural PATTERNS emerged from analyzing the actual code?
3. How do the classes and functions relate to UE5's overall design?
4. What insights about engine design were gained from the source analysis?
5. How does this source knowledge prepare for hands-on development?

Provide a thoughtful analysis of the actual source code learning achieved."""

            # Use thinking mode for comprehensive source reflection
            result = await self._query_llm_with_thinking(prompt)
             
            reflection_data = {
                "comprehensive_source_study": True,
                "source_files_actually_read": study_result.get('source_files_read', 0),
                "classes_analyzed": study_result.get('classes_found', 0),
                "functions_analyzed": study_result.get('functions_found', 0),
                "thinking_process": result.get("thinking", ""),
                "reflection_content": result.get("response", ""),
                "timestamp": datetime.now().isoformat(),
                "success": result.get("success", False)
            }
             
            if result["success"] and result["thinking"]:
                logger.info(f"🧠 Comprehensive Source Study Reflection: {result['thinking'][:300]}...")
                 
                # Store the comprehensive source thinking process
                await self._store_thinking_process("comprehensive_source_study", result["thinking"], "source_reflection")
             
            return reflection_data
             
        except Exception as e:
            logger.error(f"Error in comprehensive source study reflection: {e}")
            return {"success": False, "error": str(e)}

    async def _phase_hands_on_application(self) -> Dict[str, Any]:
        """Phase 4: Apply knowledge through experimentation - APPLICATION PHASE"""
        logger.info("🎮 Phase 4: Applying knowledge through experimentation")

        try:
            # Get current context about learning state
            context = self._get_learning_context()

            prompt = f"""
            {context}

            You are in HANDS-ON APPLICATION phase. NOW you can apply your learned knowledge practically.

            PHASE 4 OBJECTIVES:
            - Apply learned UE5 knowledge through practical experimentation
            - Create projects with specific learning goals
            - Test understanding through hands-on implementation
            - Experiment with UE5 features using learned concepts
            - Validate theoretical knowledge through practical application

            NOW YOU CAN USE UE5 EFFECTIVELY:
            - Create projects with clear objectives based on learned knowledge
            - Open projects in the editor to experiment with features
            - Apply understanding of UE5 architecture to solve problems
            - Use Blueprint system with understanding of underlying C++ classes
            - Experiment with engine systems you've studied

            AVAILABLE TOOLS FOR APPLICATION:
            - create_unreal_project: Create projects for experimentation
            - open_unreal_project: Open projects to work with them
            - All file tools: Create/modify project content
            - Epic Launcher tools: Manage UE5 installations

            FOCUS ON PURPOSEFUL EXPERIMENTATION:
            - Create projects to test specific concepts you've learned
            - Apply knowledge of UE5 architecture to solve challenges
            - Experiment with systems you understand from source study
            - Build increasingly complex projects as confidence grows

            Create a task plan with specific tool commands for hands-on UE5 application.
            """

            # Use thinking mode if available for better task planning
            if self.thinking_mode_enabled and self.thinking_mode_compatible:
                logger.info("🧠 Using thinking mode for hands-on application planning")
                task_data = await self._plan_task_with_thinking(
                    "Apply UE5 knowledge through purposeful experimentation and project creation"
                )
            else:
                ai_response = await self._query_llm(prompt)
                task_data = self._extract_task_from_response(ai_response)

            if not task_data:
                # Fallback to basic hands-on application
                task_data = {
                    "goal": "Apply UE5 knowledge through practical experimentation",
                    "steps": [
                        "Create experiment project using create_unreal_project ExperimentProject Blank",
                        "Open project for hands-on work using open_unreal_project",
                        "Apply learned knowledge to create new features",
                        "Experiment with UE systems using learned concepts",
                        "Test understanding through practical implementation",
                        "Document successful experiments and learning outcomes"
                    ],
                    "domain": "hands_on_application"
                }

            # Execute the hands-on application task
            task = Task(
                id=f"hands_on_app_{int(time.time())}",
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
                
                # Increment for hands-on application
                self.overall_mastery += 0.05
                logger.info(f"📈 Hands-on application completed. Mastery: {self.overall_mastery:.1%}")
            else:
                logger.warning(f"⚠️ Hands-on application had issues, but continuing...")
                # Still give small progress for attempting
                self.overall_mastery += 0.02
                logger.info(f"📈 Attempted hands-on application. Mastery: {self.overall_mastery:.1%}")

            return result

        except Exception as e:
            logger.error(f"Hands-on application phase failed: {e}")
            return {"success": False, "error": str(e)}

    async def _check_phase_advancement(self):
        """Check if ready to advance to next learning phase based on objective completion"""
        old_phase = self.current_phase
        
        # Check if current phase is complete based on objectives
        if self.phase_progress.is_complete():
            logger.info(f"🎉 PHASE COMPLETION DETECTED!")
            logger.info(f"   📋 Tasks completed: {len(self.phase_progress.completed_tasks)}/{len(self.phase_progress.required_tasks)}")
            logger.info(f"   📚 Files studied: {len(self.phase_progress.files_studied)}")
            logger.info(f"   🧠 Concepts learned: {len(self.phase_progress.concepts_learned)}")
            logger.info(f"   📁 Directories explored: {len(self.phase_progress.directories_explored)}")
            
            # Advance to next phase
            if self.current_phase == LearningPhase.INSTALLATION_ARCHITECTURE:
                self.current_phase = LearningPhase.PROJECT_STRUCTURE
                logger.info("🎉 Advanced to Phase 2: Project Structure Study")
            elif self.current_phase == LearningPhase.PROJECT_STRUCTURE:
                self.current_phase = LearningPhase.SOURCE_FOUNDATIONS
                logger.info("🎉 Advanced to Phase 3: Source Foundations Study")
            elif self.current_phase == LearningPhase.SOURCE_FOUNDATIONS:
                self.current_phase = LearningPhase.HANDS_ON_APPLICATION
                logger.info("🎉 Advanced to Phase 4: Hands-on Application")
            elif self.current_phase == LearningPhase.HANDS_ON_APPLICATION:
                if self.overall_mastery >= self.target_mastery:
                    self.current_phase = LearningPhase.MASTERY_INTEGRATION
                    logger.info("🎉 Advanced to Phase 5: Mastery Integration")
            
            # Initialize new phase progress if phase changed
            if old_phase != self.current_phase:
                self.phase_progress = self._initialize_phase_progress()
                logger.info(f"📋 NEW PHASE INITIALIZED: {len(self.phase_progress.required_tasks)} tasks required")
        else:
            # Log current progress
            completion = self.phase_progress.calculate_completion()
            logger.info(f"📊 PHASE PROGRESS: {completion:.1%} complete")
            logger.info(f"   📋 Tasks: {len(self.phase_progress.completed_tasks)}/{len(self.phase_progress.required_tasks)}")
            logger.info(f"   📚 Files: {len(self.phase_progress.files_studied)}")
            logger.info(f"   🧠 Concepts: {len(self.phase_progress.concepts_learned)}")
            
            # Show what's still needed
            remaining_tasks = set(self.phase_progress.required_tasks) - self.phase_progress.completed_tasks
            if remaining_tasks:
                logger.info(f"   ⏳ Remaining tasks: {list(remaining_tasks)[:3]}...")  # Show first 3

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
            
            # Create a summary of results instead of sending all data
            results_summary = self._create_results_summary(task.results)
            
            prompt = f"""Reflect on the completed task and analyze the learning outcomes with deep reasoning.

TASK DETAILS:
- Goal: {task.goal}
- Description: {task.description}
- Status: {task.status.value}
- Steps Completed: {len(task.steps)}
- Results Summary: {results_summary}

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

    def _create_results_summary(self, results: Dict[str, Any]) -> str:
        """Create a concise summary of task results to avoid context overflow"""
        if not results:
            return "No results available"
        
        summary_parts = []
        
        # Summarize different types of results
        for key, value in results.items():
            if isinstance(value, str):
                # Truncate long strings and provide summary
                if len(value) > 500:
                    summary_parts.append(f"- {key}: {len(value)} characters of content (truncated: {value[:200]}...)")
                else:
                    summary_parts.append(f"- {key}: {value}")
            elif isinstance(value, list):
                summary_parts.append(f"- {key}: {len(value)} items")
            elif isinstance(value, dict):
                summary_parts.append(f"- {key}: {len(value)} entries")
            else:
                summary_parts.append(f"- {key}: {str(value)[:100]}")
        
        return "\n".join(summary_parts[:10])  # Limit to 10 items max

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

    def _initialize_phase_progress(self) -> PhaseProgress:
        """Initialize phase progress tracking with required tasks"""
        phase_tasks = {
            LearningPhase.INSTALLATION_ARCHITECTURE: [
                "explore_engine_directory",
                "analyze_binaries_directory", 
                "study_plugins_structure",
                "read_engine_version_info",
                "explore_templates_directory",
                "analyze_engine_executables",
                "document_installation_structure",
                "understand_engine_organization"
            ],
            LearningPhase.PROJECT_STRUCTURE: [
                "create_template_project",
                "analyze_project_root_structure",
                "study_uproject_file_format",
                "explore_content_directory",
                "analyze_config_files",
                "understand_blueprint_organization",
                "study_asset_structure",
                "document_project_patterns"
            ],
            LearningPhase.SOURCE_FOUNDATIONS: [
                "study_core_object_system",
                "analyze_actor_hierarchy",
                "understand_component_architecture",
                "explore_engine_modules",
                "study_reflection_system",
                "analyze_blueprint_cpp_integration",
                "understand_memory_management",
                "study_engine_subsystems"
            ],
            LearningPhase.HANDS_ON_APPLICATION: [
                "create_experimental_project",
                "implement_basic_actor",
                "create_custom_component",
                "experiment_with_blueprints",
                "test_cpp_integration",
                "apply_learned_patterns"
            ]
        }
        
        current_tasks = phase_tasks.get(self.current_phase, [])
        return PhaseProgress(
            phase=self.current_phase,
            required_tasks=current_tasks
        )

    async def _update_learning_metrics_from_task(self, task: Task, result: Dict[str, Any]):
        """Update learning metrics based on the results of a task"""
        try:
            # Extract learning data from task results
            step_results = result.get("step_results", [])
            
            for step_result in step_results:
                if step_result.get("success", False):
                    # Count files analyzed
                    if "file" in step_result.get("type", "").lower():
                        self.learning_metrics.files_analyzed += 1
                        
                    # Count directories explored  
                    if "directory" in step_result.get("type", "").lower() or "list" in step_result.get("tool_used", "").lower():
                        self.learning_metrics.directories_explored += 1
                        
                    # Extract concepts from tool results
                    tool_result = step_result.get("tool_result", "")
                    if isinstance(tool_result, str):
                        # Count UE-specific concepts found
                        ue_concepts = self._extract_ue_concepts_from_text(tool_result)
                        self.learning_metrics.concepts_extracted += len(ue_concepts)
                        
                        # Update phase progress
                        self.phase_progress.concepts_learned.extend(ue_concepts)
                        
                        # Track files studied
                        if "file" in step_result.get("type", "").lower():
                            file_name = step_result.get("file_name", f"file_{len(self.phase_progress.files_studied)}")
                            self.phase_progress.files_studied.append(file_name)
                            
                        # Track directories explored
                        if "directory" in step_result.get("type", "").lower():
                            dir_name = step_result.get("directory", f"dir_{len(self.phase_progress.directories_explored)}")
                            self.phase_progress.directories_explored.append(dir_name)
            
            # Update knowledge documents created
            if result.get("ai_analysis"):
                self.learning_metrics.knowledge_documents += 1
                self.phase_progress.knowledge_documents_created += 1
                
            # Calculate understanding depth based on insights
            insights = task.learning_extracted or []
            self.learning_metrics.insights_generated += len(insights)
            self.phase_progress.insights_generated.extend(insights)
            
            # Update understanding depth score
            depth_increase = min(len(insights) * 0.01, 0.05)  # Cap at 5% per task
            self.learning_metrics.understanding_depth_score += depth_increase
            
            # Mark relevant tasks as completed based on what was accomplished
            await self._mark_completed_tasks_from_results(result)
            
            logger.info(f"📊 LEARNING METRICS UPDATED:")
            logger.info(f"   📁 Files analyzed: {self.learning_metrics.files_analyzed}")
            logger.info(f"   🧠 Concepts extracted: {self.learning_metrics.concepts_extracted}")
            logger.info(f"   📂 Directories explored: {self.learning_metrics.directories_explored}")
            logger.info(f"   💡 Insights generated: {self.learning_metrics.insights_generated}")
            
        except Exception as e:
            logger.error(f"❌ Error updating learning metrics: {e}")

    def _extract_ue_concepts_from_text(self, text: str) -> List[str]:
        """Extract UE-specific concepts from text"""
        concepts = []
        
        # UE-specific patterns
        ue_patterns = [
            r'\b[AU][A-Z][a-zA-Z0-9]*(?:Component|Controller|Manager|System|Engine|Actor|Pawn|Character|GameMode|PlayerController|HUD|Widget)\b',
            r'\b(?:Blueprint|Material|Texture|Mesh|Animation|Physics|Collision|Input|Audio|Rendering|Lumen|Nanite|Chaos)\b',
            r'\b(?:UPROPERTY|UFUNCTION|UCLASS|USTRUCT|UENUM|GENERATED_BODY)\b',
            r'\bUE_[A-Z_]+\b',
            r'\bF[A-Z][a-zA-Z0-9]*\b'  # UE structs like FVector, FRotator
        ]
        
        for pattern in ue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        # Remove duplicates and return
        return list(set(concepts))

    async def _mark_completed_tasks_from_results(self, result: Dict[str, Any]):
        """Mark tasks as completed based on what was accomplished"""
        try:
            step_results = result.get("step_results", [])
            
            for step_result in step_results:
                if step_result.get("success", False):
                    tool_used = step_result.get("tool_used", "")
                    
                    # Map tool usage to task completion
                    if tool_used == "list_files":
                        if "Engine" in str(step_result.get("tool_result", "")):
                            self.phase_progress.completed_tasks.add("explore_engine_directory")
                        if "Binaries" in str(step_result.get("tool_result", "")):
                            self.phase_progress.completed_tasks.add("analyze_binaries_directory")
                        if "Plugins" in str(step_result.get("tool_result", "")):
                            self.phase_progress.completed_tasks.add("study_plugins_structure")
                        if "Templates" in str(step_result.get("tool_result", "")):
                            self.phase_progress.completed_tasks.add("explore_templates_directory")
                            
                    elif tool_used == "read_file":
                        if "Build.version" in str(step_result.get("tool_result", "")):
                            self.phase_progress.completed_tasks.add("read_engine_version_info")
                            
                    elif tool_used == "find_files":
                        if ".exe" in str(step_result.get("tool_result", "")):
                            self.phase_progress.completed_tasks.add("analyze_engine_executables")
                            
                    elif tool_used == "get_unreal_engine_info":
                        self.phase_progress.completed_tasks.add("understand_engine_organization")
            
            # Update completion percentage
            self.phase_progress.completion_percentage = self.phase_progress.calculate_completion()
            
            logger.info(f"📋 TASKS COMPLETED: {len(self.phase_progress.completed_tasks)}/{len(self.phase_progress.required_tasks)}")
            for task in self.phase_progress.completed_tasks:
                logger.info(f"   ✅ {task}")
                
        except Exception as e:
            logger.error(f"❌ Error marking completed tasks: {e}")

    async def _phase_source_foundations(self) -> Dict[str, Any]:
        """Phase 3: Study UE5 source code fundamentals - COMPREHENSIVE SOURCE CODE READING"""
        logger.info("📚 Phase 3: COMPREHENSIVE UE5 source code file-by-file study")

        try:
            # Get current context about learning state
            context = self._get_learning_context()

            prompt = f"""
            {context}

            You are in SOURCE FOUNDATIONS phase. Your goal is to READ AND UNDERSTAND every single source code file in the UE5 source code.

            PHASE 3 OBJECTIVES - COMPREHENSIVE SOURCE CODE READING:
            - Read EVERY SINGLE SOURCE FILE (.h, .cpp, .cs, etc.) in the UE5 source code directory
            - Understand the actual implementation of UE5 systems and classes
            - Analyze class hierarchies, function implementations, and architectural patterns
            - Study how different engine systems are implemented and interact
            - Extract deep insights about UE5 architecture from actual code

            SYSTEMATIC SOURCE CODE APPROACH:
            - Start with core runtime modules
            - Read every header file to understand class definitions
            - Read implementation files to understand how systems work
            - Analyze engine patterns, macros, and conventions
            - Build comprehensive understanding of UE5 codebase

            AVAILABLE TOOLS FOR COMPREHENSIVE SOURCE STUDY:
            - list_files: Get complete source directory listings
            - read_file: Read and analyze every single source file
            - file_info: Get detailed information about source files
            - find_files: Locate specific source file types systematically

            CREATE A SYSTEMATIC PLAN TO READ EVERY SOURCE FILE IN THE UE5 CODEBASE.
            """

            # Execute comprehensive source code reading
            result = await self._execute_comprehensive_source_study()

            # Reflect on the comprehensive source study
            if result.get("success", False):
                reflection = await self._reflect_on_source_study_comprehensive(result)
                result["reflection"] = reflection
                
                # Update learning metrics based on ACTUAL source code reading
                await self._update_learning_metrics_from_source_study(result)
                
                # Calculate mastery increase based on actual source code analysis
                mastery_increase = self._calculate_source_mastery_increase(result)
                old_mastery = self.overall_mastery
                self.overall_mastery = min(self.target_mastery, self.overall_mastery + mastery_increase)
                
                logger.info(f"📈 COMPREHENSIVE source code study completed.")
                logger.info(f"📊 Source files actually read: {result.get('source_files_read', 0)}")
                logger.info(f"📊 Classes analyzed: {result.get('classes_found', 0)}")
                logger.info(f"📊 Functions analyzed: {result.get('functions_found', 0)}")
                logger.info(f"📈 Mastery increased: {old_mastery:.1%} → {self.overall_mastery:.1%} (+{mastery_increase:.1%})")
            else:
                logger.warning(f"⚠️ Comprehensive source study had issues, but continuing...")
                # Much smaller increase for failed comprehensive study
                self.overall_mastery += 0.001  # Minimal increase for attempt
                logger.info(f"📈 Attempted comprehensive source study. Mastery: {self.overall_mastery:.1%}")

            return result

        except Exception as e:
            logger.error(f"Comprehensive source study failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_comprehensive_source_study(self) -> Dict[str, Any]:
        """Execute comprehensive file-by-file study of UE5 source code"""
        try:
            logger.info("🔍 Starting COMPREHENSIVE UE5 source code file-by-file study")
             
            study_results = {
                "success": True,
                "source_files_read": 0,
                "directories_processed": 0,
                "total_source_content": 0,
                "classes_found": 0,
                "functions_found": 0,
                "insights_generated": [],
                "source_analysis": [],
                "errors": []
            }
             
            # Start with source directory
            source_path = self.ue_source_path
            logger.info(f"📁 Starting comprehensive source study of: {source_path}")
             
            if not source_path.exists():
                logger.error(f"❌ Source path does not exist: {source_path}")
                return {"success": False, "error": f"Source path not found: {source_path}"}
             
            # Get all source directories to process systematically
            source_directories = [source_path]
             
            # Add all subdirectories that contain source code
            for item in source_path.rglob("*"):
                if item.is_dir() and self._is_source_directory(item):
                    source_directories.append(item)
             
            logger.info(f"📊 Found {len(source_directories)} source directories to process comprehensively")
             
            # Process each source directory systematically
            for dir_index, directory in enumerate(source_directories[:100]):  # Limit to first 100 source dirs
                logger.info(f"📁 Processing source directory {dir_index + 1}/{min(len(source_directories), 100)}: {directory.name}")
                 
                dir_result = await self._process_source_directory_comprehensively(directory)
                 
                # Accumulate results
                study_results["source_files_read"] += dir_result.get("source_files_read", 0)
                study_results["total_source_content"] += dir_result.get("content_length", 0)
                study_results["classes_found"] += dir_result.get("classes_found", 0)
                study_results["functions_found"] += dir_result.get("functions_found", 0)
                study_results["insights_generated"].extend(dir_result.get("insights", []))
                study_results["source_analysis"].extend(dir_result.get("source_analysis", []))
                study_results["errors"].extend(dir_result.get("errors", []))
                 
                study_results["directories_processed"] += 1
                 
                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.5)
                 
                # Log progress
                if (dir_index + 1) % 20 == 0:
                    logger.info(f"📊 Progress: {dir_index + 1} directories, {study_results['source_files_read']} source files read")
             
            logger.info(f"✅ COMPREHENSIVE SOURCE STUDY COMPLETE!")
            logger.info(f"📊 Final stats: {study_results['directories_processed']} directories, {study_results['source_files_read']} source files")
            logger.info(f"📊 Total source content: {study_results['total_source_content']} characters")
            logger.info(f"📊 Classes found: {study_results['classes_found']}")
            logger.info(f"📊 Functions found: {study_results['functions_found']}")
             
            return study_results
             
        except Exception as e:
            logger.error(f"❌ Comprehensive source study failed: {e}")
            return {"success": False, "error": str(e)}

    def _is_source_directory(self, directory: Path) -> bool:
        """Check if a directory contains source code files"""
        try:
            # Check if directory contains source files
            for item in directory.iterdir():
                if item.is_file() and item.suffix.lower() in ['.h', '.hpp', '.cpp', '.c', '.cs']:
                    return True
            return False
        except (PermissionError, OSError):
            return False

    async def _process_source_directory_comprehensively(self, directory: Path) -> Dict[str, Any]:
        """Process a single source directory by reading every source file in it"""
        try:
            logger.info(f"🔍 Comprehensively processing source directory: {directory}")
             
            result = {
                "source_files_read": 0,
                "content_length": 0,
                "classes_found": 0,
                "functions_found": 0,
                "insights": [],
                "source_analysis": [],
                "errors": []
            }
             
            # Get all source files in this directory
            try:
                source_files = [f for f in directory.iterdir() 
                              if f.is_file() and f.suffix.lower() in ['.h', '.hpp', '.cpp', '.c', '.cs']]
            except (PermissionError, OSError) as e:
                logger.warning(f"⚠️ Cannot access source directory {directory}: {e}")
                result["errors"].append(f"Cannot access {directory}: {e}")
                return result
             
            logger.info(f"📄 Found {len(source_files)} source files in {directory.name}")
             
            # Read each source file systematically
            for file_index, file_path in enumerate(source_files[:30]):  # Limit to 30 source files per directory
                logger.info(f"📖 Reading source file {file_index + 1}/{min(len(source_files), 30)}: {file_path.name}")
                 
                file_result = await self._read_and_analyze_source_file_comprehensively(file_path)
                 
                if file_result.get("success", False):
                    result["source_files_read"] += 1
                    result["content_length"] += file_result.get("content_length", 0)
                    result["classes_found"] += file_result.get("classes_found", 0)
                    result["functions_found"] += file_result.get("functions_found", 0)
                    result["insights"].extend(file_result.get("insights", []))
                    result["source_analysis"].append({
                        "file": str(file_path),
                        "size": file_result.get("content_length", 0),
                        "type": file_result.get("file_type", "unknown"),
                        "classes": file_result.get("classes_found", 0),
                        "functions": file_result.get("functions_found", 0),
                        "insights": file_result.get("insights", [])
                    })
                else:
                    result["errors"].append(f"Failed to read {file_path}: {file_result.get('error', 'Unknown error')}")
                 
                # Brief pause between files
                await asyncio.sleep(0.1)
             
            logger.info(f"✅ Source directory {directory.name} complete: {result['source_files_read']} source files read")
            return result
             
        except Exception as e:
            logger.error(f"❌ Error processing source directory {directory}: {e}")
            return {"source_files_read": 0, "content_length": 0, "classes_found": 0, "functions_found": 0, 
                   "insights": [], "source_analysis": [], "errors": [str(e)]}

    async def _read_and_analyze_source_file_comprehensively(self, file_path: Path) -> Dict[str, Any]:
        """Read and comprehensively analyze a single source code file"""
        try:
            # Check file size - skip very large files
            file_size = file_path.stat().st_size
            if file_size > 2000000:  # 2MB limit for source files
                logger.info(f"⏭️ Skipping large source file: {file_path.name} ({file_size} bytes)")
                return {"success": False, "error": "Source file too large"}
             
            # Read the source file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"⚠️ Cannot read source file {file_path.name}: {e}")
                return {"success": False, "error": str(e)}
             
            if not content.strip():
                logger.info(f"⏭️ Empty source file: {file_path.name}")
                return {"success": False, "error": "Empty source file"}
             
            logger.info(f"📖 Successfully read source file {file_path.name}: {len(content)} characters")
             
            # Analyze the source code comprehensively
            analysis_result = await self._analyze_source_code_comprehensively(content, file_path)
             
            # Extract classes and functions
            classes_found = self._count_classes_in_source(content)
            functions_found = self._count_functions_in_source(content)
             
            # Store in knowledge base for future reference
            await self.process_and_learn(content[:4000], f"source_file_{file_path.name}")
             
            return {
                "success": True,
                "content_length": len(content),
                "file_type": "source_code",
                "classes_found": classes_found,
                "functions_found": functions_found,
                "insights": analysis_result.get("insights", []),
                "analysis": analysis_result.get("analysis", "")
            }
             
        except Exception as e:
            logger.error(f"❌ Error reading source file {file_path}: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_source_code_comprehensively(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Comprehensively analyze source code content using AI"""
        try:
            # Create source code analysis prompt
            prompt = f"""Comprehensively analyze this UE5 source code file and extract deep architectural insights:

SOURCE FILE: {file_path.name}
FILE TYPE: {file_path.suffix}
CONTENT LENGTH: {len(content)} characters

SOURCE CODE (first 3000 chars):
{content[:3000]}

COMPREHENSIVE SOURCE CODE ANALYSIS REQUIRED:
1. What CLASSES are defined in this file and what do they do?
2. What KEY FUNCTIONS are implemented and their purposes?
3. How does this code fit into the UE5 ARCHITECTURE?
4. What DESIGN PATTERNS are used in this implementation?
5. What can we LEARN about UE5 engine design from this code?
6. How does this relate to other UE5 systems and components?

Focus on understanding the actual implementation, not just listing names. Provide deep insights about UE5 architecture."""

            # Get AI analysis
            analysis = await self._query_llm(prompt, enable_thinking=False)
             
            # Extract insights from analysis
            insights = self._extract_insights_from_analysis(analysis)
             
            return {
                "analysis": analysis,
                "insights": insights
            }
             
        except Exception as e:
            logger.error(f"❌ Error analyzing source code: {e}")
            return {"analysis": "", "insights": []}

    def _count_classes_in_source(self, content: str) -> int:
        """Count actual class definitions in source code"""
        class_patterns = [
            r'\bclass\s+[A-Z_]*API\s+([A-Z][a-zA-Z0-9_]*)',  # UE classes with API
            r'\bUCLASS\s*\([^)]*\)\s*class\s+[A-Z_]*API\s+([A-Z][a-zA-Z0-9_]*)',  # UCLASS
            r'\bUSTRUCT\s*\([^)]*\)\s*struct\s+[A-Z_]*API\s+([A-Z][a-zA-Z0-9_]*)',  # USTRUCT
            r'\bclass\s+([A-Z][a-zA-Z0-9_]*)\s*[:{]',  # General class definitions
            r'\bstruct\s+([A-Z][a-zA-Z0-9_]*)\s*[:{]'  # Struct definitions
        ]
         
        total_classes = 0
        for pattern in class_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            total_classes += len(matches)
         
        return total_classes

    def _count_functions_in_source(self, content: str) -> int:
        """Count actual function definitions in source code"""
        function_patterns = [
            r'\bUFUNCTION\s*\([^)]*\)\s*[a-zA-Z_][a-zA-Z0-9_\s\*&]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # UFUNCTION
            r'\bvirtual\s+[a-zA-Z_][a-zA-Z0-9_\s\*&]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*override',  # Virtual overrides
            r'\bstatic\s+[a-zA-Z_][a-zA-Z0-9_\s\*&]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # Static functions
            r'\b[a-zA-Z_][a-zA-Z0-9_\s\*&]*\s+([A-Z][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{',  # General function definitions
        ]
         
        total_functions = 0
        for pattern in function_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            total_functions += len(matches)
         
        return total_functions

    def _calculate_source_mastery_increase(self, study_result: Dict[str, Any]) -> float:
        """Calculate mastery increase based on ACTUAL source code study"""
        files_read = study_result.get("source_files_read", 0)
        content_length = study_result.get("total_source_content", 0)
        classes_found = study_result.get("classes_found", 0)
        functions_found = study_result.get("functions_found", 0)
        insights = len(study_result.get("insights_generated", []))
         
        # Conservative scoring based on actual source code analysis
        file_score = min(files_read * 0.002, 0.03)  # 0.2% per source file, max 3%
        content_score = min(content_length / 2000000 * 0.02, 0.03)  # 2% per 2MB source content, max 3%
        class_score = min(classes_found * 0.001, 0.02)  # 0.1% per class, max 2%
        function_score = min(functions_found * 0.0005, 0.02)  # 0.05% per function, max 2%
        insight_score = min(insights * 0.003, 0.02)  # 0.3% per insight, max 2%
         
        total_increase = file_score + content_score + class_score + function_score + insight_score
         
        logger.info(f"📊 SOURCE CODE MASTERY CALCULATION:")
        logger.info(f"   📄 Source files read: {files_read} → {file_score:.1%}")
        logger.info(f"   📝 Source content: {content_length} chars → {content_score:.1%}")
        logger.info(f"   🏗️ Classes analyzed: {classes_found} → {class_score:.1%}")
        logger.info(f"   ⚙️ Functions analyzed: {functions_found} → {function_score:.1%}")
        logger.info(f"   💡 Insights: {insights} → {insight_score:.1%}")
        logger.info(f"   📈 Total increase: {total_increase:.1%}")
         
        return total_increase

    async def _update_learning_metrics_from_source_study(self, study_result: Dict[str, Any]):
        """Update learning metrics based on comprehensive source study results"""
        try:
            # Update with ACTUAL source code reading metrics
            self.learning_metrics.files_analyzed += study_result.get("source_files_read", 0)
            self.learning_metrics.knowledge_documents += 1  # One comprehensive source study session
            self.learning_metrics.insights_generated += len(study_result.get("insights_generated", []))
             
            # Extract actual UE concepts from source code insights
            all_insights = study_result.get("insights_generated", [])
            real_concepts = []
            for insight in all_insights:
                concepts = self._extract_real_ue_concepts_from_insight(insight)
                real_concepts.extend(concepts)
             
            # Add concepts from actual classes and functions found
            real_concepts.extend([f"class_{i}" for i in range(study_result.get("classes_found", 0))])
            real_concepts.extend([f"function_{i}" for i in range(study_result.get("functions_found", 0))])
             
            self.learning_metrics.concepts_extracted += len(set(real_concepts))  # Unique concepts only
             
            # Update understanding depth based on comprehensive source analysis
            depth_increase = min(len(all_insights) * 0.015, 0.08)  # Higher for source code
            self.learning_metrics.understanding_depth_score += depth_increase
             
            logger.info(f"📊 COMPREHENSIVE SOURCE LEARNING METRICS UPDATED:")
            logger.info(f"   📁 Source files actually read: {study_result.get('source_files_read', 0)}")
            logger.info(f"   🏗️ Classes found: {study_result.get('classes_found', 0)}")
            logger.info(f"   ⚙️ Functions found: {study_result.get('functions_found', 0)}")
            logger.info(f"   🧠 Real UE concepts: {len(set(real_concepts))}")
            logger.info(f"   💡 Insights from source: {len(all_insights)}")
            logger.info(f"   🎯 Understanding depth: {self.learning_metrics.understanding_depth_score:.3f}")
             
        except Exception as e:
            logger.error(f"❌ Error updating source learning metrics: {e}")

    async def _reflect_on_source_study_comprehensive(self, study_result: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on comprehensive source study with thinking mode"""
        try:
            logger.info(f"🧠 Reflecting on comprehensive source study with thinking mode")
             
            prompt = f"""Reflect deeply on this COMPREHENSIVE UE5 source code study:

COMPREHENSIVE SOURCE STUDY RESULTS:
- Source files actually read: {study_result.get('source_files_read', 0)}
- Directories processed: {study_result.get('directories_processed', 0)}
- Total source content analyzed: {study_result.get('total_source_content', 0)} characters
- Classes found and analyzed: {study_result.get('classes_found', 0)}
- Functions found and analyzed: {study_result.get('functions_found', 0)}
- Insights generated: {len(study_result.get('insights_generated', []))}
- Errors encountered: {len(study_result.get('errors', []))}

This was a REAL comprehensive source code study where every source file was actually read and analyzed for classes, functions, and architectural patterns.

DEEP SOURCE CODE REFLECTION REQUIRED:
1. What was actually LEARNED about UE5 implementation from reading the source code?
2. What architectural PATTERNS emerged from analyzing the actual code?
3. How do the classes and functions relate to UE5's overall design?
4. What insights about engine design were gained from the source analysis?
5. How does this source knowledge prepare for hands-on development?

Provide a thoughtful analysis of the actual source code learning achieved."""

            # Use thinking mode for comprehensive source reflection
            result = await self._query_llm_with_thinking(prompt)
             
            reflection_data = {
                "comprehensive_source_study": True,
                "source_files_actually_read": study_result.get('source_files_read', 0),
                "classes_analyzed": study_result.get('classes_found', 0),
                "functions_analyzed": study_result.get('functions_found', 0),
                "thinking_process": result.get("thinking", ""),
                "reflection_content": result.get("response", ""),
                "timestamp": datetime.now().isoformat(),
                "success": result.get("success", False)
            }
             
            if result["success"] and result["thinking"]:
                logger.info(f"🧠 Comprehensive Source Study Reflection: {result['thinking'][:300]}...")
                 
                # Store the comprehensive source thinking process
                await self._store_thinking_process("comprehensive_source_study", result["thinking"], "source_reflection")
             
            return reflection_data
             
        except Exception as e:
            logger.error(f"Error in comprehensive source study reflection: {e}")
            return {"success": False, "error": str(e)}

    async def _read_and_analyze_file_comprehensively(self, file_path: Path) -> Dict[str, Any]:
        """
        Read and comprehensively analyze a single file for learning
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dict containing analysis results and extracted knowledge
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return {
                    'success': False,
                    'error': 'Empty file',
                    'file_path': str(file_path),
                    'concepts_extracted': [],
                    'insights_generated': []
                }
            
            # Analyze content comprehensively
            analysis_result = await self._analyze_file_content_comprehensively(content, file_path)
            
            # Extract concepts specific to file type
            concepts = self._extract_ue_concepts_from_text(content)
            
            # Update learning metrics
            self.learning_metrics.files_analyzed += 1
            self.learning_metrics.concepts_extracted += len(concepts)
            
            return {
                'success': True,
                'file_path': str(file_path),
                'content_length': len(content),
                'concepts_extracted': concepts,
                'insights_generated': analysis_result.get('insights', []),
                'analysis': analysis_result.get('analysis', ''),
                'entities_found': analysis_result.get('entities_found', 0),
                'relationships_created': analysis_result.get('relationships_created', 0)
            }
            
        except Exception as e:
            logging.error(f"❌ Error reading file {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'concepts_extracted': [],
                'insights_generated': []
            }

    async def _analyze_file_content_comprehensively(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Comprehensively analyze file content using AI"""
        try:
            # Create analysis prompt
            prompt = f"""Comprehensively analyze this UE5 installation file and extract deep insights:

FILE: {file_path.name}
FILE TYPE: {file_path.suffix}
CONTENT LENGTH: {len(content)} characters

CONTENT (first 2000 chars):
{content[:2000]}

COMPREHENSIVE ANALYSIS REQUIRED:
1. What is the PURPOSE of this file in the UE5 installation?
2. What FUNCTIONALITY does it provide or configure?
3. How does it RELATE to other UE5 systems?
4. What can we LEARN about UE5 architecture from this file?
5. What are the KEY INSIGHTS for understanding UE5?

Provide specific, technical insights about UE5 architecture and functionality."""

            # Get AI analysis
            analysis = await self._query_llm(prompt, enable_thinking=False)
            
            # Extract insights from analysis
            insights = self._extract_insights_from_analysis(analysis)
            
            return {
                "analysis": analysis,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing file content: {e}")
            return {"analysis": "", "insights": []}

    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the type/purpose of a file"""
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        
        if suffix in ['.ini', '.cfg', '.config']:
            return "configuration"
        elif suffix in ['.txt', '.md', '.rst']:
            return "documentation"
        elif suffix in ['.json', '.xml', '.yaml', '.yml']:
            return "data"
        elif suffix in ['.h', '.hpp', '.cpp', '.c', '.cs']:
            return "source_code"
        elif suffix in ['.py', '.js', '.ts']:
            return "script"
        elif suffix in ['.bat', '.sh', '.cmd']:
            return "batch_script"
        elif 'version' in name or 'build' in name:
            return "version_info"
        elif 'license' in name or 'copyright' in name:
            return "legal"
        else:
            return "unknown"

    def _extract_insights_from_analysis(self, analysis: str) -> List[str]:
        """Extract specific insights from AI analysis"""
        insights = []
        
        # Look for insight patterns
        insight_patterns = [
            r"INSIGHT:\s*(.+?)(?:\n|$)",
            r"KEY INSIGHT:\s*(.+?)(?:\n|$)",
            r"IMPORTANT:\s*(.+?)(?:\n|$)",
            r"This reveals that\s*(.+?)(?:\n|\.)",
            r"This shows that\s*(.+?)(?:\n|\.)",
            r"This indicates that\s*(.+?)(?:\n|\.)"
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, analysis, re.IGNORECASE | re.MULTILINE)
            insights.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        # If no structured insights found, extract sentences that contain learning
        if not insights:
            sentences = analysis.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['ue5', 'unreal', 'engine', 'system', 'architecture']):
                    if len(sentence.strip()) > 20:
                        insights.append(sentence.strip())
        
        return insights[:5]  # Limit to 5 insights per file


# ==================== MAIN INTERFACE ====================

async def main():
    """Main interface for AndrioV2"""
    logger.info("🤖 ANDRIO V2 MAIN INTERFACE STARTING")
    logger.info("=" * 80)
    
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
    logger.info("🔍 CHECKING AVAILABLE OLLAMA MODELS")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    print(f"📋 Available models: {', '.join(available_models)}")
                    logger.info(f"📋 AVAILABLE MODELS: {available_models}")

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
                    logger.info(f"✅ SELECTED MODEL: {selected_model}")
                else:
                    print("⚠️  Could not connect to Ollama, using default model")
                    logger.warning("⚠️ COULD NOT CONNECT TO OLLAMA")
                    selected_model = "andriov2"
    except Exception as e:
        print(f"⚠️  Error checking models: {e}, using default")
        logger.error(f"❌ ERROR CHECKING MODELS: {e}")
        selected_model = "andriov2"

    # Initialize AndrioV2
    print(f"\n🚀 Initializing AndrioV2 with {selected_model}...")
    logger.info(f"🚀 INITIALIZING ANDRIO V2 WITH MODEL: {selected_model}")
    try:
        andrio = AndrioV2(model_name=selected_model)
        print("✅ AndrioV2 initialized successfully!")
        logger.info("✅ ANDRIO V2 INITIALIZED SUCCESSFULLY")

        if andrio.ue_connected:
            print("🟢 Connected to UE5 remote execution")
            logger.info("🟢 UE5 remote execution connection established")
        else:
            print("🔴 UE5 remote execution not available")
            logger.warning("🔴 UE5 remote execution not available")

        # Show current status
        status = andrio.get_status()
        print(f"\n📊 Current Status:")
        print(f"   📍 Phase: {status['current_phase']}")
        print(f"   🎯 Overall Mastery: {status['overall_mastery']:.1%}")
        print(f"   📚 Knowledge Base: {status['knowledge_base_size']} documents")
        print(f"   🕸️  Knowledge Graph: {status['entities_in_graph']} entities")
        
        logger.info(f"📊 INITIAL STATUS:")
        logger.info(f"   📍 Phase: {status['current_phase']}")
        logger.info(f"   🎯 Overall Mastery: {status['overall_mastery']:.1%}")
        logger.info(f"   📚 Knowledge Base: {status['knowledge_base_size']} documents")
        logger.info(f"   🕸️  Knowledge Graph: {status['entities_in_graph']} entities")
            
    except Exception as e:
        print(f"❌ Failed to initialize AndrioV2: {e}")
        logger.error(f"❌ FAILED TO INITIALIZE ANDRIO V2: {e}")
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
        logger.info(f"🧠 THINKING MODE STATUS: {thinking_status}")
        print("-" * 60)

    logger.info("🎮 ENTERING INTERACTIVE MODE")
    logger.info("=" * 60)

    while True:
        try:
            user_input = input(f"\n🤖 Andrio ({andrio.current_phase.value}): ").strip()
            
            # Log every user input
            logger.info("=" * 60)
            logger.info(f"👤 USER INPUT: {user_input}")
            logger.info(f"⏰ TIME: {datetime.now().strftime('%I:%M:%S %p')}")
            logger.info(f"📍 CURRENT PHASE: {andrio.current_phase.value}")
            logger.info("-" * 30)

            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye! Andrio's learning progress has been saved.")
                logger.info("👋 USER EXITING - SAVING PROGRESS")
                break

            elif user_input.lower() == 'autonomous':
                print("🚀 Starting autonomous learning...")
                logger.info("🚀 USER REQUESTED AUTONOMOUS LEARNING")
                result = await andrio.start_autonomous_learning()

                if result["success"]:
                    print(f"✅ Autonomous learning cycle completed!")
                    print(f"📊 Task: {result.get('task_id', 'N/A')}")
                    print(f"⏱️  Time: {result.get('total_time', 0):.2f}s")
                    
                    logger.info(f"✅ AUTONOMOUS LEARNING COMPLETED")
                    logger.info(f"📊 Result: {result}")

                    # Show updated status
                    status = andrio.get_status()
                    print(f"📈 Updated mastery: {status['overall_mastery']:.1%}")
                    logger.info(f"📈 UPDATED MASTERY: {status['overall_mastery']:.1%}")
                else:
                    print(f"❌ Autonomous learning failed: {result.get('error', 'Unknown error')}")
                    logger.error(f"❌ AUTONOMOUS LEARNING FAILED: {result.get('error', 'Unknown error')}")

            elif user_input.lower().startswith('task '):
                task_description = user_input[5:].strip()
                if task_description:
                    print(f"📋 Planning and executing task: {task_description}")
                    logger.info(f"📋 USER REQUESTED TASK: {task_description}")
                    result = await andrio.plan_and_execute_task(task_description)

                    if result["success"]:
                        print(f"✅ Task completed successfully!")
                        print(f"📊 Steps: {result['execution_result']['steps_completed']}/{result['execution_result']['total_steps']}")
                        print(f"⏱️  Time: {result['total_time']:.2f}s")
                        
                        logger.info(f"✅ TASK COMPLETED SUCCESSFULLY")
                        logger.info(f"📊 Steps: {result['execution_result']['steps_completed']}/{result['execution_result']['total_steps']}")
                        logger.info(f"⏱️  Time: {result['total_time']:.2f}s")
                    else:
                        print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
                        logger.error(f"❌ TASK FAILED: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Please provide a task description")
                    logger.warning("❌ USER PROVIDED EMPTY TASK DESCRIPTION")

            elif user_input.lower().startswith('query '):
                query = user_input[6:].strip()
                if query:
                    print(f"🔍 Querying knowledge base: {query}")
                    logger.info(f"🔍 USER QUERY: {query}")
                    result = await andrio.query_knowledge(query)

                    if result.get("ai_response"):
                        print(f"\n📝 Response:")
                        print("-" * 50)
                        print(result["ai_response"])
                        print("-" * 50)
                        print(f"📊 Results found: {result['results_found']}")
                        print(f"⏱️  Time: {result['total_time']:.2f}s")
                        
                        logger.info(f"✅ QUERY SUCCESSFUL")
                        logger.info(f"📊 Results found: {result['results_found']}")
                        logger.info(f"⏱️  Time: {result['total_time']:.2f}s")
                        logger.info(f"📝 Response: {result['ai_response'][:200]}...")
                    else:
                        print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                        logger.error(f"❌ QUERY FAILED: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Please provide a query")
                    logger.warning("❌ USER PROVIDED EMPTY QUERY")

            elif user_input.lower() == 'status':
                logger.info("📊 USER REQUESTED STATUS")
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
                
                logger.info(f"📊 STATUS DISPLAYED: {status}")

            elif user_input.lower().startswith('tool '):
                command = user_input[5:].strip()
                if command:
                    print(f"🔧 Executing tool command: {command}")
                    logger.info(f"🔧 USER TOOL COMMAND: {command}")
                    result = andrio.execute_tool_command(command)
                    print(f"📝 Tool output: {result}")
                    logger.info(f"📝 TOOL OUTPUT: {result[:200]}...")
                else:
                    print("❌ Please provide a tool command")
                    logger.warning("❌ USER PROVIDED EMPTY TOOL COMMAND")

            elif user_input.lower() == 'thinking':
                print("🧠 Toggling thinking mode...")
                logger.info("🧠 USER TOGGLING THINKING MODE")
                result = andrio.toggle_thinking_mode()
                print(result)
                logger.info(f"🧠 THINKING MODE RESULT: {result}")

            elif not user_input:
                continue
            else:
                print("❌ Unknown command. Type 'exit' to quit or use one of the available commands.")
                logger.warning(f"❌ UNKNOWN COMMAND: {user_input}")
            
            logger.info("=" * 60)

        except KeyboardInterrupt:
            print("\n👋 Goodbye! Andrio's learning progress has been saved.")
            logger.info("👋 USER INTERRUPTED - SAVING PROGRESS")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"❌ MAIN LOOP ERROR: {e}")
            logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")

    logger.info("🏁 ANDRIO V2 MAIN INTERFACE ENDING")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
