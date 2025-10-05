"""
Orchestrator Utilities

This module contains utility classes and functions that support the orchestrator
but are not core to its functionality:
- Logging utilities (ColoredFormatter, setup_colored_logging)
- LLM implementations (BaseLLM, AzureOpenAILLM, LocalOpenAILLM, MockLLM)
- Configuration dataclass (OrchestratorConfig)
- LLM factory function (create_llm)
"""

import json
import re
import logging
import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Type, Any

# Optional dependencies - imported when needed
try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AzureOpenAI = None
    OpenAI = None
    OPENAI_AVAILABLE = False

if TYPE_CHECKING:
    from pydantic import BaseModel


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to the log level name
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


def setup_colored_logging(log_to_file: bool = False, log_file_path: str = "orchestrator.log") -> None:
    """Setup colored logging for the orchestrator package."""
    # Create a colored formatter
    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create a plain formatter for file logging
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Check if we already have a colored handler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, '_is_colored', False):
            return  # Already setup
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        if isinstance(handler, (logging.StreamHandler, logging.FileHandler)):
            logger.removeHandler(handler)
    
    # Create and configure the colored console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    setattr(console_handler, '_is_colored', True)  # Mark as colored handler
    
    # Add the console handler to the logger
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    logger.setLevel(logging.DEBUG)  # Changed from WARNING to DEBUG to show all messages


@dataclass
class LLMConfig:
    """Base configuration for a single LLM."""
    name: str  # Unique identifier for this LLM instance
    llm_type: str  # azure_openai, local_openai, mock
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Azure OpenAI specific
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: str = "2024-02-01"
    azure_deployment_name: str = "gpt-4"
    
    # Local OpenAI compatible (Ollama, vLLM, etc.)
    local_endpoint: str = "http://localhost:11434/v1"
    local_api_key: Optional[str] = None
    local_model: str = "llama2"
    
    def __post_init__(self):
        """Load configuration from environment variables if not provided."""
        # Azure OpenAI settings
        if self.llm_type == "azure_openai":
            if self.azure_endpoint is None:
                self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            if self.azure_api_key is None:
                self.azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
            if azure_api_version := os.getenv('AZURE_OPENAI_API_VERSION'):
                self.azure_api_version = azure_api_version
            if azure_deployment := os.getenv('LLM_MODEL_NAME'):
                self.azure_deployment_name = azure_deployment
        
        # Local OpenAI settings
        elif self.llm_type == "local_openai":
            if local_endpoint := os.getenv('LOCAL_URL'):
                self.local_endpoint = local_endpoint
            if self.local_api_key is None:
                self.local_api_key = os.getenv('LOCAL_API_KEY')
            if local_model := os.getenv('LOCAL_MODEL'):
                self.local_model = local_model


@dataclass
class OrchestratorConfig:
    """
    Flexible configuration for orchestrators supporting arbitrary LLM role combinations
    and orchestrator-specific parameters.
    
    Uses dictionaries for both LLM roles and orchestrator-specific settings to avoid
    polluting the core config class with orchestrator-specific parameters.
    """
    
    # Multiple LLM configurations - required
    llms: Dict[str, LLMConfig]
    
    # Core settings shared by all orchestrators
    mcp_data_path: str = str(Path(__file__).resolve().parent.parent / "data" / "mcp_registry_w_embedding.json")
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    
    # Orchestrator-specific configurations
    # This dictionary allows each orchestrator type to define its own parameters
    # without modifying the core configuration class
    orchestrator_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Process configuration and set defaults."""
        # Process each LLM config for environment variables
        for llm_config in self.llms.values():
            llm_config.__post_init__()
        
        # Initialize orchestrator_params if not provided
        if self.orchestrator_params is None:
            self.orchestrator_params = self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters used by most orchestrators."""
        return {
            # Common retrieval parameters
            "server_top_k": 3,
            "tool_top_k": 5,
            "similarity_threshold": 0.3,
        }
    
    def get_llm_config(self, role: str) -> LLMConfig:
        """Get LLM configuration for a specific role."""
        if role not in self.llms:
            # Fallback to router if role not found
            return self.llms.get("router") or list(self.llms.values())[0]
        return self.llms[role]
    
    def add_llm(self, role: str, llm_config: LLMConfig):
        """Add or update an LLM configuration for a specific role."""
        self.llms[role] = llm_config
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get an orchestrator-specific parameter."""
        if self.orchestrator_params is None:
            return default
        return self.orchestrator_params.get(key, default)
    
    def set_param(self, key: str, value: Any):
        """Set an orchestrator-specific parameter."""
        if self.orchestrator_params is None:
            self.orchestrator_params = {}
        self.orchestrator_params[key] = value
    
    def update_params(self, params: Dict[str, Any]):
        """Update multiple orchestrator parameters."""
        if self.orchestrator_params is None:
            self.orchestrator_params = {}
        self.orchestrator_params.update(params)
    
    @classmethod
    def create(cls, llms: Dict[str, LLMConfig], orchestrator_params: Optional[Dict[str, Any]] = None, **kwargs) -> 'OrchestratorConfig':
        """
        Create orchestrator configuration with arbitrary LLM role combinations
        and orchestrator-specific parameters.
        
        Args:
            llms: Dictionary mapping role names to LLM configurations
            orchestrator_params: Dictionary of orchestrator-specific parameters
            **kwargs: Additional configuration parameters
            
        Example:
            config = OrchestratorConfig.create(
                llms={
                    "router": router_llm,
                    "query_rewriter": rewriter_llm,
                    "reranker": reranker_llm
                },
                orchestrator_params={
                    "query_expansion_count": 5,
                    "rerank_top_k": 15
                }
            )
        """
        return cls(llms=llms, orchestrator_params=orchestrator_params, **kwargs)
    
    @classmethod
    def create_mcp_zero(cls, router_llm: LLMConfig, retriever_llm: LLMConfig, 
                       conversational_llm: LLMConfig, **kwargs) -> 'OrchestratorConfig':
        """
        Create configuration for MCP-Zero orchestrator.
        
        MCP-Zero uses:
        - router: Query classification and final tool selection
        - retriever: Tool retrieval and ranking
        - conversational: Conversational responses
        
        MCP-Zero specific parameters:
        - server_top_k: Number of top servers to consider (default: 3)
        - tool_top_k: Number of top tools to return (default: 5)
        - similarity_threshold: Minimum similarity threshold (default: 0.3)
        """
        llms = {
            "router": router_llm,
            "retriever": retriever_llm,
            "conversational": conversational_llm
        }
        
        # MCP-Zero specific parameters (can be overridden via kwargs)
        mcp_zero_params = {
            "server_top_k": 3,
            "tool_top_k": 5,
            "similarity_threshold": 0.3,
        }
        
        # Override with any provided parameters
        if 'orchestrator_params' in kwargs:
            mcp_zero_params.update(kwargs.pop('orchestrator_params'))
        
        return cls.create(llms, orchestrator_params=mcp_zero_params, **kwargs)
    
    @classmethod
    def create_toolshed(cls, router_llm: LLMConfig, decomposer_llm: LLMConfig,
                     query_rewriter_llm: LLMConfig, query_expander_llm: LLMConfig, 
                     reranker_llm: LLMConfig, conversational_llm: LLMConfig, **kwargs) -> 'OrchestratorConfig':
        """
        Create configuration for ToolShed orchestrator.
        
        ToolShed uses:
        - router: Query classification and final tool selection
        - query_rewriter: Query cleaning and normalization
        - query_expander: Query expansion into multiple variations
        - reranker: LLM-based tool reranking
        - conversational: Conversational responses
        
        ToolShed specific parameters:
        - server_top_k: Number of top servers to consider (default: 3)
        - tool_top_k: Number of top tools to return (default: 5)
        - similarity_threshold: Minimum similarity threshold (default: 0.3)
        - query_expansion_count: Number of query variations to generate (default: 3)
        - rerank_top_k: Number of tools to consider for reranking (default: 10)
        """
        llms = {
            "router": router_llm,
            "decomposer": decomposer_llm,
            "query_rewriter": query_rewriter_llm,
            "query_expander": query_expander_llm,
            "reranker": reranker_llm,
            "conversational": conversational_llm
        }
        
        # ToolShed specific parameters (can be overridden via kwargs)
        toolshed_params = {
            "server_top_k": 3,
            "tool_top_k": 5,
            "similarity_threshold": 0.3,
            "query_expansion_count": 3,
            "rerank_top_k": 10,
        }
        
        # Override with any provided parameters
        if 'orchestrator_params' in kwargs:
            toolshed_params.update(kwargs.pop('orchestrator_params'))
        
        return cls.create(llms, orchestrator_params=toolshed_params, **kwargs)


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response to prompt."""
        pass
    
    def generate_structured(self, prompt: str, schema: Type['BaseModel']) -> str:
        """
        Generate structured response using Pydantic schema.
        
        Args:
            prompt: The prompt to send to the LLM
            schema: Pydantic model class for the expected response structure
            
        Returns:
            JSON string that conforms to the schema
            
        Raises:
            ValueError: If the response cannot be parsed into the schema
        """
        # Default implementation: generate regular response and try to parse
        response = self.generate(prompt)
        return self._parse_response_to_schema(response, schema)
    
    def _parse_response_to_schema(self, response: str, schema: Type['BaseModel']) -> str:
        """Parse a response string and extract JSON that conforms to schema."""
        import json
        
        # Try to extract JSON from response
        json_text = self._extract_json_from_response(response)
        if not json_text:
            raise ValueError(f"No JSON found in response: {response}")
        
        try:
            return json_text
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
        except Exception as e:
            raise ValueError(f"Failed to validate response against schema: {e}")
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON from response - same logic as orchestrator."""
        def find_balanced_json(text: str) -> Optional[str]:
            """Find the first balanced JSON object in text."""
            stack = []
            start_idx = None
            i = 0
            
            while i < len(text):
                char = text[i]
                
                if char == '{':
                    if not stack:
                        start_idx = i
                    stack.append(char)
                elif char == '}' and stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        # Found complete JSON object
                        json_text = text[start_idx:i+1]
                        try:
                            # Test if it's valid JSON
                            json.loads(json_text)
                            return json_text
                        except json.JSONDecodeError:
                            # Continue searching
                            start_idx = None
                i += 1
            
            return None
        
        # Remove any /* */ style comments first
        import re
        cleaned_text = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
        
        # Try to find balanced JSON
        json_text = find_balanced_json(cleaned_text)
        return json_text
    
    async def generate_async(self, prompt: str) -> str:
        """Async version of generate. Default implementation calls sync version."""
        return self.generate(prompt)


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI LLM implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.AzureOpenAILLM')
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Azure OpenAI client."""
        if not OPENAI_AVAILABLE or AzureOpenAI is None:
            raise ImportError("openai package required for Azure OpenAI. Install with: pip install openai")
        
        if not self.config.azure_endpoint or not self.config.azure_api_key:
            raise ValueError(
                "Azure OpenAI requires azure_endpoint and azure_api_key. "
                "Set them as environment variables or pass them in config."
            )
        
        self.client = AzureOpenAI(
            azure_endpoint=self.config.azure_endpoint,
            api_key=self.config.azure_api_key,
            api_version=self.config.azure_api_version
        )
        self.logger.info(f"Azure OpenAI client initialized for model {self.config.azure_deployment_name}")
    
    def generate(self, prompt: str) -> str:
        """Generate response using Azure OpenAI."""
        try:
            if not hasattr(self, 'client') or self.client is None:
                raise RuntimeError("Azure OpenAI client not initialized")
                
            response = self.client.chat.completions.create(
                model=self.config.azure_deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content or ""
        
        except Exception as e:
            self.logger.error(f"Azure OpenAI generation failed: {e}")
            return f"Error: {str(e)}"
    
    def generate_structured(self, prompt: str, schema: Type['BaseModel']) -> str:
        """Generate structured response using OpenAI's structured output feature."""
        try:
            if not hasattr(self, 'client') or self.client is None:
                raise RuntimeError("Azure OpenAI client not initialized")
            
            response = self.client.beta.chat.completions.parse(
                model=self.config.azure_deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format=schema
            )
            
            response_content = response.choices[0].message.content or ""
            if not response_content:
                raise ValueError("Empty response from OpenAI")
            
            return response_content
        
        except Exception as e:
            self.logger.error(f"Azure OpenAI structured generation failed: {e}")
            # Fallback to regular generation and parsing
            self.logger.info("Falling back to regular generation with JSON parsing")
            return super().generate_structured(prompt, schema)


class LocalOpenAILLM(BaseLLM):
    """Local OpenAI-compatible LLM implementation (Ollama, vLLM, etc.)."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.LocalOpenAILLM')
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the local LLM client."""
        if not OPENAI_AVAILABLE or OpenAI is None:
            raise ImportError("openai package required for local LLM. Install with: pip install openai")
        
        self.client = OpenAI(
            base_url=self.config.local_endpoint,
            api_key=self.config.local_api_key or "not-needed"  # Many local APIs don't need real keys
        )
        self.logger.info(f"Local OpenAI client initialized at {self.config.local_endpoint} with model {self.config.local_model}")
    
    def generate(self, prompt: str) -> str:
        """Generate response using local OpenAI-compatible endpoint."""
        try:
            if not hasattr(self, 'client') or self.client is None:
                raise RuntimeError("Local OpenAI client not initialized")
                
            response = self.client.chat.completions.create(
                model=self.config.local_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content or ""
        
        except Exception as e:
            self.logger.error(f"Local LLM generation failed: {e}")
            return f"Error: {str(e)}"
    
    def generate_structured(self, prompt: str, schema: Type['BaseModel']) -> str:
        """Generate structured response using OpenAI's structured output feature."""
        try:
            if not hasattr(self, 'client') or self.client is None:
                raise RuntimeError("Local OpenAI client not initialized")
            
            response = self.client.beta.chat.completions.parse(
                model=self.config.local_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format=schema
            )
            
            response_content = response.choices[0].message.content or ""
            if not response_content:
                raise ValueError("Empty response from local LLM")
            
            return response_content
        
        except Exception as e:
            self.logger.error(f"Local LLM structured generation failed: {e}")
            # Fallback to regular generation and parsing
            self.logger.info("Falling back to regular generation with JSON parsing")
            return super().generate_structured(prompt, schema)


class MockLLM(BaseLLM):
    """Mock LLM for testing purposes."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.MockLLM')
    
    def generate(self, prompt: str) -> str:
        """Generate mock responses based on prompt content."""
        prompt_lower = prompt.lower()
        
        if "query classifier" in prompt_lower or "thought:" in prompt_lower:
            # Handle ReAct-style classification prompt
            if "query:" in prompt_lower:
                query_part = prompt.split("Query:")[-1].strip().lower()
                
                # More sophisticated classification using ReAct format
                conversational_patterns = [
                    "what is", "what are", "who is", "explain", "define", 
                    "tell me about", "capital of", "difference between",
                    "how does", "why is", "when did", "where is"
                ]
                
                tool_patterns = [
                    "get", "fetch", "retrieve", "download", "send", "create", 
                    "read", "write", "execute", "weather", "file", "email", 
                    "search", "find", "look up", "check", "analyze", "process"
                ]
                
                # Return ReAct-style response
                if any(pattern in query_part for pattern in tool_patterns):
                    return """Thought: The user is requesting an action that requires external tools or services to complete. They want to perform a specific task rather than just get information.

Action: Classify as TOOLS

Observation: The query contains action words that indicate the need for tool usage to accomplish the requested task.

Answer: TOOLS"""
                else:
                    return """Thought: The user is asking for general information or knowledge that can be answered conversationally without requiring external tools.

Action: Classify as CONVERSATIONAL

Observation: The query is seeking knowledge or explanations that don't require specific tool execution.

Answer: CONVERSATIONAL"""
            
            # Fallback for non-standard prompts
            return "CONVERSATIONAL"
        
        elif "tool_assistant" in prompt_lower:
            # Generate appropriate tool requests based on query content
            if "weather" in prompt_lower:
                return """<tool_assistant>
server: weather service providing meteorological data and forecasts
tool: get current weather conditions and forecasts for specified locations
</tool_assistant>"""
            elif "email" in prompt_lower or "send" in prompt_lower:
                return """<tool_assistant>
server: communication and messaging service
tool: send email messages to specified recipients
</tool_assistant>"""
            elif "file" in prompt_lower or "read" in prompt_lower:
                return """<tool_assistant>
server: file system management service
tool: read and manage files and documents
</tool_assistant>"""
            elif "search" in prompt_lower or "web" in prompt_lower:
                return """<tool_assistant>
server: web search and information retrieval service
tool: search web content and return relevant results
</tool_assistant>"""
            elif "database" in prompt_lower or "sql" in prompt_lower:
                return """<tool_assistant>
server: database management and query service
tool: execute database queries and manage data
</tool_assistant>"""
            else:
                return """<tool_assistant>
server: general purpose utility service
tool: process user request with appropriate functionality
</tool_assistant>"""
        
        elif "select" in prompt_lower and "json" in prompt_lower:
            # Tool selection - parse the available tools and select the most relevant one
            import json
            import re
            
            # Extract the query from the prompt
            query_match = re.search(r'Query:\s*(.+?)(?=\n|$)', prompt, re.IGNORECASE)
            query = query_match.group(1).strip() if query_match else ""
            
            # Extract tool candidates from the prompt
            tools_section = re.search(r'Available tools:(.+?)(?=Select the most|$)', prompt, re.DOTALL | re.IGNORECASE)
            if tools_section:
                tools_text = tools_section.group(1).strip()
                
                # Parse tools (looking for tool entries)
                tool_entries = re.findall(r'(\d+)\.\s*(\w+.*?)(?=\n\d+\.|$)', tools_text, re.DOTALL)
                
                if tool_entries:
                    # Simple matching based on query keywords
                    query_lower = query.lower()
                    best_tool = tool_entries[0]  # Default to first tool
                    
                    # Try to find a better match
                    for entry in tool_entries:
                        idx, tool_info = entry
                        tool_info_lower = tool_info.lower()
                        
                        # Simple keyword matching
                        if ("weather" in query_lower and "weather" in tool_info_lower) or \
                           ("file" in query_lower and "file" in tool_info_lower) or \
                           ("email" in query_lower and ("email" in tool_info_lower or "mail" in tool_info_lower)) or \
                           ("search" in query_lower and "search" in tool_info_lower) or \
                           ("database" in query_lower and ("database" in tool_info_lower or "sql" in tool_info_lower)):
                            best_tool = entry
                            break
                    
                    idx, tool_info = best_tool
                    # Extract tool name from tool_info
                    tool_name_match = re.search(r'^(\w+)', tool_info.strip())
                    tool_name = tool_name_match.group(1) if tool_name_match else "selected_tool"
                    
                    result = {
                        "tool_name": tool_name,
                        "server": "mock_server",
                        "arguments": {},
                        "reasoning": f"Selected {tool_name} as it best matches the query requirements"
                    }
                    return json.dumps(result, indent=2)
            
            # Fallback if no tools found
            return '''{"tool_name": "none", "server": "none", "arguments": {}, "reasoning": "No specific tools found, using fallback selection"}'''
        
        else:
            # Conversational response
            return "I can help you with that! This is a response based on my knowledge."
    
    def generate_structured(self, prompt: str, schema: Type['BaseModel']) -> str:
        """Generate structured response for testing - parse regular response into schema."""
        # Generate the regular response
        response = self.generate(prompt)
        
        # Try to parse it into the schema
        try:
            return self._parse_response_to_schema(response, schema)
        except Exception as e:
            self.logger.warning(f"MockLLM failed to parse response into schema: {e}")
            # Generate a minimal valid response for the schema
            return self._generate_fallback_json(schema)
    
    def _generate_fallback_json(self, schema: Type['BaseModel']) -> str:
        """Generate a minimal valid JSON response for the given schema."""
        # Import here to avoid circular import
        from .schemas import ToolSelection
        
        if schema == ToolSelection:
            return '{"tool_name": "mock_tool", "server": "mock_server", "arguments": {}, "reasoning": "Mock response generated for testing purposes"}'
        else:
            # For other schemas, try to create minimal JSON
            return '{"reasoning": "Mock response for testing"}'


def create_llm(llm_config: LLMConfig) -> BaseLLM:
    """Factory function to create LLM from LLMConfig."""
    if llm_config.llm_type == "azure_openai":
        return AzureOpenAILLM(llm_config)
    elif llm_config.llm_type == "local_openai":
        return LocalOpenAILLM(llm_config)
    elif llm_config.llm_type == "mock":
        return MockLLM()
    else:
        raise ValueError(f"Unsupported LLM type: {llm_config.llm_type}. Supported types: azure_openai, local_openai, mock")


def create_multiple_llms(config: OrchestratorConfig) -> Dict[str, BaseLLM]:
    """Create multiple LLMs based on configuration."""
    llms = {}
    for role, llm_config in config.llms.items():
        llms[role] = create_llm(llm_config)
    
    return llms


# Export both factory functions for compatibility
__all__ = [
    'LLMConfig', 'OrchestratorConfig', 'BaseLLM', 'MockLLM', 
    'AzureOpenAILLM', 'LocalOpenAILLM', 'setup_colored_logging',
    'create_llm', 'create_multiple_llms',
    'CLASSIFICATION_PROMPT', 'TOOL_SELECTION_PROMPT'
]


# Prompt templates for classification, tool transformation, and selection
CLASSIFICATION_PROMPT = """You are a query classifier. Your task is to determine if a user query requires external tools/actions or can be answered using your existing knowledge.

Use the ReAct framework to analyze the query step by step:

**Thought Process:**
1. **Analyze**: What is the user asking for?
2. **Consider**: Does this require real-time data, external actions, or access to systems/files?
3. **Evaluate**: Can I answer this with my training knowledge alone?
4. **Decide**: Tools needed or conversational response?

**Classification Rules:**

**TOOLS Required** when the query involves:
- Real-time or current data (weather, stock prices, news)
- File system operations (read, write, create, delete files)
- External services (send email, make API calls, web searches)
- System operations (execute commands, manage processes)
- Database operations (queries, updates)
- Communication actions (send messages, notifications)
- Data retrieval from specific sources
- Actions that modify external state

**CONVERSATIONAL Response** when the query involves:
- General knowledge questions
- Explanations of concepts, theories, or how things work
- Creative tasks (writing, brainstorming, jokes)
- Analysis or comparison of known information
- Mathematical calculations or logical reasoning
- Advice or recommendations based on general principles
- Historical facts or established information

**Examples:**

TOOLS:
- "Get the current weather in Tokyo" (real-time data)
- "Read my notes.txt file" (file system access)
- "Send an email to John about the meeting" (external action)
- "Search for the latest Python tutorials" (web search)
- "Download the sales report from the server" (external data retrieval)
- "Execute ls command in the terminal" (system operation)
- "Get the current Bitcoin price" (real-time financial data)

CONVERSATIONAL:
- "What is the capital of France?" (general knowledge)
- "Explain quantum physics" (concept explanation)
- "How does machine learning work?" (educational content)
- "Tell me a joke" (creative content)
- "What are the benefits of exercise?" (general advice)
- "Compare Python and JavaScript" (known information comparison)
- "Calculate 15% of 200" (mathematical operation)

**Analysis Process:**
Query: "{query}"

Thought: Let me analyze this query step by step.
- What is being asked? [Identify the core request]
- Does this require external data or actions? [Yes/No reasoning]
- Can I answer this with my existing knowledge? [Yes/No reasoning]
- Final decision: [TOOLS or CONVERSATIONAL]

Action: Provide only the classification result.

Respond with only: TOOLS or CONVERSATIONAL"""


TOOL_SELECTION_PROMPT = """You are an AI assistant. Your task is to select the most appropriate tool from the list below based on the user's original query. 

IMPORTANT:
- Your response MUST be a **single valid JSON object**.
- Do NOT include any text, explanations, markdown code blocks, or commentary outside the JSON.
- Do NOT add trailing commas, comments, or any non-JSON content.
- Use double quotes for all keys and string values.
- Follow the schema exactly as shown below. Do NOT add or remove keys.
- For "server", use the exact "Server Name" from the tool list, NOT the "Server Description".

Expected JSON schema:
{{
    "tool_name": "<selected_tool_name>",
    "server": "<selected_server_name>",
    "arguments_kv": [
        {{"key": "<argument_name>", "value_json": "<json_encoded_value>"}},
        {{"key": "<argument_name>", "value_json": "<json_encoded_value>"}}
    ],
    "reasoning": "<brief explanation>"
}}

Instructions:
- If a suitable tool exists, fill in "tool_name" with the exact tool name and "server" with the exact server name.
- For "arguments_kv", provide a list of key-value pairs where each value is JSON-encoded:
  * For strings: "value_json": "\\"text\\""
  * For numbers: "value_json": "42"
  * For booleans: "value_json": "true"
  * For objects: "value_json": "{{\\"key\\": \\"value\\"}}"
  * For arrays: "value_json": "[1, 2, 3]"
- If no arguments are needed, use an empty array: "arguments_kv": []
- If no tool is suitable, set "tool_name" and "server" to "no", use empty arguments_kv, and explain why in "reasoning".
- Do not include any text outside the JSON object.
- Do not add comments or formatting outside the JSON.

Original query: {query}

Available tools:
{tools_list}
"""



QUERY_DECOMPOSITION_PROMPT = """
You are an expert at breaking down a complex user query into a list of self-contained, actionable, and tool-callable sub-tasks.
Your response must be a JSON object with a single key "sub_queries", which contains a list of strings.

- Each sub-task in the list must be a direct command that a tool can execute.
- Do NOT create sub-tasks for asking purely informational questions like "what is..." or "how does...".
- Information about the method, format, or constraints (e.g., "using API X", "in JSON format") must be kept within the main actionable sub-task.
- If the user's query is already a single actionable command, return it as a single-item list.

---
**Example 1: Complex Query**
User Query: "What's the weather like in New York tomorrow, and can you also find me a top-rated Italian restaurant near Times Square?"
Your Response:
{{"sub_queries": ["Get the weather forecast for tomorrow in New York City", "Find a top-rated Italian restaurant near Times Square"]}}

---
**Example 2: Simple Actionable Query**
User Query: "Can you please calculate the NPV for my project?"
Your Response:
{{"sub_queries": ["Calculate the NPV for the project"]}}

---
**Example 3: Tricky Query with Method/Constraint (VERY IMPORTANT)**
User Query: "Can you help me create floor plan views in Autodesk Revit for levels 1 and 2 using the JSON-RPC 2.0 method?"
Your Response:
{{"sub_queries": ["Create floor plan views in Autodesk Revit for levels 1 and 2 using the JSON-RPC 2.0 method"]}}
---

Now, decompose the following user query.

**User Query:** "{user_question}"
**Your Response:**
"""