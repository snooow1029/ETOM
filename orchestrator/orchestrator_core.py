"""
BaseOrchestrator - Abstract Base Class

Provides common infrastructure for orchestrator implementations.
Concrete orchestrators inherit from this and implement process_query.

Architecture:
- Handles LLM initialization and management
- Provides MCP data loading and processing
- Offers role-based LLM calling interface
- Defines abstract process_query interface

Usage:
    from orchestrator import BaseOrchestrator
    
    class MyOrchestrator(BaseOrchestrator):
        def process_query(self, query: str) -> str:
            # Implement your orchestration logic
            return self._call_llm(query, "conversational")
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, TYPE_CHECKING, Union, overload
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from pydantic import BaseModel

# Import utilities from separate module
from .orchestrator_utils import (
    OrchestratorConfig, 
    LLMConfig,
    create_llm, 
    create_multiple_llms,
    setup_colored_logging
)


class BaseOrchestrator(ABC):
    """
    Abstract base class for orchestrator implementations.
    
    Provides common infrastructure:
    - LLM initialization and management
    - MCP data loading and processing
    - Role-based LLM calling
    - Status reporting
    
    Subclasses must implement process_query().
    """

    def __init__(self,
                 config: OrchestratorConfig,
                 mcp_data_path: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize the orchestrator.
        
        Args:
            config: OrchestratorConfig object with all settings
            mcp_data_path: Path to JSON file containing MCP server data (overrides config)
            embedding_model_name: Sentence transformer model name (overrides config)
            **kwargs: Additional parameters that override config.orchestrator_params
        """
        # Setup colored logging
        setup_colored_logging()
        
        # Setup logging with the actual class's module name
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        
        # Store config and create LLMs
        self.config = config
        
        # Role-based LLM management
        if config.llms is not None and len(config.llms) > 0:
            # Multi-LLM configuration
            self.llms = create_multiple_llms(config)
            
            # Validate required LLM roles for this orchestrator
            self._validate_required_llms()
        else:
            # No LLMs configured
            raise ValueError("No LLM configurations provided in OrchestratorConfig.llms")
        
        # Use config values, but allow parameter overrides
        self.server_top_k = kwargs.get('server_top_k', config.get_param('server_top_k', 3))
        self.tool_top_k = kwargs.get('tool_top_k', config.get_param('tool_top_k', 5))
        self.similarity_threshold = kwargs.get('similarity_threshold', config.get_param('similarity_threshold', 0.3))
        
        # Determine paths and models
        mcp_data_path = mcp_data_path or config.mcp_data_path
        embedding_model_name = embedding_model_name or config.embedding_model_name
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device='cuda')
        except Exception as e:
            self.logger.warning(f"Could not load embedding model on GPU (cuda): {e}. Falling back to CPU.")
            self.embedding_model = SentenceTransformer(embedding_model_name, device='cpu')
        
        # Load and process MCP data
        self.servers_data = self._load_mcp_data(mcp_data_path)
    
    def get_required_llm_roles(self) -> List[str]:
        """
        Return list of required LLM roles for this orchestrator.
        Subclasses should override this to specify their requirements.
        """
        return []  # Base class has no specific requirements
    
    def _validate_required_llms(self):
        """Validate that all required LLM roles are configured."""
        required = self.get_required_llm_roles()
        missing = [role for role in required if role not in self.llms]
        if missing:
            raise ValueError(f"Missing required LLM roles for {self.__class__.__name__}: {missing}")
        
        self.logger.info(f"Validated LLM roles: {list(self.llms.keys())}")
        
    def _call_llm(self, prompt: str, llm_role: str = "router", response_schema: Optional[Type['BaseModel']] = None) -> str:
        """
        Helper method to call specific LLM whether sync or async.
        
        Args:
            prompt: The prompt to send to the LLM
            llm_role: Which LLM role to use (any role defined in config.llms)
            response_schema: Optional Pydantic schema for structured output
            
        Returns:
            String response (JSON format when schema is provided)
        """
        # Get the appropriate LLM for the role
        if llm_role not in self.llms:
            # Fallback to first available LLM with warning
            available_roles = list(self.llms.keys())
            fallback_role = available_roles[0] if available_roles else None
            if fallback_role:
                self.logger.warning(f"LLM role '{llm_role}' not found. Using '{fallback_role}' as fallback.")
                llm = self.llms[fallback_role]
            else:
                raise ValueError(f"No LLMs configured and role '{llm_role}' not found")
        else:
            llm = self.llms[llm_role]

        try:
            # Use structured output if schema provided
            if response_schema:
                if hasattr(llm, 'generate_structured'):
                    return llm.generate_structured(prompt, response_schema)
                else:
                    # Fallback: regular generation then parse
                    response = llm.generate(prompt)
                    return llm._parse_response_to_schema(response, response_schema)
            
            # Regular generation
            if hasattr(llm, 'generate'):
                return llm.generate(prompt)
            elif hasattr(llm, 'generate_async'):
                # Try to detect if we're in an async context
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                    # We're in an async context - this is problematic for sync interface
                    # For now, return a default response
                    self.logger.warning("Cannot call async LLM from sync context")
                    return "Unable to process request due to async/sync mismatch"
                except RuntimeError:
                    # No running loop, safe to use asyncio.run()
                    import asyncio
                    return asyncio.run(llm.generate_async(prompt))
            else:
                raise ValueError("LLM must have either 'generate' or 'generate_async' method")
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return f"Error: {str(e)}"
        
    @abstractmethod
    def process_query(self, query: str) -> str:
        """
        Abstract method for processing user queries.
        
        Must be implemented by concrete orchestrator subclasses.
        
        Args:
            query: User's input query
            
        Returns:
            Response string (either conversational or tool call JSON)
        """
        pass
    
    def _load_mcp_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load MCP server data and compute embeddings if needed."""
        self.logger.info(f"Loading MCP data from: {data_path}")
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load MCP data: {e}")
            return []

        # Handle different file structures
        if isinstance(data, dict) and 'servers' in data:
            # New consolidated format
            servers = data['servers']
            self.logger.info(f"Loaded consolidated format with {len(servers)} servers")
        elif isinstance(data, list):
            # Old direct list format
            servers = data
            self.logger.info(f"Loaded list format with {len(servers)} servers")
        else:
            raise ValueError(f"Unsupported data format in {data_path}")
        
        # Process servers and tools
        for server in servers:
            # Handle different field naming conventions
            server_name = server.get('server_name') or server.get('name', 'Unknown')
            server_desc = server.get('server_description') or server.get('description', '')
            server_category = server.get('categories', []) or server.get('category', '')
            
            # Create server description text for embedding
            if isinstance(server_category, list):
                category_text = ' '.join(server_category)
            else:
                category_text = str(server_category)
            
            server_text = f"{server_desc} {category_text}".strip()
            
            # Use existing embedding if available, otherwise compute it
            if 'server_embedding' in server:
                self.logger.debug(f"Using existing server embedding for {server_name}")
            else:
                self.logger.debug(f"Computing server embedding for {server_name}")
                server['server_embedding'] = self.embedding_model.encode(server_text, show_progress_bar=False)
            
            # Process tools
            tools = server.get('tools', [])
            for tool in tools:
                tool_name = tool.get('tool_name') or tool.get('name', 'Unknown')
                tool_desc = tool.get('tool_description') or tool.get('description', '')
                
                tool_text = f"{tool_name} {tool_desc}".strip()
                
                # Use existing embedding if available, otherwise compute it
                if 'tool_embedding' in tool:
                    self.logger.debug(f"Using existing tool embedding for {tool_name}")
                else:
                    self.logger.debug(f"Computing tool embedding for {tool_name}")
                    tool['tool_embedding'] = self.embedding_model.encode(tool_text, show_progress_bar=False)
                
                # Standardize field names for consistency
                tool['name'] = tool_name
                tool['description'] = tool_desc
                tool['schema'] = tool.get('input_schema') or tool.get('schema', {})
            
            # Standardize server field names
            server['name'] = server_name
            server['description'] = server_desc
        
        self.logger.info(f"Processed {len(servers)} servers with embeddings")
        return servers
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status information."""
        llm_info = {}
        for role, llm in self.llms.items():
            llm_info[role] = type(llm).__name__
        
        return {
            'servers_loaded': len(self.servers_data),
            'total_tools': sum(len(server.get('tools', [])) for server in self.servers_data),
            'config': {
                'server_top_k': self.server_top_k,
                'tool_top_k': self.tool_top_k,
                'similarity_threshold': self.similarity_threshold,
                'embedding_model': self.embedding_model.get_sentence_embedding_dimension(),
                'mcp_data_path': self.config.mcp_data_path
            },
            'llm_instances': llm_info
        }


# Export both the base class and concrete implementation
__all__ = ['BaseOrchestrator']
