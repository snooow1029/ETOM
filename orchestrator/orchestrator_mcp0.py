"""
MCP-Zero Orchestrator Implementation

Implements the MCP-zero inspired orchestrator architecture by inheriting from BaseOrchestrator.

Architecture:
1. Router LLM determines if query needs tools or conversational response
2. If tools needed: Retriever LLM transforms query to tool request format
3. Embedding similarity search finds relevant tools (two-stage: server → tool)
4. Router LLM selects final tool and returns tool call schema

Usage:
    from orchestrator import MCPZeroOrchestrator, OrchestratorConfig, LLMConfig
    
    # Create multi-LLM config
    config = OrchestratorConfig.create_config(
        router_llm=LLMConfig(
            name="router",
            llm_type="azure_openai",
            azure_deployment_name="gpt-4"
        )
    )
    orchestrator = MCPZeroOrchestrator(config=config)
    
    result = orchestrator.process_query("What's the weather in Tokyo?")
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import base class and utilities
from .orchestrator_core import BaseOrchestrator
from .orchestrator_utils import (
    CLASSIFICATION_PROMPT,
    TOOL_SELECTION_PROMPT,
    QUERY_DECOMPOSITION_PROMPT,
)


class MCPZeroOrchestrator(BaseOrchestrator):
    """
    MCP-Zero implementation of orchestrator.
    
    Uses a 4-step process:
    1. Query classification (router LLM)
    2. Tool request transformation (retriever LLM)
    3. Embedding similarity search for tool discovery
    4. Final tool selection (router LLM)
    
    Required LLM roles:
    - router: Query classification and final tool selection
    - retriever: Tool retrieval and ranking
    - conversational: Conversational responses
    """
    
    def get_required_llm_roles(self) -> List[str]:
        """Return required LLM roles for MCP-Zero orchestrator."""
        return ["router", "retriever", "conversational", "decomposer"]
    
    # MCP-Zero specific prompt templates
    TOOL_TRANSFORMATION_PROMPT = """You are an AI assistant that helps users by analyzing their requests and identifying appropriate tools. Your task is to identify both the SERVER (platform/service domain) and the specific TOOL (operation type + target) that would best address the user's request.

When a user asks you to perform a task, you should:
1. Carefully read and understand the user's request
2. Identify the key requirements and intentions in the request
3. Determine the information of the server from user's request (SERVER)
4. Determine what specific tool from user's request (TOOL)
5. Respond using ONLY the following format:

<tool_assistant>
server: [brief description of the server/platform from user's request]
tool: [brief description of the specific tool from user's request]
</tool_assistant>

Your response should be concise but descriptive.
Remember to ONLY provide the server and tool descriptions within the <tool_assistant> tags. DO NOT provide any additional explanation or commentary outside the <tool_assistant> tags.

User request: {query}"""


    def process_query(self, query: str) -> str:
        """
        Main entry point. Process a user query and return response.
        
        Args:
            query: User's input query
            
        Returns:
            Either a conversational response or tool call JSON string
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Step 1: Router LLM - classify query
            classification = self._classify_query(query)
            self.logger.info(f"Query classification: {classification}")
            
            if classification == "CONVERSATIONAL":
                # Generate direct conversational response
                response = self._generate_conversational_response(query)
                return response
            
            # Step 1.5: decompose query
            querys = self._decompose_query(query)
            # Step 2: Retriever LLM - transform to tool request
            answer_tool_chains = []
            for query in querys:
                tool_request = self._transform_to_tool_request(query)
                self.logger.info(f"Tool request: {tool_request}")
            
                if not tool_request:
                    return "I couldn't understand what tools you need for this request."
                
                # Step 3: Embedding similarity - retrieve relevant tools
                candidate_tools = self._retrieve_tools(tool_request)
                self.logger.info(f"Retrieved {len(candidate_tools)} candidate tools")
                
                if not candidate_tools:
                    return "I couldn't find any suitable tools for your request."
                
                # Step 4: Router LLM - select final tool
                final_tool = self._select_final_tool(query, candidate_tools)
                answer_tool_chains.append(final_tool)
            # return final_tool
            return answer_tool_chains

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return f"I encountered an error processing your request: {str(e)}"
    
    def _classify_query(self, query: str) -> str:
        """Determine if query needs tools or conversational response."""
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            response = self._call_llm(prompt, "router")
            response = response.strip().upper()
            
            if "TOOLS" in response:
                return "TOOLS"
            elif "CONVERSATIONAL" in response:
                return "CONVERSATIONAL"
            else:
                # Default to conversational if unclear
                return "CONVERSATIONAL"
                
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return "CONVERSATIONAL"
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose query into multiple variations for retrieval."""
        # Import here to avoid circular import
        from .schemas import QueryDecomposition
        
        # Use query expansion prompt to generate variations
        prompt = QUERY_DECOMPOSITION_PROMPT.format(
            user_question=query
        )
        
        try:
            # Use structured output for reliable JSON response
            response = self._call_llm(prompt, "decomposer", response_schema=QueryDecomposition)
            data = json.loads(response)
            
            sub_queries = data.get("sub_queries", [])
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                self.logger.info(f"Decomposed query '{query}' into: {sub_queries}")
                return sub_queries
            else:
                self.logger.warning(f"LLM returned unexpected format for decomposition. Falling back.")
                # 如果格式不對，回傳原始查詢作為備案
                return [query]
                
        except Exception as e:
            self.logger.error(f"Error in query decomposition: {e}. Falling back to original query.")
            return [query]

    def _transform_to_tool_request(self, query: str) -> Optional[Dict[str, str]]:
        """Transform user query into structured tool request."""
        prompt = self.TOOL_TRANSFORMATION_PROMPT.format(query=query)
        
        try:
            response = self._call_llm(prompt, "retriever")
            
            # Extract tool_assistant content
            tool_content = self._extract_tool_assistant_content(response)
            return tool_content
            
        except Exception as e:
            self.logger.error(f"Tool transformation failed: {e}")
            return None
    
    def _retrieve_tools(self, tool_request: Dict[str, str]) -> List[Dict[str, Any]]:
        """Retrieve relevant tools using embedding similarity."""
        server_desc = tool_request.get('server', '')
        tool_desc = tool_request.get('tool', '')
        
        if not server_desc or not tool_desc:
            return []
        
        # Step 1: Find top-k servers by server description similarity
        server_embedding = self.embedding_model.encode([server_desc], show_progress_bar=False)
        server_similarities = []
        
        for server in self.servers_data:
            server_embed = server['server_embedding']
            similarity = cosine_similarity(server_embedding, np.array([server_embed]))[0][0]
            
            if similarity >= self.similarity_threshold:
                server_similarities.append((server, similarity))
        
        # Sort by similarity and take top-k
        server_similarities.sort(key=lambda x: x[1], reverse=True)
        top_servers = server_similarities[:self.server_top_k]
        
        # Step 2: Find top-k tools from top servers by tool description similarity
        tool_embedding = self.embedding_model.encode([tool_desc], show_progress_bar=False)
        candidate_tools = []
        
        for server, server_sim in top_servers:
            for tool in server['tools']:
                tool_embed = tool['tool_embedding']
                tool_similarity = cosine_similarity(tool_embedding, np.array([tool_embed]))[0][0]
                if tool_similarity >= self.similarity_threshold:
                    # New scoring formula:
                    # final_score = (server_sim * tool_similarity) * max(server_sim, tool_similarity)
                    final_score = tool_similarity
                    candidate_tools.append({
                        'tool_name': tool['name'],
                        'tool_description': tool['description'],
                        'tool_schema': tool.get('schema', {}),
                        'server_name': server['name'],
                        'server_description': server['description'],
                        'server_similarity': server_sim,
                        'tool_similarity': tool_similarity,
                        'combined_similarity': final_score
                    })
        # Sort by new combined similarity and take top-k
        candidate_tools.sort(key=lambda x: x['combined_similarity'], reverse=True)
        return candidate_tools[:self.tool_top_k]
    
    def _select_final_tool(self, original_query: str, candidate_tools: List[Dict[str, Any]]) -> str:
        """Select the most appropriate tool from candidates using structured output."""
        # Import the schema
        from .schemas import ToolSelection

        # Format tools for prompt
        tools_list = ""
        for i, tool in enumerate(candidate_tools, 1):
            tools_list += f"{i}. Tool: {tool['tool_name']}\n"
            tools_list += f"   Server Name: {tool['server_name']}\n"
            tools_list += f"   Description: {tool['tool_description']}\n"
            tools_list += f"   Server Description: {tool['server_description']}\n"
            tools_list += f"   Similarity: {tool['combined_similarity']:.3f}\n\n"

        prompt = TOOL_SELECTION_PROMPT.format(
            query=original_query,
            tools_list=tools_list
        )

        try:
            # Use structured output for reliable JSON response
            selection = self._call_llm(prompt, "router", response_schema=ToolSelection)

            result = json.loads(selection)

            for tool in candidate_tools:
                if tool['tool_name'] == result['tool_name']:
                    result['schema'] = tool['tool_schema']
                    break

            return json.dumps(result, indent=2)


        except Exception as e:
            self.logger.error(f"Structured tool selection failed: {e}")
            try:
                fallback_response = self._call_llm(prompt, "router")
                if isinstance(fallback_response, str):
                    return fallback_response
                else:
                    return "Tool selection failed and fallback did not return a string."
            except Exception as inner_e:
                self.logger.error(f"Fallback tool selection also failed: {inner_e}")
                return f"Error selecting tool: {str(inner_e)}"
    
    
    def _generate_conversational_response(self, query: str) -> str:
        """Generate a direct conversational response."""
        try:
            # return self._call_llm(query, "conversational")
            return ['{\n  "tool_name": "No",\n  "server": "No",\n  "arguments_kv": [],\n  "reasoning": "null"\n}']
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _extract_tool_assistant_content(self, text: str) -> Optional[Dict[str, str]]:
        """Extract content from <tool_assistant> tags."""
        pattern = r'<tool_assistant>(.*?)</tool_assistant>'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return None
        
        content = match.group(1).strip()
        
        # Parse server and tool descriptions
        server_match = re.search(r'server:\s*(.+)', content, re.IGNORECASE)
        tool_match = re.search(r'tool:\s*(.+)', content, re.IGNORECASE)
        
        if server_match and tool_match:
            return {
                'server': server_match.group(1).strip(),
                'tool': tool_match.group(1).strip()
            }
        
        return None
