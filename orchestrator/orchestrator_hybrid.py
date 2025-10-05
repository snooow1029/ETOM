"""
HybridOrchestrator - Advanced RAG-Tool Fusion Implementation

Implements the Hybrid methodology for enhanced tool retrieval using:
- Pre-retrieval: Pre-enhanced tool documents with hypothetical questions and key topics
- Intra-retrieval: Query rewriting, expansion, and multi-query retrieval
- Post-retrieval: LLM-based reranking for optimal tool selection

Based on the Hybrid paper's Advanced RAG-Tool Fusion approach.

Note: Tool document enhancement is performed during data preparation phase,
not at runtime for better performance.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .orchestrator_core import BaseOrchestrator
from .orchestrator_utils import (
    CLASSIFICATION_PROMPT, 
    # TOOL_TRANSFORMATION_PROMPT, 
    TOOL_SELECTION_PROMPT,
    QUERY_DECOMPOSITION_PROMPT,
    setup_colored_logging
)


class HybridOrchestrator(BaseOrchestrator):
    """
    Hybrid-based orchestrator implementing Advanced RAG-Tool Fusion.
    
    Key differences from MCPZeroOrchestrator:
    1. Query rewriting and expansion for better retrieval
    2. Multi-query retrieval with different query variations
    3. LLM-based reranking for final tool selection
    
    Uses the same MCP data format as MCPZeroOrchestrator:
    - tool['name'], tool['description'], tool['tool_embedding']
    - server['name'], server['description'], server['server_embedding']
    """

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
    
    # Hybrid-specific prompts based on paper examples
    QUERY_EXPANSION_PROMPT = """You are an expert at converting user questions to {num_variations} sentence variations that target different keywords and nuanced approaches with the goal to embed this query in a vector database to retrieve relevant tools across various industries.

Your goal is to craft {num_variations} nuanced sentence variations that target different aspects of understanding or solving the query.
For example, one sentence could focus on a detailed aspect of the user query, while another is more broad to cover more ground when embedding these sentences to retrieve the most relevant tools for the user query.

Before you start, understand this from a practical standpoint: The user question can be matched to a range of tools or solutions within the system, and your crafted variations should optimize for breadth and specificity. Write out your approach and plan for tackling this, then provide the {num_variations} sentences you would craft for the user question.

Think through your approach step by step, be intelligent, take a deep breath.

USER QUESTION: {user_question}

YOUR APPROACH, REASONING, AND {num_variations} SENTENCES:"""

    RERANK_PROMPT = """OK here are the results:
USER QUESTION EMBEDDED AND RETRIEVED TOOLS:
{user_question_results}
{variation_results}
===================
Based on these results, rank the top {top_k} most relevant tools to solve the user question. Just return the {top_k} tool names for each relevant tool.

Return as a JSON list: ["tool_name_1", "tool_name_2", ...]"""

    def __init__(self, config, **kwargs):
        """Initialize Hybrid orchestrator with pre-enhanced tool embeddings."""
        super().__init__(config, **kwargs)
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Hybrid-specific configurations
        self.query_expansion_count = kwargs.get('query_expansion_count', config.get_param('query_expansion_count', 3))
        self.rerank_top_k = kwargs.get('rerank_top_k', config.get_param('rerank_top_k', 10))
        
        self.logger.info("HybridOrchestrator initialized with pre-enhanced tool embeddings")
    
    def get_required_llm_roles(self) -> List[str]:
        """Return required LLM roles for Hybrid orchestrator."""
        return ["router", "query_rewriter", "query_expander", "reranker", "conversational", "decomposer"]
    
    def process_query(self, query: str) -> str:
        """
        Process user query using Hybrid Advanced RAG-Tool Fusion methodology.
        
        Flow:
        1. Query classification (TOOLS vs CONVERSATIONAL)
        2. If TOOLS: Hybrid retrieval pipeline
        3. Final tool selection or conversational response
        """
        self.logger.info(f"Processing query: {query}")
        
        # Step 1: Query Classification
        classification = self._classify_query(query)
        self.logger.info(f"Query classification: {classification}")
        
        if classification == "CONVERSATIONAL":
            # Handle conversational queries
            return self._handle_conversational_query(query)

        # Step 1: Query rewriting (cleaning and normalization)
        rewritten_query = self._transform_to_tool_request(query)
        self.logger.info(f"Rewritten query: {rewritten_query}")
        
        queries = self._decompose_query(rewritten_query)
        # Step 2: Hybrid Tool Retrieval Pipeline
        answer_tool_chains = []
        for q in queries:
            candidate_tools = self._retrieve_tools_hybrid(q)
            self.logger.info(f"Retrieved {len(candidate_tools)} candidate tools")
            
            if not candidate_tools:
                return "No suitable tools found for your request."
            # Step 3: Final Tool Selection
            
            answer_tool_chains.append(self._select_final_tool(q, candidate_tools))
        # print(self._select_final_tool(q, candidate_tools))
        # print(f"Final tool selection for query '{q}': {answer_tool_chains}")
        #print(answer_tool_chains)
        return answer_tool_chains
    
    def _classify_query(self, query: str) -> str:
        """Classify query as TOOLS or CONVERSATIONAL (same as MCP-Zero)."""
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        response = self._call_llm(prompt, "router")
        self.logger.info(f"Classification response: {response}")
        # Extract classification from response
        if "TOOLS" in response.upper():
            return "TOOLS"
        elif "CONVERSATIONAL" in response.upper():
            return "CONVERSATIONAL"
        else:
            # Default to TOOLS if unclear
            return "TOOLS"
    
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

    def _retrieve_tools_hybrid(self, query: str) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval pipeline:
        1. Query rewriting (cleaning and normalization)
        2. Query expansion (create multiple variations)
        3. Multi-query retrieval
        4. LLM-based reranking
        """
        # # Step 1: Query rewriting (cleaning and normalization)
        # rewritten_query = self._transform_to_tool_request(query)
        # self.logger.info(f"Rewritten query: {rewritten_query}")
        
        # Step 2: Query expansion (create multiple variations)
        query_variations = self._expand_query(query)
        self.logger.info(f"Generated {len(query_variations)} query variations")
        
        # Step 3: Multi-query retrieval
        all_retrieved_tools = []
        original_results = self._retrieve_tools_for_query(query)
        all_retrieved_tools.extend(original_results)
        
        variation_results = {}
        for i, variation in enumerate(query_variations):
            tools = self._retrieve_tools_for_query(variation)
            all_retrieved_tools.extend(tools)
            variation_results[f"sentence_{i+1}"] = tools
        
        # Remove duplicates based on tool_name + server_name
        unique_tools = self._remove_duplicate_tools(all_retrieved_tools)
        self.logger.info(f"Retrieved {len(unique_tools)} unique tools from all variations")
        
        # Step 4: LLM-based reranking with context from all retrievals
        if len(unique_tools) > self.tool_top_k:
            reranked_tools = self._rerank_tools_with_context(
                unique_tools, query, original_results, variation_results
            )
            return reranked_tools[:self.tool_top_k]
        
        return unique_tools
    
    def _transform_to_tool_request(self, query: str) -> str:
        """Transform user query into structured tool request."""
        prompt = self.TOOL_TRANSFORMATION_PROMPT.format(query=query)
        
        try:
            response = self._call_llm(prompt, "retriever")
            
            # Extract tool_assistant content
            tool_content = self._extract_tool_assistant_content(response)
            return tool_content
            
        except Exception as e:
            self.logger.error(f"Tool transformation failed: {e}")
            return query  # Fallback to original query if transformation fails
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Step 2: Query expansion to generate multiple variations.
        Uses query_expander LLM to create diverse approaches for better tool retrieval.
        """
        prompt = self.QUERY_EXPANSION_PROMPT.format(
            num_variations=self.query_expansion_count,
            user_question=query
        )
        
        response = self._call_llm(prompt, "query_expander")
        
        # Parse variations from response
        variations = []
        lines = response.strip().split('\n')
        
        # Look for numbered sentences in the response
        current_sentence = ""
        for line in lines:
            line = line.strip()
            # Look for numbered variations (1., 2., 3., etc.)
            if re.match(r'^\d+\.?\s+', line):
                # If we have a previous sentence, add it
                if current_sentence:
                    variations.append(current_sentence.strip())
                # Start new sentence
                current_sentence = re.sub(r'^\d+\.?\s+', '', line).strip()
            elif current_sentence and line:
                # Continue building the current sentence
                current_sentence += " " + line
        
        # Add the last sentence if it exists
        if current_sentence:
            variations.append(current_sentence.strip())
        
        # Clean up variations and remove empty ones
        variations = [v for v in variations if v and len(v) > 10]  # Filter out very short variations
        
        # Limit to configured count
        return variations[:self.query_expansion_count]
    
    def _retrieve_tools_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve tools for a single query using tool embeddings (same as MCP-Zero)."""
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
        candidate_tools = []
        
        for server in self.servers_data:
            server_name = server.get('name', 'Unknown')
            server_desc = server.get('description', '')
            
            for tool in server.get('tools', []):
                try:
                    # Use tool_embedding (same as MCP-Zero)
                    tool_embedding = tool.get('tool_embedding')
                    if tool_embedding is None:
                        self.logger.warning(f"No tool embedding for tool {tool.get('name', 'Unknown')}")
                        continue
                    
                    # Calculate similarity
                    similarity = cosine_similarity(
                        np.array(query_embedding).reshape(1, -1), 
                        np.array(tool_embedding).reshape(1, -1)
                    )[0][0]
                    
                    if similarity >= self.similarity_threshold:
                        candidate_tools.append({
                            'tool_name': tool.get('name', ''),
                            'tool_description': tool.get('description', ''),
                            'server_name': server_name,
                            'server_description': server_desc,
                            'tool_schema': tool.get('schema', {}),
                            'similarity': similarity,
                            'query_variation': query
                        })
                
                except Exception as e:
                    self.logger.warning(f"Error processing tool similarity: {e}")
        
        # Sort by similarity
        candidate_tools.sort(key=lambda x: x['similarity'], reverse=True)
        
        return candidate_tools[:self.rerank_top_k]  # Return more for reranking
    
    def _remove_duplicate_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tools based on tool_name + server_name combination."""
        seen = set()
        unique_tools = []
        
        for tool in tools:
            key = (tool['tool_name'], tool['server_name'])
            if key not in seen:
                seen.add(key)
                unique_tools.append(tool)
        
        return unique_tools
    
    def _rerank_tools_with_context(self, tools: List[Dict[str, Any]], original_query: str, 
                                  original_results: List[Dict[str, Any]], 
                                  variation_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Use LLM-based reranking with context from all query variations."""
        
        # Format original results
        user_question_results = self._format_tools_for_rerank(original_results[:5])
        
        # Format variation results
        variation_sections = ""
        for i in range(self.query_expansion_count):
            sentence_key = f"sentence_{i+1}"
            if sentence_key in variation_results:
                results = variation_results[sentence_key][:5]
                formatted = self._format_tools_for_rerank(results)
                variation_sections += f"SENTENCE {i+1} EMBEDDED AND RETRIEVED TOOLS:\n{formatted}\n"
        
        prompt = self.RERANK_PROMPT.format(
            user_question_results=user_question_results,
            variation_results=variation_sections,
            top_k=min(self.tool_top_k, len(tools))
        )
        
        try:
            response = self._call_llm(prompt, "reranker")
            
            # Parse reranked tool names
            reranked_names = self._parse_rerank_names_response(response)
            
            # Reorder tools based on reranking by tool names
            reranked_tools = []
            used_tools = set()
            
            for tool_name in reranked_names:
                for tool in tools:
                    tool_key = (tool['tool_name'], tool['server_name'])
                    if tool['tool_name'] == tool_name and tool_key not in used_tools:
                        reranked_tools.append(tool)
                        used_tools.add(tool_key)
                        break
            
            # Add any tools that weren't included in reranking
            for tool in tools:
                tool_key = (tool['tool_name'], tool['server_name'])
                if tool_key not in used_tools:
                    reranked_tools.append(tool)
            
            self.logger.info(f"Reranked {len(reranked_tools)} tools using context")
            return reranked_tools
            
        except Exception as e:
            self.logger.warning(f"Context-based reranking failed: {e}. Using original order.")
            return tools
    
    def _format_tools_for_rerank(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for reranking prompt."""
        if not tools:
            return "No tools found."
        
        formatted = ""
        for tool in tools:
            formatted += f"- {tool['tool_name']} (Server: {tool['server_name']}): {tool['tool_description']}\n"
        return formatted.strip()
    
    def _parse_rerank_names_response(self, response: str) -> List[str]:
        """Parse tool names from reranking response."""
        try:
            # Try to extract JSON array
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                tool_names = json.loads(json_str)
                return [name for name in tool_names if isinstance(name, str)]
            
            # Fallback: look for tool names in the text
            tool_names = []
            lines = response.strip().split('\n')
            for line in lines:
                # Look for lines that might contain tool names
                if ':' in line or '-' in line:
                    # Extract potential tool name
                    parts = re.split(r'[:\-]', line)
                    if len(parts) > 1:
                        potential_name = parts[0].strip().strip('"\'')
                        if potential_name and len(potential_name) > 2:
                            tool_names.append(potential_name)
            
            return tool_names[:self.tool_top_k]
            
        except Exception as e:
            self.logger.warning(f"Failed to parse rerank names response: {e}")
            return []
    
    def _select_final_tool(self, query: str, candidate_tools: List[Dict[str, Any]]) -> str:
        """Select final tool from candidates (same logic as MCP-Zero)."""
        # Import here to avoid circular import
        from .schemas import ToolSelection, QueryDecomposition

        # Format tools for prompt
        tools_list = ""
        for i, tool in enumerate(candidate_tools, 1):
            tools_list += f"{i}. Tool: {tool['tool_name']}\n"
            tools_list += f"   Server Name: {tool['server_name']}\n"
            tools_list += f"   Description: {tool['tool_description']}\n"
            tools_list += f"   Server Description: {tool['server_description']}\n"
            tools_list += f"   Similarity: {tool['similarity']:.3f}\n\n"

        prompt = TOOL_SELECTION_PROMPT.format(
            query=query,
            tools_list=tools_list
        )

        try:
            # Use structured output for reliable JSON response
            selection = self._call_llm(prompt, "router", response_schema=ToolSelection)
            result = json.loads(selection)

            # Add schema information for selected tool
            for tool in candidate_tools:
                if tool['tool_name'] == result['tool_name']:
                    result['schema'] = tool['tool_schema']
                    break

            return json.dumps(result, indent=2)

        except Exception as e:
            self.logger.error(f"Tool selection failed: {e}")
            return f"Error in tool selection: {str(e)}"
    
    def _handle_conversational_query(self, query: str) -> str:
        """Handle conversational queries (same as MCP-Zero)."""
        #response = self._call_llm(f"Please provide a helpful response to: {query}", "conversational")
        
        # return response
        return ['{\n  "tool_name": "No",\n  "server": "No",\n  "arguments_kv": [],\n  "reasoning": "null"\n}']
    
    def get_status(self) -> Dict[str, Any]:
        """Get Hybrid orchestrator status."""
        base_status = super().get_status()
        
        # Add Hybrid-specific status
        base_status['orchestrator_type'] = 'Hybrid'
        base_status['hybrid_config'] = {
            'query_expansion_count': self.query_expansion_count,
            'rerank_top_k': self.rerank_top_k,
            'total_tools': sum(
                len(server.get('tools', []))
                for server in self.servers_data
            )
        }
        
        return base_status
    
    def _extract_tool_assistant_content(self, text: str) -> str:
        """Extract content from <tool_assistant> tags."""
        pattern = r'<tool_assistant>(.*?)</tool_assistant>'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return text
        
        content = match.group(1).strip()
        
        # Parse server and tool descriptions
        server_match = re.search(r'server:\s*(.+)', content, re.IGNORECASE)
        tool_match = re.search(r'tool:\s*(.+)', content, re.IGNORECASE)
        
        if server_match and tool_match:
            return tool_match.group(1).strip()
        
        return text


# Export the Hybrid orchestrator
__all__ = ['HybridOrchestrator']
