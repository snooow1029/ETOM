"""
ReACT Orchestrator Implementation

Implements the ReACT inspired orchestrator architecture by inheriting from BaseOrchestrator.

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

# Import base class and utilities
from .orchestrator_core import BaseOrchestrator
from .orchestrator_utils import (
    CLASSIFICATION_PROMPT,
    TOOL_SELECTION_PROMPT,
    QUERY_DECOMPOSITION_PROMPT,
)


class REACTOrchestrator(BaseOrchestrator):
    """
    ReACT implementation of orchestrator with three-stage tool selection.

    Architecture:
    1. Query classification (router LLM): Determine if query needs tools or conversational response
    2. Server selection (router LLM): LLM selects top-k relevant servers from server pool based on descriptions
    3. Tool selection (router LLM): LLM selects top-k2 tools from selected servers based on descriptions
    4. Final tool selection (router LLM): Choose the best tool using structured LLM reasoning
    
    Required LLM roles:
    - router: Query classification, server selection, tool selection, and final tool selection
    - conversational: Conversational responses when no tools needed
    
    Configuration parameters:
    - server_top_k: Number of top servers to select in stage 2
    - tool_top_k: Number of top tools to select in stage 3 (k2)
    """
    
    def __init__(self,
                 config,
                 mcp_data_path=None,
                 **kwargs):
        """
        Initialize GPT-OSS orchestrator without embedding model.
        
        Args:
            config: OrchestratorConfig object with all settings
            mcp_data_path: Path to JSON file containing MCP server data (overrides config)
            **kwargs: Additional parameters that override config.orchestrator_params
        """
        # Store config and create LLMs
        self.config = config
        
        # Role-based LLM management
        if config.llms is not None and len(config.llms) > 0:
            # Multi-LLM configuration
            from .orchestrator_utils import create_multiple_llms
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
        
        # Determine MCP data path
        mcp_data_path = mcp_data_path or config.mcp_data_path
        
        # Skip embedding model initialization for GPT-OSS
        self.embedding_model = None
        logging.debug("REACT orchestrator: Skipping embedding model initialization (uses LLM-based selection)")
        
        # Load and process MCP data
        self.servers_data = self._load_mcp_data(mcp_data_path)
    
    def get_required_llm_roles(self) -> List[str]:
        """Return required LLM roles for GPT-OSS orchestrator."""
        return ["router", "conversational"]
    
    def _validate_required_llms(self):
        """Validate that all required LLM roles are configured."""
        required = self.get_required_llm_roles()
        missing = [role for role in required if role not in self.llms]
        if missing:
            raise ValueError(f"Missing required LLM roles for {self.__class__.__name__}: {missing}")
        logging.debug(f"All required LLM roles configured: {required}")
    
    def _load_mcp_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load MCP server data from JSON file."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                servers_data = data
            elif isinstance(data, dict) and 'servers' in data:
                servers_data = data['servers']
            else:
                raise ValueError(f"Invalid MCP data format in {data_path}")
            
            logging.debug(f"Loaded {len(servers_data)} MCP servers from {data_path}")
            return servers_data
        except Exception as e:
            logging.error(f"Failed to load MCP data from {data_path}: {e}")
            raise
    
    def _call_llm(self, prompt: str, llm_role: str, response_schema=None) -> str:
        """Call LLM with the given prompt and role."""
        if llm_role not in self.llms:
            raise ValueError(f"LLM role '{llm_role}' not configured")
        
        llm = self.llms[llm_role]
        
        if response_schema:
            return llm.generate_structured(prompt, response_schema)
        else:
            return llm.generate(prompt)

    # GPT-OSS specific prompt templates
    NEEDLE_SELECTION_PROMPT = """You are an AI assistant that helps users by analyzing their requests and identifying appropriate tools. Your task is to identify both the SERVER (platform/service domain) and the specific TOOL (operation type + target) that would best address the user's request.

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

    SERVER_SELECTION_PROMPT = """You are an AI assistant that helps users by analyzing their requests and selecting the most relevant platforms or services (servers) to fulfill the user's needs.

Given:
- A user request: {query}
- A list of available servers, each with a name and a brief description

Your task:
1. Carefully read and understand the user's request.
2. Review the list of servers and their descriptions.
3. Select the top-{top_k} servers that are most relevant to the user's request.
4. Respond using ONLY the following format:

<server_selection>
["server_name_1", "server_name_2", ..., "server_name_{top_k}"]
</server_selection>

Your response should be a JSON array of server names, ordered by relevance (most relevant first).
Remember to ONLY provide the JSON array within the <server_selection> tags. DO NOT provide any additional explanation or commentary outside the tags.

User request: {query}

Available servers:
{server_list}"""

    TOOL_SELECTION_FROM_SERVERS_PROMPT = """You are an AI assistant that helps users by analyzing their requests and selecting the single most relevant tool from a given set of servers to fulfill the user's needs.

Given:
- A user request: {query}
- A list of tools from selected servers, each with server name, tool name, and tool description

Your task:
1. Carefully read and understand the user's request.
2. Review all tools from the selected servers and their descriptions.
3. Select the ONE tool that is most relevant and appropriate for the user's request.
4. Respond using ONLY the following format:

<tool_selection>
"tool_name"
</tool_selection>

Your response should be a single tool name (as a string, not an array).
Remember to ONLY provide the tool name within the <tool_selection> tags. DO NOT provide any additional explanation or commentary outside the tags.

User request: {query}

Available tools from selected servers:
{tool_list}"""

    # Multi-turn planning prompts
    PLANNING_PROMPT = """You are an AI assistant that helps users by breaking down complex tasks into smaller, actionable steps using available tools.

Given:
- Original user request: {original_query}
- Current context and previous actions: {context}
- Current step in the plan

Your task:
1. Analyze the current progress and what still needs to be done
2. Determine the next specific action needed to progress toward the goal
3. Decide if the task is complete or if more steps are needed
4. If the query does not require tools or cannot be solved with available context/tools, set status to "finish"
5. Respond using ONLY the following format:

<planning>
{{
    "status": "continue" or "finish",
    "next_action": "description of what needs to be done next (if status is continue)",
    "reasoning": "brief explanation of why this action is needed, why the task is complete, why no tools are needed, or why the problem cannot be solved"
}}
</planning>

Important: Set status to "finish" if:
- The task is complete
- The query does not require any tools (e.g., conversational questions, general information requests)
- The problem cannot be solved with current available tools or context

Remember to ONLY provide the JSON within the <planning> tags.

Original request: {original_query}
Current context: {context}"""


    def _classify_error(self, error_message: str, stage: str) -> str:
        """
        Classify errors into specific categories for better debugging and analysis.
        
        Args:
            error_message: The error message to classify
            stage: The stage where the error occurred (planning, server_selection, tool_selection, execution)
            
        Returns:
            Error classification string
        """
        error_msg_lower = error_message.lower()
        
        # Network and connection errors
        if any(keyword in error_msg_lower for keyword in ['error code: 500', 'connection', 'timeout', 'network', 'http']):
            return f"{stage}_network_error"
        
        # LLM response format errors
        if any(keyword in error_msg_lower for keyword in ['no <server_selection> tags found', 'no <tool_selection> tags found', 'no <planning> tags found']):
            return f"{stage}_format_error"
        
        # LLM validation errors
        if any(keyword in error_msg_lower for keyword in ['did not return valid', 'invalid format', 'parsing failed', 'json']):
            return f"{stage}_validation_error"
        
        # Tool/server not found errors
        if any(keyword in error_msg_lower for keyword in ['not found', 'no suitable', 'no matching']):
            return f"{stage}_not_found_error"
        
        # LLM generation errors
        if any(keyword in error_msg_lower for keyword in ['generation failed', 'llm failed', 'model error']):
            return f"{stage}_llm_error"
        
        # Planning specific errors
        if stage == 'planning' and any(keyword in error_msg_lower for keyword in ['max turns', 'planning']):
            return "planning_iteration_error"
        
        # Generic stage error
        return f"{stage}_generic_error"

    def _record_error(self, stage: str, error_message: str, context: Optional[dict] = None) -> dict:
        """
        Record and classify an error with detailed information.
        
        Args:
            stage: The stage where error occurred
            error_message: The error message
            context: Additional context information
            
        Returns:
            Error record dictionary
        """
        error_classification = self._classify_error(error_message, stage)
        
        error_record = {
            'error_stage': stage,
            'error_classification': error_classification,
            'error_message': error_message,
            'timestamp': self._get_current_timestamp(),
        }
        
        if context:
            error_record['context'] = context
            
        logging.debug(f"[{error_classification}] {stage}: {error_message}")
        return error_record

    def _get_current_timestamp(self) -> str:
        """Get current timestamp for error recording."""
        import datetime
        return datetime.datetime.now().isoformat()

    def _is_genuine_level_5_scenario(self, planning_response: dict, conversation_context: list, react_count: int) -> bool:
        """
        Determine if this is a genuine Level 5 scenario (no tools needed) or a process error.
        
        Args:
            planning_response: The planning response that triggered 'finish' status
            conversation_context: The conversation history
            react_count: Current react cycle count
            
        Returns:
            True if this is genuinely Level 5, False if it's likely a process error
        """
        reasoning = planning_response.get('reasoning', '').lower()
        
        # Strong indicators of genuine Level 5 scenario
        strong_level_5_phrases = [
            'no tools needed', 'not require tools', 'does not need tools',
            'conversational question', 'general information', 'direct answer',
            'can be answered directly', 'no external tools required'
        ]
        
        # Check for strong Level 5 indicators in reasoning
        has_strong_level_5_signal = any(phrase in reasoning for phrase in strong_level_5_phrases)
        
        # Additional context analysis
        context_indicators = {
            'early_finish': react_count <= 1,  # Finished very early, likely Level 5
            'no_tool_attempts': len([c for c in conversation_context if c.get('selected_tool')]) == 0,
            'no_server_attempts': len([c for c in conversation_context if c.get('selected_servers')]) == 0,
            'reasoning_length': len(reasoning) > 20,  # Substantial reasoning provided
        }
        
        # Count failed attempts vs no attempts
        failed_attempts = len([c for c in conversation_context if c.get('status') == 'failed'])
        successful_attempts = len([c for c in conversation_context if c.get('status') == 'success'])
        
        # Decision logic
        if has_strong_level_5_signal:
            # Strong textual evidence for Level 5
            if context_indicators['early_finish'] and context_indicators['no_tool_attempts']:
                logging.debug("Level 5 Analysis: Strong signal + early finish + no tool attempts = GENUINE Level 5")
                return True
            elif context_indicators['reasoning_length']:
                logging.debug("Level 5 Analysis: Strong signal + substantial reasoning = GENUINE Level 5")
                return True
        
        # Weaker indicators check
        weak_level_5_phrases = [
            'cannot be solved', 'no available tools', 'not possible with tools'
        ]
        has_weak_level_5_signal = any(phrase in reasoning for phrase in weak_level_5_phrases)
        
        if has_weak_level_5_signal:
            if failed_attempts == 0 and context_indicators['early_finish']:
                logging.debug("Level 5 Analysis: Weak signal + no failures + early finish = GENUINE Level 5")
                return True
            elif failed_attempts > 0:
                logging.debug(f"Level 5 Analysis: Weak signal + {failed_attempts} failures = PROCESS ERROR")
                return False
        
        # If we have multiple failed attempts, it's likely a process error
        if failed_attempts >= 2:
            logging.debug(f"Level 5 Analysis: {failed_attempts} failed attempts = PROCESS ERROR")
            return False
        
        # If reasoning is very short or generic, likely an error
        if len(reasoning) < 15:
            logging.debug("Level 5 Analysis: Very short reasoning = PROCESS ERROR")
            return False
        
        # Default: if unsure and it's early with no attempts, lean towards Level 5
        if context_indicators['early_finish'] and context_indicators['no_tool_attempts']:
            logging.debug("Level 5 Analysis: Early finish + no attempts = LIKELY Level 5")
            return True
        
        logging.debug("Level 5 Analysis: Ambiguous case = PROCESS ERROR (conservative)")
        return False

    def process_query(self, query, top_k=5, top_k2=1, max_turns=1, num_attempts=3):
        """
        Process a query using React@N: Generate N different solution paths and succeed if any one works.
        
        React@N Design:
        - Goal: Test if agent can solve problem in at least 1 out of N attempts
        - Core: Generate N different solution trajectories 
        - Success: At least one successful path among N attempts
        
        Args:
            num_attempts: Number of different solution paths to try (N in React@N)
        """
        all_attempts = []  # Store all N attempts
        original_query = query
        
        # Initial debug information
        # print(f" React@N Processing Started")
        # print(f" Original Query: {original_query}")
        # print(f"  Parameters: top_k={top_k}, top_k2={top_k2}, max_turns={max_turns}")
        # print(f" React@N: Generating {num_attempts} different solution paths")
        
        # Try N different solution paths
        for attempt_num in range(1, num_attempts + 1):
            # print(f"\n{'='*80}")
            # print(f"🔄 REACT@N ATTEMPT {attempt_num}/{num_attempts}")
            # print(f"{'='*80}")
            
            # Each attempt gets its own conversation context
            conversation_context = []
            react_count = 0
            max_reacts = max_turns * 3  # Allow multiple React cycles per attempt
            error_log = []  # Track errors for this attempt
            attempt_successful = False
            
            logging.debug(f"Starting independent solution path #{attempt_num}")
            
            while react_count < max_reacts and not attempt_successful:
                react_count += 1
                
                # THINK: Observe current state + Plan next action
                context_str = self._format_context(conversation_context)
                # print(f" THINK: Analyzing context and planning next action...")
                # print(f" Context: {len(context_str)} chars")
                
                try:
                    # Add attempt diversity: use different prompting strategies for different attempts
                    if attempt_num == 1:
                        planning_response = self._get_planning_decision(original_query, context_str)
                    elif attempt_num == 2:
                        # Second attempt: more conservative approach
                        planning_response = self._get_planning_decision(
                            f"[Conservative approach] {original_query}", context_str
                        )
                    else:
                        # Third+ attempt: creative/alternative approach
                        planning_response = self._get_planning_decision(
                            f"[Alternative approach] {original_query}", context_str
                        )
                    
                    # print(f" THINK: Analysis completed")
                    # print(f" Planning result: {planning_response}")
                except Exception as e:
                    error_record = self._record_error('planning', str(e), {'react_cycle': react_count, 'attempt': attempt_num})
                    error_log.append(error_record)
                    planning_response = None
                    logging.debug(f"Planning failed - {str(e)}")
                
                # Check if planning failed for this cycle
                if not planning_response:
                    logging.debug(f"React cycle {react_count} failed at planning - trying next cycle")
                    continue
                
                # Check if task is complete
                if planning_response.get('status') == 'finish':
                    # print(f"🏁 COMPLETE: Task analysis shows completion for attempt {attempt_num}")
                    
                    # Task is complete - check if we have successful tools
                    last_successful_tool = self._extract_last_successful_tool(conversation_context)
                    
                    if last_successful_tool:
                        attempt_successful = True
                        attempt_result = {
                            'attempt_number': attempt_num,
                            'status': 'success',
                            'result': last_successful_tool,
                            'react_cycles_used': react_count,
                            'error_log': error_log,
                            'conversation_context': conversation_context
                        }
                        all_attempts.append(attempt_result)
                        logging.debug(f"Attempt {attempt_num} SUCCESSFUL! Used {react_count} React cycles")
                        break  # Success! No need to continue this attempt
                    else:
                        # Analyze if this is truly a Level 5 scenario or process error
                        is_genuine_level_5 = self._is_genuine_level_5_scenario(
                            planning_response, conversation_context, react_count
                        )
                        
                        if is_genuine_level_5:
                            # This is a genuine Level 5 scenario - query doesn't need tools
                            attempt_successful = True
                            level_5_result = {
                                'tool_name': 'No',
                                'server_name': 'No',
                                'server': 'No', 
                                'reasoning': planning_response.get('reasoning', 'Query does not require tools'),
                                'schema': {},
                                'level_5_compatible': True,
                                'no_tools_needed': True,
                                'confidence': 'high'  # High confidence this is Level 5
                            }
                            attempt_result = {
                                'attempt_number': attempt_num,
                                'status': 'success',
                                'result': level_5_result,
                                'react_cycles_used': react_count,
                                'error_log': error_log,
                                'conversation_context': conversation_context
                            }
                            all_attempts.append(attempt_result)
                            logging.debug(f"Attempt {attempt_num} SUCCESSFUL (Level 5 - No tools needed)! Used {react_count} React cycles")
                            break
                        else:
                            # No successful tools found and not Level 5 scenario
                            error_record = self._record_error('completion', 
                                f'Attempt {attempt_num} completed but no successful tool selections', 
                                {'react_cycle': react_count, 'attempt': attempt_num})
                            error_log.append(error_record)
                            logging.debug(f"Attempt {attempt_num} completed but no successful tools found")
                            continue  # Continue to next cycle
                
                # Continue with ACT phase if not finished
                next_action = planning_response.get('next_action', '')
                if not next_action:
                    error_record = self._record_error('planning', 
                        f'No next action provided by planning in attempt {attempt_num}', 
                        {'react_cycle': react_count, 'attempt': attempt_num})
                    error_log.append(error_record)
                    logging.debug(f"THINK: No next action determined - trying next cycle")
                    continue
                
                # print(f" THINK: Next action determined: {next_action}")
                # print(f" THINK: Reasoning: {planning_response.get('reasoning', 'No reasoning provided')}")
                    
                # ACT Phase 1: Select top-k servers for this action
                # print(f" ACT: Selecting servers and tools for action: {next_action}")
                # print(f" ACT: Phase 1 - Server Selection (top_k={top_k})")
                
                try:
                    top_servers = self._select_top_servers(next_action, top_k)
                    logging.debug(f"ACT: Server selection completed - {len(top_servers)} servers selected")
                    for i, server in enumerate(top_servers, 1):
                        logging.debug(f"   {i}. {server.get('server_name', 'Unknown')} - {server.get('server_description', '')[:100]}")
                except Exception as e:
                    error_record = self._record_error('server_selection', str(e), 
                        {'react_cycle': react_count, 'action': next_action, 'attempt': attempt_num})
                    error_log.append(error_record)
                    top_servers = []
                    logging.debug(f"ACT: Server selection failed - {str(e)}")
                
                if not top_servers:
                    error_record = self._record_error('server_selection', 'No suitable servers found', {'react_cycle': react_count, 'action': next_action})
                    error_log.append(error_record)
                    logging.debug(f"ACT: No suitable servers found, continuing to next React cycle")
                    conversation_context.append({
                        'react_cycle': react_count,
                        'action': next_action,
                        'status': 'failed',
                        'error': 'No suitable servers found',
                        'error_record': error_record
                    })
                    continue
            
                # ACT Phase 2: Select the best tool from selected servers
                # print(f"🎬 ACT: Phase 2 - Tool Selection (from {len(top_servers)} servers)")
                
                try:
                    selected_tool = self._select_tools_from_servers(next_action, top_servers, top_k2)
                    if selected_tool:
                        logging.debug(f"ACT: Tool selection completed - {selected_tool.get('tool_name', 'Unknown')} selected")
                        logging.debug(f"Selected tool details: {selected_tool}")
                    else:
                        logging.debug(f"ACT: Tool selection returned None")
                except Exception as e:
                    error_record = self._record_error('tool_selection', str(e), {'react_cycle': react_count, 'action': next_action, 'servers': [s['server_name'] for s in top_servers]})
                    error_log.append(error_record)
                    selected_tool = None
                    logging.debug(f"ACT: Tool selection failed - {str(e)}")
                
                if not selected_tool:
                    error_record = self._record_error('tool_selection', 'No suitable tool found', {'react_cycle': react_count, 'action': next_action, 'servers': [s['server_name'] for s in top_servers]})
                    error_log.append(error_record)
                    logging.debug(f"ACT: No suitable tool found, continuing to next React cycle")
                    conversation_context.append({
                        'react_cycle': react_count,
                        'action': next_action,
                        'servers': top_servers,
                        'status': 'failed',
                        'error': 'No suitable tool found',
                        'error_record': error_record
                    })
                    continue
                
                # ACT Phase 3: Execute the selected tool
                logging.debug(f"ACT: Phase 3 - Tool Execution")
                execution_start = self._get_current_timestamp()
                
                try:
                    execution_result = self._execute_tool_step(original_query, next_action, selected_tool['tool_name'])
                    exec_status = execution_result.get('execution_status', 'unknown')
                    logging.debug(f"ACT: Tool execution completed with status: {exec_status}")
                except Exception as e:
                    error_record = self._record_error('execution', str(e), {'react_cycle': react_count, 'action': next_action, 'tool': selected_tool.get('tool_name', '')})
                    error_log.append(error_record)
                    execution_result = {
                        'execution_status': 'failed',
                        'result': f'Execution error: {str(e)}',
                        'error_record': error_record
                    }
                    logging.debug(f"ACT: Tool execution failed - {str(e)}")
                
                # REFLECT: Update conversation context and analyze results
                react_status = 'success' if execution_result.get('execution_status') in ['ready_for_execution', 'success'] else 'failed'
                
                conversation_context.append({
                    'react_cycle': react_count,
                    'action': next_action,
                    'planning_reasoning': planning_response.get('reasoning', ''),
                    'selected_servers': top_servers,
                    'selected_tool': selected_tool,
                    'execution_result': execution_result,
                    'status': react_status,
                    'debug_info': {
                        'servers_found': len(top_servers),
                        'tool_found': selected_tool is not None,
                        'react_cycle': react_count,
                        'attempt_number': attempt_num
                    }
                })
                
                # DEBUG: Full React cycle summary
                logging.debug(f"DEBUG: React Cycle {react_count} Summary:")
                logging.debug(f"   Action: {next_action}")
                logging.debug(f"   Servers: {[s.get('server_name') for s in top_servers]}")
                logging.debug(f"   Tool: {selected_tool.get('tool_name', 'None') if selected_tool else 'None'}")
                logging.debug(f"   Status: {react_status}")
                logging.debug(f"   Execution Status: {execution_result.get('execution_status', 'unknown')}")
                
                # REFLECT: Context analysis
                successful_reacts = len([t for t in conversation_context if t.get('status') == 'success'])
                logging.debug(f"REFLECT: Progress - {successful_reacts}/{react_count} successful React cycles")
                
                # End of React cycle loop for this attempt
                
            # End of while loop - this attempt finished without success
            if not attempt_successful:
                logging.debug(f"Attempt {attempt_num} failed after {react_count} React cycles")
                attempt_result = {
                    'attempt_number': attempt_num,
                    'status': 'failed',
                    'react_cycles_used': react_count,
                    'error_log': error_log,
                    'conversation_context': conversation_context,
                    'reason': 'Max React cycles reached without success'
                }
                all_attempts.append(attempt_result)
            
            # End of attempt loop
            
            # React@N Final Analysis: Check if any attempt succeeded
            logging.debug(f"\n{'='*80}")
            logging.debug(f"React@N FINAL ANALYSIS")
            logging.debug(f"{'='*80}")
            
            successful_attempts = [att for att in all_attempts if att['status'] == 'success']
            
            if successful_attempts:
                # At least one attempt succeeded - React@N passes!
                best_attempt = successful_attempts[0]  # Use first successful attempt
                logging.debug(f"React@N SUCCESS: {len(successful_attempts)}/{num_attempts} attempts succeeded")
                logging.debug(f"Best attempt: #{best_attempt['attempt_number']} ({best_attempt['react_cycles_used']} cycles)")
                
                # Return the successful result
                result = best_attempt['result']
                
                # Ensure result is always a list for evaluation compatibility
                if not isinstance(result, list):
                    # Convert single tool result to list format for Level 3 compatibility
                    if isinstance(result, dict):
                        # Create a single-item list with JSON string format expected by Level 3
                        tool_step = {
                            "tool_name": result.get('tool_name', ''),
                            "server": result.get('server_name', result.get('server', '')),
                            "arguments_kv": [],
                            "reasoning": result.get('reasoning', 'Single React cycle result'),
                            "schema": result.get('schema', {}),
                            "step_id": 0,
                            "dependencies": []
                        }
                        result = [json.dumps(tool_step, indent=2)]
                    else:
                        # Handle unexpected format
                        result = []
                
                # Add react_n_info to the list if it's not empty
                if result:
                    react_n_info = {
                        'total_attempts': num_attempts,
                        'successful_attempts': len(successful_attempts),
                        'failed_attempts': num_attempts - len(successful_attempts),
                        'best_attempt_number': best_attempt['attempt_number'],
                        'best_attempt_cycles': best_attempt['react_cycles_used'],
                        'all_attempts_summary': [
                            {
                                'attempt': att['attempt_number'],
                                'status': att['status'],
                                'cycles': att['react_cycles_used']
                            }
                            for att in all_attempts
                        ]
                    }
                    # Store react_n_info separately as it cannot be added to list
                    logging.debug(f"React@N Info: {react_n_info}")
                logging.debug(f'final result:{result}')
                return result
            else:
                # All attempts failed - React@N fails
                logging.debug(f"React@N FAILURE: 0/{num_attempts} attempts succeeded")
                for i, attempt in enumerate(all_attempts, 1):
                    logging.debug(f"   Attempt {i}: Failed after {attempt['react_cycles_used']} cycles")
                
                # Return empty list for evaluation compatibility (Level 3 expects list)
                return []
    
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
            logging.debug(f"Classification failed: {e}")
            return "CONVERSATIONAL"
    
    def _select_top_servers(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Stage 1: Select top-k servers from server pool using LLM reasoning.
        
        Args:
            query: User's original query
            top_k: Number of top servers to select (optional, uses self.server_top_k if not provided)
            
        Returns:
            List of top-k server dictionaries selected by LLM
        """
        # Use provided top_k or fall back to instance variable
        if top_k is None:
            top_k = self.server_top_k
            
        try:
            # Format server list for prompt
            server_list = ""
            for i, server in enumerate(self.servers_data, 1):
                server_list += f"{i}. {server['server_name']}: {server['server_description']}\n"
            
            # Use SERVER_SELECTION_PROMPT with LLM
            prompt = self.SERVER_SELECTION_PROMPT.format(
                query=query,
                top_k=top_k,
                server_list=server_list
            )
            # print(f'prompt for LLM to choose server: {prompt}')
            # Call LLM to select servers
            response = self._call_llm(prompt, "router")
            # print(f'response after calling LLM to choose server: {response}')

            # Extract server selection from response
            selected_server_names = self._extract_server_selection(response)
            
            if not selected_server_names:
                error_msg = "LLM did not return valid server selection"
                logging.debug(error_msg)
                # This will be caught and classified as server_selection_validation_error
                raise ValueError(error_msg)
            
            # Find corresponding server objects
            selected_servers = []
            for server_name in selected_server_names:
                for server in self.servers_data:
                    if server['server_name'] == server_name:
                        # Only keep necessary fields for each tool
                        minimal_tools = []
                        for tool in server.get('tools', []):
                            minimal_tools.append({
                                'tool_name': tool.get('name', tool.get('tool_name', '')),
                                'tool_description': tool.get('description', tool.get('tool_description', '')),
                                # 'input_schema': tool.get('input_schema', {})
                            })
                        selected_servers.append({
                            'server_name': server['server_name'],
                            'server_description': server['server_description'],
                            'tools': minimal_tools
                        })
                        break
            
            if not selected_servers:
                error_msg = f"No matching servers found for selected names: {selected_server_names}"
                logging.debug(error_msg)
                raise ValueError(error_msg)
            
            logging.debug(f"LLM selected {len(selected_servers)} servers from {len(self.servers_data)} available servers")
            for server in selected_servers:
                logging.debug(f"  - {server['server_name']}")
            
            return selected_servers[:top_k]  # Ensure we don't exceed top_k
            
        except Exception as e:
            # Re-raise the exception to be caught and classified by the calling method
            raise e
    
    def _extract_server_selection(self, response: str) -> List[str]:
        """Extract server names from LLM response with <server_selection> tags."""
        try:
            # Extract content from <server_selection> tags
            pattern = r'<server_selection>(.*?)</server_selection>'
            match = re.search(pattern, response, re.DOTALL)
            
            if not match:
                error_msg = "No <server_selection> tags found in LLM response"
                logging.debug(error_msg)
                raise ValueError(error_msg)
            
            content = match.group(1).strip()
            logging.debug(f'content in extract server selection:{content}')
            # Parse JSON array
            server_names = json.loads(content)
            
            if isinstance(server_names, list) and all(isinstance(name, str) for name in server_names):
                return server_names
            else:
                error_msg = "Server selection is not a valid list of strings"
                logging.debug(error_msg)
                raise ValueError(error_msg)
                
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse server selection JSON: {e}"
            logging.debug(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error extracting server selection: {e}"
            logging.debug(error_msg)
            raise ValueError(error_msg)

    def _select_tools_from_servers(self, query: str, top_servers: List[Dict[str, Any]], top_k2: Optional[int] = None) -> Dict[str, Any]:
        """
        Stage 2: Select the single best tool from the selected servers using LLM reasoning.
        
        Args:
            query: User's original query
            top_servers: List of selected servers from stage 1
            top_k2: Unused parameter (kept for compatibility)
            
        Returns:
            Single tool dictionary selected by LLM, or None if no tool selected
        """
        try:
            # Format tool list for prompt
            tool_list = ""
            tool_counter = 1
            all_tools = []
            
            # Collect all tools from selected servers
            for server in top_servers:
                for tool in server.get('tools', []):
                    tool_list += f"{tool_counter}. Server: {server['server_name']}\n"
                    tool_list += f"   Tool: {tool['tool_name']}\n"
                    tool_list += f"   Description: {tool['tool_description']}\n\n"
                    
                    # Keep track of all tools for later matching
                    all_tools.append({
                        'tool_name': tool['tool_name'],
                        'tool_description': tool['tool_description'],
                        'tool_schema': tool.get('input_schema', {}),
                        'server_name': server['server_name'],
                        'server_description': server['server_description']
                    })
                    tool_counter += 1
            
            if not all_tools:
                error_msg = "No tools available from selected servers"
                raise ValueError(error_msg)
            
            # Use TOOL_SELECTION_FROM_SERVERS_PROMPT with LLM (single tool selection)
            prompt = self.TOOL_SELECTION_FROM_SERVERS_PROMPT.format(
                query=query,
                tool_list=tool_list
            )
            # print(f'prompt for LLM to choose tool: {prompt}')
            # Call LLM to select the best tool
            response = self._call_llm(prompt, "router")
            
            # Extract single tool selection from response
            selected_tool_name = self._extract_single_tool_selection(response)
            
            if not selected_tool_name:
                error_msg = "LLM did not return valid tool selection"
                raise ValueError(error_msg)
            
            # Find corresponding tool object
            for tool in all_tools:
                if tool['tool_name'] == selected_tool_name:
                    logging.debug(f"LLM selected tool: {tool['tool_name']} from server: {tool['server_name']}")
                    return tool
            
            # Tool name not found
            error_msg = f"Selected tool '{selected_tool_name}' not found in available tools"
            raise ValueError(error_msg)
            
        except Exception as e:
            # Re-raise the exception to be caught and classified by the calling method
            raise e
    
    def _extract_single_tool_selection(self, response: str) -> str:
        """Extract single tool name from LLM response with <tool_selection> tags."""
        try:
            # Extract content from <tool_selection> tags
            pattern = r'<tool_selection>(.*?)</tool_selection>'
            match = re.search(pattern, response, re.DOTALL)
            
            if not match:
                error_msg = "No <tool_selection> tags found in LLM response"
                logging.debug(error_msg)
                raise ValueError(error_msg)
            
            content = match.group(1).strip()
            logging.debug(f'content in extract single tool selection: {content}')
            
            # Remove quotes if present and return the tool name
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            elif content.startswith("'") and content.endswith("'"):
                content = content[1:-1]
            
            if not content:
                error_msg = "Empty tool selection in LLM response"
                raise ValueError(error_msg)
            
            return content.strip()
                
        except Exception as e:
            error_msg = f"Error extracting single tool selection: {e}"
            logging.debug(error_msg)
            raise ValueError(error_msg)
    
    def _generate_conversational_response(self, query: str) -> str:
        """Generate a direct conversational response."""
        try:
            return self._call_llm(query, "conversational")
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

    def _get_tools_from_servers(self, selected_servers):
        """Get all tools from selected servers"""
        tools_from_servers = []
        for server in selected_servers:
            for tool in server.get('tools', []):
                tools_from_servers.append({
                    'tool_name': tool.get('tool_name'),
                    'tool_description': tool.get('tool_description'),
                    'input_schema': tool.get('input_schema', {}),
                    'server_name': server.get('server_name'),
                    'server_description': server.get('server_description')
                })
        return tools_from_servers

    def _format_context(self, conversation_context):
        """Format conversation context for planning prompt"""
        if not conversation_context:
            return "No previous actions taken."
        
        context_parts = []
        for react_cycle in conversation_context:
            cycle_key = react_cycle.get('react_cycle', react_cycle.get('turn', 'Unknown'))
            cycle_summary = f"React Cycle {cycle_key}: {react_cycle['action']}"
            if react_cycle['status'] == 'success':
                cycle_summary += f" -> Success: {react_cycle['execution_result'].get('result', 'Completed')}"
            else:
                cycle_summary += f" -> Failed: {react_cycle.get('error', 'Unknown error')}"
            context_parts.append(cycle_summary)
        
        return "\n".join(context_parts)
    
    def _get_planning_decision(self, original_query, context_str):
        """Get planning decision using LLM"""
        try:
            prompt = self.PLANNING_PROMPT.format(
                original_query=original_query,
                context=context_str
            )
            
            response = self._call_llm(prompt, "router")
            # print(f'Planning response: {response}')
            # Parse planning response
            if "<planning>" in response and "</planning>" in response:
                planning_content = response.split("<planning>")[1].split("</planning>")[0].strip()
                return json.loads(planning_content)
            
        except Exception as e:
            logging.debug(f"Error in planning decision: {e}")
        
        return None
    
    def _execute_tool_step(self, original_query, current_action, selected_tool):
        """Return tool call information without LLM simulation to avoid model bias"""
        try:
            # Find tool details and server name
            tool_details = None
            server_name = None
            for server_data in self.servers_data:
                for tool in server_data.get('tools', []):
                    if tool.get('tool_name') == selected_tool:
                        tool_details = tool
                        server_name = server_data.get('server_name', '')
                        break
                if tool_details:
                    break
            
            if not tool_details:
                return {
                    'tool_name': selected_tool,
                    'execution_status': 'failed',
                    'result': 'Tool not found in server data',
                    'context_update': 'Failed to locate tool'
                }
            
            # Return structured tool call information without LLM simulation
            return {
                'tool_name': selected_tool,
                'execution_status': 'ready_for_execution',
                'result': f'Tool {selected_tool} selected for action: {current_action}',
                'context_update': f'Selected {selected_tool} to perform: {current_action}',
                'tool_call_schema': {
                    'tool_name': selected_tool,
                    'tool_description': tool_details.get('tool_description', ''),
                    'input_schema': tool_details.get('input_schema', {}),
                    'server_name': server_name,
                    'action_context': current_action
                }
            }
            
        except Exception as e:
            logging.debug(f"Error preparing tool step: {e}")
            return {
                'tool_name': selected_tool,
                'execution_status': 'failed',
                'result': f'Tool preparation error: {str(e)}',
                'context_update': 'Tool preparation failed'
            }
    
    def _extract_final_result(self, conversation_context):
        """Extract final result from conversation context"""
        if not conversation_context:
            return "No actions completed."
        
        successful_actions = [cycle for cycle in conversation_context if cycle['status'] == 'success']
        
        result_parts = []
        for cycle in successful_actions:
            action = cycle['action']
            tool = cycle['selected_tool']
            result = cycle['execution_result'].get('result', 'Completed')
            react_num = cycle.get('react_cycle', 'Unknown')
            result_parts.append(f"React {react_num} - Action: {action} | Tool: {tool} | Result: {result}")
        
        return "\n".join(result_parts) if result_parts else "No successful actions completed."
    
    def _extract_last_successful_tool(self, conversation_context):
        """Extract tool selection information in evaluation-compatible format with multi-React path"""
        if not conversation_context:
            return None
        
        # Collect all successful tools in order
        successful_tools = []
        for cycle in conversation_context:
            if cycle.get('status') == 'success' and cycle.get('execution_result'):
                execution_result = cycle['execution_result']
                tool_schema = execution_result.get('tool_call_schema', {})
                selected_tool = cycle.get('selected_tool', {})
                
                # Handle both old format (string) and new format (dict)
                if isinstance(selected_tool, dict):
                    tool_name = selected_tool.get('tool_name', '')
                    server_name = selected_tool.get('server_name', '')
                else:
                    tool_name = str(selected_tool)
                    server_name = tool_schema.get('server_name', '')
                
                successful_tools.append({
                    'react_cycle': cycle.get('react_cycle', cycle.get('turn', 'Unknown')),
                    'action': cycle.get('action', ''),
                    'tool_name': tool_schema.get('tool_name', tool_name),
                    'server_name': tool_schema.get('server_name', server_name),
                    'planning_reasoning': cycle.get('planning_reasoning', ''),
                    'selected_servers': [s.get('server_name', '') for s in cycle.get('selected_servers', [])],
                    'selected_tool_info': selected_tool,  # Store the complete tool info
                    'tool_schema': tool_schema
                })
        
        # Check if this is a genuine Level 5 scenario (more sophisticated analysis)
        has_genuine_level_5_reasoning = False
        level_5_reasoning = ""
        
        # Analyze all planning reasoning in conversation
        for cycle in conversation_context:
            reasoning = cycle.get('planning_reasoning', '')
            if reasoning:
                # Use stronger Level 5 indicators
                strong_level_5_phrases = [
                    'no tools needed', 'not require tools', 'does not need tools',
                    'conversational question', 'general information', 'direct answer',
                    'can be answered directly', 'no external tools required'
                ]
                
                if any(phrase in reasoning.lower() for phrase in strong_level_5_phrases):
                    has_genuine_level_5_reasoning = True
                    level_5_reasoning = reasoning
                    break
        
        # Count failed attempts to distinguish from process errors
        failed_attempts = len([c for c in conversation_context if c.get('status') == 'failed'])
        tool_attempts = len([c for c in conversation_context if c.get('selected_tool')])
        
        if not successful_tools:
            # If no successful tools, check if it's genuinely Level 5 or process error
            if has_genuine_level_5_reasoning and failed_attempts <= 1 and tool_attempts == 0:
                # Genuine Level 5: strong reasoning + minimal/no failures + no tool attempts
                return {
                    'tool_name': 'No',
                    'server_name': 'No', 
                    'server': 'No',
                    'reasoning': level_5_reasoning or 'Query does not require tools',
                    'schema': {},
                    'level_5_compatible': True,
                    'no_tools_needed': True,
                    'confidence': 'high'
                }
            elif has_genuine_level_5_reasoning and failed_attempts == 0:
                # Possible Level 5: strong reasoning + no failures
                return {
                    'tool_name': 'No',
                    'server_name': 'No', 
                    'server': 'No',
                    'reasoning': level_5_reasoning or 'Query does not require tools',
                    'schema': {},
                    'level_5_compatible': True,
                    'no_tools_needed': True,
                    'confidence': 'medium'
                }
            # Otherwise, it's likely a process error, return None
            return None
        
        # For multi-React scenarios: Return list of JSON strings with complete tool information
        if len(successful_tools) > 1:
            # Return list of JSON strings for Level 3 evaluation compatibility
            tools_json_list = []
            for i, tool_info in enumerate(successful_tools):
                tool_step = {
                    "tool_name": tool_info['tool_name'],
                    "server": tool_info['server_name'],
                    "arguments_kv": [],  # Empty for now - could be populated from action context
                    "reasoning": f"Multi-React step {i+1}: {tool_info['action']}. {tool_info['planning_reasoning']}",
                    "schema": tool_info['tool_schema'].get('input_schema', {}),
                    "step_id": i,  # Add step ID for dependency tracking
                    "dependencies": [i-1] if i > 0 else []  # Sequential dependency
                }
                # Convert to formatted JSON string as expected by evaluation
                tools_json_list.append(json.dumps(tool_step, indent=2))
            return tools_json_list
        
        # Single tool case: Return single tool object for Level 1/2 compatibility
        last_tool = successful_tools[-1]
        last_cycle = None
        for cycle in reversed(conversation_context):
            if cycle.get('status') == 'success' and cycle.get('execution_result'):
                last_cycle = cycle
                break
        
        if last_cycle is None:
            # Fallback if no successful cycle found
            return {
                'tool_name': last_tool['tool_name'],
                'server_name': last_tool['server_name'],
                'server': last_tool['server_name'],
                'reasoning': f"Multi-React final selection: {last_tool['action']}",
                'schema': last_tool['tool_schema'].get('input_schema', {}),
                'action_context': last_tool['action'],
                'multi_react_path': successful_tools,
                'tool_path_summary': ' -> '.join([f"R{t['react_cycle']}:{t['tool_name']}" for t in successful_tools]),
                'react_info': {
                    'total_successful_reacts': len(successful_tools),
                    'final_react_number': last_tool['react_cycle'],
                    'planning_reasoning': last_tool['planning_reasoning'],
                    'selected_servers': last_tool['selected_servers'],
                    'selected_tool_details': last_tool['selected_tool_info']
                }
            }
        
        execution_result = last_cycle['execution_result']
        tool_schema = execution_result.get('tool_call_schema', {})
        selected_tool = last_cycle.get('selected_tool', {})
        
        # Extract server name and tool name
        if isinstance(selected_tool, dict):
            tool_name = tool_schema.get('tool_name', selected_tool.get('tool_name', ''))
            server_name = tool_schema.get('server_name', selected_tool.get('server_name', ''))
        else:
            tool_name = tool_schema.get('tool_name', str(selected_tool))
            server_name = tool_schema.get('server_name', '')
        
        # Return in format compatible with evaluation script, with tool path information
        return {
            'tool_name': tool_name,
            'server_name': server_name,
            'server': server_name,  # Add for evaluation compatibility
            'reasoning': f"Multi-React final selection: {last_cycle.get('action', '')}",
            'schema': tool_schema.get('input_schema', {}),
            'action_context': tool_schema.get('action_context', ''),
            'multi_react_path': successful_tools,  # Complete tool path sequence
            'tool_path_summary': ' -> '.join([f"R{t['react_cycle']}:{t['tool_name']}" for t in successful_tools]),
            'react_info': {
                'total_successful_reacts': len(successful_tools),
                'final_react_number': last_tool['react_cycle'],
                'planning_reasoning': last_tool['planning_reasoning'],
                'selected_servers': last_tool['selected_servers'],
                'selected_tool_details': selected_tool
            }
        }
