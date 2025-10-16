import json
import random
import time
import argparse
from typing import Any, Dict, List
import re
from pathlib import Path
import os
from openai.types.chat import ChatCompletionMessageParam
from util import (
    setup_environment, get_vllm_client, get_azure_openai_client,
    call_vllm_api, call_azure_openai
)

# Load environment variables
setup_environment("../.env")

# --- Configuration & Client Initialization ---
vllm_client, VLLM_MODEL_NAME = get_vllm_client(
    base_url="http://localhost:8060/v1",
    model_name="Qwen/Qwen3-4B-Instruct-2507"
)

# Azure OpenAI for query generation (GPT-4.1)
azure_client = get_azure_openai_client()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Debug: Print Azure OpenAI configuration
print(f"Azure OpenAI Configuration:")
print(f"  Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"  API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
print(f"  Deployment: {AZURE_DEPLOYMENT}")
print(f"  API Key: {'***' + os.getenv('AZURE_OPENAI_API_KEY', '')[-4:] if os.getenv('AZURE_OPENAI_API_KEY') else 'None'}")
print()

FEASIBILITY_PROMPT_TEMPLATE = """
# ROLE & GOAL
You are a creative product manager designing tasks for a powerful AI assistant. Your task is to determine if a given combination of real-world services can form a logical user workflow.

# CONTEXT: PROVIDED SERVICES
Here are the services available for the potential task:
{services_description}

# YOUR TASK: FEASIBILITY CHECK
First, analyze the provided services. Can you imagine a realistic, common user scenario that would require **combining ALL** of these services? The workflow must be logical.

Respond with your boolean judgement (`true` or `false`) **wrapped in `<is_feasible></is_feasible>` tags**, followed by a brief, one-sentence reason for your decision on a new line.

# EXAMPLE 1 (Feasible)
## Services: [A GitHub Server, A Slack Server]
## Your Output:
<is_feasible>true</is_feasible>
A user might want to get notifications on Slack for new issues created in their GitHub repository.

# EXAMPLE 2 (Not Feasible)
## Services: [A Weather Forecast Server, A File Encryption Server]
## Your Output:
<is_feasible>false</is_feasible>
There is no common, logical workflow that directly connects weather forecasting with file encryption.

---

# YOUR CHECK NOW
## Services: [As provided in the context above]
## Your Output:
"""

def get_generation_prompt(sampled_servers):
    """Generate the prompt with actual tool information from sampled servers"""
    
    # Extract tool information from sampled servers
    tools_info = []
    for server in sampled_servers:

        server_name = server.get('name', 'Unknown')
        server_desc = server.get('description', '') or server.get('summary', 'No description')
        tools = server.get('tools', [])
        
        server_info = f"Server: {server_name}\n"
        server_info += f"Description: {server_desc}\n"
        server_info += "Available Tools:\n"
        
        for tool in tools:
            if isinstance(tool, dict):
                tool_name = tool.get('tool_name', 'unknown')
                tool_desc = tool.get('tool_description', 'No description available')
                input_schema = tool.get('input_schema', {})
                server_info += f"  - {tool_name}: {tool_desc}\n"
                if input_schema:
                    # Simplify input schema to only include properties and required fields
                    simplified_schema = {}
                    if 'required' in input_schema:
                        simplified_schema['required'] = input_schema['required']
                    
                    if simplified_schema:
                        server_info += f"    required parameter: {json.dumps(simplified_schema['required'], indent=6)}\n"
            else:
                tool_name = str(tool)
                server_info += f"  - {tool_name}: No description available\n"
        
        tools_info.append(server_info)
    
    available_tools_section = '\n\n'.join(tools_info)
    
    prompt = f"""# SCENARIO GENERATION
Based on the services and tools provided below, please generate a user scenario.

# AVAILABLE SERVICES AND TOOLS:
{available_tools_section}

# HIGH-QUALITY EXAMPLES
Here are examples of well-structured queries and their corresponding ground truth tools:

## Example 1:
{{"query": "Get detailed information about the \\"My Favorite Chair\\" object within a Blender scene and check the PolyHaven integration status.", "ground_truth_tools_count": 2, "ground_truth_tools": [{{"tool_id": "Tripo MCP Server::get_object_info", "server_name": "Tripo MCP Server", "tool_name": "get_object_info", "description": "Tool for get_object_info functionality provided by Tripo MCP Server", "dependencies": []}}, {{"tool_id": "Tripo MCP Server::get_polyhaven_status", "server_name": "Tripo MCP Server", "tool_name": "get_polyhaven_status", "description": "Tool for get_polyhaven_status functionality provided by Tripo MCP Server", "dependencies": []}}]}}

## Example 2:
{{"query": "Scan the content of the file \\"main.py\\" using Semgrep, then retrieve the detailed results, check the status, and finally, get a list of supported languages for security vulnerability detection.", "ground_truth_tools_count": 4, "ground_truth_tools": [{{"tool_id": "Semgrep MCP Server::start_scan_from_content", "server_name": "Semgrep MCP Server", "tool_name": "start_scan_from_content", "description": "Tool for start_scan_from_content functionality provided by Semgrep MCP Server", "dependencies": []}}, {{"tool_id": "Semgrep MCP Server::get_scan_results", "server_name": "Semgrep MCP Server", "tool_name": "get_scan_results", "description": "Tool for get_scan_results functionality provided by Semgrep MCP Server", "dependencies": [0]}}, {{"tool_id": "Semgrep MCP Server::get_scan_status", "server_name": "Semgrep MCP Server", "tool_name": "get_scan_status", "description": "Tool for get_scan_status functionality provided by Semgrep MCP Server", "dependencies": [0]}}, {{"tool_id": "Semgrep MCP Server::get_supported_languages", "server_name": "Semgrep MCP Server", "tool_name": "get_supported_languages", "description": "Tool for get_supported_languages functionality provided by Semgrep MCP Server", "dependencies": []}}]}}

# INSTRUCTIONS
Based on the services and tools provided above, create a scenario following these guidelines:

1. **Specific and Actionable Query**: Write a detailed user query that includes:
   - Specific output formats (PDF, PPT, CSV, image, etc.) when relevant
   - File names and paths when relevant (e.g., /root/pdf/report.pdf, /home/user/data.csv)
   - Clear deliverables and requirements
   - Concrete subjects or targets (e.g., specific companies, topics, data sources)

2. **Ground Truth Tools**: For each tool needed in the workflow:
   - Use the exact tool names from the available services listed above
   - Set tool_id as "ServerName::tool_name"
   - Copy the description directly from the service definition or use the pattern "Tool for [tool_name] functionality provided by [server_name]"
   - Add dependencies as array of indices (0-based) if one tool depends on output from another
   - Include only essential tools needed for the workflow

3. **Natural Integration**: Ensure the query requires tools from ALL provided services in a logical workflow that a real user might request.

4. **Output Format**: Return your response as a valid JSON object with the exact structure shown in the examples above.

## Your Scenario Generation:"""
    
    return prompt

QUALITY_CONTROL_PROMPT_TEMPLATE = """
You are a professional query quality assessment and improvement expert. Please evaluate whether the following query contains the necessary parameters for executing the required tools, and improve the query when needed.

Original Query:
{query}

Expected tools and their parameter requirements:
{tools_description}

EVALUATION CRITERIA:

1. SOLVABILITY (1-10):
   - 10: All required data is provided, tools perfectly match needs, clear success criteria
   - 8-9: Task is clearly solvable with the given tools, minor ambiguities acceptable
   - 6-7: Mostly solvable but some steps may be challenging or unclear
   - 4-5: Significant gaps in tool coverage or data requirements
   - 1-3: Task cannot be meaningfully completed with available tools

   Consider:
   - Are all necessary tools available?
   - Is all required data provided (no external dependencies)?
   - Can the agent achieve the stated goal with these tools based on the function and output of the tools?
   - Are success criteria clear and measurable?

2. UTILITY (1-10):
   - 10: Critical business/research value, addresses real-world problem perfectly
   - 8-9: Strong practical value, useful for decision-making or operations
   - 6-7: Moderate value, interesting but not critical
   - 4-5: Limited practical value, mostly academic exercise
   - 1-3: Trivial or artificial task with no real-world application

   Consider:
   - Does this address a real business or research need?
   - Would the results be actionable and valuable?
   - Is the complexity justified by the outcome?
   - Does it test meaningful agent capabilities?

Provide scores and brief feedback in JSON format:
{{
  "solvability_score": <number 1-10>,
  "utility_score": <number 1-10>,
  "solvability_feedback": "Brief explanation of solvability assessment",
  "utility_feedback": "Brief explanation of utility assessment"
}}
"""

# --- Helper Functions ---

def load_and_group_servers(registry_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    try:
        with registry_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to load registry file '{registry_path}': {exc}") from exc

    servers = registry.get("servers", [])
    category_map: Dict[str, List[Dict[str, Any]]] = {}

    for server in servers:
        categories = server.get("categories") or ["unknown"]
        main_category = categories[0]
        canonical_server = {
            "name": server.get("server_name") or server.get("name", "N/A"),
            "description": server.get("server_description") or server.get("description"),
            "summary": server.get("server_summary") or server.get("summary"),
            "categories": categories,
            "tools": server.get("tools", []),
        }
        category_map.setdefault(main_category, []).append(canonical_server)

    return category_map

def format_services_for_prompt(servers: List[Dict]) -> str:
    descriptions = []
    for i, server in enumerate(servers):
        desc = f"Service #{i+1}:\n"
        desc += f"- Name: {server.get('name', 'N/A')}\n"
        desc += f"- Description: {server.get('description') or server.get('summary', 'No description.')}"
        descriptions.append(desc)
    return "\n\n".join(descriptions)

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from response that may contain extra text"""
    try:
        # Try to find JSON content between curly braces
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        else:
            # Fallback: try parsing the entire response
            return json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from a response that may contain extra text"""
    # Try to find JSON in the response
    start_idx = response.find('{')
    if start_idx == -1:
        raise ValueError("No JSON found in response")
    
    # Find the matching closing brace
    brace_count = 0
    for i, char in enumerate(response[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = response[start_idx:i+1]
                return json.loads(json_str)
    
    raise ValueError("No complete JSON found in response")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Level 4 scenarios by combining servers from different categories.")
    default_registry = Path(__file__).resolve().parents[1] / "data" / "mcp_registry.json"
    parser.add_argument("--registry", "-r", type=Path, default=default_registry, help="Path to the consolidated MCP registry JSON file.")
    parser.add_argument("--num-scenarios", "-n", type=int, default=20, help="Total number of scenarios to attempt to generate.")
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=Path("../data/level4_data.json"),
        help="Output file for generated scenarios.",
    )
    args = parser.parse_args()

    # 1. Load and group servers by category
    print("--- Step 1: Loading and grouping servers by category... ---")
    category_map = load_and_group_servers(args.registry.resolve())
    # Filter out categories with no servers
    valid_categories = [cat for cat, srvs in category_map.items() if srvs]
    print(f"Found {len(valid_categories)} valid categories.")
    output_path = args.output_file.resolve()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            try:
                generated_scenarios = json.load(f)
                print(f"📄 Loaded {len(generated_scenarios)} existing scenarios from {output_path}")
            except json.JSONDecodeError:
                print("⚠️ Existing output file is not valid JSON. Starting fresh.")
                generated_scenarios = []
    else:
        generated_scenarios = []

    
    print(f"\n--- Step 2: Attempting to generate {args.num_scenarios} scenarios... ---")
    for i in range(args.num_scenarios):
        print(f"\n--- [Attempt {i+1}/{args.num_scenarios}] ---")
        
        # 2. Sample 2-4 different categories, then one server from each
        num_to_sample = random.randint(2, 4)
        if len(valid_categories) < num_to_sample:
            print("  ! Not enough categories to sample from. Skipping.")
            continue
            
        sampled_categories = random.sample(valid_categories, num_to_sample)
        sampled_servers = [random.choice(category_map[cat]) for cat in sampled_categories]
        
        server_names = [s.get('name') for s in sampled_servers]
        print(f"  > Sampled combination of {num_to_sample} servers: {server_names}")

        # 3. Phase 1: Feasibility Check
        services_desc_text = format_services_for_prompt(sampled_servers)
        feasibility_prompt = FEASIBILITY_PROMPT_TEMPLATE.format(services_description=services_desc_text)
        
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": feasibility_prompt}
        ]
        
        # Convert messages to prompt and call util function directly
        prompt = "\n".join([msg.get("content", "") for msg in messages if msg.get("content")])
        response_text = call_vllm_api(
            prompt=prompt,
            client=vllm_client,
            model_name=VLLM_MODEL_NAME,
            max_tokens=100,
            temperature=0.0
        )
        
        # 使用正則表達式解析 <is_feasible> 標籤
        is_feasible = False
        match = re.search(r'<is_feasible>(.*?)</is_feasible>', response_text, re.DOTALL)
        
        if match:
            result_str = match.group(1).strip().lower()
            is_feasible = (result_str == 'true')
            # 打印出 LLM 給出的理由，方便偵錯
            reason = response_text.split('</is_feasible>')[-1].strip()
            print(f"  > Feasibility check result: {is_feasible}. Reason: {reason}")
        else:
            print(f"  ! Warning: Feasibility check did not return <is_feasible> tag. Response: {response_text}")
        # 4. Phase 2: Scenario Generation (if feasible)
        if is_feasible:
            # Generate prompt with actual tool information from sampled servers
            generation_prompt = get_generation_prompt(sampled_servers)
            print(f"  > Generated task with tools from {len(sampled_servers)} servers")
            
            # Create messages for GPT-4
            messages_for_generation = [{"role": "user", "content": generation_prompt}]
            # Convert messages to prompt and call util function directly
            prompt = "\n".join([msg.get("content", "") for msg in messages_for_generation if msg.get("content")])
            generation_response = call_azure_openai(
                prompt_content=prompt,
                client=azure_client,
                model_id=AZURE_DEPLOYMENT,
                max_tokens=1024,
                temperature=0.7
            )

            # Parse JSON response directly
            try:
                scenario_json = extract_json_from_response(generation_response)
                query = scenario_json.get("query", "")
                ground_truth_tools = scenario_json.get("ground_truth_tools", [])
                ground_truth_tools_count = scenario_json.get("ground_truth_tools_count", len(ground_truth_tools))

                # print(f'scenario_json:{scenario_json}')
                if query and ground_truth_tools:
                    # 5. Phase 3: Quality Control (JSON version)
                    print(f"  > Starting quality control assessment...")
                    quality_passed = False
                    retry_count = 0
                    max_retries = 3

                    # Build tools description with correct info from sampled servers
                    tools_description_parts = []
                    for tool in ground_truth_tools:
                        # print(f'tool:{tool}')
                        tool_id = tool.get("tool_id", "")
                        server_name = tool.get("server_name", "")
                        tool_name = tool.get("tool_name", "")
                        
                        # Find the original tool definition to get correct description and input schema
                        correct_description = tool.get("description", "")
                        input_schema = "No input schema available"
                        
                        for server in sampled_servers:
                            if server.get('name') == server_name:
                                server_tools = server.get('tools', [])
                                for server_tool in server_tools:
                                    if isinstance(server_tool, dict) and server_tool.get('tool_name') == tool_name:
                                        # Use the correct description from the server
                                        correct_description = server_tool.get('tool_description', server_tool.get('description', correct_description))
                                        
                                        # Get input schema
                                        input_schema_dict = server_tool.get('input_schema', {})
                                        if input_schema_dict:
                                            # Extract required fields for clarity
                                            required_fields = input_schema_dict.get('required', [])
                                            
                                            if required_fields:
                                                input_schema = f"Required parameters: {required_fields}\n"
                                        else:
                                            input_schema = "No input schema available"
                                        break
                                break
                        
                        tool_desc = f"Tool: {tool_id}\n"
                        tool_desc += f"Server: {server_name}\n"
                        tool_desc += f"Description: {correct_description}\n"
                        tool_desc += f"{input_schema}\n"
                        tools_description_parts.append(tool_desc)
                    
                    tools_description = '\n'.join(tools_description_parts)

                    while not quality_passed and retry_count < max_retries:
                        quality_prompt = QUALITY_CONTROL_PROMPT_TEMPLATE.format(
                            query=query,
                            tools_description=tools_description
                        )
                        # print(f'quality_prompt:{quality_prompt}')
                        quality_messages = [{"role": "user", "content": quality_prompt}]
                        # Convert messages to prompt and call util function directly
                        prompt = "\n".join([msg.get("content", "") for msg in quality_messages if msg.get("content")])
                        quality_response = call_azure_openai(
                            prompt_content=prompt,
                            client=azure_client,
                            model_id=AZURE_DEPLOYMENT,
                            max_tokens=512,
                            temperature=0.0
                        )
                        
                        # Extract JSON from response (handle extra text)
                        try:
                            qc_json = extract_json_from_response(quality_response)
                        except Exception as e:
                            print(f"  ! Failed to parse QC JSON: {e}")
                            print(f"  ! Raw response: {quality_response}")
                            break

                        solvability_score = qc_json.get("solvability_score", 0)
                        utility_score = qc_json.get("utility_score", 0)
                        solvability_feedback = qc_json.get("solvability_feedback", "")
                        utility_feedback = qc_json.get("utility_feedback", "")

                        print(f"  > Quality Assessment (Attempt {retry_count + 1}):")
                        print(f"    - Solvability: {solvability_score}/10")
                        print(f"    - Utility: {utility_score}/10")
                        print(f"    - Solvability Feedback: {solvability_feedback}")
                        print(f"    - Utility Feedback: {utility_feedback}")
                        retry_count += 1

                        # Pass if both solvability and utility are >= 7
                        if solvability_score >= 7 and utility_score >= 7:
                            quality_passed = True
                            
                            # Use LLM output scenario directly with additional metadata
                            scenario = scenario_json.copy()
                            scenario["query_id"] = len(generated_scenarios) + 1
                            scenario["quality_assessment"] = {
                                "solvability_score": solvability_score,
                                "utility_score": utility_score,
                                "solvability_feedback": solvability_feedback,
                                "utility_feedback": utility_feedback
                            }
                            generated_scenarios.append(scenario)
                            print(f"  ✅ Scenario generated successfully with quality score: Solvability={solvability_score}, Utility={utility_score}")
                        else:
                            
                            print(f"  ⚠️ Quality check failed. Regenerating scenario (retry {retry_count}/{max_retries})...")
                            
                            # Regenerate the scenario with a different temperature to get variation
                            print(f"  > Regenerating scenario with higher temperature...")
                            generation_prompt = get_generation_prompt(sampled_servers)
                            messages_for_generation = [{"role": "user", "content": generation_prompt}]
                            # Convert messages to prompt and call util function directly
                            prompt = "\n".join([msg.get("content", "") for msg in messages_for_generation if msg.get("content")])
                            generation_response = call_azure_openai(
                                prompt_content=prompt,
                                client=azure_client,
                                model_id=AZURE_DEPLOYMENT,
                                max_tokens=1024,
                                temperature=0.8 + retry_count * 0.1
                            )
                            
                            try:
                                scenario_json = extract_json_from_response(generation_response)
                                query = scenario_json.get("query", "")
                                ground_truth_tools = scenario_json.get("ground_truth_tools", [])
                                ground_truth_tools_count = scenario_json.get("ground_truth_tools_count", len(ground_truth_tools))
                                
                                if not query or not ground_truth_tools:
                                    print(f"  ! New scenario generation failed. Query or tools missing.")
                                    continue
                                    
                                # Rebuild tools description with new tools
                                tools_description_parts = []
                                for tool in ground_truth_tools:
                                    tool_id = tool.get("tool_id", "")
                                    server_name = tool.get("server_name", "")
                                    tool_name = tool.get("tool_name", "")
                                    
                                    # Find the original tool definition
                                    correct_description = tool.get("description", "")
                                    input_schema = "No input schema available"
                                    
                                    for server in sampled_servers:
                                        if server.get('name') == server_name:
                                            server_tools = server.get('tools', [])
                                            for server_tool in server_tools:
                                                if isinstance(server_tool, dict) and server_tool.get('tool_name') == tool_name:
                                                    correct_description = server_tool.get('tool_description', server_tool.get('description', correct_description))
                                                    
                                                    input_schema_dict = server_tool.get('input_schema', {})
                                                    if input_schema_dict:
                                                        required_fields = input_schema_dict.get('required', [])
                                                        if required_fields:
                                                            input_schema = f"Required parameters: {required_fields}\n"

                                                    else:
                                                        input_schema = "No input schema available"
                                                    break
                                            break
                                    
                                    tool_desc = f"Tool: {tool_id}\n"
                                    tool_desc += f"Server: {server_name}\n"
                                    tool_desc += f"Description: {correct_description}\n"
                                    tool_desc += f"{input_schema}\n"
                                    tools_description_parts.append(tool_desc)
                                
                                tools_description = '\n'.join(tools_description_parts)
                                print(f"  > Regenerated scenario. New query: {query[:100]}...")
                            
                            except Exception as e:
                                print(f"  ! Failed to parse regenerated scenario JSON: {e}")
                                continue

                    if not quality_passed:
                        print(f"  ❌ Scenario failed quality control after {max_retries} attempts.")

            except Exception as e:
                print(f"  ! Failed to parse scenario generation JSON: {e}")
                print(f"  ! Raw response: {generation_response}")
        else:
            print(f"  ⚠️ Scenario is not feasible. Skipping.")

        # 6. Save progress intermittently
        if len(generated_scenarios) % 10 == 0:
            print(f"\n--- Saving progress: {len(generated_scenarios)} scenarios generated. ---")
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(generated_scenarios, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    # Final save
    print(f"\n\n✅ Generation complete. Total scenarios created: {len(generated_scenarios)}.")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(generated_scenarios, f, ensure_ascii=False, indent=2)
    print(f"All results saved to '{output_path}'")