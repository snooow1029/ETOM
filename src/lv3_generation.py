#!/usr/bin/env python3
"""End-to-end Level 3 dataset generation pipeline."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
from openai.types.chat import ChatCompletionMessageParam
from util import (
    setup_environment, get_azure_openai_client, call_azure_openai,
    find_repo_root
)
from openai import AzureOpenAI

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(CURRENT_DIR)

setup_environment(REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Prompts shared across the pipeline
# ---------------------------------------------------------------------------

VERIFIER_PROMPT = """# ROLE & GOAL
You are a pragmatic product manager and expert AI system analyst. Your task is to evaluate if a given sequence of tool calls (a "tool chain") represents a logical, efficient, and realistic user workflow.

# INSTRUCTIONS
Analyze the tool chain based on the following CRITICAL criteria:

1.  **Direct Sequential Dependency (Most Important)**:
	- Does a later step **depend on the output** of an earlier step? A true sequence cannot be parallel.
	- Reject chains where all tools could be executed independently because they only rely on a common, user-provided input.

2.  **Logical Data Flow & Resource Lifecycle**:
	- Are resources (`files`, `users`, etc.) only acted upon *after* they are created or found, and *before* they are deleted?

3.  **Meaningful & Non-Trivial Goal**:
	- Does the workflow achieve a valuable result? Avoid self-canceling or trivial chains (e.g., `create` then `delete`).

4.  **Informational Efficiency**:
	- Is each step necessary? Avoid chains where an early step's output is made redundant by a later step.

5.  **Coherent User Intent**:
	- Does the entire chain serve a single, focused user goal? Reject chains that combine unrelated functionalities.

6.  **Final Judgement**: Based on ALL the criteria above, decide if the tool chain is **"LOGICAL"** or **"ILLOGICAL"**. A chain must pass ALL checks to be considered LOGICAL.

7.  **Format Your Response**: Structure your output as follows:
	- First, provide the boolean judgement (`LOGICAL` or `ILLOGICAL`) **wrapped in `<judgement></judgement>` tags**.
	- Second, on a new line, provide a **brief, one-sentence explanation** for your decision, **wrapped in `<reason></reason>` tags**.

# YOUR TASK
## Tool Chain:
{formatted_tool_chain}
## Your Output:
"""

GENERATOR_PROMPT = """
# ROLE & GOAL
You are a creative writer for an AI benchmark. Your task is to generate a single, high-level user query and a corresponding Chain of Thought (COT) for a given tool chain.

# CRITICAL INSTRUCTIONS
1.  **Create a Concrete Scenario**: To make the query executable, invent a specific, realistic, but fictional entity for the task. For example, if the task is about a file, invent a filename like `"quarterly_report.pdf"`. If it's about a repository, invent a name like `"my-new-app"`.
2.  **Generate a Specific, Executable Query**: The user query MUST be a direct command that includes the fictional entity you created. It should be a complete, self-contained task.
3.  **Reflect the Workflow**: The query should ONLY describe the final goal, but its goal must implicitly require all steps in the provided tool chain.
4.  **Generate a Chain of Thought (COT)**: Write a step-by-step reasoning process explaining how an AI would break down your specific query into the provided tool chain.
5.  **Format**:
	- The user query MUST be wrapped in `<query></query>` tags.
	- The Chain of Thought MUST be wrapped in `<COT></COT>` tags.

# YOUR TASK
## Tool Chain:
{formatted_tool_chain}
## Your Output:"""

DEPENDENCY_ANALYZER_PROMPT = """
# ROLE & GOAL
You are an expert AI system architect. Your task is to analyze a chain of tools and determine their execution dependencies to create a dependency graph.

# INSTRUCTIONS
1.  Analyze the provided `Tool Chain` in the context of the `User Query`. Each tool is numbered and includes a name and description.
2.  For each tool, determine if it **strictly requires the output** of any preceding tools. A dependency exists only if a tool consumes data produced by another.
3.  Tools that only depend on the initial user query are parallel starting points.
4.  **Format your output as a single JSON object** inside a ```json code block. The object must have one key, `"dependency_graph"`, which is a list of nodes.
5.  Each node must have: `"id"` (integer), `"tool_name"` (string), and `"dependencies"` (a list of integer IDs this tool depends on).

# YOUR TASK
## User Query: "{query}"
## Tool Chain:
{formatted_tool_chain}
## Your Output:
"""


# ---------------------------------------------------------------------------
# Tool chain extraction (Step 1)
# ---------------------------------------------------------------------------

def canonical_tool_name(tool: Dict[str, Any]) -> Optional[str]:
	return tool.get("tool_name") or tool.get("name")


def simplify_param_name(name: str) -> str:
	return name.lower().removesuffix("s").replace("_", "-")


def get_params_from_schema(schema: Dict[str, Any]) -> Set[str]:
	if not isinstance(schema, dict) or "properties" not in schema:
		return set()
	properties = schema.get("properties", {})
	return {simplify_param_name(prop) for prop in properties.keys()}


def find_tool_chains_in_server(server_data: Dict[str, Any]) -> List[List[str]]:
	tools = server_data.get("tools", [])
	if not tools:
		return []

	graph = nx.DiGraph()
	tool_names = [canonical_tool_name(tool) for tool in tools if canonical_tool_name(tool)]
	graph.add_nodes_from(tool_names)

	for i, tool_a in enumerate(tools):
		tool_a_name = canonical_tool_name(tool_a)
		if not tool_a_name:
			continue

		output_params_a = get_params_from_schema(tool_a.get("output_schema", {}))
		if not output_params_a:
			continue

		for j, tool_b in enumerate(tools):
			if i == j:
				continue
			tool_b_name = canonical_tool_name(tool_b)
			if not tool_b_name:
				continue
			input_params_b = get_params_from_schema(tool_b.get("input_schema", {}))
			if input_params_b and input_params_b.issubset(output_params_a):
				graph.add_edge(tool_a_name, tool_b_name)

	all_chains: List[List[str]] = []
	nodes = list(graph.nodes)
	for source in nodes:
		for target in nodes:
			if source == target:
				continue
			for path in nx.all_simple_paths(graph, source=source, target=target):
				if len(path) >= 2:
					all_chains.append(path)

	return all_chains


def filter_by_endpoints(chains: List[List[str]]) -> List[List[str]]:
	start_keywords = ["start", "create", "new", "list", "search", "find", "get", "restore"]
	filtered = []
	for chain in chains:
		start_tool_lower = chain[0].lower()
		if any(keyword in start_tool_lower for keyword in start_keywords):
			filtered.append(chain)
	return filtered


def filter_maximal_paths(chains: List[List[str]]) -> List[List[str]]:
	if not chains:
		return []
	chains.sort(key=len, reverse=True)
	maximal: List[List[str]] = []
	for chain in chains:
		chain_str = ",".join(chain)
		if any(chain_str in ",".join(existing) for existing in maximal):
			continue
		maximal.append(chain)
	return maximal


def format_chain_details(chain: List[str], server_data: Dict[str, Any]) -> Dict[str, Any]:
	tool_map = {
		canonical_tool_name(tool): tool
		for tool in server_data.get("tools", [])
		if canonical_tool_name(tool)
	}

	tool_details = []
	for tool_name in chain:
		tool_info = tool_map.get(tool_name)
		if tool_info:
			tool_details.append(
				{
					"tool_name": tool_name,
					"description": tool_info.get("tool_description")
					or tool_info.get("description", ""),
				}
			)

	return {"chain": chain, "tools_in_chain": tool_details}


def generate_tool_chains_from_registry(registry_path: Path, *, verbose: bool = True) -> List[Dict[str, Any]]:
	registry_path = registry_path.resolve()
	if not registry_path.exists():
		raise FileNotFoundError(f"Registry file '{registry_path}' does not exist.")

	try:
		with registry_path.open("r", encoding="utf-8") as handle:
			registry_data = json.load(handle)
	except (json.JSONDecodeError, OSError) as exc:
		raise ValueError(f"Unable to load registry '{registry_path}': {exc}") from exc

	servers = registry_data.get("servers", [])
	if not isinstance(servers, list):
		raise ValueError("Registry JSON missing 'servers' list.")

	all_chains: List[Dict[str, Any]] = []

	for idx, server_data in enumerate(servers, start=1):
		server_name = (
			server_data.get("server_name")
			or server_data.get("name")
			or f"server_{idx}"
		)
		if verbose:
			print(f"\n--- Analyzing server: {server_name} ---")

		raw_chains = find_tool_chains_in_server(server_data)
		if not raw_chains:
			if verbose:
				print("  > No raw tool chains found.")
			continue

		endpoint_filtered = filter_by_endpoints(raw_chains)
		final_chains = filter_maximal_paths(endpoint_filtered)

		if verbose:
			print(f"  > Found {len(final_chains)} refined chains.")

		for chain in final_chains:
			all_chains.append(
				{
					"source_file": server_data.get("source_file"),
					"server_name": server_name,
					"chain_details": format_chain_details(chain, server_data),
				}
			)

	if verbose:
		print(
			"\n\n✅ Processing complete. Found a total of "
			f"{len(all_chains)} refined chains across all servers."
		)

	return all_chains


# ---------------------------------------------------------------------------
# Azure client helpers shared by steps 2 and 3
# ---------------------------------------------------------------------------

def build_azure_client() -> Tuple[AzureOpenAI, str]:
	"""Use util.py function but maintain API compatibility."""
	client = get_azure_openai_client()
	deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
	if not deployment:
		raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT in environment.")
	return client, deployment





# ---------------------------------------------------------------------------
# Level 3 query generation (Step 2)
# ---------------------------------------------------------------------------

def format_chain_for_prompt(tools_in_chain: List[Dict[str, Any]]) -> str:
	formatted = []
	for i, tool in enumerate(tools_in_chain):
		formatted.append(
			f"- Step {i + 1} ({tool.get('tool_name')}): \"{tool.get('description')}\""
		)
	return "\n".join(formatted)


def verify_chain_logic(client: AzureOpenAI, model_id: str, formatted_chain: str) -> bool:
	prompt = VERIFIER_PROMPT.format(formatted_tool_chain=formatted_chain)
	response_text = call_azure_openai(
		prompt_content=prompt,
		client=client,
		model_id=model_id,
		max_tokens=256,
		temperature=0.0
	)
	matches = re.findall(r"<judgement>(.*?)</judgement>", response_text, re.DOTALL)
	if matches:
		judgement = matches[-1].strip().upper()
		return judgement == "LOGICAL"
	return False


def generate_query_and_cot(client: AzureOpenAI, model_id: str, formatted_chain: str) -> Dict[str, str]:
	prompt = GENERATOR_PROMPT.format(formatted_tool_chain=formatted_chain)
	response_text = call_azure_openai(
		prompt_content=prompt,
		client=client,
		model_id=model_id,
		max_tokens=2048,
		temperature=0.7
	)

	query_match = re.search(r"<query>(.*?)</query>", response_text, re.DOTALL)
	cot_match = re.search(r"<COT>(.*?)</COT>", response_text, re.DOTALL)

	query = query_match.group(1).strip() if query_match else "PARSE_ERROR: QUERY_NOT_FOUND"
	cot = cot_match.group(1).strip() if cot_match else "PARSE_ERROR: COT_NOT_FOUND"

	return {"query": query, "cot": cot}


def _chain_key_from_sequence(server_name: Optional[str], sequence: Sequence[str]) -> Tuple[str, Tuple[str, ...]]:
	return (server_name or "", tuple(sequence))


def _chain_key_from_chain(chain_data: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
	server_name = chain_data.get("server_name")
	tools_in_chain = chain_data.get("chain_details", {}).get("tools_in_chain", [])
	sequence = [tool.get("tool_name") for tool in tools_in_chain if tool.get("tool_name")]
	return _chain_key_from_sequence(server_name, sequence)


def _chain_key_from_task(task: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
	server_name = task.get("server_name")
	if task.get("ground_truth_sequence"):
		sequence = task["ground_truth_sequence"]
	elif task.get("ground_truth_tools"):
		sequence = [tool.get("tool_name") for tool in task["ground_truth_tools"] if tool.get("tool_name")]
	else:
		sequence = []
	return _chain_key_from_sequence(server_name, sequence)


def generate_level3_tasks_from_chains(
	chains: List[Dict[str, Any]],
	azure_client: AzureOpenAI,
	azure_deployment: str,
	*,
	existing_tasks: Optional[List[Dict[str, Any]]] = None,
	temp_output: Optional[Path] = None,
	flush_every: int = 10,
	sleep_seconds: float = 1.0,
	verbose: bool = True,
) -> List[Dict[str, Any]]:
	final_tasks: List[Dict[str, Any]] = list(existing_tasks or [])
	processed_keys = {_chain_key_from_task(task) for task in final_tasks}

	for idx, chain_data in enumerate(chains, start=1):
		server_name = chain_data.get("server_name")
		tools_in_chain = chain_data.get("chain_details", {}).get("tools_in_chain", [])

		if verbose:
			print(f"\n--- [{idx}/{len(chains)}] Processing Chain from: {server_name} ---")

		if len(tools_in_chain) < 2:
			if verbose:
				print("  > Skipping chain with less than 2 tools.")
			continue

		key = _chain_key_from_chain(chain_data)
		if key in processed_keys:
			if verbose:
				print("  > Chain already processed. Skipping.")
			continue

		formatted_chain = format_chain_for_prompt(tools_in_chain)
		if verbose:
			print(formatted_chain)

		if not verify_chain_logic(azure_client, azure_deployment, formatted_chain):
			if verbose:
				print("  > Logic is ILLOGICAL. Skipping this chain.")
			continue

		if verbose:
			print("  > Logic is VERIFIED. Proceeding to generate query...")

		generation_result = generate_query_and_cot(azure_client, azure_deployment, formatted_chain)
		if "PARSE_ERROR" in generation_result["query"] or "PARSE_ERROR" in generation_result["cot"]:
			if verbose:
				print("  ! Skipping due to parsing error in generation result.")
			continue

		ground_truth_sequence = [tool.get("tool_name") for tool in tools_in_chain if tool.get("tool_name")]
		ground_truth_tools: List[Dict[str, Any]] = []
		for tool in tools_in_chain:
			tool_name = tool.get("tool_name")
			if not tool_name:
				continue
			ground_truth_tools.append(
				{
					"tool_id": f"{server_name}::{tool_name}",
					"server_name": server_name,
					"tool_name": tool_name,
					"description": tool.get("description", ""),
					"dependencies": [],
				}
			)

		level3_task = {
			"level": 3,
			"source_file": chain_data.get("source_file"),
			"server_name": server_name,
			"level_3_queries": generation_result["query"],
			"query": generation_result["query"],
			"chain_of_thought": generation_result["cot"],
			"ground_truth_sequence": ground_truth_sequence,
			"ground_truth_tools": ground_truth_tools,
			"ground_truth_tools_count": len(ground_truth_tools),
		}

		final_tasks.append(level3_task)
		processed_keys.add(key)

		if verbose:
			print(f"  ✅ Generated Query: {generation_result['query']}")

		if temp_output and flush_every > 0 and len(final_tasks) % flush_every == 0:
			temp_output.parent.mkdir(parents=True, exist_ok=True)
			with temp_output.open("w", encoding="utf-8") as handle:
				json.dump(final_tasks, handle, ensure_ascii=False, indent=2)

		if sleep_seconds > 0:
			time.sleep(sleep_seconds)

	return final_tasks


# ---------------------------------------------------------------------------
# Dependency augmentation (Step 3)
# ---------------------------------------------------------------------------

def format_tool_chain_with_desc(tools: List[Dict[str, Any]]) -> str:
	formatted = []
	for i, tool in enumerate(tools):
		formatted.append(
			f"- {i + 1} ({tool.get('tool_name')}): \"{tool.get('description', '')}\""
		)
	return "\n".join(formatted)


def extract_json_from_response(response_text: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
	json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
	json_str = json_match.group(1) if json_match else response_text

	try:
		parsed = json.loads(json_str)
	except json.JSONDecodeError:
		return None, "JSONDecodeError"

	dependency_graph = parsed.get("dependency_graph") if isinstance(parsed, dict) else None
	if not isinstance(dependency_graph, list):
		return None, "Invalid JSON structure"
	return dependency_graph, None


def analyze_dependencies(
	client: AzureOpenAI,
	model_id: str,
	query: str,
	tools: List[Dict[str, Any]],
	*,
	verbose: bool,
) -> Tuple[List[Dict[str, Any]], str]:
	formatted_chain = format_tool_chain_with_desc(tools)
	prompt = DEPENDENCY_ANALYZER_PROMPT.format(query=query, formatted_tool_chain=formatted_chain)
	response_text = call_azure_openai(
		prompt_content=prompt,
		client=client,
		model_id=model_id,
		max_tokens=1024,
		temperature=0.0
	)
	if verbose:
		print(f"  > Dependency Analyzer Response: {response_text.strip()}")

	dependency_graph, error = extract_json_from_response(response_text)
	if dependency_graph is None:
		return [], error or "Unknown error"
	return dependency_graph, ""


def visualize_graph(dependency_graph: List[Dict[str, Any]]) -> str:
	if not dependency_graph:
		return "No dependency data."

	tool_map = {node["id"]: node.get("tool_name", "") for node in dependency_graph}
	children_map = {node["id"]: [] for node in dependency_graph}

	for node in dependency_graph:
		for dep_id in node.get("dependencies", []):
			if dep_id in children_map:
				children_map[dep_id].append(node["id"])

	root_nodes = [node["id"] for node in dependency_graph if not node.get("dependencies")]
	if not root_nodes and dependency_graph:
		root_nodes = [dependency_graph[0]["id"]]

	lines: List[str] = []

	def build_tree(node_id: int, prefix: str) -> None:
		children = children_map.get(node_id, [])
		is_parallel = " (parallel)" if len(children) > 1 else ""
		lines.append(f"{prefix}{node_id}: {tool_map.get(node_id, 'Unknown')}{is_parallel}")

		for idx, child_id in enumerate(children):
			is_last = idx == len(children) - 1
			connector = "└── " if is_last else "├── "
			new_prefix = prefix + ("    " if is_last else "│   ")
			build_tree(child_id, new_prefix + connector)

	for root_id in sorted(root_nodes):
		build_tree(root_id, "")

	return "\n".join(lines)


def add_dependencies_to_tools(
	tools: List[Dict[str, Any]],
	dependency_graph: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	dependency_map: Dict[str, List[int]] = {}
	for node in dependency_graph:
		tool_name = node.get("tool_name")
		if not tool_name:
			continue
		dependencies = node.get("dependencies", [])
		dep_indices = []
		for dep_id in dependencies:
			dep_index = dep_id - 1
			if 0 <= dep_index < len(tools):
				dep_indices.append(dep_index)
		dependency_map[tool_name] = dep_indices

	updated_tools: List[Dict[str, Any]] = []
	for tool in tools:
		tool_copy = dict(tool)
		tool_name = tool.get("tool_name")
		if tool_name is None:
			tool_copy["dependencies"] = []
		else:
			tool_copy["dependencies"] = dependency_map.get(tool_name, [])
		updated_tools.append(tool_copy)

	return updated_tools


def build_sequential_graph(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	dependency_graph: List[Dict[str, Any]] = []
	for idx, tool in enumerate(tools):
		tool_name = tool.get("tool_name", f"tool_{idx + 1}")
		dependencies = [idx] if idx > 0 else []
		dependency_graph.append(
			{
				"id": idx + 1,
				"tool_name": tool_name,
				"dependencies": dependencies,
			}
		)
	return dependency_graph


def _ensure_dependencies_list(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	normalized: List[Dict[str, Any]] = []
	for tool in tools:
		tool_copy = dict(tool)
		if not isinstance(tool_copy.get("dependencies"), list):
			tool_copy["dependencies"] = []
		normalized.append(tool_copy)
	return normalized


def _task_key(task: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
	server_name = task.get("server_name") or ""
	if task.get("ground_truth_sequence"):
		sequence = task["ground_truth_sequence"]
	elif task.get("ground_truth_tools"):
		sequence = [tool.get("tool_name") for tool in task["ground_truth_tools"] if tool.get("tool_name")]
	else:
		sequence = []
	return server_name, tuple(sequence)


def enrich_task_with_dependencies(
	task: Dict[str, Any],
	azure_client: AzureOpenAI,
	azure_deployment: str,
	*,
	verbose: bool,
) -> Tuple[Dict[str, Any], bool]:
	query = task.get("query") or task.get("level_3_queries") or ""
	tools = task.get("ground_truth_tools", [])

	updated_task = dict(task)
	normalized_tools = _ensure_dependencies_list(tools)

	if len(normalized_tools) <= 1:
		dependency_graph = build_sequential_graph(normalized_tools)
		visualization = visualize_graph(dependency_graph)
		updated_task["ground_truth_tools"] = [dict(tool, dependencies=[]) for tool in normalized_tools]
		updated_task["ground_truth_tools_count"] = len(normalized_tools)
		updated_task["dependency_graph"] = {
			"json_format": dependency_graph,
			"visualized_format": visualization,
		}
		updated_task["dependency_visualization"] = visualization
		return updated_task, False

	dependency_graph, error = analyze_dependencies(
		azure_client,
		azure_deployment,
		query,
		normalized_tools,
		verbose=verbose,
	)

	if error:
		if verbose:
			print(f"  ! Error during dependency analysis: {error}. Using sequential fallback.")
		dependency_graph = build_sequential_graph(normalized_tools)

	visualization = visualize_graph(dependency_graph)
	updated_tools = add_dependencies_to_tools(normalized_tools, dependency_graph)

	updated_task["ground_truth_tools"] = updated_tools
	updated_task["ground_truth_tools_count"] = len(updated_tools)
	updated_task["dependency_graph"] = {
		"json_format": dependency_graph,
		"visualized_format": visualization,
	}
	updated_task["dependency_visualization"] = visualization

	return updated_task, True


def augment_tasks_with_dependencies(
	tasks: List[Dict[str, Any]],
	azure_client: AzureOpenAI,
	azure_deployment: str,
	*,
	existing_tasks: Optional[List[Dict[str, Any]]] = None,
	temp_output: Optional[Path] = None,
	flush_every: int = 100,
	sleep_seconds: float = 1.5,
	verbose: bool = True,
) -> List[Dict[str, Any]]:
	final_tasks: List[Dict[str, Any]] = list(existing_tasks or [])
	processed_keys = {_task_key(task) for task in final_tasks}

	for idx, task in enumerate(tasks, start=1):
		if verbose:
			print(f"\n--- [{idx}/{len(tasks)}] Processing task for server: {task.get('server_name')} ---")

		key = _task_key(task)
		if key in processed_keys:
			if verbose:
				print("  > Task already processed. Skipping.")
			continue

		updated_task, azure_called = enrich_task_with_dependencies(
			task,
			azure_client,
			azure_deployment,
			verbose=verbose,
		)

		final_tasks.append(updated_task)
		processed_keys.add(key)

		if temp_output and flush_every > 0 and len(final_tasks) % flush_every == 0:
			temp_output.parent.mkdir(parents=True, exist_ok=True)
			with temp_output.open("w", encoding="utf-8") as handle:
				json.dump(final_tasks, handle, ensure_ascii=False, indent=2)

		if azure_called and sleep_seconds > 0:
			time.sleep(sleep_seconds)

	for query_id, task in enumerate(final_tasks, start=1):
		task["query_id"] = query_id

	return final_tasks


# ---------------------------------------------------------------------------
# Final dataset normalization
# ---------------------------------------------------------------------------


def _normalize_dependencies_list(dependencies: Any) -> List[int]:
	if not isinstance(dependencies, list):
		return []
	normalized: List[int] = []
	for dep in dependencies:
		if isinstance(dep, int):
			normalized.append(dep)
			continue
		try:
			normalized.append(int(dep))
		except (TypeError, ValueError):
			continue
	return normalized


def convert_tasks_to_final_dataset(tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
	final_entries: List[Dict[str, Any]] = []
	for idx, task in enumerate(tasks, start=1):
		query_text = (task.get("query") or task.get("level_3_queries") or "").strip()
		visualization = task.get("dependency_visualization") or ""
		if not visualization:
			graph_info = task.get("dependency_graph")
			if isinstance(graph_info, dict):
				visualization = str(graph_info.get("visualized_format") or "").strip()

		normalized_tools: List[Dict[str, Any]] = []
		for tool in task.get("ground_truth_tools", []) or []:
			tool_name = tool.get("tool_name", "")
			server_name = tool.get("server_name", "")
			tool_id = tool.get("tool_id")
			if not tool_id:
				if server_name and tool_name:
					tool_id = f"{server_name}::{tool_name}"
				else:
					tool_id = tool_name or server_name

			normalized_tools.append(
				{
					"tool_id": tool_id,
					"server_name": server_name,
					"tool_name": tool_name,
					"description": tool.get("description", ""),
					"dependencies": _normalize_dependencies_list(tool.get("dependencies")),
				}
			)

		final_entries.append(
			{
				"query_id": idx,
				"query": query_text,
				"ground_truth_tools_count": len(normalized_tools),
				"ground_truth_tools": normalized_tools,
				"dependency_visualization": visualization,
			}
		)

	return final_entries


# ---------------------------------------------------------------------------
# JSON helpers and CLI orchestration
# ---------------------------------------------------------------------------


def load_json_if_exists(path: Path) -> Optional[Any]:
	if not path.exists():
		return None
	try:
		with path.open("r", encoding="utf-8") as handle:
			return json.load(handle)
	except (json.JSONDecodeError, OSError) as exc:
		print(f"  ! Warning: Unable to load JSON from '{path}': {exc}")
		return None


def write_json(path: Path, data: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the Level 3 generation pipeline end-to-end.")
	default_registry = REPO_ROOT / "data" / "mcp_registry_w_schemas.json"
	default_chains = CURRENT_DIR / "all_tool_chains.json"
	default_tasks = CURRENT_DIR / "level3_tasks.json"
	default_final = REPO_ROOT / "queries" / "level_3.json"

	parser.add_argument("--registry", type=Path, default=default_registry, help="Path to the MCP registry JSON file.")
	parser.add_argument("--chains-output", type=Path, default=default_chains, help="Path to write the extracted tool chains JSON.")
	parser.add_argument("--tasks-output", type=Path, default=default_tasks, help="Path to write the generated Level 3 tasks JSON.")
	parser.add_argument("--final-output", type=Path, default=default_final, help="Path to write the dependency-augmented tasks JSON.")

	parser.add_argument("--skip-chains", action="store_true", help="Skip the tool chain extraction step and reuse the existing chains file.")
	parser.add_argument("--skip-generation", action="store_true", help="Skip the Level 3 query generation step and reuse the existing tasks file.")
	parser.add_argument("--skip-dependencies", action="store_true", help="Skip the dependency augmentation step and reuse the existing final file.")
	parser.add_argument("--resume", action="store_true", help="Resume generation and dependency steps using existing outputs to avoid reprocessing.")

	parser.add_argument("--sleep-step2", type=float, default=1.0, help="Seconds to sleep between Azure calls during query generation.")
	parser.add_argument("--sleep-step3", type=float, default=1.5, help="Seconds to sleep between Azure calls during dependency augmentation.")
	parser.add_argument("--flush-step2", type=int, default=10, help="Flush intermediate query generation results every N tasks (0 to disable).")
	parser.add_argument("--flush-step3", type=int, default=100, help="Flush intermediate dependency results every N tasks (0 to disable).")

	parser.add_argument("--tasks-temp", type=Path, default=None, help="Optional custom checkpoint path for query generation.")
	parser.add_argument("--deps-temp", type=Path, default=None, help="Optional custom checkpoint path for dependency augmentation.")

	parser.add_argument("--quiet", action="store_true", help="Reduce logging output.")
	return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)
	verbose = not args.quiet

	registry_path = args.registry.resolve()
	chains_output = args.chains_output.resolve()
	tasks_output = args.tasks_output.resolve()
	final_output = args.final_output.resolve()

	if not registry_path.exists() and not args.skip_chains:
		print(f"Error: Registry file '{registry_path}' does not exist.")
		return 1

	# Step 1: Extract tool chains.
	if args.skip_chains:
		existing_chains = load_json_if_exists(chains_output)
		if not isinstance(existing_chains, list):
			print("Error: Unable to load existing chains. Remove --skip-chains or regenerate the file.")
			return 1
		chains: List[Dict[str, Any]] = existing_chains
		if verbose:
			print(f"Skipping chain extraction. Loaded {len(chains)} chains from '{chains_output}'.")
	else:
		chains = generate_tool_chains_from_registry(registry_path, verbose=verbose)
		write_json(chains_output, chains)
		if verbose:
			print(f"Saved {len(chains)} refined chains to '{chains_output}'.")

	# Determine if we need Azure OpenAI access.
	needs_azure = not args.skip_generation or not args.skip_dependencies
	if needs_azure:
		try:
			azure_client, azure_deployment = build_azure_client()
		except ValueError as exc:
			print(f"Error: {exc}")
			return 1
	else:
		azure_client = None  # type: ignore[assignment]
		azure_deployment = ""

	# Step 2: Generate Level 3 tasks.
	tasks: List[Dict[str, Any]]
	tasks_temp = args.tasks_temp.resolve() if args.tasks_temp else tasks_output.with_name(f"{tasks_output.stem}_checkpoint.json")

	if args.skip_generation:
		existing_tasks = load_json_if_exists(tasks_output)
		if not isinstance(existing_tasks, list):
			print("Error: Unable to load existing Level 3 tasks. Remove --skip-generation or regenerate the file.")
			return 1
		tasks = existing_tasks
		if verbose:
			print(f"Skipping query generation. Loaded {len(tasks)} tasks from '{tasks_output}'.")
	else:
		existing_tasks_list: List[Dict[str, Any]] = []
		if args.resume:
			loaded = load_json_if_exists(tasks_output)
			if isinstance(loaded, list):
				existing_tasks_list = loaded
				if verbose:
					print(f"  > Resuming query generation with {len(existing_tasks_list)} existing tasks.")

		assert azure_client is not None  # Narrow type for static analysis.
		tasks = generate_level3_tasks_from_chains(
			chains,
			azure_client,
			azure_deployment,
			existing_tasks=existing_tasks_list,
			temp_output=tasks_temp,
			flush_every=max(args.flush_step2, 0),
			sleep_seconds=max(args.sleep_step2, 0.0),
			verbose=verbose,
		)

		write_json(tasks_output, tasks)
		if tasks_temp.exists():
			tasks_temp.unlink()
		if verbose:
			print(f"Saved {len(tasks)} Level 3 tasks to '{tasks_output}'.")

	# Step 3: Augment with dependency graphs.
	final_temp = args.deps_temp.resolve() if args.deps_temp else final_output.with_name(f"{final_output.stem}_checkpoint.json")

	if args.skip_dependencies:
		existing_final = load_json_if_exists(final_output)
		if not isinstance(existing_final, list):
			print("Error: Unable to load existing dependency-augmented tasks. Remove --skip-dependencies or regenerate the file.")
			return 1
		final_dataset = existing_final
		if verbose:
			print(f"Skipping dependency augmentation. Loaded {len(final_dataset)} tasks from '{final_output}'.")
	else:
		existing_final_list: List[Dict[str, Any]] = []
		if args.resume:
			resume_source: Optional[Path] = None
			if final_temp.exists():
				resume_source = final_temp
			elif args.deps_temp is None:
				legacy_checkpoint = CURRENT_DIR / "post_chain_checkpoint.json"
				if legacy_checkpoint.exists():
					resume_source = legacy_checkpoint
			if resume_source is not None:
				loaded = load_json_if_exists(resume_source)
				if isinstance(loaded, list):
					existing_final_list = loaded
					if verbose:
						print(f"  > Resuming dependency augmentation with {len(existing_final_list)} existing tasks from '{resume_source}'.")
			elif verbose:
				print("  > Resume requested but no dependency checkpoint found. Starting from scratch.")

		assert azure_client is not None  # Narrow type for static analysis.
		dependency_augmented_tasks = augment_tasks_with_dependencies(
			tasks,
			azure_client,
			azure_deployment,
			existing_tasks=existing_final_list,
			temp_output=final_temp,
			flush_every=max(args.flush_step3, 0),
			sleep_seconds=max(args.sleep_step3, 0.0),
			verbose=verbose,
		)

		final_dataset = convert_tasks_to_final_dataset(dependency_augmented_tasks)
		write_json(final_output, final_dataset)
		if final_temp.exists():
			final_temp.unlink()
		if verbose:
			print(f"Saved {len(final_dataset)} dependency-augmented tasks to '{final_output}'.")

	if verbose:
		print("\n🎉 Level 3 generation pipeline complete.")

	return 0


if __name__ == "__main__":
	sys.exit(main())
