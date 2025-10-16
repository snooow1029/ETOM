"""Generate missing output schemas for tools listed in the MCP registry.

This variant reads the consolidated `mcp_registry.json` (or the embedding-augmented
variant) instead of scanning the legacy `mcp_json_data/` directory. It mirrors the
original batching behaviour but operates entirely in-memory on the registry file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration & Client Initialization ---
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

VLLM_BASE_URL = os.getenv("LOCAL_URL", "http://localhost:800/v1")
VLLM_API_KEY = os.getenv("LOCAL_API_KEY", "moe")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

# Batch processing configuration
MAX_CONCURRENT_TOOLS = 3  # Number of tools per server processed concurrently
API_RATE_LIMIT_DELAY = 0.1  # API call interval (seconds)

# --- Prompt template ---
INFER_OUTPUT_PROMPT = """
# ROLE & GOAL
You are an expert API designer. Your task is to analyze a tool's function, considering its inputs, and infer its most likely output parameters. You must structure your response as a valid JSON Schema object.

# INSTRUCTIONS
1.  Read the tool's name, description, and its required inputs carefully.
2.  **Reason based on the inputs**: The tool's outputs are often logically related to its inputs. For example, if the input is an ID of an object, the output will likely contain details *about that specific object*.
3.  Identify the key pieces of information (entities, IDs, lists, statuses) that the tool would logically *produce* or *return*.
4.  For each piece of information, define it as a parameter with a `type` and a `description`.
5.  Your final output MUST be the inferred JSON schema object **wrapped in `<output_schema></output_schema>` tags**.
6.  If the tool description implies no specific output (e.g., it just performs an action like "delete" or "send"), provide an empty properties object: `{{"properties": {{}}}}`.

# EXAMPLE 1: Listing items (no specific input ID)
## Tool Name: "list_repositories"
## Tool Description: "Lists all repositories for the authenticated user."
## Tool Input Schema:
No specific inputs required.
## Your Output:
<output_schema>
{{
  "properties": {{
    "repositories": {{
      "type": "array",
      "items": {{ 
        "type": "object",
        "properties": {{
          "id": {{"type": "string", "description": "The unique ID of the repository."}},
          "name": {{"type": "string", "description": "The name of the repository."}}
        }}
      }},
      "description": "A list of repository objects found for the user."
    }}
  }}
}}
</output_schema>

# EXAMPLE 2: Getting details (takes a specific input ID)
## Tool Name: "get_issue_details"
## Tool Description: "Gets the details of a specific issue in a repository."
## Tool Input Schema:
- Parameter: `repo_id` (type: string, required)
  Description: The ID of the repository where the issue exists.
- Parameter: `issue_number` (type: integer, required)
  Description: The number of the issue to retrieve.
## Your Output:
<output_schema>
{{
  "properties": {{
    "issue_id": {{ "type": "string", "description": "The ID of the issue." }},
    "issue_title": {{ "type": "string", "description": "The title of the issue." }},
    "issue_body": {{ "type": "string", "description": "The main content of the issue." }},
    "author": {{ "type": "string", "description": "The username of the issue creator." }}
  }}
}}
</output_schema>

---

# YOUR TASK
## Tool Name: "{tool_name}"
## Tool Description: "{tool_description}"
## Tool Input Schema:
{simplified_input_schema_text}
## Your Output:
"""


# --- Helper functions ---

def default_registry_path() -> Path:
    """Return the default registry path relative to this script."""
    return Path(__file__).resolve().parent / "mcp_registry.json"

def default_output_path(input_path: Path) -> Path:
    """Return the default output path, which is the same directory as the input."""
    return input_path.parent / "mcp_registry_w_schemas.json"


def call_vllm_api(prompt: str) -> str:
    """Invoke the local vLLM endpoint with basic rate limiting."""
    time.sleep(API_RATE_LIMIT_DELAY)
    try:
        response = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0,
            top_p=0.95,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - network errors are logged
        print(f"  ! Error calling vLLM API: {exc}")
        return ""


def simplify_schema_for_prompt(schema: MutableMapping[str, Any]) -> str:
    """Convert a JSON Schema object into a compact text snippet for the prompt."""
    if not schema or not isinstance(schema, MutableMapping):
        return "No specific inputs required."

    properties = schema.get("properties")
    if not isinstance(properties, MutableMapping) or not properties:
        return "No defined parameters."

    required = set(schema.get("required", []))
    simplified_parts: List[str] = []
    for name, details in properties.items():
        if not isinstance(details, MutableMapping):
            continue
        param_type = details.get("type", "any")
        description = details.get("description", "")
        flag = "required" if name in required else "optional"
        part = f"- Parameter: `{name}` (type: {param_type}, {flag})"
        if description:
            part += f"\n  Description: {description}"
        simplified_parts.append(part)

    return "\n".join(simplified_parts) if simplified_parts else "No defined parameters."


def parse_output_schema(response_text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"<output_schema>(.*?)</output_schema>", response_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def process_single_tool(tool_data: Dict[str, Any], tool_index: int) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Infer missing output schema for a single tool.

    Returns the updated tool (if any) alongside the index for reinsertion.
    """
    if tool_data.get("output_schema"):
        return tool_index, None

    tool_name = tool_data.get("tool_name") or tool_data.get("name") or f"tool_{tool_index}"
    print(f"    > Processing tool '{tool_name}' (index {tool_index})")

    simplified_input = simplify_schema_for_prompt(tool_data.get("input_schema", {}))
    prompt = INFER_OUTPUT_PROMPT.format(
        tool_name=tool_name,
        tool_description=tool_data.get("tool_description") or tool_data.get("description") or "",
        simplified_input_schema_text=simplified_input,
    )

    response_text = call_vllm_api(prompt)
    inferred_schema = parse_output_schema(response_text)

    if inferred_schema is None:
        print(f"    ! Could not parse schema for tool '{tool_name}'. Skipping.")
        return tool_index, None

    updated_tool = dict(tool_data)
    updated_tool["output_schema"] = inferred_schema
    print(f"    ✓ Successfully inferred schema for tool '{tool_name}'")
    return tool_index, updated_tool


def process_tools_batch(tools: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """Batch-process tools for a single server using a thread pool."""
    missing = [(idx, tool) for idx, tool in enumerate(tools) if not tool.get("output_schema")]
    if not missing:
        return []

    print(f"  > Found {len(missing)} tools needing schema inference")
    results: List[Tuple[int, Dict[str, Any]]] = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TOOLS) as executor:
        future_to_index = {
            executor.submit(process_single_tool, tool, idx): idx for idx, tool in missing
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                tool_index, updated_tool = future.result()
                if updated_tool is not None:
                    results.append((tool_index, updated_tool))
            except Exception as exc:  # pragma: no cover - logged for visibility
                print(f"    ! Error processing tool at index {idx}: {exc}")

    return results


def process_servers(servers: List[Dict[str, Any]]) -> int:
    """Iterate through all servers, enriching tools with inferred output schemas."""
    updated_servers = 0
    total = len(servers)
    for idx, server in enumerate(servers, start=1):
        server_name = server.get("server_name") or server.get("name") or f"server_{idx}"
        print(f"\n--- [{idx}/{total}] Processing server: {server_name} ---")
        tools = server.get("tools", [])
        if not tools:
            print("  > No tools found. Skipping.")
            continue

        updates = process_tools_batch(tools)
        for tool_index, updated_tool in updates:
            server["tools"][tool_index] = updated_tool

        if updates:
            updated_servers += 1
    return updated_servers


def load_registry(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_registry(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer missing output schemas for tools listed in an MCP registry file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_registry_path(),
        help="Path to the consolidated MCP registry JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(default_registry_path()),
        help="Optional output path. Defaults to overwriting the input file if not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Registry file not found: {input_path}")

    print(f"Loading registry from {input_path} ...")
    registry = load_registry(input_path)
    servers = registry.get("servers")
    if not isinstance(servers, list):
        raise ValueError("Registry JSON is missing the 'servers' list.")

    start_time = time.time()
    updated_servers = process_servers(servers)
    elapsed = time.time() - start_time

    if updated_servers:
        save_registry(registry, output_path)
        print(
            f"\n✅ Schema inference complete. Updated {updated_servers} servers in "
            f"{elapsed:.2f} seconds. Output saved to {output_path}"
        )
    else:
        print("\n✅ No updates were necessary; all tools already have output schemas.")


if __name__ == "__main__":
    main()
