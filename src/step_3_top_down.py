import os
import json
import re
from pathlib import Path
from util import (
    setup_environment, get_azure_openai_client, call_azure_openai,
    parse_xml_tag, find_repo_root
)
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
import torch

# Get script directory and navigate to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load environment variables from .env file
env_path = os.path.join(project_root, ".env")
setup_environment(env_path)

# Get Azure OpenAI credentials from environment variables
client_openai = get_azure_openai_client()
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# --- 2. LLM Related Functions ---

LLM_JUDGE_PROMPT_TEMPLATE_XML = """
# ROLE & GOAL
You are a highly intelligent AI routing engine. Your task is to analyze user queries and determine which of the provided candidate tools can fully and accurately resolve the request.

# CONTEXT
## User Query:
{query_list_text}

## Candidate Tools:
Here is a list of tools that might be able to help. Each tool is identified by a unique `tool_id`.

{candidate_tools_text}

# INSTRUCTIONS
1.  Read the user query to understand the core intent and requirements.
2.  For each candidate tool, evaluate if its function directly matches this core intent.
3.  A tool is a match ONLY IF its core function directly corresponds to the user's request. Do not select tools that are only partially related.
4.  Your final output MUST be a list of `tool_id`s for the tools you have selected.
5.  Each `tool_id` must be on a new line.
6.  The entire list of selected `tool_id`s MUST be wrapped in `<selected_tools>` and `</selected_tools>` tags.
7.  If you determine that NONE of the candidate tools are a suitable match, you must return an empty tag pair, like this: `<selected_tools></selected_tools>`.

# EXAMPLE
## User Query: "Create a new git repository for my project 'alpha'."
## Candidate Tools:
... (list of tools) ...
## Your Output (Example):
<selected_tools>
GitHub MCP Server::create_repository
GitLab MCP Server::create_project
</selected_tools>
"""

def parse_selected_tools_xml(response_text: str) -> set:
    """Parse the tool_id list from XML tags and return a set"""
    if not response_text:
        return set()
    
    match = re.search(r'<selected_tools>(.*?)</selected_tools>', response_text, re.DOTALL)
    
    if not match:
        print(f"  [Warning] Could not find <selected_tools> tag in LLM response.")
        return set()
    
    content = match.group(1).strip()
    if not content:
        return set()
        
    # Split by newlines and filter out empty lines
    tool_ids = {line.strip() for line in content.split('\n') if line.strip()}
    return tool_ids

# --- 3. RAG Knowledge Base Setup ---
class ToolKnowledgeBase:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B', device='cuda'):
        # Set device (e.g., CUDA:1)
        self.device = device
        if torch.cuda.is_available() and device.startswith('cuda'):
            print(f"Using device: {device}")
            self.model = SentenceTransformer(model_name, device=device)
        else:
            print("CUDA not available or device not specified, using CPU")
            self.model = SentenceTransformer(model_name)
            self.device = 'cpu'
        
        self.tools = []
        self.tool_embeddings = None
        print(f"SentenceTransformer model '{model_name}' loaded on device: {self.device}")

    def build_index_from_consolidated_json(self, consolidated_json_path: Path):
        """Build tool index from consolidated JSON file"""
        print(f"Building tool index from: {consolidated_json_path}")
        
        try:
            with open(consolidated_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            servers = data.get("servers", [])
            print(f"Found {len(servers)} servers in consolidated file")
            
            for server in servers:
                server_name = server.get("server_name", "Unknown Server")
                
                for tool in server.get("tools", []):
                    tool_name = tool.get("tool_name", "unknown_tool")
                    description = tool.get("tool_description", "")
                    # Combine name and description as index content
                    content_to_embed = f"Tool: {tool_name}. Description: {description}"
                    self.tools.append({
                        "tool_id": f"{server_name}::{tool_name}",
                        "content": content_to_embed,
                        "description": description
                    })
        except Exception as e:
            print(f"  [Error] Failed to process consolidated JSON: {e}")
            import traceback
            traceback.print_exc()
            raise

        if not self.tools:
            raise ValueError("No tools found in the consolidated JSON file.")

        print(f"Found {len(self.tools)} tools. Now creating embeddings...")
        contents = [tool['content'] for tool in self.tools]
        # Ensure embeddings are computed on the correct device
        self.tool_embeddings = self.model.encode(contents, show_progress_bar=True, device=self.device)
        print(f"Embeddings created successfully on device: {self.device}")

    def search(self, query: str, top_k: int):
        if self.tool_embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index_from_consolidated_json() first.")
        
        # Ensure query embedding is computed on the correct device
        query_embedding = self.model.encode([query], device=self.device)
        similarities = cosine_similarity(query_embedding, self.tool_embeddings)[0]
        
        # Get top_k indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.tools[i] for i in top_k_indices]

# --- 4. Evaluation Main Process ---
def main():
    # --- Preparation ---
    # Get script directory and navigate to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    consolidated_json_path = project_root / "data/mcp_registry.json"
    evaluation_file = script_dir / "tool_sets_with_queries.json"
    output_file = project_root / "eval_result/evaluation_results.jsonl"
    TOP_K_RAG = 10

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
        if torch.cuda.device_count() > 1:
            print("Using CUDA device 1 (cuda:1)")
            device = 'cuda:1'
        else:
            print("Only one GPU available, using cuda:0")
            device = 'cuda'
    else:
        print("CUDA is not available, using CPU")
        device = 'cpu'

    # 1. Build RAG knowledge base (using specified CUDA device)
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    kb = ToolKnowledgeBase(model_name=embedding_model_name, device=device)
    kb.build_index_from_consolidated_json(consolidated_json_path)
    
    # Create a quick lookup dictionary from tool_id to description
    tool_id_to_desc_map = {tool['tool_id']: tool['description'] for tool in kb.tools}

    # 2. Load evaluation data
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        evaluation_data = json.load(f)

    evaluation_results_list = []
    
    # --- Execute Evaluation Loop ---
    for i, item in enumerate(evaluation_data):
        print(f"\n--- Evaluating Query {item['set_id']} ({i+1}/{len(evaluation_data)}) ---")
        print(f"Raw Query Field: {item}")
        raw_query_field = item['generated_queries']
        try:
            query_list = ast.literal_eval(raw_query_field)
            if not isinstance(query_list, list):
                query_list = [str(raw_query_field)]
        except (ValueError, SyntaxError):
            query_list = [raw_query_field]
        
        # ground_truth_tools now contains both IDs and descriptions
        ground_truth_tools = {gt['tool_id']: gt['description'] for gt in item['tools']}
        ground_truth_tool_ids = set(ground_truth_tools.keys())

        print(f"Ground Truth Tools: {ground_truth_tool_ids}")
        
        # RAG retrieval and LLM judgment
        rag_query = query_list[0]
        retrieved_tools = kb.search(rag_query, top_k=TOP_K_RAG)
        # retrieved_tools is a list of dictionaries, each containing 'tool_id' and 'description'
        
        # LLM judgment section
        query_list_text = "\n".join([f"- {q}" for q in query_list])
        candidate_tools_text = "\n".join(
            [f"- tool_id: {t['tool_id']}\n  description: {t['description']}" for t in retrieved_tools]
        )
        prompt = LLM_JUDGE_PROMPT_TEMPLATE_XML.format(
            query_list_text=query_list_text,
            candidate_tools_text=candidate_tools_text
        )
        llm_response_text = call_azure_openai(prompt, client=client_openai, model_id=deployment_name, max_tokens=512, temperature=0)
        if not llm_response_text:
            print("  [Error] LLM did not return a response. Skipping this query.")
            continue
        llm_selected_tool_ids = parse_selected_tools_xml(llm_response_text)
        print(f"LLM Selected Tools: {llm_selected_tool_ids}")
        
        # Calculate and compare metrics
        true_positives = len(llm_selected_tool_ids.intersection(ground_truth_tool_ids))
        precision = true_positives / len(llm_selected_tool_ids) if len(llm_selected_tool_ids) > 0 else 0
        recall = true_positives / len(ground_truth_tool_ids) if len(ground_truth_tool_ids) > 0 else 0
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

        # Format detailed results with descriptions
        def format_tool_details(tool_ids_set):
            return [
                {"tool_id": tid, "description": tool_id_to_desc_map.get(tid, "N/A - Description not found")}
                for tid in sorted(list(tool_ids_set))
            ]

        # Record results
        result_item = {
            "query_id": item['set_id'],
            "query": raw_query_field,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "true_positives": true_positives,
                "is_rag_hit": len(set(t['tool_id'] for t in retrieved_tools).intersection(ground_truth_tool_ids)) > 0
            },
            "ground_truth_details": format_tool_details(ground_truth_tool_ids),
            "rag_retrieved_details": format_tool_details(set(t['tool_id'] for t in retrieved_tools)),
            "llm_selected_details": format_tool_details(llm_selected_tool_ids)
        }
        evaluation_results_list.append(result_item)

    # --- 5. Output results as JSON Lines (.jsonl) ---
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in evaluation_results_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Use Pandas to quickly calculate summary metrics
    df = pd.DataFrame([res['metrics'] for res in evaluation_results_list])
    avg_precision = df['precision'].mean()
    avg_recall = df['recall'].mean()
    rag_hit_rate = df['is_rag_hit'].mean()

    print("\n--- Evaluation Summary ---")
    print(f"Total Queries Evaluated: {len(df)}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"RAG Hit Rate (at least one correct tool in Top-{TOP_K_RAG}): {rag_hit_rate:.4f}")
    print(f"Detailed results saved to '{output_file}' in JSONL format.")

if __name__ == "__main__":
    main()