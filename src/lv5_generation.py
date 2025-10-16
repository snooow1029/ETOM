import os
import json
import pandas as pd
import numpy as np

try:
    import hdbscan
except ImportError:
    print("hdbscan not found. Please run 'pip install hdbscan'.")
    exit()

from sklearn.metrics.pairwise import cosine_similarity
import random
import re
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path
from util import (
    setup_environment, get_vllm_client, get_azure_openai_client,
    call_vllm_api, call_azure_openai_with_usage, find_repo_root,
    extract_tagged_content
)

REPO_ROOT = find_repo_root()
env_path = REPO_ROOT / ".env"
setup_environment(env_path)

# Initialize clients
client, LOCAL_MODEL = get_vllm_client()
client_gpt = get_azure_openai_client()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

LLM_AGENTS = {
    "proposer": LOCAL_MODEL,
    "red_team": LOCAL_MODEL,
    "judge": LOCAL_MODEL
}

AGENT_CONFIGS = {
    "proposer": {"temperature": 0.7},
    "red_team": {"temperature": 0.8, "top_p": 1.0},
    "judge": {"temperature": 0.1, "top_p": 0.7}
}

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')

CONFIG = {
    "TOOL_REGISTRY_PATH": REPO_ROOT / "data" / "mcp_registry_w_embedding.json",
    "OUTPUT_DIR": REPO_ROOT / "data" / "level5_data",
    "HDBSCAN_MIN_CLUSTER_SIZE": 5,
    "HDBSCAN_MIN_SAMPLES": 2,
    "CANDIDATE_QUERIES_PER_GAP": 10,
    "AZURE_DEPLOYMENT_RED_TEAM": AZURE_OPENAI_DEPLOYMENT,
    "DENSE_TOP_K": 5,
    "SPARSE_TOP_K": 5
}

def call_vllm_api_local(prompt: str, agent_name: str = "proposer") -> str:
    """Wrapper for util.call_vllm_api using agent-specific configs."""
    temp_config = AGENT_CONFIGS[agent_name]
    
    return call_vllm_api(
        prompt=prompt,
        client=client,
        model_name=LOCAL_MODEL,
        max_tokens=4096,
        temperature=temp_config.get("temperature", 0.7),
        top_p=temp_config.get("top_p", 0.9)
    )
    
def call_azure_api(prompt: str, agent_name: str = "proposer") -> tuple[str, dict]:
    """Wrapper for util.call_azure_openai_with_usage."""
    deployment_name = CONFIG[f"AZURE_DEPLOYMENT_{agent_name.upper()}"]
    
    return call_azure_openai_with_usage(
        prompt_content=prompt,
        client=client_gpt,
        model_id=deployment_name,
        max_tokens=4096,
        temperature=0.7
    )
    
def get_query_embedding(text: str) -> np.ndarray:
    return model.encode(text)

def load_tools_and_embeddings(registry_path: Path) -> pd.DataFrame:
    registry_path = Path(registry_path)
    print(f"Starting to load tools from {registry_path}...")

    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: Could not load registry '{registry_path}': {exc}")
        return pd.DataFrame()

    tools_list = []
    for server in registry.get("servers", []):
        server_name = server.get("server_name") or server.get("name", "unknown_server")
        for tool in server.get("tools", []):
            description = tool.get("tool_description") or tool.get("description")
            embedding = tool.get("tool_embedding")
            tool_name = tool.get("tool_name") or tool.get("name")
            if description and embedding and tool_name:
                try:
                    embedding_array = np.array(embedding, dtype=float)
                except (ValueError, TypeError):
                    print(
                        f"Warning: Skipping tool '{tool_name}' from '{server_name}' due to invalid embedding."
                    )
                    continue

                tools_list.append(
                    {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "description": description,
                        "embedding": embedding_array,
                        "tool_id": f"{server_name}::{tool_name}",
                    }
                )

    df = pd.DataFrame(tools_list)
    if not df.empty:
        print(f"Successfully loaded {len(df)} tools with embeddings.")
    else:
        print("Warning: No tools with embeddings were found in the registry.")
    return df

def run_phase1_capability_mapping(df: pd.DataFrame, output_dir: str) -> str:
    """
    Phase 1: Groups tools using HDBSCAN and LLM ensemble for robust labeling.
    """
    print("\n--- Phase 1: Building a Self-Consistent and Granular Capability Map ---")
    
    embeddings_matrix = np.vstack(df['embedding'].to_list())
    normalized_embeddings = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=CONFIG['HDBSCAN_MIN_CLUSTER_SIZE'],
        min_samples=CONFIG['HDBSCAN_MIN_SAMPLES'],
        metric='euclidean',
        cluster_selection_epsilon=0.5
    )
    df['cluster_id'] = clusterer.fit_predict(normalized_embeddings)
    
    num_clusters = len(set(df['cluster_id'])) - (1 if -1 in df['cluster_id'] else 0)
    num_noise = np.sum(df['cluster_id'] == -1)
    print(f"Clustered tools into {num_clusters} groups with {num_noise} noise points using HDBSCAN.")
    capability_map = {}
    for cluster_id in sorted(df['cluster_id'].unique()):
        if cluster_id == -1: continue 
        
        cluster_df = df[df['cluster_id'] == cluster_id]
        sample_size = min(len(cluster_df), 10)
        # We should check if the cluster is empty after sampling, although it shouldn't happen
        if sample_size == 0: continue

        sampled_descriptions = cluster_df.sample(sample_size, random_state=42)['description'].tolist()
        descriptions_text = "\n- ".join(sampled_descriptions)
        
        prompt = f"""
# ROLE & GOAL
You are an AI system architect. Your task is to analyze a group of tool descriptions and create a concise, high-level capability label for their shared function (e.g., "Manage Git Repositories").

# TOOL DESCRIPTIONS:
- {descriptions_text}

# YOUR OUTPUT
Wrap the final label in <label></label> tags.
"""
        labels = [extract_tagged_content(call_vllm_api_local(prompt, agent_name=agent), "label") for agent in LLM_AGENTS.keys()]
        
        judge_prompt = f"""
# CONTEXT
Three AI annotators have proposed the following labels for the same group of tools:
- Label A: "{labels[0]}"
- Label B: "{labels[1]}"
- Label C: "{labels[2]}"
# YOUR TASK
Analyze these labels. Are they semantically synonymous? If so, choose the most clear and concise one. If not, state "DISAGREEMENT".
# YOUR OUTPUT
Wrap your final chosen label OR the word "DISAGREEMENT" in <consensus_label></consensus_label> tags.
"""
        consensus_label = extract_tagged_content(call_vllm_api_local(judge_prompt, agent_name="judge"), "consensus_label")

        if "DISAGREEMENT" not in consensus_label.upper():
            capability_map[str(cluster_id)] = consensus_label
            print(f"Cluster {cluster_id} -> Consensus Label: {consensus_label}")
        else:
            print(f"  ! Cluster {cluster_id} -> DISAGREEMENT among agents. Using first label as fallback: {labels[0]}")
            capability_map[str(cluster_id)] = labels[0]
    
    filepath = os.path.join(output_dir, "capability_map.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(capability_map, f, indent=2)
    print(f"Capability map saved to {filepath}")
    return filepath

def run_phase2_gap_analysis(capability_map_path: str, output_dir: str) -> str:
    """
    Phase 2: Uses Proposer vs Red Team debate to identify gaps.
    """
    print("\n--- Phase 2: Identifying Gaps via AI Debate ---")
    prompt_brainstorm = """
    ROLE & GOAL
    As a world-class product manager for a general-purpose AI assistant, your goal is to brainstorm a diverse list of common digital tasks users might want to perform.
    INSTRUCTIONS
    List 100 distinct categories of digital tasks.
    Cover a broad range of areas like productivity, entertainment, information retrieval, and e-commerce.
    Output the result as a JSON list of strings.
    EXAMPLE
    ["Travel Planning", "E-commerce Shopping", "Food & Local Services"]
    YOUR OUTPUT
    Wrap the JSON list inside <universal_tasks></universal_tasks> tags.
    """
    try:
        response_text = call_vllm_api_local(prompt_brainstorm)
        tasks_json_str = extract_tagged_content(response_text, "universal_tasks")
        print(tasks_json_str)
        universal_tasks = re.findall(r'"(.*?)"', tasks_json_str)

        universal_tasks = re.findall(r'"(.*?)"', tasks_json_str)
        print(f"Brainstormed {len(universal_tasks)} universal task categories.")
    except Exception as e:
        print(f"  ! Error brainstorming universal tasks: {e}")
        return None
        
    with open(capability_map_path, 'r', encoding='utf-8') as f:
        capability_map = json.load(f)
    our_capabilities = list(capability_map.values())
    capability_text = "\n- ".join(our_capabilities)
    
    df_caps = pd.DataFrame(capability_map.values(), columns=['capability_label'])
    df_caps['embedding'] = [get_query_embedding(label) for label in df_caps['capability_label']]
    all_caps_embeddings = np.vstack(df_caps['embedding'].to_list())
    capability_text = "\n- ".join(df_caps['capability_label'])

    capability_gaps = []
    for task in universal_tasks:
        proposer_prompt = f"""
# CONTEXT
AI Capabilities:
- {capability_text}
# QUESTION
Can the AI fully perform the task of **"{task}"**? Answer YES or NO.
"""
        proposer_verdict = call_vllm_api_local(proposer_prompt, agent_name="proposer").strip().upper()

        if "NO" in proposer_verdict:
            task_embedding = get_query_embedding(task).reshape(1, -1)
            similarities = cosine_similarity(task_embedding, all_caps_embeddings)[0]
            top_k_indices = np.argsort(similarities)[-10:][::-1]
            relevant_caps_df = df_caps.iloc[top_k_indices]
            relevant_caps_text = "\n- ".join(relevant_caps_df['capability_label'])

            red_team_prompt = f"""
# ROLE & GOAL
You are a pragmatic and skeptical senior engineer, NOT a creative writer. Your goal is to find *realistic and direct* flaws in the claim that a task is impossible.

# CONTEXT
A "Proposer" AI claims the task **"{task}"** is impossible.
You are ONLY allowed to use the following **most relevant capabilities** to challenge this claim:
- {relevant_caps_text}

# YOUR TASK & STRICT RULES
Analyze if a challenge is valid based on these rules:
1.  **Rule of Direct Relevance**: The capability's primary function must be directly applicable to the task. Do NOT stretch the meaning of a capability. (e.g., A 'hashing' tool cannot be used for 'booking IDs').
2.  **Rule of Realistic Workflow**: A proposed solution must represent a simple, logical workflow that a user would find genuinely helpful. Do not chain more than 2-3 capabilities in a speculative way.
3.  **Final Decision**: Based on these strict rules, is there a valid, practical challenge to the "impossible" claim?

# YOUR OUTPUT
Provide a JSON object with two keys: {{"is_challenge_valid": boolean, "reasoning": "If valid, describe the realistic workflow. If invalid, state 'No practical or direct solution found.'"}}.
Wrap the JSON object in <challenge></challenge> tags.
"""
            raw_response, usage = call_azure_api(red_team_prompt, agent_name="red_team")
            print(usage)
            json_str = extract_tagged_content(raw_response, "challenge")

            try:
                challenge_judgement = json.loads(json_str)
                if not challenge_judgement.get('is_challenge_valid', True):
                    capability_gaps.append(task)
                    print(f"  ✅ Confirmed Gap: {task}. Red Team found no practical challenge.")
                else:
                    print(f"  ❌ Discarded Gap: {task}. Red Team proposed a challenge: {challenge_judgement.get('reasoning')}")
            except json.JSONDecodeError:
                print(f"  ! Warning: Red Team produced invalid JSON for task '{task}'. Assuming no valid challenge.")
                capability_gaps.append(task)

    filepath = os.path.join(output_dir, "capability_gaps.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(capability_gaps, f, indent=2)
    print(f"List of {len(capability_gaps)} confirmed capability gaps saved to {filepath}")
    return filepath

def run_phase3_query_generation(gaps_path: str, caps_path: str, output_dir: str) -> str:
    """
    Phase 3: Uses personas to generate diverse queries.
    """
    print("\n--- Phase 3: Generating Diverse Candidate Queries ---")
    
    with open(gaps_path, 'r', encoding='utf-8') as f:
        capability_gaps = json.load(f)
    with open(caps_path, 'r', encoding='utf-8') as f:
        our_capabilities = list(json.load(f).values())
    
    level5_candidate_queries = {}
    personas = ["a business analyst", "a creative writer", "a student", "a casual user"]
    
    for gap in capability_gaps:
        all_queries_for_gap = []
        for persona in personas:
            prompt_generate = f"""
# ROLE & GOAL
You are acting as **{persona}**. Your goal is to generate {CONFIG['CANDIDATE_QUERIES_PER_GAP'] // len(personas)} realistic user queries for a task category you know an AI assistant cannot handle.
# CONTEXT
The assistant CANNOT do: "{gap}".
# YOUR TASK
Generate specific, natural-sounding queries that fall into the unsupported category: "{gap}".
# YOUR OUTPUT
List each query on a new line, starting with '-'. Wrap the list in <queries></queries> tags.
"""
            try:
                response = call_vllm_api_local(prompt_generate)
                print(f'response in phase 3 :{response}')
                queries_block = extract_tagged_content(response, "queries")
                queries = [line.strip()[1:].strip() for line in queries_block.split('\n') if line.strip().startswith('-')]
                all_queries_for_gap.extend(queries)
            except Exception as e:
                print(f"  ! Error generating queries for '{gap}' with persona '{persona}': {e}")
        
        level5_candidate_queries[gap] = all_queries_for_gap
        print(f"Generated {len(all_queries_for_gap)} queries for gap: '{gap}'.")

    filepath = os.path.join(output_dir, "level5_candidate_queries.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(level5_candidate_queries, f, indent=2)
    print(f"Candidate queries saved to {filepath}")
    return filepath

def run_phase4_verification(candidates_path: str, df_tools: pd.DataFrame, output_dir: str) -> str:
    """
    Phase 4: Uses three-tier judgment schema and self-consistency check for verification.
    Support continuation from existing dataset.
    """
    print("\n--- Phase 4: Structured Verification and Self-Correction ---")
    with open(candidates_path, 'r', encoding='utf-8') as f:
        candidate_data = json.load(f)
    
    filepath = os.path.join(output_dir, "level5_final_dataset.json")
    final_level5_dataset = []
    processed_queries = set()
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            final_level5_dataset = existing_data if isinstance(existing_data, list) else []
            processed_queries = {item.get('query', '') for item in final_level5_dataset}
            print(f"📂 Loaded existing dataset: {len(final_level5_dataset)} queries")
            print(f"🔄 Will continue from where left off, skipping {len(processed_queries)} processed queries")
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing dataset: {e}")
            print("🆕 Starting fresh dataset")
    else:
        print("🆕 Starting new dataset")
        
    all_tool_embeddings = np.vstack(df_tools['embedding'].to_list())

    new_queries_processed = 0
    skipped_queries = 0
    total_candidates = sum(len(queries) for queries in candidate_data.values())
    processed_count = 0
    
    for category, queries in candidate_data.items():
        print(f"\n🏷️  Processing category: {category} ({len(queries)} queries)")
        for query in queries:
            processed_count += 1
            progress_percent = (processed_count / total_candidates) * 100
            
            # Skip if query already processed
            if query in processed_queries:
                skipped_queries += 1
                print(f"  ⏭️  [{processed_count}/{total_candidates}] ({progress_percent:.1f}%) SKIPPED: \"{query[:50]}...\"")
                continue
                
            try:
                query_embedding = get_query_embedding(query).reshape(1, -1)
                similarities = cosine_similarity(query_embedding, all_tool_embeddings)[0]
                top_k_indices = np.argsort(similarities)[-CONFIG['DENSE_TOP_K']:][::-1]
                candidate_tools_df = df_tools.iloc[top_k_indices]
                
                candidate_tools_text = ""
                for _, row in candidate_tools_df.iterrows():
                    candidate_tools_text += f"- Tool: {row['tool_name']}, Description: {row['description']}\n"
                
                judge_prompt = f"""
# ROLE & GOAL
You are a meticulous system evaluator. Analyze if the user query can be solved by the candidate tools and provide a structured JSON output.

# USER QUERY: "{query}"
# CANDIDATE TOOLS:
{candidate_tools_text}
# YOUR TASK
Follow these reasoning steps and output a single JSON object:
1.  **Query Intent Analysis**: What is the user's primary, concrete goal?
2.  **Direct Solution Analysis**: Is there any tool specifically designed for this intent?
3.  **Partial Solution Analysis**: If no direct tool exists, could generic tools plausibly assist the user?
4.  **Final Verdict**: Choose one: 'Directly Solvable', 'Partially Solvable', 'Out-of-Scope'.

# YOUR OUTPUT
Wrap the JSON object in <judgement></judgement> tags.
"""
                response_str = call_vllm_api_local(judge_prompt, agent_name="judge")
                json_str = extract_tagged_content(response_str, "judgement")
                judgement = json.loads(json_str)
                
                if judgement.get('final_verdict') == 'Out-of-Scope':
                    consistency_prompt = f"""
# CONTEXT
You previously determined a query was 'Out-of-Scope'. Now, try one more time.
# TASK
Write a simple step-by-step plan to solve this user query using ONLY the tools provided. If it's impossible, just say "IMPOSSIBLE".

# USER QUERY: "{query}"
# AVAILABLE TOOLS:
{candidate_tools_text}
# YOUR PLAN:
"""
                    consistency_check_response = call_vllm_api_local(consistency_prompt, agent_name="proposer")
                    
                    if "IMPOSSIBLE" in consistency_check_response.upper():
                        final_level5_dataset.append({
                            "query": query, "category": category, "ground_truth": []
                        })
                        new_queries_processed += 1
                        print(f"  ✅ [{processed_count}/{total_candidates}] ({progress_percent:.1f}%) PASSED: \"{query[:50]}...\"")
                    else:
                        print(f"  ❌ [{processed_count}/{total_candidates}] ({progress_percent:.1f}%) FAILED CONSISTENCY: \"{query[:50]}...\"")
                else:
                    print(f"  ❌ [{processed_count}/{total_candidates}] ({progress_percent:.1f}%) REJECTED ({judgement.get('final_verdict', 'N/A')}): \"{query[:50]}...\"")
            except Exception as e:
                print(f"  ⚠️  [{processed_count}/{total_candidates}] ({progress_percent:.1f}%) ERROR: \"{query[:50]}...\": {e}")
                
            # Save progress periodically (every 10 queries)
            if processed_count % 10 == 0:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(final_level5_dataset, f, indent=2)
                print(f"  💾 Progress saved: {len(final_level5_dataset)} queries in dataset")

    # Save updated dataset
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(final_level5_dataset, f, indent=2)
    
    print(f"\n📊 Processing Summary:")
    print(f"  • Total queries in dataset: {len(final_level5_dataset)}")
    print(f"  • New queries processed: {new_queries_processed}")
    print(f"  • Skipped (already processed): {skipped_queries}")
    print(f"  • Final output saved to: {filepath}")
    return filepath


def main():
    """Main function to execute all phases in sequence, with caching and refresh logic."""
    parser = argparse.ArgumentParser(description="Generate Level 5 dataset with caching.")
    parser.add_argument(
        '--refresh',
        nargs='*',
        type=int,
        choices=[1, 2, 3, 4],
        help="List of phase numbers to re-run, e.g., --refresh 2 4"
    )
    args = parser.parse_args()
    phases_to_refresh = set(args.refresh) if args.refresh is not None else set()

    output_dir = CONFIG['OUTPUT_DIR'] if isinstance(CONFIG['OUTPUT_DIR'], Path) else Path(CONFIG['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)

    caps_path = output_dir / "capability_map.json"
    gaps_path = output_dir / "capability_gaps.json"
    candidates_path = output_dir / "level5_candidate_queries.json"
    final_dataset_path = output_dir / "level5_final_dataset.json"

    df_tools: pd.DataFrame | None = None

    if 1 in phases_to_refresh or not caps_path.exists():
        df_tools = load_tools_and_embeddings(CONFIG['TOOL_REGISTRY_PATH'])
        if df_tools.empty:
            print("No tools were loaded. Terminating the program.")
            return
        run_phase1_capability_mapping(df_tools, str(output_dir))
    else:
        print("--- Phase 1: Skipped. Loading from cache. ---")

    if not caps_path.exists():
        print("Error: Phase 1 did not produce an output file. Cannot continue.")
        return

    if 2 in phases_to_refresh or not gaps_path.exists():
        run_phase2_gap_analysis(str(caps_path), str(output_dir))
    else:
        print("--- Phase 2: Skipped. Loading from cache. ---")

    if not gaps_path.exists():
        print("Error: Phase 2 did not produce an output file. Cannot continue.")
        return

    if 3 in phases_to_refresh or not candidates_path.exists():
        run_phase3_query_generation(str(gaps_path), str(caps_path), str(output_dir))
    else:
        print("--- Phase 3: Skipped. Loading from cache. ---")

    if not candidates_path.exists():
        print("Error: Phase 3 did not produce an output file. Cannot continue.")
        return

    if 4 in phases_to_refresh or not final_dataset_path.exists():
        print("--- Phase 4: Running fresh verification ---")
    else:
        print("--- Phase 4: Continuing from existing dataset ---")

    if df_tools is None or df_tools.empty:
        df_tools = load_tools_and_embeddings(CONFIG['TOOL_REGISTRY_PATH'])
        if df_tools.empty:
            print("No tools were loaded. Terminating the program.")
            return

    run_phase4_verification(str(candidates_path), df_tools, str(output_dir))

    print("\n🎉 All phases completed successfully!")


if __name__ == "__main__":
    main()