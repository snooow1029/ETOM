import os
import json
import numpy as np
from typing import List, Dict, Any, Set, Tuple
import openai
from collections import defaultdict
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
from util import setup_environment

class ToolEquivalenceFinder:
    """
    A robust pipeline for finding functionally equivalent tools using embedding similarity 
    and LLM verification with proper transitivity handling.
    """
    
    def __init__(self, embedding_threshold: float = 0.7, top_k: int = 10, 
                 vllm_base_url: str = "http://localhost:8000/v1", 
                 vllm_model_name: str = "your-model-name", 
                 test_mode: bool = False, test_sample_size: int = 50,
                 max_parallel_llm_calls: int = 10, 
                 checkpoint_file: str = "pipeline_checkpoint.pkl"):
        """
        Initialize the equivalence finder.
        
        Args:
            embedding_threshold: Minimum cosine similarity to consider tools as candidates
            top_k: Maximum number of similar tools to consider per tool
            vllm_base_url: Base URL for vLLM server (default: http://localhost:8000/v1)
            vllm_model_name: Name of the model to use with vLLM
            test_mode: If True, only process a small sample of tools for testing
            test_sample_size: Number of tools to sample when in test mode
            max_parallel_llm_calls: Maximum number of parallel LLM API calls
            checkpoint_file: File to save progress for resumability
        """
        self.embedding_threshold = embedding_threshold
        self.top_k = top_k
        self.vllm_base_url = vllm_base_url
        self.vllm_model_name = vllm_model_name
        self.test_mode = test_mode
        self.test_sample_size = test_sample_size
        self.max_parallel_llm_calls = max_parallel_llm_calls
        self.checkpoint_file = checkpoint_file
        self.tools_data = []
        self.tool_lookup = {}
        
    def load_tools_with_embeddings(self, consolidated_json_path: str) -> List[Dict[str, Any]]:
        """Load all tools with their embeddings from consolidated JSON file."""
        all_tools = []
        print(f"Loading tools from '{consolidated_json_path}'...")
        
        try:
            with open(consolidated_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract servers from the consolidated structure
            servers = data.get("servers", [])
            print(f"Found {len(servers)} servers in consolidated file")
            
            for server in servers:
                server_name = server.get("server_name")
                if not server_name:
                    continue
                
                for tool in server.get("tools", []):
                    # Check for either 'tool_embedding' or 'tool_description_embedding'
                    embedding_key = None
                    if 'tool_embedding' in tool:
                        embedding_key = 'tool_embedding'
                    elif 'tool_description_embedding' in tool:
                        embedding_key = 'tool_description_embedding'
                    
                    if embedding_key and tool.get('tool_name'):
                        all_tools.append({
                            "server_name": server_name,
                            "tool_name": tool['tool_name'],
                            "description": tool.get('tool_description', ''),
                            "embedding": np.array(tool[embedding_key]),
                            "source_file": consolidated_json_path,
                            "tool_id": f"{server_name}::{tool['tool_name']}"  # Unique identifier
                        })
            
            print(f"Successfully loaded {len(all_tools)} tools with embeddings")
            
        except Exception as e:
            print(f"Error loading consolidated JSON file: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Apply test mode sampling if enabled
        if self.test_mode and len(all_tools) > self.test_sample_size:
            import random
            random.seed(5)  # For reproducible results
            all_tools = random.sample(all_tools, self.test_sample_size)
            print(f"TEST MODE: Sampled {len(all_tools)} tools for testing")
        else:
            print(f"Successfully loaded {len(all_tools)} tools with embeddings.")
            
        self.tools_data = all_tools
        # Create lookup dictionary once for efficient access
        self.tool_lookup = {tool['tool_id']: tool for tool in self.tools_data}
        return all_tools

    def phase1_collect_candidates(self) -> Set[Tuple[str, str, float]]:
        """
        OPTIMIZED Phase 1: Use vectorized operations for similarity computation.
        Returns set of (tool_id_1, tool_id_2, similarity_score) tuples.
        """
        print("Phase 1: Collecting candidates with vectorized similarity computation...")
        
        # Extract embeddings into a matrix for vectorized operations
        embeddings_matrix = np.vstack([tool['embedding'] for tool in self.tools_data])
        tool_ids = [tool['tool_id'] for tool in self.tools_data]
        
        print(f"Computing similarity matrix for {len(self.tools_data)} tools...")
        
        # Compute all pairwise similarities at once using sklearn
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Find candidate pairs efficiently
        candidate_pairs = set()
        n_tools = len(self.tools_data)
        
        print("Extracting candidate pairs...")
        with tqdm(total=n_tools, desc="Processing tools") as pbar:
            for i in range(n_tools):
                # Get similarities for this tool
                similarities = similarity_matrix[i]
                
                # Find indices above threshold (excluding self)
                valid_indices = np.where(
                    (similarities >= self.embedding_threshold) & 
                    (np.arange(n_tools) != i)
                )[0]
                
                # Get top-k similar tools
                if len(valid_indices) > self.top_k:
                    top_indices = valid_indices[np.argsort(similarities[valid_indices])[-self.top_k:]]
                else:
                    top_indices = valid_indices
                
                # Add pairs with consistent ordering
                for j in top_indices:
                    pair = tuple(sorted([tool_ids[i], tool_ids[j]]))
                    candidate_pairs.add((pair[0], pair[1], similarities[j]))
                    
                pbar.update(1)
        
        print(f"Found {len(candidate_pairs)} candidate pairs for LLM verification.")
        return candidate_pairs

    async def phase2_llm_verification(self, candidate_pairs: Set[Tuple[str, str, float]]) -> Set[Tuple[str, str]]:
        """
        OPTIMIZED Phase 2: Parallel LLM verification with progress tracking.
        Returns set of verified equivalent pairs.
        """
        print("Phase 2: Parallel LLM verification of candidate pairs...")
        
        # Check for existing progress
        if os.path.exists(self.checkpoint_file):
            print("Found checkpoint file. Loading previous progress...")
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                verified_pairs = set(checkpoint_data.get('verified_pairs', []))
                processed_pairs = set(checkpoint_data.get('processed_pairs', []))
        else:
            verified_pairs = set()
            processed_pairs = set()
        
        # Filter out already processed pairs
        remaining_pairs = [
            (tool_id_a, tool_id_b, sim) for tool_id_a, tool_id_b, sim in candidate_pairs
            if (tool_id_a, tool_id_b) not in processed_pairs
        ]
        
        print(f"Processing {len(remaining_pairs)} remaining pairs...")
        
        # Process in parallel batches
        semaphore = asyncio.Semaphore(self.max_parallel_llm_calls)
        
        async def verify_pair_with_semaphore(pair_data):
            async with semaphore:
                return await self._verify_pair_async(pair_data)
        
        # Process all pairs
        with tqdm(total=len(remaining_pairs), desc="LLM verification") as pbar:
            batch_size = 50  # Process in batches to avoid memory issues
            
            for i in range(0, len(remaining_pairs), batch_size):
                batch = remaining_pairs[i:i + batch_size]
                batch_tasks = [verify_pair_with_semaphore(pair) for pair in batch]
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for pair_data, result in zip(batch, batch_results):
                    tool_id_a, tool_id_b, similarity = pair_data
                    
                    if isinstance(result, Exception):
                        print(f"Error verifying {tool_id_a} vs {tool_id_b}: {result}")
                        is_equivalent = False
                    else:
                        is_equivalent = result
                    
                    processed_pairs.add((tool_id_a, tool_id_b))
                    if is_equivalent:
                        verified_pairs.add((tool_id_a, tool_id_b))
                    
                    pbar.update(1)
                
                # Save checkpoint periodically
                if i % (batch_size * 5) == 0:  # Every 5 batches
                    self._save_checkpoint(verified_pairs, processed_pairs)
        
        # Final checkpoint save
        self._save_checkpoint(verified_pairs, processed_pairs)
        
        print(f"LLM verified {len(verified_pairs)} equivalent pairs.")
        return verified_pairs

    async def _verify_pair_async(self, pair_data: Tuple[str, str, float]) -> bool:
        """Async version of LLM verification."""
        tool_id_a, tool_id_b, similarity = pair_data
        
        tool_a = self.tool_lookup[tool_id_a]
        tool_b = self.tool_lookup[tool_id_b]
        
        # Create prompt (same as before but optimized structure)
        tools = sorted([tool_a, tool_b], key=lambda x: x['tool_name'])
        first_tool, second_tool = tools[0], tools[1]
        
        prompt = f"""Analyze these two tools and determine if they achieve the same end goal with the same scope and depth.

Tool A: {first_tool['tool_name']} ({first_tool['server_name']})
Description: {first_tool['description']}

Tool B: {second_tool['tool_name']} ({second_tool['server_name']})  
Description: {second_tool['description']}

Two tools are functionally equivalent if they achieve the same END RESULT with similar scope:
- Focus on WHAT they accomplish AND the breadth/depth of that accomplishment
- Consider the user's intent and the comprehensiveness of the output
- Different implementation methods are acceptable (different servers, models, APIs)
- But the scope, depth, and purpose of the end result should be equivalent

Answer "YES" if they produce the same type AND scope of result for users, "NO" if they have different scope, depth, or purpose.

Answer:"""
        
        try:
            response = await self._call_llm_async(prompt)
            
            # Handle reasoning models
            if "<think>" in response and "</think>" in response:
                think_end = response.rfind("</think>")
                if think_end != -1:
                    actual_response = response[think_end + 8:].strip()
                else:
                    actual_response = response
            else:
                actual_response = response
            
            return actual_response.strip().upper().startswith("YES")
            
        except Exception as e:
            print(f"Error in async LLM verification: {e}")
            return False

    async def _call_llm_async(self, prompt: str) -> str:
        """Async LLM call using aiohttp."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.vllm_model_name,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that analyzes tool functionality. Always provide clear YES/NO answers followed by brief explanations."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0,
                "stop": None
            }
            
            async with session.post(
                f"{self.vllm_base_url}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer EMPTY"}
            ) as response:
                result = await response.json()
                return result['choices'][0]['message']['content'] or ""

    def _save_checkpoint(self, verified_pairs: Set, processed_pairs: Set):
        """Save progress checkpoint."""
        checkpoint_data = {
            'verified_pairs': list(verified_pairs),
            'processed_pairs': list(processed_pairs),
            'timestamp': str(asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0)
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    def phase3_create_equivalence_sets(self, verified_pairs: Set[Tuple[str, str]]) -> List[Set[str]]:
        """
        Phase 3: Create equivalence sets using Union-Find algorithm.
        Handles transitivity automatically.
        """
        print("Phase 3: Creating equivalence sets...")
        
        # Union-Find implementation
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Process all verified equivalent pairs
        for tool_a, tool_b in verified_pairs:
            union(tool_a, tool_b)
        
        # Group tools by their root parent
        groups = defaultdict(set)
        all_tools = {tool['tool_id'] for tool in self.tools_data}
        
        for tool_id in all_tools:
            root = find(tool_id)
            groups[root].add(tool_id)
        
        # Convert to list of sets, filter out singletons if desired
        equivalence_sets = [group for group in groups.values() if len(group) > 1]
        
        print(f"Created {len(equivalence_sets)} equivalence sets from {len(all_tools)} total tools.")
        return equivalence_sets

    async def run_pipeline_async(self, consolidated_json_path: str) -> List[Set[str]]:
        """
        Run the complete async pipeline to find tool equivalence sets.
        """
        print("Starting Tool Equivalence Pipeline...")
        
        # Load tools
        self.load_tools_with_embeddings(consolidated_json_path)
        
        # Phase 1: Vectorized candidate collection
        candidates = self.phase1_collect_candidates()
        
        # Phase 2: Parallel LLM verification  
        verified = await self.phase2_llm_verification(candidates)
        
        # Phase 3: Create equivalence sets
        equivalence_sets = self.phase3_create_equivalence_sets(verified)
        
        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        
        return equivalence_sets

    def run_pipeline(self, consolidated_json_path: str) -> List[Set[str]]:
        """
        Sync wrapper for the async pipeline.
        """
        return asyncio.run(self.run_pipeline_async(consolidated_json_path))

    def save_results(self, equivalence_sets: List[Set[str]], output_file: str):
        """Save equivalence sets to JSON file."""
        results = []
        for i, eq_set in enumerate(equivalence_sets):
            set_data = {
                "set_id": i + 1,
                "size": len(eq_set),
                "tools": []
            }
            
            for tool_id in eq_set:
                tool = self.tool_lookup[tool_id]
                set_data["tools"].append({
                    "tool_id": tool_id,
                    "server_name": tool['server_name'],
                    "tool_name": tool['tool_name'],
                    "description": tool['description'][:200] + "..." if len(tool['description']) > 200 else tool['description']
                })
            
            results.append(set_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_file}")

    def run_test_pipeline(self, consolidated_json_path: str) -> List[Set[str]]:
        """
        Run a quick test of the pipeline with a small sample of tools.
        """
        print("=" * 60)
        print("RUNNING TEST PIPELINE")
        print("=" * 60)
        
        # Temporarily enable test mode
        original_test_mode = self.test_mode
        original_sample_size = self.test_sample_size
        
        self.test_mode = True
        if self.test_sample_size > 100:  # Keep test really small
            self.test_sample_size = 100
            
        try:
            equivalence_sets = self.run_pipeline(consolidated_json_path)
            
            # Print test results
            print("\n" + "=" * 60)
            print("TEST RESULTS SUMMARY")
            print("=" * 60)
            print(f"✓ Pipeline completed successfully")
            print(f"✓ Processed {len(self.tools_data)} tools")
            print(f"✓ Found {len(equivalence_sets)} equivalence sets")
            
            if equivalence_sets:
                print(f"\nSample equivalence sets found:")
                for i, eq_set in enumerate(equivalence_sets[:3], 1):  # Show first 3 sets
                    print(f"\nSet {i} ({len(eq_set)} tools):")
                    for tool_id in eq_set:
                        print(f"  - {tool_id}")
                    if i >= 3:  # Limit output
                        break
                if len(equivalence_sets) > 3:
                    print(f"\n... and {len(equivalence_sets) - 3} more sets")
            else:
                print("No equivalence sets found in test sample")
                
            print(f"\n🎯 Test completed! Ready to run full pipeline.")
            
        finally:
            # Restore original settings
            self.test_mode = original_test_mode
            self.test_sample_size = original_sample_size
            
        return equivalence_sets


if __name__ == "__main__":
    # Get script directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load environment variables from .env file
    env_path = os.path.join(project_root, ".env")
    setup_environment(env_path)
    
    # Configuration from .env
    consolidated_json_path = "../data/mcp_registry_w_embedding.json"
    vllm_base_url = os.getenv("LOCAL_URL", "http://localhost:8000/v1")
    # vllm_model_name = os.getenv("LOCAL_MODEL", "Qwen/Qwen3-8B")
    vllm_model_name = "Qwen/Qwen3-8B"
    
    print(f"Using vLLM Base URL: {vllm_base_url}")
    print(f"Using vLLM Model: {vllm_model_name}")
    
    # Choose test mode or full pipeline
    RUN_TEST = False  # Set to False to run full pipeline
    
    if RUN_TEST:
        print("Running TEST MODE...")
        
        # Test run with small sample
        test_finder = ToolEquivalenceFinder(
            embedding_threshold=0.8,  # Lower threshold for testing
            top_k=10,  # Fewer candidates for testing
            vllm_base_url=vllm_base_url,
            vllm_model_name=vllm_model_name,
            test_mode=True,
            test_sample_size=100,  # Small sample for quick testing
            max_parallel_llm_calls=5,  # Conservative for testing
            checkpoint_file="test_checkpoint.pkl"
        )
        
        test_results = test_finder.run_test_pipeline(consolidated_json_path)
        
        # Save test results
        test_finder.save_results(test_results, "test_equivalence_sets_2.json")
        
    else:
        print("Running FULL PIPELINE...")
        
        # Full pipeline with optimizations
        finder = ToolEquivalenceFinder(
            embedding_threshold=0.8,  # Adjust based on test results
            top_k=10,  # Adjust based on test results  
            vllm_base_url=vllm_base_url,
            vllm_model_name=vllm_model_name,
            test_mode=False,  # Full dataset
            max_parallel_llm_calls=10,  # Adjust based on your vLLM server capacity
            checkpoint_file="pipeline_checkpoint.pkl"
        )
        
        equivalence_sets = finder.run_pipeline(consolidated_json_path)
        
        # Print results
        print(f"\n=== FINAL RESULTS ===")
        print(f"Found {len(equivalence_sets)} equivalence sets:")
        
        for i, eq_set in enumerate(equivalence_sets, 1):
            print(f"\nSet {i} ({len(eq_set)} tools):")
            for tool_id in eq_set:
                print(f"  - {tool_id}")
        
        # Save results
        finder.save_results(equivalence_sets, "tool_equivalence_sets.json")
