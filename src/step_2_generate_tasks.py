#!/usr/bin/env python3

import os
import json
import asyncio
import aiohttp
import random
from typing import List, Dict, Any
from tqdm import tqdm
import pickle
from util import setup_environment

class QueryGenerator:
    """
    Generate queries that can be solved by tool equivalence sets.
    """
    
    def __init__(self, 
                 vllm_base_url: str = "http://localhost:8001/v1",
                 vllm_model_name: str = "Qwen/Qwen3-8B",
                 checkpoint_file: str = "query_generation_checkpoint.pkl",
                 output_file: str = "tool_sets_with_queries.json",
                 temperature: float = 0.7,  # Higher temperature for more diverse queries
                 max_tokens: int = 2000,
                 delay_between_calls: float = 0.0,
                 checkpoint_interval: int = 10,
                 test_mode: bool = False,
                 test_sets: int = 5,
                 queries_per_set: int = 3):  # Generate multiple queries per set
        """
        Initialize the query generator.
        
        Args:
            vllm_base_url: Base URL for vLLM server
            vllm_model_name: Name of the model to use with vLLM
            checkpoint_file: File to save progress for resumability
            output_file: Output file for results
            temperature: Temperature for LLM generation (higher = more diverse)
            max_tokens: Maximum tokens for LLM response
            delay_between_calls: Seconds to wait between API calls
            checkpoint_interval: Save progress every N sets
            test_mode: If True, only process first few sets
            test_sets: Number of sets to process in test mode
            queries_per_set: Number of queries to generate per tool set
        """
        self.vllm_base_url = vllm_base_url
        self.vllm_model_name = vllm_model_name
        self.checkpoint_file = checkpoint_file
        self.output_file = output_file
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay_between_calls = delay_between_calls
        self.checkpoint_interval = checkpoint_interval
        self.test_mode = test_mode
        self.test_sets = test_sets
        self.queries_per_set = queries_per_set
        
    def load_equivalence_sets(self, file_path: str) -> List[Dict[str, Any]]:
        """Load tool equivalence sets from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_checkpoint(self, processed_sets: List[Dict[str, Any]], current_index: int):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'processed_sets': processed_sets,
            'current_index': current_index
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def _load_checkpoint(self) -> tuple:
        """Load progress from checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            return data['processed_sets'], data['current_index']
        return [], 0
    
    async def _call_llm_async(self, prompt: str) -> str:
        """Make async call to vLLM server with retry mechanism."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": self.vllm_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.vllm_base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=600)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            content = result["choices"][0]["message"]["content"].strip()
                            if content:  # Ensure non-empty response
                                return content
                            else:
                                raise Exception("Empty response from server")
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
                            
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait before retry: 1s, 2s, 4s
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    print(f"  Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  All {max_retries} attempts failed: {str(e)}")
                    raise
        
        # This should never be reached due to the raise above
        raise Exception("Unexpected error in retry logic")
    
    def _create_query_generation_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Create prompt for generating queries that can be solved by the tool set."""
        tools_info = []
        for tool in tools:
            tools_info.append(f"- {tool['tool_name']} ({tool['server_name']}): {tool['description']}")
        
        tools_text = "\n".join(tools_info)
        
        prompt = f"""The following {len(tools)} tools are functionally equivalent and can solve the same types of problems. Generate specific user queries that could be solved by ANY of these tools.

Tools:
{tools_text}

Your task:
1. Generate {self.queries_per_set} diverse, realistic user queries that could be solved by these tools
2. Make each query specific and actionable (not too general)
3. Ensure the queries represent real user needs/scenarios
4. Vary the complexity and context of the queries
5. Focus on what users would actually ask for

Format your response as a JSON array of strings:
["Query 1 text here", "Query 2 text here", "Query 3 text here"]

Examples of good queries for different tool types:
- For image generation tools: "Create a realistic portrait of a golden retriever wearing a red bandana"
- For web scraping tools: "Extract all product prices from an e-commerce website homepage"
- For file conversion tools: "Convert my PowerPoint presentation to PDF format"
- For API tools: "Get current weather data for Tokyo, Japan"

Generate queries that these specific tools could handle:"""
        
        return prompt
    
    async def generate_queries_for_set(self, tool_set: Dict[str, Any]) -> List[str]:
        """Generate queries for a single tool set."""
        tools = tool_set['tools']
        prompt = self._create_query_generation_prompt(tools)
        
        try:
            response = await self._call_llm_async(prompt)
            
            # Handle reasoning models with <think> tags
            if "<think>" in response and "</think>" in response:
                think_end = response.rfind("</think>")
                if think_end != -1:
                    actual_response = response[think_end + 8:].strip()
                else:
                    actual_response = response
            else:
                actual_response = response
            
            # Try to parse JSON response
            try:
                import json
                queries = json.loads(actual_response.strip())
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    return queries
                else:
                    print(f"Warning: Invalid JSON format for set {tool_set['set_id']}")
                    return [actual_response.strip()]
            except json.JSONDecodeError:
                # If JSON parsing fails, treat as single query
                print(f"Warning: Could not parse JSON for set {tool_set['set_id']}, treating as single query")
                return [actual_response.strip()]
            
        except Exception as e:
            print(f"Failed to generate queries for set {tool_set['set_id']}: {e}")
            return [f"Error generating queries: {str(e)}"]
    
    async def generate_queries_for_all_sets(self, equivalence_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate queries for all tool sets."""
        # Load existing progress if available
        processed_sets, start_index = self._load_checkpoint()
        
        print(f"Generating queries for {len(equivalence_sets)} tool sets...")
        if start_index > 0:
            print(f"Resuming from set {start_index + 1}")
        
        # Continue from where we left off
        remaining_sets = equivalence_sets[start_index:]
        
        with tqdm(total=len(equivalence_sets), initial=start_index, desc="Generating queries") as pbar:
            for i, tool_set in enumerate(remaining_sets):
                current_index = start_index + i
                
                try:
                    # Generate queries for this set
                    queries = await self.generate_queries_for_set(tool_set)
                    
                    # Add queries to the set data
                    enhanced_set = tool_set.copy()
                    enhanced_set['generated_queries'] = queries
                    processed_sets.append(enhanced_set)
                    
                    pbar.update(1)
                    
                    # Save checkpoint periodically
                    if (i + 1) % self.checkpoint_interval == 0:
                        self._save_checkpoint(processed_sets, current_index + 1)
                    
                    # Small delay to be nice to the API
                    await asyncio.sleep(self.delay_between_calls)
                    
                except Exception as e:
                    print(f"Error processing set {tool_set['set_id']}: {e}")
                    # Add error entry to maintain order
                    enhanced_set = tool_set.copy()
                    enhanced_set['general_task'] = f"Error: {str(e)}"
                    processed_sets.append(enhanced_set)
                    pbar.update(1)
        
        # Final checkpoint save
        self._save_checkpoint(processed_sets, len(equivalence_sets))
        
        return processed_sets
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save the analyzed results to output file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {self.output_file}")
    
    async def run_generation(self, input_file: str):
        """Run the complete query generation pipeline."""
        print("Loading tool equivalence sets...")
        equivalence_sets = self.load_equivalence_sets(input_file)
        
        # Test mode: only process first few sets
        if self.test_mode:
            equivalence_sets = equivalence_sets[:self.test_sets]
            print(f"TEST MODE: Processing only first {len(equivalence_sets)} sets")
        
        print(f"Loaded {len(equivalence_sets)} equivalence sets")
        
        print(f"\nGenerating {self.queries_per_set} queries per set...")
        results = await self.generate_queries_for_all_sets(equivalence_sets)
        
        print("\nSaving results...")
        self.save_results(results)
        
        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        
        print("\nQuery generation complete!")
        print(f"Processed {len(results)} sets")
        
        # Print some example results
        print("\nExample results:")
        for i, result in enumerate(results[:3]):
            print(f"Set {result['set_id']} ({result['size']} tools):")
            for j, query in enumerate(result.get('generated_queries', []), 1):
                print(f"  Query {j}: {query}")
            print()


async def main():
    """Main function to run the analysis."""
    
    # Get script directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load environment variables from .env file
    env_path = os.path.join(project_root, ".env")
    setup_environment(env_path)
    
    # Configuration from .env
    vllm_base_url = os.getenv("LOCAL_URL", "http://localhost:8040/v1")
    # vllm_model_name = os.getenv("LOCAL_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    vllm_model_name = "Qwen/Qwen3-8B"
    print(f"Using vLLM Base URL: {vllm_base_url}")
    print(f"Using vLLM Model: {vllm_model_name}")
    
    input_file = "tool_equivalence_sets.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        print("Please ensure the tool equivalence sets file exists.")
        return
    
    # Initialize query generator
    generator = QueryGenerator(
        vllm_base_url=vllm_base_url,
        vllm_model_name=vllm_model_name,
        checkpoint_file="query_generation_checkpoint.pkl",
        output_file="tool_sets_with_queries.json",
        test_mode=True,
        queries_per_set=3
    )
    
    # Run query generation
    await generator.run_generation(input_file)


if __name__ == "__main__":
    asyncio.run(main())
