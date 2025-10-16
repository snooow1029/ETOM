import os
import json
from pathlib import Path
from util import (
    setup_environment, get_vllm_client, get_azure_openai_client,
    call_vllm_api, call_azure_openai, parse_xml_tag, 
    parse_multiple_xml_tags, format_schema_for_prompt
)
from lv1_lv2_prompts import (
    PLATFORM_ID_PROMPT,
    TASK_TYPE_PROMPT,
    GENERATOR_L1_PROMPT_FINAL,
    VERIFIER_PROMPT,
    PLATFORM_COUPLING_CLASSIFIER_PROMPT,
    GENERATOR_L2_PROMPT,
    VERIFIER_L2_PROMPT,
    USER_FACING_CLASSIFIER_PROMPT
)

# Load environment variables
setup_environment()

# --- Configuration & Client Initialization ---
client_local, VLLM_MODEL_NAME = get_vllm_client()
client_openai = get_azure_openai_client()

# --- Main Processing Pipeline ---

def process_consolidated_server(server_data: dict) -> dict:
    """
    Process a single server from consolidated JSON data and generate L1/L2 queries.
    
    This function:
    1. Identifies the platform from server description
    2. Classifies each tool (final_goal vs middleware, user_facing vs system_facing)
    3. Generates and verifies Level 1 (platform-specific) queries
    4. Generates and verifies Level 2 (natural language) queries
    
    Args:
        server_data (dict): Server data from consolidated JSON
        
    Returns:
        dict: Dictionary of generated queries keyed by server/tool combinations,
              or None if no queries were generated
    """
    server_name = server_data.get("server_name", "unknown_server")
    
    # Convert consolidated format to original single-file format
    converted_data = {
        "name": server_name,
        "description": server_data.get("server_description", ""),
        "summary": server_data.get("server_summary", ""),
        "tools": []
    }
    
    # Convert tool format
    for tool in server_data.get("tools", []):
        converted_tool = {
            "name": tool.get("tool_name", "unknown_tool"),
            "description": tool.get("tool_description", ""),
            "input_schema": tool.get("input_schema", {})
        }
        converted_data["tools"].append(converted_tool)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SERVER: {server_name}")
    print(f"Description: {converted_data.get('description', 'N/A')[:100]}{'...' if len(converted_data.get('description', '')) > 100 else ''}")
    print(f"Tools found: {len(converted_data['tools'])}")
    print(f"{'='*60}")
    
    # Use original processing logic but don't write back to file
    try:
        # --- Step 1: Platform identification ---
        print("\n STEP 1: Platform Identification")
        text_to_analyze = converted_data.get('description', '') + ' ' + converted_data.get('summary', '')

        prompt = PLATFORM_ID_PROMPT.format(text_to_analyze=text_to_analyze)
        response = call_azure_openai(
            prompt_content=prompt,
            client=client_openai,
            model_id="gpt-4.1",
            max_tokens=256,
            temperature=0.0
        )
        platform_name = parse_xml_tag(response, 'keyword')

        # Add platform identification result to data object
        identified_platform = platform_name if platform_name and platform_name != 'N/A' else 'N/A'
        converted_data['platform_specificity'] = identified_platform
        
        if identified_platform == 'N/A':
            print(f"    No specific platform identified - skipping query generation")
        else:
            print(f"    Platform identified: {identified_platform}")
        
        server_queries = {}
        tools = converted_data.get('tools', [])
        if not tools:
            print("    WARNING: No tools found in this server")
            return None
        print(f"\n STEP 2: Tool Analysis & Classification")
        
        processed_tools = 0
        skipped_tools = 0
        
        for tool in tools:
            tool_name = tool.get('name', 'unknown_tool')
            tool_desc = tool.get('description', '')
            processed_tools += 1
            
            # --- Step 2a: Task type classification ---
            prompt = TASK_TYPE_PROMPT.format(tool_name=tool_name, tool_description=tool_desc)
            response = call_vllm_api(prompt, client=client_local, model_name=VLLM_MODEL_NAME)
            task_type = parse_xml_tag(response, 'task_type')

            # If LLM failed to return valid classification, provide default value to avoid errors
            if task_type not in ['final_goal', 'middleware']:
                print(f"       Could not determine task type - defaulting to 'middleware'")
                task_type = 'middleware'
                skipped_tools += 1
                continue

            # print(f"       Classifying Task type: {task_type}")
            
            if task_type == 'middleware':
                # print("        Skipping middleware tool")
                skipped_tools += 1
                continue
            
            # --- Step 2b: User orientation classification ---
            # print("       Classifying user orientation...")
            prompt = USER_FACING_CLASSIFIER_PROMPT.format(tool_name=tool_name, tool_description=tool_desc)
            response = call_azure_openai(
                prompt_content=prompt,
                client=client_openai,
                model_id="gpt-4.1",
                max_tokens=256,
                temperature=0.0
            )
            classification = parse_xml_tag(response, 'classification')

            # If classification failed or classified as system_facing, skip query generation
            if classification != 'user_facing':
                # print(f"       Tool classified as '{classification}' - skipping")
                tool['user_orientation'] = classification if classification else 'unknown'
                skipped_tools += 1
                continue
            
            tool['user_orientation'] = 'user_facing'
            print(f"       User orientation: user_facing")
            # --- Query generation logic (only for platform-specific and final_goal tools) ---
            if converted_data['platform_specificity'] != 'N/A' and task_type == 'final_goal':
                print(f"\n       QUERY GENERATION for {tool_name}")
                print(f"         Platform: {identified_platform} | Type: final_goal | Orientation: user_facing")
                
                verified_l1_queries = []
                verified_l2_queries = []

                # --- Level 1 Generation and Verification ---
                print("\n       LEVEL 1: Platform-Specific Queries")
                
                l1_gen_prompt = GENERATOR_L1_PROMPT_FINAL.format(
                    platform_name=platform_name,
                    server_name=converted_data.get('name', 'unknown_server'),
                    tool_name=tool_name,
                    tool_description=tool_desc
                )

                l1_gen_response = call_vllm_api(l1_gen_prompt, client=client_local, model_name=VLLM_MODEL_NAME)
                generated_l1_queries = parse_multiple_xml_tags(l1_gen_response, 'query')
                print(f"          Generated {len(generated_l1_queries)} candidate L1 queries")

                for idx, query in enumerate(generated_l1_queries, 1):
                    print(f"          Verifying L1 query {idx}/{len(generated_l1_queries)}")
                    verifier_prompt = VERIFIER_PROMPT.format(
                        query=query, tool_name=tool_name, 
                        platform_name=platform_name, tool_description=tool_desc
                    )
                    verifier_response = call_vllm_api(verifier_prompt, client=client_local, model_name=VLLM_MODEL_NAME)
                    is_match_str = parse_xml_tag(verifier_response, 'is_match')
                    
                    if is_match_str == 'true':
                        verified_l1_queries.append(query)
                        print(f"          VERIFIED: '{query[:60]}{'...' if len(query) > 60 else ''}'")
                    else:
                        print(f"          REJECTED: '{query[:60]}{'...' if len(query) > 60 else ''}'")
                        if verifier_response and len(verifier_response.splitlines()) > 1:
                            print(f"             Reason: {verifier_response.splitlines()[1]}")
                
                print(f"          L1 Results: {len(verified_l1_queries)}/{len(generated_l1_queries)} queries verified")
                # --- Level 2 Generation and Verification ---
                print("\n       LEVEL 2: Natural Language Queries")

                # Step 2.1: Determine platform coupling
                print("          Analyzing platform coupling...")
                coupling_prompt = PLATFORM_COUPLING_CLASSIFIER_PROMPT.format(
                    platform_name=platform_name,
                    tool_name=tool_name,
                    tool_description=tool_desc
                )
                coupling_response = call_vllm_api(coupling_prompt, client=client_local, model_name=VLLM_MODEL_NAME)
                coupling_type = parse_xml_tag(coupling_response, 'coupling')

                print(f"          Coupling type: {coupling_type}")

                # Step 2.2: Set generation rules based on coupling
                if coupling_type == 'tightly_coupled':
                    platform_mention_rule = f"You MUST mention the platform '{platform_name}' because the function is iconic to it."
                    verify_rule = f"The query MUST mention the platform '{platform_name}'."
                    print(f"          Rule: MUST mention platform '{platform_name}'")
                else: # generic_concept or unknown
                    platform_mention_rule = f"You MUST NOT mention the platform '{platform_name}'. The function is a generic concept."
                    verify_rule = f"The query MUST NOT mention the platform '{platform_name}'."
                    print(f"          Rule: MUST NOT mention platform '{platform_name}'")

                # Step 2.3: Use dynamic prompt to generate queries, including schema information
                formatted_schema = format_schema_for_prompt(tool.get('input_schema', {}))
                l2_gen_prompt = GENERATOR_L2_PROMPT.format(
                    platform_name=platform_name,
                    tool_name=tool_name,
                    tool_description=tool_desc,
                    formatted_schema=formatted_schema,
                    platform_mention_rule=platform_mention_rule
                )

                l2_gen_response = call_vllm_api(l2_gen_prompt, client=client_local, model_name=VLLM_MODEL_NAME)
                generated_l2_queries = parse_multiple_xml_tags(l2_gen_response, 'query')
                print(f"          Generated {len(generated_l2_queries)} candidate L2 queries")

                for idx, query in enumerate(generated_l2_queries, 1):
                    print(f"          Verifying L2 query {idx}/{len(generated_l2_queries)}")
                    verifier_prompt = VERIFIER_L2_PROMPT.format(
                        query=query, 
                        tool_name=tool_name, 
                        platform_name=platform_name, 
                        tool_description=tool_desc,
                        platform_mention_rule=verify_rule
                    )
                    verifier_response = call_vllm_api(verifier_prompt, client=client_local, model_name=VLLM_MODEL_NAME)
                    is_match_str = parse_xml_tag(verifier_response, 'is_match')
                    
                    if is_match_str == 'true':
                        verified_l2_queries.append(query)
                        print(f"          VERIFIED: '{query[:60]}{'...' if len(query) > 60 else ''}'")
                    else:
                        print(f"          REJECTED: '{query[:60]}{'...' if len(query) > 60 else ''}'")
                        explanation = verifier_response.split('</is_match>')[-1].strip()
                        if explanation:
                            print(f"             Reason: {explanation}")

                print(f"         L2 Results: {len(verified_l2_queries)}/{len(generated_l2_queries)} queries verified")

                if verified_l1_queries or verified_l2_queries:
                    unique_tool_key = f"{converted_data.get('name', 'unknown_server')}/{tool_name}"
                    server_queries[unique_tool_key] = {
                        "level_1_queries": verified_l1_queries,
                        "level_2_queries": verified_l2_queries
                    }
                    print(f"         Saved queries for {unique_tool_key}")
                else:
                    print(f"           No verified queries generated for {tool_name}")
            else:
                print(f"        Skipping query generation (platform: {converted_data['platform_specificity']}, type: {tool.get('task_type', 'unknown')})")
                skipped_tools += 1
        
        return server_queries

    except Exception as e:
        print(f" ERROR processing server '{server_name}': {e}")
        print(f"   Stack trace: {str(e)}")
        return None

def main():
    """
    Main function to read from consolidated JSON and execute the L1/L2 query generation pipeline.
    
    This function:
    1. Reads consolidated MCP server data
    2. Processes each server through the query generation pipeline
    3. Saves results in batches to prevent data loss
    4. Provides detailed progress reporting
    """
    consolidated_file = Path("../data/mcp_registry.json")
    output_file = Path("../data/tests_L1_L2.json")

    # --- Configuration ---
    BATCH_SIZE = 50
    SAMPLE_LIMIT = 5  # Limit number of servers to process
    
    print(" STARTING L1/L2 QUERY GENERATION PIPELINE")
    print(f" Input file: {consolidated_file}")
    print(f" Output file: {output_file}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f" Sample limit: {SAMPLE_LIMIT}")
    print("="*70)
    
    # Check if consolidated file exists
    if not consolidated_file.exists():
        print(f" ERROR: Consolidated file '{consolidated_file}' not found.")
        return

    # Read consolidated JSON
    try:
        print(f" Reading consolidated JSON file...")
        with open(consolidated_file, 'r', encoding='utf-8') as f:
            consolidated_data = json.load(f)
    except Exception as e:
        print(f" ERROR: Could not read consolidated file: {e}")
        return

    servers = consolidated_data.get("servers", [])
    total_servers = len(servers)
    
    # Apply sample limit
    if SAMPLE_LIMIT and total_servers > SAMPLE_LIMIT:
        servers = servers[:SAMPLE_LIMIT]
        print(f"  Sample limit applied: Processing {SAMPLE_LIMIT}/{total_servers} servers")
    
    processed_servers = len(servers)
    
    if not servers:
        print(" No servers found in consolidated data.")
        return

    print(f" Ready to process {processed_servers} servers")
    print("="*70)

    all_generated_data = {}
    servers_processed_since_save = 0
    successful_processing = 0
    failed_processing = 0
    
    for i, server_data in enumerate(servers):
        server_name = server_data.get("server_name", f"server_{i}")
        print(f"\n [{i+1}/{processed_servers}] Processing: {server_name}")
        
        result = process_consolidated_server(server_data)
        
        # If queries were generated, update main dictionary
        if result:
            all_generated_data.update(result)
            successful_processing += 1
            print(f" Successfully processed {server_name} - {len(result)} query sets generated")
        else:
            failed_processing += 1
            print(f"  No queries generated for {server_name}")
        
        servers_processed_since_save += 1

        # Batch save logic
        if servers_processed_since_save >= BATCH_SIZE or (i + 1) == processed_servers:
            print("\n" + " BATCH SAVE OPERATION".center(70, "="))
            print(f" Progress: {i + 1}/{processed_servers} servers processed")
            
            if all_generated_data:
                print(f" Saving {len(all_generated_data)} total query entries to '{output_file.name}'...")
            else:
                print(f" No queries generated, saving empty file to '{output_file.name}'...")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_generated_data, f, indent=2, ensure_ascii=False)
                print(f" SUCCESS: Progress saved to {output_file}")
            except Exception as e:
                print(f" CRITICAL ERROR during batch save: {e}")
            
            # Reset counter
            servers_processed_since_save = 0
            print("="*70 + "\n")

    # Final summary
    if all_generated_data:
        print(f" Results saved to: {output_file}")
        print(f" Final summary: {successful_processing} successful, {failed_processing} failed, {len(all_generated_data)} total query sets")
    else:
        print(f" No queries were generated. Empty file saved to: {output_file}")
        print(f" Final summary: {successful_processing} successful, {failed_processing} failed")



if __name__ == "__main__":
    main()