"""
Common utility functions and configurations for the MSC_Bench project.
This module contains shared functions used across multiple pipeline scripts.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
def setup_environment(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

# --- API Client Configuration & Initialization ---

def get_vllm_client(base_url: Optional[str] = None, api_key: Optional[str] = None, 
                   model_name: Optional[str] = None) -> tuple[OpenAI, str]:
    """
    Initialize and return vLLM client and model name.
    
    Args:
        base_url: vLLM base URL, defaults to LOCAL_URL env var
        api_key: API key, defaults to LOCAL_API_KEY env var  
        model_name: Model name, defaults to LOCAL_MODEL env var
        
    Returns:
        tuple: (OpenAI client, model_name)
    """
    vllm_base_url = base_url or os.getenv("LOCAL_URL", "http://localhost:8000/v1")
    vllm_api_key = api_key or os.getenv("LOCAL_API_KEY", "moe")
    vllm_model_name = model_name or os.getenv("LOCAL_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    client = OpenAI(base_url=vllm_base_url, api_key=vllm_api_key)
    return client, vllm_model_name

def get_azure_openai_client() -> AzureOpenAI:
    """
    Initialize and return Azure OpenAI client.
    
    Returns:
        AzureOpenAI: Configured Azure OpenAI client
    """
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    if not azure_api_key or not azure_endpoint:
        raise ValueError("Azure OpenAI credentials not found in environment variables")
    
    return AzureOpenAI(
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )

# --- API Call Functions ---

def call_vllm_api(prompt: str, client: OpenAI = None, model_name: str = None, 
                  max_tokens: int = 512, temperature: float = 1.0, 
                  top_p: float = 0.9, frequency_penalty: float = 0.0, 
                  presence_penalty: float = 0.0) -> str:
    """
    Call the vLLM API for text generation.
    
    Args:
        prompt: The input prompt for the model
        client: OpenAI client instance, creates default if None
        model_name: Model name, uses default if None
        max_tokens: Maximum tokens in response
        temperature: Temperature for response generation
        top_p: Top-p sampling parameter
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        
    Returns:
        str: Generated response text or empty string on error
    """
    if client is None:
        client, model_name = get_vllm_client()
    elif model_name is None:
        _, model_name = get_vllm_client()
        
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            frequency_penalty=frequency_penalty, 
            presence_penalty=presence_penalty
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return ""

def call_azure_openai(prompt_content: str, client: AzureOpenAI = None, 
                     model_id: str = None, max_tokens: int = 512, 
                     temperature: float = 0.0) -> str:
    """
    Call Azure OpenAI Chat Completions API.
    
    Args:
        prompt_content: The prompt to send
        client: Azure OpenAI client instance, creates default if None
        model_id: Azure deployment name, defaults to env var
        max_tokens: Maximum tokens in response
        temperature: Temperature for response generation
        
    Returns:
        str: Generated response text or empty string on error
    """
    if client is None:
        client = get_azure_openai_client()
    
    if model_id is None:
        model_id = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    
    messages = [{"role": "user", "content": prompt_content}]

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"Error calling Azure OpenAI API: {e}")
        return ""

def call_azure_openai_with_usage(prompt_content: str, client: AzureOpenAI = None, 
                                model_id: str = None, max_tokens: int = 4096, 
                                temperature: float = 0.7) -> tuple[str, Dict[str, int]]:
    """
    Call Azure OpenAI API and return response with token usage information.
    
    Args:
        prompt_content: The prompt to send
        client: Azure OpenAI client instance, creates default if None
        model_id: Azure deployment name, defaults to env var
        max_tokens: Maximum tokens in response
        temperature: Temperature for response generation
        
    Returns:
        tuple: (response_content, usage_dict)
    """
    if client is None:
        client = get_azure_openai_client()
    
    if model_id is None:
        model_id = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    
    messages = [{"role": "user", "content": prompt_content}]

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        content = response.choices[0].message.content or ""
        usage = response.usage
        
        if usage:
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
        else:
            usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
        return content, usage_dict

    except Exception as e:
        print(f"Error calling Azure OpenAI API: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# --- Utility Functions ---

def parse_xml_tag(text: str, tag_name: str) -> Optional[str]:
    """
    Parse content from specified XML tag in text.
    
    Args:
        text: Text containing XML tags
        tag_name: Name of the XML tag to extract
        
    Returns:
        str: Content within the tag or None if not found
    """
    if not text:
        return None
    match = re.search(f'<{tag_name}>(.*?)</{tag_name}>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_multiple_xml_tags(text: str, tag_name: str) -> List[str]:
    """
    Parse all specified XML tags content from text.
    
    Args:
        text: Text containing XML tags
        tag_name: Name of the XML tags to extract
        
    Returns:
        list: List of content within all matching tags
    """
    if not text:
        return []
    return re.findall(f'<{tag_name}>(.*?)</{tag_name}>', text, re.DOTALL)

def format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """
    Convert JSON Schema to LLM-friendly text format.
    
    Args:
        schema: JSON schema dictionary
        
    Returns:
        str: Formatted schema description
    """
    if not schema or 'properties' not in schema:
        return "This tool does not specify any inputs."
        
    lines = ["Inputs required by the tool:"]
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    for name, prop in properties.items():
        is_required = " (required)" if name in required else ""
        prop_type = prop.get('type', 'any')
        description = prop.get('description', 'No description.')
        lines.append(f"- `{name}`{is_required}: {description} (Type: {prop_type})")
        
    if not properties:
        return "This tool takes no specific inputs."
        
    return "\n".join(lines)

def extract_tagged_content(text: str, tag: str) -> str:
    """
    Extract content from XML-style tags, with fallback to original text.
    
    Args:
        text: Text containing XML tags
        tag: Name of the XML tag to extract
        
    Returns:
        str: Content within the tag or original text if not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        print(f"Warning: Could not find content within <{tag}> tags in the response.")
        return text
    return match.group(1).strip()

# --- File and Path Utilities ---

def find_repo_root(start_path: Path = None) -> Path:
    """
    Find the repository root by looking for .env file.
    
    Args:
        start_path: Starting path to search from, defaults to current file's parent
        
    Returns:
        Path: Repository root path
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
        
    current = start_path
    while current != current.parent:
        if (current / ".env").exists():
            return current
        current = current.parent
    # Fallback to assuming it's two levels up from src
    return start_path.parent

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        dict: Loaded JSON data or empty dict on error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file '{file_path}': {e}")
        return {}

def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file '{file_path}': {e}")
        return False

# --- Legacy compatibility functions ---
# These maintain compatibility with existing code patterns

def get_vllm_config():
    """Get vLLM configuration from environment variables."""
    return {
        "base_url": os.getenv("LOCAL_URL", "http://localhost:8000/v1"),
        "api_key": os.getenv("LOCAL_API_KEY", "moe"),
        "model_name": os.getenv("LOCAL_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    }

def get_azure_config():
    """Get Azure OpenAI configuration from environment variables."""
    return {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    }