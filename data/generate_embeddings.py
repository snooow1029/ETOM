#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATA_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = DATA_DIR / "mcp_registry.json"
REGISTRY_EMBED_PATH = DATA_DIR / "mcp_registry_w_embedding.json"


def load_embedding_model():
    """Load the embedding model from environment variable (AutoTokenizer + AutoModel).

    Returns:
        tuple: (tokenizer, model, device)
    """
    model_name = os.getenv('EMBEDDING_MODEL_NAME', 'Qwen/Qwen3-Embedding-0.6B')

    print(f"Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Using device: {device}")
    return tokenizer, model, device


def generate_text_embedding(text, tokenizer, model, device, max_length=512):
    """Generate a single embedding vector for the provided text.

    Args:
        text (str): Input text.
        tokenizer: Hugging Face tokenizer instance.
        model: Hugging Face model instance.
        device: torch.device where model and inputs should be placed.
        max_length (int): Maximum token length for truncation.

    Returns:
        list: Floating point embedding vector.
    """
    # Handle empty or None inputs
    if not text or text.strip() == "":
        text = "No description available"

    # Truncate overly long text to avoid extreme tokenization
    if len(text) > max_length * 4:
        text = text[:max_length * 4]

    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean of last_hidden_state as a sentence embedding
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy().tolist()[0]

    except Exception as e:
        print(f"Error generating embedding: {e}")
        print(f"Problematic text (truncated): {text[:100]}...")
        # Return a zero vector as a fallback (dimension chosen conservatively)
        return [0.0] * 512


def create_server_text(server):
    """Construct a single descriptive string for a server to embed.

    The generated text concatenates name, description, summary and categories
    to provide a compact, searchable representation for embedding.
    """
    parts = []

    if server.get('server_name'):
        parts.append(f"Server: {server['server_name']}")

    if server.get('server_description'):
        parts.append(f"Description: {server['server_description']}")

    if server.get('server_summary'):
        parts.append(f"Summary: {server['server_summary']}")

    if server.get('categories'):
        categories_str = ", ".join(server['categories'])
        parts.append(f"Categories: {categories_str}")

    return " | ".join(parts)


def create_tool_text(tool):
    """Construct a compact descriptive string for a tool.

    Includes tool name, description and a short representation of the
    input schema (if present) to improve retrieval quality.
    """
    parts = []

    if tool.get('tool_name'):
        parts.append(f"Tool: {tool['tool_name']}")

    if tool.get('tool_description'):
        parts.append(f"Description: {tool['tool_description']}")

    if tool.get('input_schema'):
        try:
            schema = tool['input_schema']
            if isinstance(schema, dict) and 'properties' in schema:
                props = []
                for key, value in schema['properties'].items():
                    if isinstance(value, dict) and 'description' in value:
                        props.append(f"{key}: {value['description']}")
                if props:
                    # Limit the amount of schema text to keep embeddings focused
                    parts.append(f"Parameters: {'; '.join(props[:5])}")
        except Exception as e:
            print(f"Warning: failed to process input_schema: {e}")

    return " | ".join(parts)


def process_consolidated_mcp():
    """Read the consolidated registry, create embeddings and save the result.

    Returns:
        str: Path to the written output file.
    """
    input_path = REGISTRY_PATH
    output_path = REGISTRY_EMBED_PATH

    tokenizer, model, device = load_embedding_model()

    print("Reading input registry...")
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    servers = data.get('servers', [])
    print(f"Found {len(servers)} servers in the registry")

    print("Generating server embeddings...")
    for i, server in enumerate(tqdm(servers, desc="Server Embeddings")):
        server_text = create_server_text(server)
        server_embedding = generate_text_embedding(server_text, tokenizer, model, device)
        server['server_embedding'] = server_embedding

        tools = server.get('tools', [])
        for tool in tools:
            tool_text = create_tool_text(tool)
            tool_embedding = generate_text_embedding(tool_text, tokenizer, model, device)
            tool['tool_embedding'] = tool_embedding

        # Periodically save a temporary checkpoint to reduce data loss risk
        if (i + 1) % 50 == 0:
            print(f"\nProcessed {i + 1} servers — saving intermediate checkpoint...")
            tmp_path = output_path.parent / f"{output_path.name}.tmp_{i+1}"
            with tmp_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nWriting final output to: {output_path}")
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_servers = len(servers)
    total_tools = sum(len(server.get('tools', [])) for server in servers)

    print(f"\n✅ Finished processing")
    print(f"   • Total servers: {total_servers}")
    print(f"   • Total tools: {total_tools}")
    print(f"   • Output file: {output_path}")

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   • File size: {file_size:.2f} MB")

    return str(output_path)


def test_embedding_quality(output_path):
    """Basic checks to validate generated embeddings.

    Prints dimensions and numeric ranges for the first few servers/tools.
    """
    print("\n🔍 Testing embedding quality...")

    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    servers = data.get('servers', [])

    for i, server in enumerate(servers[:3]):
        server_name = server.get('server_name', f'Server {i}')
        server_embedding = server.get('server_embedding', [])

        print(f"\n📊 {server_name}:")
        print(f"   • Server embedding dimension: {len(server_embedding)}")
        if server_embedding:
            print(f"   • Server embedding range: [{min(server_embedding):.4f}, {max(server_embedding):.4f}]")

        tools = server.get('tools', [])
        if tools:
            tool = tools[0]
            tool_embedding = tool.get('tool_embedding', [])
            tool_name = tool.get('tool_name', 'Unknown Tool')
            print(f"   • First tool ({tool_name}) embedding dimension: {len(tool_embedding)}")
            if tool_embedding:
                print(f"   • Tool embedding range: [{min(tool_embedding):.4f}, {max(tool_embedding):.4f}]")


if __name__ == "__main__":
    # Verify required package availability
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Please install transformers: pip install transformers")
        exit(1)

    try:
        output_path = process_consolidated_mcp()
        test_embedding_quality(output_path)
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()