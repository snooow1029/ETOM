# MSC_Bench: Multi-Level Server-Tool Selection Benchmark

A comprehensive evaluation framework for testing tool selection capabilities across multiple complexity levels, designed to assess the performance of various orchestrator implementations in multi-tool environments.

## Overview

MSC_Bench provides a standardized platform for evaluating how well different orchestration systems can select and sequence tools to complete complex tasks. The benchmark supports five levels of complexity, from simple single-tool selection to complex multi-tool dependency graphs.

## Key Features

- **Multi-Level Evaluation**: Five distinct complexity levels (1-5) covering single tools, tool sets, tool sequences, and complex dependency graphs
- **Multiple Orchestrator Support**: Evaluation framework for ToolShed, MCP-Zero, REACT, and Hybrid orchestrators
- **Comprehensive Metrics**: Level-specific evaluation metrics including exact match, F1-score, precision, and recall
- **Parallel Processing**: Configurable batch processing and multi-threading for efficient evaluation
- **Checkpoint System**: Resume interrupted evaluations with automatic progress saving
- **Extensible Architecture**: Easy addition of new orchestrators and evaluation levels

## Project Structure

```
MSC_Bench/
├── data/                         # Data and query files
│   ├── queries/                  # Evaluation queries by level
│   │   ├── level_1.json          # Single tool selection queries
│   │   ├── level_2.json          # Tool set selection queries
│   │   ├── level_3.json          # Tool sequence queries
│   │   ├── level_4.json          # Complex tool graph queries
│   │   └── level_5.json          # Advanced dependency queries
│   ├── mcp_registry.json         # Tool registry without embeddings
│   └── generate_embeddings.py    # Script to generate tool embeddings
├── orchestrator/                 # Orchestrator implementations
│   ├── orchestrator_core.py      # Base orchestrator class
│   ├── orchestrator_hybrid.py    # Hybrid orchestrator implementation
│   ├── orchestrator_mcp0.py      # MCP-Zero orchestrator
│   ├── orchestrator_react.py     # REACT orchestrator
│   ├── orchestrator_toolshed.py  # ToolShed orchestrator
│   ├── orchestrator_utils.py     # Utility functions and configurations
│   └── schemas.py                # Pydantic schemas for structured output
├── eval/                         # Evaluation framework
│   ├── eval.py                   # Generalized evaluation script
│   ├── graph_evaluator.py        # Graph-based evaluation for complex levels
├── eval_result/                  # Evaluation results and checkpoints
├── requirements.txt              # Python dependencies
├── .env.example                  # example to setup backbone llm
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MSC_Bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Evaluation Levels

### Level 1: Single Tool Selection
- **Objective**: Select the correct single tool for a given query
- **Evaluation**: Exact match between predicted and ground truth tool
- **Example**: "Get weather information" → `weather_tool`

### Level 2: Tool Set Selection
- **Objective**: Select any tool from a set of equivalent tools
- **Evaluation**: Binary success if any equivalent tool is selected
- **Example**: "Search the web" → `google_search` or `bing_search`

### Level 3: Tool Sequence
- **Objective**: Select tools in the correct sequence
- **Evaluation**: Sequence accuracy with dependency validation
- **Example**: "Analyze code and generate report" → `analyze_tool` → `report_tool`

### Level 4: Tool Graph with Equivalence
- **Objective**: Handle complex tool graphs with equivalent tool sets
- **Evaluation**: Graph F1-score with equivalence set matching
- **Example**: Multi-step data processing with alternative tool paths

### Level 5: Advanced Dependencies
- **Objective**: Complex dependency graphs with conditional tool selection
- **Evaluation**: Advanced graph metrics with conditional logic validation
- **Example**: Dynamic workflow with branching tool selection

## Supported Orchestrators

### ToolShed (TS)
Multi-LLM pipeline with specialized components:
- Router: Initial query classification
- Decomposer: Query decomposition
- Query Rewriter: Query optimization
- Query Expander: Query enhancement
- Reranker: Tool ranking
- Conversational: Final response generation

### MCP-Zero (MCP0)
Three-LLM architecture:
- Router: Query routing
- Retriever: Tool retrieval with embeddings
- Conversational: Response generation

### REACT
Reasoning and Acting approach:
- Iterative reasoning with tool use
- Dynamic tool selection based on intermediate results

### Hybrid
Combined approach:
- Leverages strengths of multiple orchestrator types
- Adaptive strategy selection

## Usage

### Quick Start

Run evaluation with interactive mode:
```bash
cd eval
python eval.py
```

### Command Line Usage

```bash
# Single level evaluation
python eval.py --level 1 --orchestrator TS --mode subset

# Full evaluation with custom parameters
python eval.py --level 3 --orchestrator MCP0 --mode full --batch_size 20 --max_workers 4

# Multiple level evaluation
for level in 1 2 3 4 5; do
    python eval.py --level $level --orchestrator TS --mode subset
done
```

### Programmatic Usage

```python
from eval.eval import run_evaluation, EvaluationConfig

config = EvaluationConfig(
    level=2,
    orchestrator="TS",
    mode="full",
    batch_size=15,
    max_workers=6
)

results = run_evaluation(config)
```

## Evaluation Metrics

### Level-Specific Metrics

- **Level 1**: Exact match accuracy
- **Level 2**: Correct prediction rate
- **Level 3**: Sequence accuracy with dependency validation
- **Level 4**: Graph F1-score with equivalence set matching
- **Level 5**: Advanced graph metrics with conditional validation

### Common Metrics

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Exact Match**: Perfect prediction rate

## Data Format

### Query Format
```json
{
  "query_id": 1,
  "query": "Task description",
  "ground_truth_tools_count": 2,
  "ground_truth_tools": [
    {
      "tool_id": "server::tool_name",
      "server_name": "server",
      "tool_name": "tool_name",
      "description": "Tool description"
    }
  ]
}
```

### Result Format
```json
{
  "query_id": 1,
  "status": "success",
  "predicted_tools": ["server::tool_name"],
  "metrics": {
    "exact_match": true,
    "f1_score": 1.0,
    "precision": 1.0,
    "recall": 1.0
  }
}
```

## Advanced Features

### Checkpoint System
- Automatic progress saving every 5 batches
- Resume interrupted evaluations
- Emergency checkpoint on errors
- Clear checkpoint on completion

### Parallel Processing
- Configurable batch sizes
- Multi-threading support
- Resource usage optimization
- Progress tracking with ETA

### Graph Evaluation
- Global relative order validation
- Fault-tolerant dependency checking
- Equivalent function set support
- Detailed debugging information

## Extending the Framework

### Adding New Orchestrators

1. Implement orchestrator class:
```python
class NewOrchestrator(BaseOrchestrator):
    def process_query(self, query: str) -> str:
        # Implementation
        return result
```

2. Register in orchestrator factory:
```python
def create_orchestrator(orchestrator_type: str, config: EvaluationConfig):
    if orchestrator_type == "NEW":
        return NewOrchestrator(config)
```

### Adding New Evaluation Levels

1. Create evaluation strategy:
```python
class Level6Strategy(EvaluationStrategy):
    def evaluate(self, query_id, result, ground_truth):
        # Level-specific evaluation logic
        return evaluation_result
```

2. Register strategy and update statistics functions

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `LOCAL_LLM_BASE_URL`: Local LLM server URL

### Orchestrator Parameters
- LLM configurations (temperature, model names)
- Tool selection parameters (top-k, similarity thresholds)
- Processing parameters (batch sizes, timeouts)

## Performance Considerations

- **Batch Size**: Larger batches reduce checkpoint overhead but increase memory usage
- **Max Workers**: More workers improve speed but increase resource consumption
- **Mode Selection**: Use subset mode for testing, full mode for complete evaluation
- **Checkpoint Frequency**: Balance between progress safety and performance

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure environment variables are properly set
2. **Memory Issues**: Reduce batch_size or max_workers
3. **Import Errors**: Check Python path and package installation
4. **Checkpoint Corruption**: Delete checkpoint files to start fresh

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Citation

If you use MSC_Bench in your research, please cite:

```bibtex
@article{msc_bench_2024,
  title={MSC_Bench: A Multi-Level Tool Selection Benchmark},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.

## Contact

For questions or issues, please open an issue on the project repository or contact the maintainers.