"""
Generalized Query Evaluation Script
Supports multiple levels and orchestrator implementations.
Easily extensible for new levels and orchestrators.
"""
import logging
import os
import sys
import json
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator import (
    ToolShedOrchestrator, MCPZeroOrchestrator, REACTOrchestrator, HybridOrchestrator,
    OrchestratorConfig, LLMConfig, setup_colored_logging
)

# Handle both direct execution and package import
try:
    from .graph_evaluator import GraphEvaluator
except ImportError:
    from graph_evaluator import GraphEvaluator

load_dotenv()

# Setup logging
setup_colored_logging()
logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)

# --- Configuration Management ---
@dataclass
class EvaluationConfig:
    level: int
    orchestrator: str
    mode: str  # 'full' or 'subset'
    batch_size: int = 10
    max_workers: int = 5
    llm_provider: str = "local_openai"  # 'azure_openai' or 'local_openai'
    orchestrator_params: Optional[Dict[str, Any]] = None  # Custom orchestrator parameters
    
    def __post_init__(self):
        # Set default orchestrator_params if not provided
        if self.orchestrator_params is None:
            if self.orchestrator.upper() == "TS":
                self.orchestrator_params = {
                    "tool_top_k": 5, 
                    "similarity_threshold": 0,
                    "query_expansion_count": 3, 
                    "rerank_top_k": 10
                }
            elif self.orchestrator.upper() == "HB":
                self.orchestrator_params = {
                    "server_top_k": 3,
                    "tool_top_k": 5, 
                    "similarity_threshold": 0,
                    "query_expansion_count": 3, 
                    "rerank_top_k": 10
                }
            elif self.orchestrator.upper() == "REACT":
                self.orchestrator_params = {
                    "num_attempts": 3
                }
            elif self.orchestrator.upper() == "MCP0":
                self.orchestrator_params = {
                    "server_top_k": 3,
                    "tool_top_k": 5
                }
            else:
                self.orchestrator_params = {}

# --- Orchestrator Factory ---
class OrchestratorFactory:
    @staticmethod
    def create(orchestrator_type: str, config: EvaluationConfig):
        """Create orchestrator instance based on type."""
        llm_type = config.llm_provider
        
        if orchestrator_type.upper() == "TS":
            orch_config = OrchestratorConfig.create(
                llms={
                    "router": LLMConfig(name="router", llm_type=llm_type, temperature=0),
                    "decomposer": LLMConfig(name="decomposer", llm_type=llm_type, temperature=0),
                    "query_rewriter": LLMConfig(name="query_rewriter", llm_type=llm_type, temperature=0.3),
                    "query_expander": LLMConfig(name="query_expander", llm_type=llm_type, temperature=0.5),
                    "reranker": LLMConfig(name="reranker", llm_type=llm_type, temperature=0),
                    "conversational": LLMConfig(name="conversational", llm_type=llm_type, temperature=0.7)
                },
                orchestrator_params=config.orchestrator_params
            )
            return ToolShedOrchestrator(config=orch_config)
        elif orchestrator_type.upper() == "HB":
            orch_config = OrchestratorConfig.create(
                llms={
                    "router": LLMConfig(name="router", llm_type=llm_type, temperature=0),
                    "decomposer": LLMConfig(name="decomposer", llm_type=llm_type, temperature=0),
                    "query_rewriter": LLMConfig(name="query_rewriter", llm_type=llm_type, temperature=0.3),
                    "query_expander": LLMConfig(name="query_expander", llm_type=llm_type, temperature=0.5),
                    "reranker": LLMConfig(name="reranker", llm_type=llm_type, temperature=0),
                    "conversational": LLMConfig(name="conversational", llm_type=llm_type, temperature=0.7)
                },
                orchestrator_params=config.orchestrator_params
            )
            return HybridOrchestrator(config=orch_config)
        elif orchestrator_type.upper() == "MCP0":
            orch_config = OrchestratorConfig.create(
                llms={
                    "router": LLMConfig(name="router", llm_type=llm_type, temperature=0),
                    "retriever": LLMConfig(name="retriever", llm_type=llm_type, temperature=0),
                    "conversational": LLMConfig(name="conversational", llm_type=llm_type, temperature=0.7),
                    "decomposer": LLMConfig(name="decomposer", llm_type=llm_type, temperature=0)
                },
                orchestrator_params=config.orchestrator_params
            )
            return MCPZeroOrchestrator(config=orch_config)
        elif orchestrator_type.upper() == "REACT":
            orch_config = OrchestratorConfig.create(
                llms={
                    "router": LLMConfig(name="router", llm_type=llm_type, temperature=0),
                    "conversational": LLMConfig(name="conversational", llm_type=llm_type, temperature=0.7),
                },
                orchestrator_params=config.orchestrator_params
            )
            return REACTOrchestrator(config=orch_config)
        else:
            raise ValueError(f"Unknown orchestrator type: {orchestrator_type}. Available: TS, HB, MCP0, REACT")

# --- Evaluation Strategy Registry ---
class EvaluationStrategy:
    """Base class for evaluation strategies."""
    def evaluate(self, query_id: str, result: Any, ground_truth_tools: List[Dict]) -> Dict[str, Any]:
        raise NotImplementedError
    
    @staticmethod
    def _extract_graph_metrics(eval_result):
        """Extract unified metrics from GraphEvaluator result.
        
        Used by Level3 and Level4 strategies to consistently process GraphEvaluator output.
        """
        return {
            'precision': eval_result.precision,
            'recall': eval_result.recall,
            'f1': eval_result.f1_score,
            'tp': eval_result.metrics['tp'],
            'fp': eval_result.metrics['fp'],
            'fn': eval_result.metrics['fn'],
            'exact_match': eval_result.exact_match
        }

    @staticmethod
    def _normalize_predictions(result: Any) -> List[Any]:
        """Normalize orchestrator result into a list."""
        if result is None:
            return []
        return result if isinstance(result, list) else [result]

    @staticmethod
    def _format_simple_ground_truth(ground_truth_tools: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Format ground truth tools for single-step graph evaluation."""
        formatted: List[List[Dict[str, Any]]] = []

        for entry in ground_truth_tools:
            if isinstance(entry, list):
                group: List[Dict[str, Any]] = []
                for option in entry:
                    normalized = EvaluationStrategy._normalize_ground_truth_tool(option)
                    if normalized:
                        group.append(normalized)
                if group:
                    formatted.append(group)
            else:
                normalized = EvaluationStrategy._normalize_ground_truth_tool(entry)
                if normalized:
                    formatted.append([normalized])
        return formatted

    @staticmethod
    def _extract_predicted_tools_info(raw_prediction_list: List[Any]) -> List[Dict[str, str]]:
        """Extract server/tool metadata from orchestrator predictions."""
        predicted_tools_info = []
        for tool_json_str in raw_prediction_list:
            if isinstance(tool_json_str, str):
                try:
                    tool_data = json.loads(tool_json_str)
                    server = tool_data.get('server', '') or tool_data.get('server_name', '')
                    tool_name = tool_data.get('tool_name', '')
                    predicted_tools_info.append({'server': server, 'tool_name': tool_name})
                except json.JSONDecodeError:
                    predicted_tools_info.append({'server': 'parse_error', 'tool_name': str(tool_json_str)})
            elif isinstance(tool_json_str, dict):
                server = tool_json_str.get('server', '') or tool_json_str.get('server_name', '')
                tool_name = tool_json_str.get('tool_name', '')
                predicted_tools_info.append({'server': server, 'tool_name': tool_name})
            else:
                predicted_tools_info.append({'server': 'unknown_format', 'tool_name': str(tool_json_str)})
        return predicted_tools_info

    @staticmethod
    def _compute_primary_matches(predicted_tools_info: List[Dict[str, str]], ground_truth_tools: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Compute tool/server match flags for the first predicted tool."""
        tool_name_match = False
        server_match = False

        if predicted_tools_info:
            pred_tool = predicted_tools_info[0]
            for gt_tool in EvaluationStrategy._flatten_ground_truth_tools(ground_truth_tools):
                if pred_tool['tool_name'] == gt_tool.get('tool_name', ''):
                    tool_name_match = True
                if pred_tool['server'] == gt_tool.get('server_name', ''):
                    server_match = True

        return {
            'tool_name_match': tool_name_match,
            'server_match': server_match
        }

    @staticmethod
    def _normalize_ground_truth_tool(tool_candidate: Any) -> Optional[Dict[str, Any]]:
        """Normalize a ground truth tool candidate into a standard dict."""
        if isinstance(tool_candidate, str):
            try:
                tool_candidate = json.loads(tool_candidate)
            except json.JSONDecodeError:
                return None

        if isinstance(tool_candidate, dict):
            server = tool_candidate.get('server_name', tool_candidate.get('server', ''))
            tool_name = tool_candidate.get('tool_name', '')
            dependencies = tool_candidate.get('dependencies', []) or []
            return {
                'server_name': server,
                'tool_name': tool_name,
                'dependencies': dependencies
            }

        return None

    @staticmethod
    def _flatten_ground_truth_tools(ground_truth_tools: List[Any]) -> List[Dict[str, Any]]:
        """Flatten nested ground truth tool structures into a simple list of dicts."""
        flattened: List[Dict[str, Any]] = []
        for entry in ground_truth_tools:
            if isinstance(entry, list):
                for option in entry:
                    normalized = EvaluationStrategy._normalize_ground_truth_tool(option)
                    if normalized:
                        flattened.append(normalized)
            else:
                normalized = EvaluationStrategy._normalize_ground_truth_tool(entry)
                if normalized:
                    flattened.append(normalized)
        return flattened

    @staticmethod
    def _format_graph_ground_truth(ground_truth_tools: List[Any], allow_efs: bool = False) -> List[List[Dict[str, Any]]]:
        """Format ground truth for GraphEvaluator across graph-based levels."""
        formatted: List[List[Dict[str, Any]]] = []

        for entry in ground_truth_tools:
            if allow_efs and isinstance(entry, list):
                group: List[Dict[str, Any]] = []
                for option in entry:
                    normalized = EvaluationStrategy._normalize_ground_truth_tool(option)
                    if normalized:
                        group.append(normalized)
                if group:
                    formatted.append(group)
            else:
                normalized = EvaluationStrategy._normalize_ground_truth_tool(entry)
                if normalized:
                    formatted.append([normalized])

        return formatted

    @staticmethod
    def _ensure_json_strings(prediction_sequence: List[Any]) -> List[str]:
        """Serialize prediction items to JSON strings for GraphEvaluator."""
        raw_predictions: List[str] = []
        for item in prediction_sequence:
            if isinstance(item, str):
                raw_predictions.append(item)
            else:
                raw_predictions.append(json.dumps(item))
        return raw_predictions

class Level1Strategy(EvaluationStrategy):
    """Level 1 evaluation: Single tool call accuracy."""
    
    def evaluate(self, query_id, result, ground_truth_tools):
        """
        Evaluate Level 1 queries using the new GraphEvaluator.
        """
        # Handle empty result
        if not result:
            return {
                'query_id': query_id,
                'status': 'empty_result_error',
                'exact_match': False,
                'tool_name_match': False,
                'server_match': False,
                'ground_truth_tools': ground_truth_tools,
                'predicted_tools': [],
                'f1_metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'exact_match': False},
                'error_message': "Level 1 queries should return 1 tool, but orchestrator returned empty result"
            }
        
        raw_prediction_list = self._normalize_predictions(result)

        # Convert ground_truth_tools to GraphEvaluator expected format
        gt_formatted = self._format_simple_ground_truth(ground_truth_tools)

        # Use GraphEvaluator for evaluation
        evaluator = GraphEvaluator(level=1)
        eval_result = evaluator.evaluate(gt_formatted, raw_prediction_list)

        # Parse detailed information of predicted tools
        predicted_tools_info = self._extract_predicted_tools_info(raw_prediction_list)

        # Calculate traditional matching metrics (for backward compatibility)
        exact_match = eval_result.exact_match
        match_flags = self._compute_primary_matches(predicted_tools_info, ground_truth_tools)
        tool_name_match = match_flags['tool_name_match']
        server_match = match_flags['server_match']
        
        # Check if should be marked as error
        status = 'tool_call'
        error_message = None
        
        if len(raw_prediction_list) > 1:
            status = 'multi_tool_error'
            error_message = f"Level 1 queries should require only 1 tool, but orchestrator returned {len(raw_prediction_list)} tools"
        
        return {
            'query_id': query_id,
            'status': status,
            'exact_match': exact_match,
            'tool_name_match': tool_name_match,
            'server_match': server_match,
            'ground_truth_tools': ground_truth_tools,
            'predicted_tools': predicted_tools_info,
            'f1_metrics': self._extract_graph_metrics(eval_result),
            'error_message': error_message,
            'graph_eval_details': eval_result.details  # Add detailed diagnostic information
        }

class Level2Strategy(EvaluationStrategy):
    """Level 2 evaluation: Tool set accuracy (any valid tool from set)."""
    
    def evaluate(self, query_id, result, ground_truth_tools):
        """
        Evaluate Level 2 queries using the new GraphEvaluator.
        """
        # Handle empty result
        if not result:
            return {
                'query_id': query_id,
                'status': 'empty_result_error',
                'exact_match': False,
                'tool_name_match': False,
                'server_match': False,
                'ground_truth_tools': ground_truth_tools,
                'predicted_tools': [],
                'ground_truth_set': [(gt.get('server_name', ''), gt.get('tool_name', '')) for gt in ground_truth_tools],
                'ground_truth_count': len(ground_truth_tools),
                'f1_metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'exact_match': False},
                'error_message': "Level 2 queries should return 1 tool, but orchestrator returned empty result"
            }
        
        raw_prediction_list = self._normalize_predictions(result)

        # Convert ground_truth_tools to GraphEvaluator expected format
        gt_formatted = self._format_simple_ground_truth(ground_truth_tools)

        # Use GraphEvaluator for evaluation
        evaluator = GraphEvaluator(level=2)
        eval_result = evaluator.evaluate(gt_formatted, raw_prediction_list)

        # Parse detailed information of predicted tools
        predicted_tools_info = self._extract_predicted_tools_info(raw_prediction_list)

        # Calculate traditional matching metrics (for backward compatibility)
        exact_match = eval_result.exact_match
        match_flags = self._compute_primary_matches(predicted_tools_info, ground_truth_tools)
        tool_name_match = match_flags['tool_name_match']
        server_match = match_flags['server_match']
        
        # Check if should be marked as error
        status = 'tool_call'
        error_message = None
        
        if len(raw_prediction_list) > 1:
            status = 'multi_tool_error'
            error_message = f"Level 2 queries should require only 1 tool, but orchestrator returned {len(raw_prediction_list)} tools"
        
        # Create ground truth set for backward compatibility
        flattened_gt = self._flatten_ground_truth_tools(ground_truth_tools)
        ground_truth_set = [(gt.get('server_name', ''), gt.get('tool_name', '')) for gt in flattened_gt]
        
        return {
            'query_id': query_id,
            'status': status,
            'exact_match': exact_match,
            'tool_name_match': tool_name_match,
            'server_match': server_match,
            'ground_truth_tools': ground_truth_tools,
            'predicted_tools': predicted_tools_info,
            'ground_truth_set': ground_truth_set,
            'ground_truth_count': len(flattened_gt) if flattened_gt else len(ground_truth_tools),
            'f1_metrics': self._extract_graph_metrics(eval_result),
            'error_message': error_message,
            'graph_eval_details': eval_result.details  # Add detailed diagnostic information
        }

class Level3Strategy(EvaluationStrategy):
    """Level 3 evaluation: Full tool graph structure accuracy."""

    def evaluate(self, query_id, result, ground_truth_tools):
        """Evaluate tool graph structure for Level 3 plans."""
        base_eval = {'query_id': query_id, 'set_id': None}
        gt_formatted = EvaluationStrategy._format_graph_ground_truth(ground_truth_tools)
        gt_flat = [tool for group in gt_formatted for tool in group]
        is_gt_empty = len(gt_flat) == 0

        if not isinstance(result, list):
            match = is_gt_empty
            status = 'conversational' if match else 'error'
            return {
                **base_eval,
                'status': status,
                'exact_match': match,
                'graph_structure_match': match,
                'error_message': 'Predicted result was not a valid tool plan list.'
            }

        if not result:
            if is_gt_empty:
                return {**base_eval, 'status': 'conversational', 'exact_match': True, 'graph_structure_match': True}
            return {
                **base_eval,
                'status': 'tool_graph',
                'exact_match': False,
                'graph_f1_score': 0.0,
                'graph_precision': 0.0,
                'graph_recall': 0.0,
                'correctly_placed_common_tools': 0,
                'common_tools_count': 0,
                'predicted_tools_count': 0,
                'total_gt_tools': len(gt_formatted),
                'graph_structure_match': False,
                'node_set_match': False,
                'predicted_graph': {'nodes': [], 'dependencies': []},
                'ground_truth_graph': {'nodes': [], 'dependencies': []}
            }

        raw_prediction_list = EvaluationStrategy._ensure_json_strings(result)
        try:
            evaluator = GraphEvaluator(level=3)
            eval_result = evaluator.evaluate(gt_formatted, raw_prediction_list)
        except Exception as e:
            logger.exception("Level3 evaluation failed")
            return {
                **base_eval,
                'status': 'error',
                'exact_match': False,
                'graph_structure_match': False,
                'error_message': str(e)
            }

        f1_metrics = EvaluationStrategy._extract_graph_metrics(eval_result)
        metrics = eval_result.metrics

        predicted_tools_info = EvaluationStrategy._extract_predicted_tools_info(raw_prediction_list)
        pred_nodes = {
            (info['server'], info['tool_name'])
            for info in predicted_tools_info
            if info['server'] and info['tool_name']
        }
        gt_nodes = {
            (tool.get('server_name', ''), tool.get('tool_name', ''))
            for group in gt_formatted for tool in group
            if tool.get('server_name') and tool.get('tool_name')
        }

        node_set_match = pred_nodes == gt_nodes
        common_tools_count = len(pred_nodes & gt_nodes)
        total_predictions = metrics.get('total_predictions', len(predicted_tools_info))
        total_gt_tools = metrics.get('total_gt_nodes', len(gt_formatted))

        gt_deps = []
        for idx, group in enumerate(gt_formatted):
            if group:
                for dep_idx in group[0].get('dependencies', []):
                    if 0 <= dep_idx < idx:
                        gt_deps.append([dep_idx, idx])

        gt_has_dependencies = any(tool.get('dependencies') for tool in gt_flat)
        pred_deps = [[i - 1, i] for i in range(1, len(predicted_tools_info))] if gt_has_dependencies else []

        is_gt_parallel = all(not tool.get('dependencies') for tool in gt_flat) if gt_flat else True
        parallel_aware_match = is_gt_parallel and node_set_match
        final_exact_match = eval_result.exact_match or parallel_aware_match

        predicted_graph_nodes = [
            [info['server'], info['tool_name']]
            for info in predicted_tools_info
            if info['server'] or info['tool_name']
        ]
        ground_truth_graph_nodes = [
            [group[0]['server_name'], group[0]['tool_name']] if group else ['', '']
            for group in gt_formatted
        ]

        return {
            **base_eval,
            'status': 'tool_graph',
            'exact_match': final_exact_match,
            'graph_f1_score': f1_metrics['f1'] * 100,
            'graph_precision': f1_metrics['precision'] * 100,
            'graph_recall': f1_metrics['recall'] * 100,
            'correctly_placed_common_tools': f1_metrics['tp'],
            'common_tools_count': common_tools_count,
            'predicted_tools_count': total_predictions,
            'total_gt_tools': total_gt_tools,
            'graph_structure_match': eval_result.exact_match,
            'node_set_match': node_set_match,
            'is_gt_parallel': is_gt_parallel,
            'parallel_aware_match': parallel_aware_match,
            'predicted_graph': {'nodes': predicted_graph_nodes, 'dependencies': pred_deps},
            'ground_truth_graph': {'nodes': ground_truth_graph_nodes, 'dependencies': gt_deps},
            'graph_eval_details': eval_result.details,
            'f1_metrics': f1_metrics
        }

class Level4Strategy(EvaluationStrategy):
    """Level 4 evaluation: Tool graph with parameter accuracy."""
    
    def evaluate(self, query_id, result, ground_truth_tools):
        """Shared graph evaluation logic for Level 4 plans."""
        base_eval = {'query_id': query_id, 'set_id': None}
        gt_formatted = EvaluationStrategy._format_graph_ground_truth(ground_truth_tools, allow_efs=True)
        gt_flat = [tool for group in gt_formatted for tool in group]
        is_gt_empty = len(gt_flat) == 0

        if not isinstance(result, list):
            match = is_gt_empty
            status = 'conversational' if match else 'error'
            return {
                **base_eval,
                'status': status,
                'exact_match': match,
                'graph_structure_match': match,
                'error_message': 'Predicted result was not a valid tool plan list.'
            }

        if not result:
            if is_gt_empty:
                return {**base_eval, 'status': 'conversational', 'exact_match': True, 'graph_structure_match': True}
            return {
                **base_eval,
                'status': 'tool_graph',
                'exact_match': False,
                'graph_f1_score': 0.0,
                'graph_precision': 0.0,
                'graph_recall': 0.0,
                'correctly_placed_common_tools': 0,
                'common_tools_count': 0,
                'predicted_tools_count': 0,
                'total_gt_tools': len(gt_formatted),
                'graph_structure_match': False,
                'node_set_match': False,
                'predicted_graph': {'nodes': [], 'dependencies': []},
                'ground_truth_graph': {'nodes': [], 'dependencies': []}
            }

        raw_prediction_list = EvaluationStrategy._ensure_json_strings(result)
        try:
            evaluator = GraphEvaluator(level=4)
            eval_result = evaluator.evaluate(gt_formatted, raw_prediction_list)
        except Exception as e:
            logger.exception("Level4 evaluation failed")
            return {
                **base_eval,
                'status': 'error',
                'exact_match': False,
                'graph_structure_match': False,
                'error_message': str(e)
            }

        f1_metrics = EvaluationStrategy._extract_graph_metrics(eval_result)
        metrics = eval_result.metrics

        predicted_tools_info = EvaluationStrategy._extract_predicted_tools_info(raw_prediction_list)
        pred_nodes = {
            (info['server'], info['tool_name'])
            for info in predicted_tools_info
            if info['server'] and info['tool_name']
        }
        gt_nodes = {
            (tool.get('server_name', ''), tool.get('tool_name', ''))
            for group in gt_formatted for tool in group
            if tool.get('server_name') and tool.get('tool_name')
        }

        node_set_match = pred_nodes == gt_nodes
        common_tools_count = len(pred_nodes & gt_nodes)
        total_predictions = metrics.get('total_predictions', len(predicted_tools_info))
        total_gt_tools = metrics.get('total_gt_nodes', len(gt_formatted))

        gt_deps = []
        for idx, group in enumerate(gt_formatted):
            if group:
                for dep_idx in group[0].get('dependencies', []):
                    if 0 <= dep_idx < idx:
                        gt_deps.append([dep_idx, idx])

        gt_has_dependencies = any(tool.get('dependencies') for tool in gt_flat)
        pred_deps = [[i - 1, i] for i in range(1, len(predicted_tools_info))] if gt_has_dependencies else []

        is_gt_parallel = all(not tool.get('dependencies') for tool in gt_flat) if gt_flat else True
        parallel_aware_match = is_gt_parallel and node_set_match
        final_exact_match = eval_result.exact_match or parallel_aware_match

        predicted_graph_nodes = [
            [info['server'], info['tool_name']]
            for info in predicted_tools_info
            if info['server'] or info['tool_name']
        ]
        ground_truth_graph_nodes = [
            [group[0]['server_name'], group[0]['tool_name']] if group else ['', '']
            for group in gt_formatted
        ]

        return {
            **base_eval,
            'status': 'tool_chain_efs',
            'exact_match': final_exact_match,
            'graph_f1_score': f1_metrics['f1'] * 100,
            'graph_precision': f1_metrics['precision'] * 100,
            'graph_recall': f1_metrics['recall'] * 100,
            'correctly_placed_common_tools': f1_metrics['tp'],
            'common_tools_count': common_tools_count,
            'predicted_tools_count': total_predictions,
            'total_gt_tools': total_gt_tools,
            'graph_structure_match': eval_result.exact_match,
            'node_set_match': node_set_match,
            'is_gt_parallel': is_gt_parallel,
            'parallel_aware_match': parallel_aware_match,
            'predicted_graph': {'nodes': predicted_graph_nodes, 'dependencies': pred_deps},
            'ground_truth_graph': {'nodes': ground_truth_graph_nodes, 'dependencies': gt_deps},
            'graph_eval_details': eval_result.details,
            'f1_metrics': f1_metrics
        }

class Level5Strategy(EvaluationStrategy):
    """Level 5 evaluation: Beyond toolset - model should output no tools (tool/server = 'No')."""
    
    def _calculate_f1_metrics(self, predicted_tools, ground_truth_tools):
        """Calculate F1, precision, recall for multi-tool predictions against ground truth."""
        # For level 5, ground_truth_tools should always be empty
        # Perfect prediction should be empty list or tools with "No" values
        if not predicted_tools:
            # Empty prediction is perfect for level 5
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        
        # Check if all predicted tools have "No" values for server and tool_name
        no_tools_count = 0
        actual_tools_count = 0
        logger.debug("Level5 - Predicted tools: %s", predicted_tools)
        for tool_json_str in predicted_tools:
            if isinstance(tool_json_str, str):
                try:
                    tool_data = json.loads(tool_json_str)
                    server = tool_data.get('server', '').lower()
                    tool_name = tool_data.get('tool_name', '').lower()
                    if server == 'no' and tool_name == 'no':
                        no_tools_count += 1
                    else:
                        actual_tools_count += 1
                except json.JSONDecodeError:
                    actual_tools_count += 1
            elif isinstance(tool_json_str, dict):
                server = (tool_json_str.get('server', '') or tool_json_str.get('server_name', '')).lower()
                tool_name = tool_json_str.get('tool_name', '').lower()
                if server == 'no' and tool_name == 'no':
                    no_tools_count += 1
                else:
                    actual_tools_count += 1
            else:
                actual_tools_count += 1
        
        # For level 5: 
        # - "No" tools are correct (true positive)
        # - Any actual tools are wrong (false positive)
        # - Since ground truth is always empty, no false negatives
        tp = no_tools_count
        fp = actual_tools_count
        fn = 0  # Ground truth is always empty
        
        precision = tp / len(predicted_tools) if len(predicted_tools) > 0 else 1.0
        recall = 1.0 if actual_tools_count == 0 else 0.0  # Perfect recall if no actual tools predicted
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate(self, query_id, result, ground_truth_tools):
        # Handle multiturn orchestrator results (always return lists)
        logger.debug("Level5 - Raw result: %s", result)
        if isinstance(result, list):
            if len(result) == 0:
                # Empty result is perfect for level 5
                return {
                    'query_id': query_id,
                    'status': 'correct_no_tools',
                    'exact_match': True,
                    'tool_name_match': True,
                    'server_match': True,
                    'ground_truth_tools': ground_truth_tools,
                    'predicted_tools': [],
                    'f1_metrics': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0},
                    'message': "Perfect: Model correctly returned no tools"
                }
            else:
                # First check if it's a direct "No" string response
                if len(result) == 1 and isinstance(result[0], str) and result[0].strip().lower() == 'no':
                    return {
                        'query_id': query_id,
                        'status': 'correct_no_tools',
                        'exact_match': True,
                        'tool_name_match': True,
                        'server_match': True,
                        'ground_truth_tools': ground_truth_tools,
                        'predicted_tools': [{'server': 'No', 'tool_name': 'No'}],
                        'f1_metrics': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 1, 'fp': 0, 'fn': 0},
                        'message': "Perfect: Model correctly returned 'No' string"
                    }
                # Calculate F1 metrics for the tools returned
                f1_metrics = self._calculate_f1_metrics(result, ground_truth_tools)
                
                # Extract predicted tools for detailed reporting
                predicted_tools_info = []
                correct_no_tools = 0
                incorrect_tools = 0
                
                for tool_json_str in result:
                    if isinstance(tool_json_str, str):
                        try:
                            tool_data = json.loads(tool_json_str)
                            server = tool_data.get('server', '')
                            tool_name = tool_data.get('tool_name', '')
                            predicted_tools_info.append({'server': server, 'tool_name': tool_name})
                            
                            if server.lower() == 'no' and tool_name.lower() == 'no':
                                correct_no_tools += 1
                            else:
                                incorrect_tools += 1
                        except json.JSONDecodeError:
                            predicted_tools_info.append({'server': 'parse_error', 'tool_name': str(tool_json_str)})
                            incorrect_tools += 1
                    elif isinstance(tool_json_str, dict):
                        server = tool_json_str.get('server', '') or tool_json_str.get('server_name', '')
                        tool_name = tool_json_str.get('tool_name', '')
                        predicted_tools_info.append({'server': server, 'tool_name': tool_name})
                        
                        if server.lower() == 'no' and tool_name.lower() == 'no':
                            correct_no_tools += 1
                        else:
                            incorrect_tools += 1
                    else:
                        predicted_tools_info.append({'server': 'unknown_format', 'tool_name': str(tool_json_str)})
                        incorrect_tools += 1
                
                # Determine status and exact match
                if incorrect_tools == 0:
                    status = 'correct_no_tools'
                    exact_match = True
                    message = f"Perfect: All {correct_no_tools} tools correctly marked as 'No'"
                else:
                    status = 'incorrect_tools_predicted'
                    exact_match = False
                    message = f"Error: Model predicted {incorrect_tools} actual tools (should be 'No'), {correct_no_tools} correct 'No' tools"
                
                return {
                    'query_id': query_id,
                    'status': status,
                    'exact_match': exact_match,
                    'tool_name_match': exact_match,
                    'server_match': exact_match,
                    'ground_truth_tools': ground_truth_tools,
                    'predicted_tools': predicted_tools_info,
                    'f1_metrics': f1_metrics,
                    'correct_no_tools': correct_no_tools,
                    'incorrect_tools': incorrect_tools,
                    'message': message
                }
        
        # Handle single result (non-list format)
        try:
            # If result is already a dict, use it directly
            if isinstance(result, dict):
                parsed_result = result
            else:
                # If result is a JSON string, parse it
                parsed_result = json.loads(result)
                
            predicted_tool = parsed_result.get('tool_name', '')
            predicted_server = parsed_result.get('server', '') or parsed_result.get('server_name', '')
            
            # For level 5, check if both tool and server are "No"
            correct_prediction = (predicted_tool.lower() == 'no' and predicted_server.lower() == 'no')
            
            # Calculate F1 metrics for single tool case
            single_tool_result = [json.dumps({'server': predicted_server, 'tool_name': predicted_tool})]
            f1_metrics = self._calculate_f1_metrics(single_tool_result, ground_truth_tools)
            
            return {
                'query_id': query_id,
                'status': 'correct_no_tools' if correct_prediction else 'incorrect_tools_predicted',
                'exact_match': correct_prediction,
                'tool_name_match': predicted_tool.lower() == 'no',
                'server_match': predicted_server.lower() == 'no',
                'ground_truth_tools': ground_truth_tools,
                'predicted_tools': [{'server': predicted_server, 'tool_name': predicted_tool}],
                'f1_metrics': f1_metrics,
                'message': "Perfect: Correctly predicted 'No' for both tool and server" if correct_prediction 
                          else f"Error: Predicted tool='{predicted_tool}', server='{predicted_server}' (should be 'No', 'No')"
            }
        except Exception as e:
            return {
                'query_id': query_id,
                'status': 'error',
                'exact_match': False,
                'ground_truth_tools': ground_truth_tools,
                'predicted_tools': [],
                'f1_metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0},
                'error_message': str(e)
            }

# --- Strategy Registry ---
EVALUATION_STRATEGIES = {
    1: Level1Strategy(),
    2: Level2Strategy(),
    3: Level3Strategy(),
    4: Level4Strategy(),
    5: Level5Strategy(),
    # TODO: Add more levels as needed
}

def get_strategy(level: int) -> EvaluationStrategy:
    if level in EVALUATION_STRATEGIES:
        return EVALUATION_STRATEGIES[level]
    else:
        raise ValueError(f"No evaluation strategy implemented for level {level}. Available levels: {list(EVALUATION_STRATEGIES.keys())}")

# --- File Path Manager ---
class FilePathManager:
    @staticmethod
    def get_query_file(level: int) -> str:
        """Get query file path for a given level, with fallback options."""
        candidates = [
            f"data/queries/level_{level}.json",
            f"../data/queries/level_{level}.json",  # From eval/ directory
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No query file found for level {level}. Tried: {candidates}")
    
    @staticmethod
    def get_checkpoint_file(level: int, orchestrator: str) -> str:
        """Get checkpoint file path."""
        os.makedirs("eval_result", exist_ok=True)
        # kept for backward compatibility when no extra params provided
        return f"eval_result/eval_checkpoint_lv{level}_{orchestrator}.json"
    
    @staticmethod
    def get_results_file(level: int, orchestrator: str) -> str:
        """Get timestamped results file path."""
        os.makedirs("eval_result", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"eval_result/eval_results_level{level}_{orchestrator}_{timestamp}.json"
    
    @staticmethod
    def get_latest_file(level: int, orchestrator: str) -> str:
        """Get latest results file path."""
        os.makedirs("eval_result", exist_ok=True)
        return f"eval_result/eval_results_level{level}_{orchestrator}_latest.json"

    @staticmethod
    def _format_params_suffix(orchestrator_params: Optional[Dict[str, Any]], model_name: Optional[str] = None) -> str:
        """Create a compact suffix from orchestrator params and optional model name.

        Example: {tool_top_k:5, similarity_threshold:0, query_expansion_count:3, rerank_top_k:10}
        -> "5_0_3_10"
        For REACT: {num_attempts: 5} -> "5"
        If model_name provided, include a sanitized short model token: "Qwen3-4B" -> "Qwen3-4B"
        """
        if not orchestrator_params:
            return ''

        # Handle REACT orchestrator parameters
        if 'num_attempts' in orchestrator_params:
            parts = [str(orchestrator_params['num_attempts'])]
        else:
            # Define expected ordering for TS/HB params
            keys_order = []
            if 'server_top_k' in orchestrator_params:
                keys_order.append('server_top_k')
            if 'tool_top_k' in orchestrator_params:
                keys_order.append('tool_top_k')
            if 'similarity_threshold' in orchestrator_params:
                keys_order.append('similarity_threshold')
            if 'query_expansion_count' in orchestrator_params:
                keys_order.append('query_expansion_count')
            if 'rerank_top_k' in orchestrator_params:
                keys_order.append('rerank_top_k')

            parts = []
            for k in keys_order:
                v = orchestrator_params.get(k)
                if isinstance(v, float):
                    # remove decimal point if integer-like
                    if v.is_integer():
                        parts.append(str(int(v)))
                    else:
                        # convert float like 0.05 to 005 for compactness
                        parts.append(str(v).replace('.', '_'))
                else:
                    parts.append(str(v))

        # optional model short token
        model_token = ''
        if model_name:
            # sanitize model name and take last part after slash
            token = model_name.split('/')[-1]
            # Replace special characters but keep dots and dashes
            token = token.replace(' ', '_')
            # For OpenAI models like "gpt-4.1", keep as is
            # For HuggingFace models, shorten if too long
            if len(token) > 25:
                token = token[:25]
            model_token = f"_{token}"

        suffix = '_'.join(parts)
        return f"_{suffix}{model_token}"

    @staticmethod
    def get_results_file_with_params(level: int, orchestrator: str, orchestrator_params: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None) -> str:
        os.makedirs("eval_result", exist_ok=True)
        timestamp = datetime.now().strftime("%m%d%H%M")
        params_suffix = FilePathManager._format_params_suffix(orchestrator_params, model_name)
        return f"eval_result/eval_results_level{level}_{orchestrator}{params_suffix}_{timestamp}.json"

    @staticmethod
    def get_checkpoint_file_with_params(level: int, orchestrator: str, orchestrator_params: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None) -> str:
        os.makedirs("eval_result", exist_ok=True)
        params_suffix = FilePathManager._format_params_suffix(orchestrator_params, model_name)
        return f"eval_result/eval_checkpoint_lv{level}_{orchestrator}{params_suffix}.json"

# --- Data Loader ---
def load_queries(level: int) -> List[Dict[str, Any]]:
    path = FilePathManager.get_query_file(level)
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} queries from {path}")
    return data

# --- Checkpoint Utilities ---
def load_checkpoint(path: str):
    """Load checkpoint from file if it exists."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                checkpoint = json.load(f)
            print(f"📂 Loaded checkpoint: {checkpoint['completed_queries']} queries completed")
            return checkpoint
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
    return None

def save_checkpoint(evaluations, query_index, total_queries, start_time, path: str, config: Optional[Dict[str, Any]] = None):
    """Save checkpoint to file. Optionally include config dict so checkpoint is self-describing."""
    checkpoint = {
        'evaluations': evaluations,
        'completed_queries': len(evaluations),
        'last_query_index': query_index,
        'total_queries': total_queries,
        'start_time': start_time,
        'last_saved': datetime.now().isoformat(),
        'progress_percentage': (len(evaluations) / total_queries) * 100 if total_queries > 0 else 0,
        'config': config or {}
    }
    try:
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        return True
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")
        return False

def clear_checkpoint(path: str):
    """Clear checkpoint file."""
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"🗑️  Cleared checkpoint file")
        except Exception as e:
            print(f"⚠️  Failed to clear checkpoint: {e}")

# --- Progress Tracking ---
def print_progress(current, total, start_time, evaluations, level: int, batch_size=None, recent_latencies=None):
    """Print detailed progress information with level-specific metrics."""
    if current == 0: 
        return
        
    elapsed = time.time() - start_time
    progress = (current / total) * 100
    avg_time_per_query = elapsed / current
    estimated_remaining = avg_time_per_query * (total - current)
    
    # Show recent latency if available (last 10 queries)
    latency_info = ""
    if recent_latencies and len(recent_latencies) > 0:
        recent_avg_seconds = sum(recent_latencies[-10:]) / min(10, len(recent_latencies))
        latency_info = f" | Avg Latency: {recent_avg_seconds:.1f}s"
    
    if evaluations:
        # Level-specific progress metrics
        if level == 1:
            exact_matches = sum(1 for e in evaluations if e.get('exact_match', False))
            exact_rate = (exact_matches / len(evaluations)) * 100
            metric_label = "Exact Match"
            metric_value = exact_rate
            exact_info = f" | Exact: {exact_rate:.1f}%"
        elif level == 2:
            exact_matches = sum(1 for e in evaluations if e.get('exact_match', False))
            exact_rate = (exact_matches / len(evaluations)) * 100
            metric_label = "Correct Tool"
            metric_value = exact_rate
            exact_info = f" | Exact: {exact_rate:.1f}%"
        elif level == 3:
            tool_graphs = [e for e in evaluations if e.get('status') == 'tool_graph']
            if tool_graphs:
                total_f1_score = sum(e.get('graph_f1_score', 0) for e in tool_graphs)
                avg_f1_score = total_f1_score / len(tool_graphs)
                exact_matches = sum(1 for e in tool_graphs if e.get('exact_match', False))
                exact_rate = (exact_matches / len(tool_graphs)) * 100
                metric_label = "Graph F1"
                metric_value = avg_f1_score
                exact_info = f" | Exact: {exact_rate:.1f}%"
            else:
                metric_label = "Graph F1"
                metric_value = 0.0
                exact_info = ""
        elif level == 4:
            tool_chain_efs = [e for e in evaluations if e.get('status') == 'tool_chain_efs']
            if tool_chain_efs:
                total_f1_score = sum(e.get('efs_f1_score', 0) for e in tool_chain_efs)
                avg_f1_score = total_f1_score / len(tool_chain_efs)
                exact_matches = sum(1 for e in tool_chain_efs if e.get('exact_match', False))
                exact_rate = (exact_matches / len(tool_chain_efs)) * 100
                metric_label = "EFS F1"
                metric_value = avg_f1_score
                exact_info = f" | Exact: {exact_rate:.1f}%"
            else:
                metric_label = "EFS F1"
                metric_value = 0.0
                exact_info = ""
        elif level == 5:
            correct_no_tools = sum(1 for e in evaluations if e.get('exact_match', False))
            correct_rate = (correct_no_tools / len(evaluations)) * 100
            metric_label = "Correct No Tools"
            metric_value = correct_rate
            exact_info = ""
        else:
            metric_label = "Accuracy"
            metric_value = 0.0
            exact_info = ""
            
        batch_info = f" | Batch: {batch_size}" if batch_size else ""
        print(f"\n📊 Progress: {current}/{total} ({progress:.1f}%) | "
              f"Elapsed: {elapsed/60:.1f}m | "
              f"ETA: {estimated_remaining/60:.1f}m | "
              f"{metric_label}: {metric_value:.1f}%{exact_info}{batch_info}{latency_info}")
    else:
        batch_info = f" | Batch: {batch_size}" if batch_size else ""
        print(f"\n📊 Progress: {current}/{total} ({progress:.1f}%) | "
              f"Elapsed: {elapsed/60:.1f}m | "
              f"ETA: {estimated_remaining/60:.1f}m{batch_info}{latency_info}")

# --- Main Evaluation Loop ---
def process_single_query(orchestrator, query_data, query_index, strategy: EvaluationStrategy, config: Optional[EvaluationConfig] = None):
    try:
        query = query_data['query']
        query_id = query_data.get('query_id', str(query_index))
        

        ground_truth_tools = query_data.get('ground_truth_tools', [])
        
        # Precise timing for orchestrator processing only
        start_time = time.perf_counter()
        
        # Handle REACT orchestrator with num_attempts parameter
        if config and config.orchestrator.upper() == "REACT" and config.orchestrator_params:
            num_attempts = config.orchestrator_params.get('num_attempts', 3)
            result = orchestrator.process_query(query, num_attempts=num_attempts)
        else:
            result = orchestrator.process_query(query)
            
        end_time = time.perf_counter()
        query_latency = end_time - start_time
        
        evaluation = strategy.evaluate(query_id, result, ground_truth_tools)
        
        # Add timing information to evaluation result
        evaluation['query_latency_seconds'] = query_latency
        
        if 'set_id' in query_data:
            evaluation['set_id'] = query_data['set_id']
            
        return (query_index, evaluation, True, None, query_latency)
        
    except Exception as e:
        # Even on error, we want to track that this query had timing issues
        error_evaluation = {
            'query_id': query_data.get('query_id', str(query_index)),
            'set_id': query_data.get('set_id'),
            'status': 'processing_error',
            'exact_match': False,
            'graph_structure_match': False,
            'error_message': str(e),
            'query_latency_seconds': None,
            'f1_metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 1}  # Count error as FN
        }
        return (query_index, error_evaluation, False, str(e), None)

def process_batch_parallel(orchestrator, batch_queries, start_index, strategy: EvaluationStrategy, max_workers=5, config: Optional[EvaluationConfig] = None):
    evaluations = [{}] * len(batch_queries)
    query_latencies = []  # Collect all successful query latencies
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_single_query, orchestrator, query_data, start_index + i, strategy, config): i
            for i, query_data in enumerate(batch_queries)
        }
        for future in as_completed(future_to_index):
            batch_index = future_to_index[future]
            try:
                query_index, evaluation, success, error_msg, query_latency = future.result()
                evaluations[batch_index] = evaluation
                
                # Collect latency data for successful queries
                if success and query_latency is not None:
                    query_latencies.append(query_latency)
                
                if not success:
                    print(f"⚠️  Error in query {query_index}: {error_msg}")
            except Exception as e:
                print(f"❌ Failed to process query at batch index {batch_index}: {e}")
                evaluations[batch_index] = {
                    'query_id': batch_queries[batch_index].get('query_id', str(batch_index)),
                    'set_id': batch_queries[batch_index].get('set_id'),
                    'status': 'executor_error',
                    'exact_match': False,
                    'graph_structure_match': False,
                    'error_message': str(e),
                    'query_latency_seconds': None
                }
    
    return [e for e in evaluations if e], query_latencies

# --- Statistics Calculation (Level-specific and Extensible) ---
def calculate_statistics(evaluations, level: int, query_latencies=None):
    """Produce streamlined statistics for serialized reports."""
    total = len(evaluations)
    if total == 0:
        return {}

    from collections import Counter
    from typing import Dict, Any

    status_counts = Counter(e.get('status', 'unknown') for e in evaluations)

    def _avg_metrics(items):
        if not items:
            return None
        keys = ['precision', 'recall', 'f1']
        avg = {k: sum(m.get(k, 0.0) for m in items) / len(items) for k in keys}
        support = {
            'tp': sum(m.get('tp', 0) for m in items),
            'fp': sum(m.get('fp', 0) for m in items),
            'fn': sum(m.get('fn', 0) for m in items)
        }
        return {'averages': avg, 'support': support, 'count': len(items)}

    summary: Dict[str, Any] = {
        'level': level,
        'total_queries': total,
        'status_counts': dict(status_counts)
    }

    performance: Dict[str, Any] = {}
    level_specific: Dict[str, Any] = {}

    if level == 1:
        exact_matches = sum(1 for e in evaluations if e.get('exact_match'))
        tool_name_matches = sum(1 for e in evaluations if e.get('tool_name_match'))
        server_matches = sum(1 for e in evaluations if e.get('server_match'))

        f1_metrics = [e['f1_metrics'] for e in evaluations if isinstance(e.get('f1_metrics'), dict)]
        averages = _avg_metrics(f1_metrics)
        performance = {
            'metric_label': 'single_tool_f1',
            'metrics': averages['averages'] if averages else {},
            'support': averages['support'] if averages else {},
            'sample_size': averages['count'] if averages else 0
        }

        summary['exact_match_rate'] = exact_matches / total

        level_specific = {
            'match_rates': {
                'tool_name': tool_name_matches / total,
                'server': server_matches / total
            },
            'error_counts': {
                'multi_tool': status_counts.get('multi_tool_error', 0),
                'empty_result': status_counts.get('empty_result_error', 0),
                'processing': status_counts.get('processing_error', 0),
            }
        }

        multi_tool_metrics = [
            e['f1_metrics'] for e in evaluations if e.get('status') == 'multi_tool_error' and isinstance(e.get('f1_metrics'), dict)
        ]
        if multi_tool_metrics:
            level_specific['multi_tool_summary'] = _avg_metrics(multi_tool_metrics)

    elif level == 2:
        exact_matches = sum(1 for e in evaluations if e.get('exact_match'))
        tool_name_matches = sum(1 for e in evaluations if e.get('tool_name_match'))
        server_matches = sum(1 for e in evaluations if e.get('server_match'))

        f1_metrics = [e['f1_metrics'] for e in evaluations if isinstance(e.get('f1_metrics'), dict)]
        averages = _avg_metrics(f1_metrics)
        performance = {
            'metric_label': 'tool_selection_f1',
            'metrics': averages['averages'] if averages else {},
            'support': averages['support'] if averages else {},
            'sample_size': averages['count'] if averages else 0
        }

        summary['exact_match_rate'] = exact_matches / total

        level_specific = {
            'match_rates': {
                'tool_name': tool_name_matches / total,
                'server': server_matches / total
            },
            'error_counts': {
                'multi_tool': status_counts.get('multi_tool_error', 0),
                'empty_result': status_counts.get('empty_result_error', 0),
                'processing': status_counts.get('processing_error', 0),
            }
        }

        multi_tool_metrics = [
            e['f1_metrics'] for e in evaluations if e.get('status') == 'multi_tool_error' and isinstance(e.get('f1_metrics'), dict)
        ]
        if multi_tool_metrics:
            level_specific['multi_tool_summary'] = _avg_metrics(multi_tool_metrics)

    elif level == 3:
        tool_graphs = [e for e in evaluations if e.get('status') == 'tool_graph']
        exact_matches = sum(1 for e in evaluations if e.get('exact_match'))
        node_set_matches = sum(1 for e in evaluations if e.get('node_set_match'))

        f1_metrics = [e['f1_metrics'] for e in tool_graphs if isinstance(e.get('f1_metrics'), dict)]
        averages = _avg_metrics(f1_metrics)
        performance = {
            'metric_label': 'tool_graph_f1',
            'metrics': averages['averages'] if averages else {},
            'support': averages['support'] if averages else {},
            'sample_size': averages['count'] if averages else 0
        }
        summary['exact_match_rate'] = exact_matches / total

        tool_graph_count = sum(count for status, count in status_counts.items() if 'tool_graph' in status)
        conversational_count = sum(count for status, count in status_counts.items() if 'conversational' in status)
        error_count = sum(count for status, count in status_counts.items() if 'error' in status)

        level_specific = {
            'response_rates': {
                'tool_graph': tool_graph_count / total,
                'conversational': conversational_count / total,
                'error': error_count / total
            },
            'structure_match_rate': exact_matches / total,
            'node_set_match_rate': node_set_matches / total
        }

    elif level == 4:
        tool_chain_efs = [e for e in evaluations if e.get('status') == 'tool_chain_efs']
        exact_matches = sum(1 for e in evaluations if e.get('exact_match'))
        node_set_matches = sum(1 for e in evaluations if e.get('efs_node_set_match'))
        structure_matches = sum(1 for e in evaluations if e.get('efs_structure_match'))
        parallel_matches = sum(1 for e in evaluations if e.get('parallel_aware_match'))

        f1_metrics = [e['f1_metrics'] for e in tool_chain_efs if isinstance(e.get('f1_metrics'), dict)]
        averages = _avg_metrics(f1_metrics)
        performance = {
            'metric_label': 'efs_chain_f1',
            'metrics': averages['averages'] if averages else {},
            'support': averages['support'] if averages else {},
            'sample_size': averages['count'] if averages else 0
        }
        summary['exact_match_rate'] = exact_matches / total

        total_efs_matches = sum(e.get('efs_matches', 0) for e in tool_chain_efs)
        total_efs_alternatives = sum(e.get('total_efs_alternatives', 0) for e in tool_chain_efs)

        tool_chain_efs_count = sum(count for status, count in status_counts.items() if 'tool_chain_efs' in status)
        conversational_count = sum(count for status, count in status_counts.items() if 'conversational' in status)
        error_count = sum(count for status, count in status_counts.items() if 'error' in status)

        level_specific = {
            'response_rates': {
                'tool_chain_efs': tool_chain_efs_count / total,
                'conversational': conversational_count / total,
                'error': error_count / total
            },
            'structure_match_rate': structure_matches / total,
            'node_set_match_rate': node_set_matches / total,
            'parallel_aware_match_rate': parallel_matches / total,
            'efs_matches': total_efs_matches,
            'efs_alternatives': total_efs_alternatives
        }

    elif level == 5:
        exact_matches = sum(1 for e in evaluations if e.get('exact_match'))
        correct_no_tools = status_counts.get('correct_no_tools', 0)
        incorrect_tools = status_counts.get('incorrect_tools_predicted', 0)

        f1_metrics = [e['f1_metrics'] for e in evaluations if isinstance(e.get('f1_metrics'), dict)]
        averages = _avg_metrics(f1_metrics)
        performance = {
            'metric_label': 'no_tool_f1',
            'metrics': averages['averages'] if averages else {},
            'support': averages['support'] if averages else {},
            'sample_size': averages['count'] if averages else 0
        }
        summary['exact_match_rate'] = exact_matches / total

        level_specific = {
            'prediction_breakdown': {
                'correct_no_tools_rate': correct_no_tools / total,
                'incorrect_tools_rate': incorrect_tools / total
            }
        }

    else:
        summary['exact_match_rate'] = sum(1 for e in evaluations if e.get('exact_match')) / total

    latency: Dict[str, Any] = {}
    if query_latencies:
        try:
            import numpy as np
            latency = {
                'count': len(query_latencies),
                'mean_seconds': float(np.mean(query_latencies)),
                'median_seconds': float(np.median(query_latencies)),
                'p95_seconds': float(np.percentile(query_latencies, 95)),
                'max_seconds': float(np.max(query_latencies))
            }
        except ImportError:
            sorted_latencies = sorted(query_latencies)
            count = len(sorted_latencies)
            latency = {
                'count': count,
                'mean_seconds': sum(sorted_latencies) / count if count else 0.0,
                'median_seconds': sorted_latencies[count // 2] if count else 0.0,
                'p95_seconds': sorted_latencies[int(0.95 * (count - 1))] if count else 0.0,
                'max_seconds': sorted_latencies[-1] if count else 0.0
            }

    result: Dict[str, Any] = {
        'summary': summary,
        'performance': performance,
        'level_specific': level_specific
    }

    if latency:
        result['latency'] = latency

    return result

# --- Main Evaluation Runner ---
def run_evaluation(config: EvaluationConfig):
    """Main evaluation runner with checkpoint support and progress tracking."""
    print(f"\n Testing Level {config.level} Query Evaluation with {config.orchestrator} Orchestrator")
    print(f" Batch size: {config.batch_size}, Workers: {config.max_workers}")
    
    # Print model and port information from .env
    local_model = os.getenv('LOCAL_MODEL', 'Not set')
    local_url = os.getenv('LOCAL_URL', 'Not set')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', 'Not set')
    embedding_model = os.getenv('EMBEDDING_MODEL_NAME', 'Qwen/Qwen3-Embedding-0.6B')
    
    print(f" LLM Provider: {config.llm_provider}")
    if config.llm_provider == 'local_openai':
        print(f" Model Config - LOCAL_MODEL: {local_model}")
        print(f" Server Config - LOCAL_URL: {local_url}")
    else:
        print(f" Azure Endpoint: {azure_endpoint}")
        print(f" Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'Not set')}")
    print(f" Embedding Model: {embedding_model}")
    
    # Load queries
    try:
        queries_data = load_queries(config.level)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return
    
    if not queries_data:
        logger.error("❌ No queries loaded. Exiting.")
        return
    
    # Setup file paths (include orchestrator params and model name when available)
    # Determine model name based on LLM provider
    if config.llm_provider == 'azure_openai':
        # For Azure, use AZURE_OPENAI_DEPLOYMENT
        model_name = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    else:
        # For local, use LOCAL_MODEL
        model_name = os.getenv('LOCAL_MODEL')
    
    if config.orchestrator_params:
        checkpoint_file = FilePathManager.get_checkpoint_file_with_params(
            config.level, config.orchestrator, config.orchestrator_params, model_name
        )
        results_file = FilePathManager.get_results_file_with_params(
            config.level, config.orchestrator, config.orchestrator_params, model_name
        )
        latest_file = FilePathManager.get_results_file_with_params(
            config.level, config.orchestrator, config.orchestrator_params, model_name
        ).replace('.json', '_latest.json')
    else:
        checkpoint_file = FilePathManager.get_checkpoint_file(config.level, config.orchestrator)
        results_file = FilePathManager.get_results_file(config.level, config.orchestrator)
        latest_file = FilePathManager.get_latest_file(config.level, config.orchestrator)
    
    # Handle checkpoint restoration
    start_time = time.time()
    checkpoint = load_checkpoint(checkpoint_file)
    
    if checkpoint:
        resume_choice = input("📋 Found existing checkpoint. Resume? (y/n): ").lower().strip()
        if resume_choice == 'y':
            evaluations = checkpoint['evaluations']
            start_index = checkpoint['last_query_index'] + 1
            original_start_time = checkpoint.get('start_time', start_time)
            print(f"▶️  Resuming from query {start_index + 1}")
        else:
            clear_checkpoint(checkpoint_file)
            evaluations, start_index, original_start_time = [], 0, start_time
            print("🔄 Starting fresh evaluation")
    else:
        evaluations, start_index, original_start_time = [], 0, start_time
        print("🆕 Starting new evaluation")
    
    # Initialize latency collection for precise timing
    all_query_latencies = []
    
    # Determine test subset
    if config.mode == 'full':
        test_subset = queries_data[start_index:]
        total_queries = len(queries_data)
        print(f"📊 Full evaluation: {len(test_subset)} remaining queries")
    else:
        end_index = min(10, len(queries_data))
        test_subset = queries_data[start_index:end_index]
        total_queries = end_index if start_index == 0 else len(evaluations) + len(test_subset)
        print(f"📊 Subset evaluation: {len(test_subset)} remaining queries")
    
    # Initialize orchestrator and strategy
    try:
        print(f"🔧 Initializing {config.orchestrator} orchestrator...")
        orchestrator = OrchestratorFactory.create(config.orchestrator, config)
        strategy = get_strategy(config.level)
        print(f"✅ Orchestrator and evaluation strategy initialized!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        return
    
    # Run evaluation with error handling
    current_index = start_index
    
    try:
        for batch_start in range(0, len(test_subset), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(test_subset))
            batch_queries = test_subset[batch_start:batch_end]

            print(f"\n🔄 Processing batch {batch_start//config.batch_size + 1} ({len(batch_queries)} queries)...")

            batch_evaluations, batch_latencies = process_batch_parallel(
                orchestrator, batch_queries, current_index, strategy, config.max_workers, config
            )

            evaluations.extend(batch_evaluations)
            all_query_latencies.extend(batch_latencies)
            current_index += len(batch_queries)

            # Save checkpoint periodically
            if len(evaluations) % (config.batch_size) == 0 or batch_end == len(test_subset):
                save_checkpoint(evaluations, current_index - 1, total_queries, original_start_time, checkpoint_file, config.__dict__)

            print_progress(len(evaluations), total_queries, original_start_time, evaluations, config.level, config.batch_size, all_query_latencies)

    except KeyboardInterrupt:
        print(f"\n⏸️  Evaluation interrupted. Saving checkpoint...")
        save_checkpoint(evaluations, current_index - 1, total_queries, original_start_time, checkpoint_file, config.__dict__)
        print(f"💾 Checkpoint saved. Resume later with the same parameters.")
        return
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        print(f'exception: {e} ')
        save_checkpoint(evaluations, current_index - 1, total_queries, original_start_time, checkpoint_file, config.__dict__)
        print(f"💾 Emergency checkpoint saved.")
        return
    
    # Generate final report
    stats = calculate_statistics(evaluations, config.level, all_query_latencies)
    total_time = time.time() - original_start_time
    
    # Calculate comprehensive timing metrics
    total_processing_time = sum(all_query_latencies) if all_query_latencies else 0
    parallelization_benefit = total_processing_time - total_time  # Positive = time saved
    
    timing_metrics = {
        'evaluation_runtime': {
            'total_seconds': total_time,
            'average_per_query_seconds': total_time / len(evaluations) if evaluations else 0,
            'description': 'Wall-clock time including overhead and parallelization benefits'
        },
        'query_processing': {
            'total_latency_seconds': total_processing_time,
            'average_latency_seconds': total_processing_time / len(all_query_latencies) if all_query_latencies else 0,
            'description': 'Pure orchestrator processing time (sequential equivalent)'
        },
        'parallelization_analysis': {
            'time_saved_seconds': parallelization_benefit,
            'average_time_saved_per_query_seconds': parallelization_benefit / len(evaluations) if evaluations else 0,
            'efficiency_gain_percentage': (parallelization_benefit / total_processing_time * 100) if total_processing_time > 0 else 0,
            'description': 'Benefits gained from parallel processing and system efficiency'
        }
    }
    
    print("\n" + "="*80)
    print(f"📊 FINAL LEVEL {config.level} {config.orchestrator.upper()} EVALUATION STATISTICS")
    print("="*80)

    summary = stats.get('summary', {})
    performance_summary = stats.get('performance', {})
    level_specific = stats.get('level_specific', {})

    print(f"Total Queries Evaluated: {summary.get('total_queries', 0)}")
    if 'status_counts' in summary:
        print(f"Status Breakdown: {summary['status_counts']}")
    if 'exact_match_rate' in summary:
        print(f"Exact Match Rate: {summary['exact_match_rate']*100:.1f}%")

    metrics = performance_summary.get('metrics', {})
    if metrics:
        print("-" * 40)
        print(f"🎯 {performance_summary.get('metric_label', 'Performance').replace('_', ' ').title()}")
        print(f"F1: {metrics.get('f1', 0):.3f}")
        print(f"Precision: {metrics.get('precision', 0):.3f}")
        print(f"Recall: {metrics.get('recall', 0):.3f}")
        support = performance_summary.get('support', {})
        if support:
            print(f"Support (TP/FP/FN): {support.get('tp', 0)}/{support.get('fp', 0)}/{support.get('fn', 0)}")

    if config.level == 1:
        match_rates = level_specific.get('match_rates', {})
        if match_rates:
            print("-" * 40)
            print("🔁 Match Rates")
            print(f"Tool Name: {match_rates.get('tool_name', 0)*100:.1f}%")
            print(f"Server: {match_rates.get('server', 0)*100:.1f}%")
        error_counts = level_specific.get('error_counts', {})
        if error_counts:
            print(f"Error Counts: {error_counts}")
        if 'multi_tool_summary' in level_specific:
            multi = level_specific['multi_tool_summary']
            metrics = multi.get('averages', {})
            if metrics:
                print("-" * 40)
                print("🧩 Multi-Tool Analysis")
                print(f"F1: {metrics.get('f1', 0):.3f}")
                print(f"Precision: {metrics.get('precision', 0):.3f}")
                print(f"Recall: {metrics.get('recall', 0):.3f}")

    elif config.level == 2:
        match_rates = level_specific.get('match_rates', {})
        if match_rates:
            print("-" * 40)
            print("� Match Rates")
            print(f"Tool Name: {match_rates.get('tool_name', 0)*100:.1f}%")
            print(f"Server: {match_rates.get('server', 0)*100:.1f}%")
        error_counts = level_specific.get('error_counts', {})
        if error_counts:
            print(f"Error Counts: {error_counts}")
        if 'multi_tool_summary' in level_specific:
            multi = level_specific['multi_tool_summary']
            metrics = multi.get('averages', {})
            if metrics:
                print("-" * 40)
                print("🧩 Multi-Tool Analysis")
                print(f"F1: {metrics.get('f1', 0):.3f}")
                print(f"Precision: {metrics.get('precision', 0):.3f}")
                print(f"Recall: {metrics.get('recall', 0):.3f}")

    elif config.level == 3:
        response_rates = level_specific.get('response_rates', {})
        if response_rates:
            print("-" * 40)
            print("📊 Response Rates")
            for name, value in response_rates.items():
                print(f"{name.replace('_', ' ').title()}: {value*100:.1f}%")
        print("-" * 40)
        print("🎯 Structural Accuracy")
        print(f"Exact Structure Match Rate: {level_specific.get('structure_match_rate', 0)*100:.1f}%")
        print(f"Node Set Match Rate: {level_specific.get('node_set_match_rate', 0)*100:.1f}%")

    elif config.level == 4:
        response_rates = level_specific.get('response_rates', {})
        if response_rates:
            print("-" * 40)
            print("📊 Response Rates")
            for name, value in response_rates.items():
                print(f"{name.replace('_', ' ').title()}: {value*100:.1f}%")
        print("-" * 40)
        print("🎯 Structural Accuracy")
        print(f"EFS Structure Match Rate: {level_specific.get('structure_match_rate', 0)*100:.1f}%")
        print(f"EFS Node Set Match Rate: {level_specific.get('node_set_match_rate', 0)*100:.1f}%")
        print(f"Parallel Aware Match Rate: {level_specific.get('parallel_aware_match_rate', 0)*100:.1f}%")
        if 'efs_matches' in level_specific:
            print("-" * 40)
            print("🔗 EFS Utilization")
            print(f"EFS Matches: {level_specific.get('efs_matches', 0)}")
            print(f"EFS Alternatives: {level_specific.get('efs_alternatives', 0)}")

    elif config.level == 5:
        breakdown = level_specific.get('prediction_breakdown', {})
        if breakdown:
            print("-" * 40)
            print("📊 Prediction Breakdown")
            print(f"Correct No-Tool Rate: {breakdown.get('correct_no_tools_rate', 0)*100:.1f}%")
            print(f"Incorrect Tools Rate: {breakdown.get('incorrect_tools_rate', 0)*100:.1f}%")

    print("-" * 40)
    print("⏱️ TIMING:")
    print(f"Total Runtime: {timing_metrics['evaluation_runtime']['total_seconds']/60:.1f} minutes")
    print(f"Average Runtime per Query: {timing_metrics['evaluation_runtime']['average_per_query_seconds']:.2f} seconds")
    if all_query_latencies:
        print(f"Average Processing per Query: {timing_metrics['query_processing']['average_latency_seconds']:.2f} seconds")
        print(f"Parallelization Efficiency: {timing_metrics['parallelization_analysis']['efficiency_gain_percentage']:.1f}% time saved")
    
    # Save detailed results
    detailed_results = {
        'statistics': stats,
        'config': config.__dict__,
        'evaluations': evaluations
    }
    
    # with open(results_file, 'w') as f: 
    #     json.dump(detailed_results, f, indent=2)
    with open(latest_file, 'w') as f: 
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to:")
    print(f"   📄 {latest_file}")
    
    # Clear checkpoint on successful completion
    clear_checkpoint(checkpoint_file)
    print(f"\n✅ Level {config.level} evaluation completed successfully!")# --- Main Command-Line Interface ---
def main():
    """Main entry point with argument parsing and interactive mode."""
    parser = argparse.ArgumentParser(
        description="Generalized Query Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generalized_query_eval.py --level 1 --orchestrator TS --mode subset
  python generalized_query_eval.py --level 3 --orchestrator MCP0 --mode full --batch_size 20
  python generalized_query_eval.py --level 4 --orchestrator TS --mode full --batch_size 10
  python generalized_query_eval.py --level 5 --orchestrator REACT --mode full
  python generalized_query_eval.py --level 2 --orchestrator HB --mode full
  python generalized_query_eval.py  # Interactive mode
        """
    )
    parser.add_argument('--level', type=int, help='Test level (1, 2, 3, 4, 5)')
    parser.add_argument('--orchestrator', type=str, help='Orchestrator type (TS, HB, MCP0, REACT)')
    parser.add_argument('--mode', type=str, choices=['full', 'subset'], default='subset', 
                       help='Evaluation mode (default: subset)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size (default: 10)')
    parser.add_argument('--max_workers', type=int, default=5, help='Max workers (default: 5)')
    parser.add_argument('--llm_provider', type=str, choices=['azure_openai', 'local_openai'], 
                       default=None, help='LLM provider (azure_openai or local_openai). If not specified, will determine from env vars.')
    
    # Orchestrator parameters
    parser.add_argument('--tool_top_k', type=int, help='Tool top k for TS/HB/MCP0 orchestrator')
    parser.add_argument('--server_top_k', type=int, help='Server top k for HB/MCP0 orchestrator')
    parser.add_argument('--similarity_threshold', type=float, help='Similarity threshold for TS/HB orchestrator')
    parser.add_argument('--query_expansion_count', type=int, help='Query expansion count for TS/HB orchestrator')
    parser.add_argument('--rerank_top_k', type=int, help='Rerank top k for TS/HB orchestrator')
    parser.add_argument('--react_attempts', type=int, help='Number of attempts for REACT orchestrator (default 3)')   
    
    args = parser.parse_args()
    
    # Handle orchestrator parameters from command line
    orchestrator_params = None
    if args.level and args.orchestrator:
        orchestrator = args.orchestrator.upper()
        if orchestrator == 'TS':
            orchestrator_params = {}
            if args.tool_top_k is not None:
                orchestrator_params['tool_top_k'] = args.tool_top_k
            if args.similarity_threshold is not None:
                orchestrator_params['similarity_threshold'] = args.similarity_threshold
            if args.query_expansion_count is not None:
                orchestrator_params['query_expansion_count'] = args.query_expansion_count
            if args.rerank_top_k is not None:
                orchestrator_params['rerank_top_k'] = args.rerank_top_k
            # If no params provided via command line, keep None to use defaults
            if not orchestrator_params:
                orchestrator_params = None
                
        elif orchestrator == 'HB':
            orchestrator_params = {}
            if args.server_top_k is not None:
                orchestrator_params['server_top_k'] = args.server_top_k
            if args.tool_top_k is not None:
                orchestrator_params['tool_top_k'] = args.tool_top_k
            if args.similarity_threshold is not None:
                orchestrator_params['similarity_threshold'] = args.similarity_threshold
            if args.query_expansion_count is not None:
                orchestrator_params['query_expansion_count'] = args.query_expansion_count
            if args.rerank_top_k is not None:
                orchestrator_params['rerank_top_k'] = args.rerank_top_k
            # If no params provided via command line, keep None to use defaults
            if not orchestrator_params:
                orchestrator_params = None
                
        elif orchestrator == 'REACT':
            orchestrator_params = {}
            if args.react_attempts is not None:
                orchestrator_params['num_attempts'] = args.react_attempts
            # If no params provided via command line, keep None to use defaults  
            if not orchestrator_params:
                orchestrator_params = None
                
        elif orchestrator == 'MCP0':
            orchestrator_params = {}
            if args.server_top_k is not None:
                orchestrator_params['server_top_k'] = args.server_top_k
            if args.tool_top_k is not None:
                orchestrator_params['tool_top_k'] = args.tool_top_k
            # If no params provided via command line, keep None to use defaults
            if not orchestrator_params:
                orchestrator_params = None
    
    # Interactive mode if essential args are missing
    if not args.level or not args.orchestrator:
        print("🤖 Generalized Query Evaluation Tool")
        print("="*50)
        print("Available Levels: 1 (single tool), 2 (tool sets), 3 (tool graphs), 4 (EFS tool chains), 5 (beyond toolset)")
        print("Available Orchestrators: TS (ToolShed), HB (Hybrid), MCP0 (MCP-Zero), REACT")
        print()
        
        # Get level
        while True:
            try:
                level_input = input("Select Level (1, 2, 3, 4, or 5): ").strip()
                args.level = int(level_input)
                if args.level not in [1, 2, 3, 4, 5]:
                    print("❌ Level must be 1, 2, 3, 4, or 5")
                    continue
                break
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get orchestrator
        while True:
            orch_input = input("Select Orchestrator (TS or HB or MCP0 or REACT): ").strip().upper()
            if orch_input in ['TS', 'HB', 'MCP0', 'REACT']:
                args.orchestrator = orch_input
                break
            else:
                print("❌ Orchestrator must be TS or HB or MCP0 or REACT")

        # Get mode
        mode_input = input("Select Mode (full/subset, default subset): ").strip().lower()
        args.mode = mode_input if mode_input in ['full', 'subset'] else 'subset'
        
        # Get batch size
        batch_input = input("Batch size (default 10): ").strip()
        args.batch_size = int(batch_input) if batch_input.isdigit() else 10
        
        # Get max workers
        workers_input = input("Max workers (default 5): ").strip()
        args.max_workers = int(workers_input) if workers_input.isdigit() else 5
        
        # Get custom orchestrator parameters for TS, HB, MCP0, and REACT
        orchestrator_params = None
        if args.orchestrator in ['TS', 'HB', 'MCP0', 'REACT']:
            print(f"\n🔧 Custom {args.orchestrator} Orchestrator Parameters:")
            print("Press Enter to use defaults or modify parameters:")
            
            if args.orchestrator == 'TS':
                print("Default TS params: tool_top_k=5, similarity_threshold=0, query_expansion_count=3, rerank_top_k=10")
                
                tool_top_k = input("  tool_top_k (default 5): ").strip()
                tool_top_k = int(tool_top_k) if tool_top_k.isdigit() else 5
                
                sim_threshold = input("  similarity_threshold (default 0): ").strip()
                sim_threshold = float(sim_threshold) if sim_threshold else 0
                
                query_exp_count = input("  query_expansion_count (default 3): ").strip()
                query_exp_count = int(query_exp_count) if query_exp_count.isdigit() else 3
                
                rerank_top_k = input("  rerank_top_k (default 10): ").strip()
                rerank_top_k = int(rerank_top_k) if rerank_top_k.isdigit() else 10
                
                orchestrator_params = {
                    "tool_top_k": tool_top_k,
                    "similarity_threshold": sim_threshold,
                    "query_expansion_count": query_exp_count,
                    "rerank_top_k": rerank_top_k
                }
                
            elif args.orchestrator == 'HB':
                print("Default HB params: server_top_k=3, tool_top_k=5, similarity_threshold=0, query_expansion_count=3, rerank_top_k=10")
                
                server_top_k = input("  server_top_k (default 3): ").strip()
                server_top_k = int(server_top_k) if server_top_k.isdigit() else 3
                
                tool_top_k = input("  tool_top_k (default 5): ").strip()
                tool_top_k = int(tool_top_k) if tool_top_k.isdigit() else 5
                
                sim_threshold = input("  similarity_threshold (default 0): ").strip()
                sim_threshold = float(sim_threshold) if sim_threshold else 0
                
                query_exp_count = input("  query_expansion_count (default 3): ").strip()
                query_exp_count = int(query_exp_count) if query_exp_count.isdigit() else 3
                
                rerank_top_k = input("  rerank_top_k (default 10): ").strip()
                rerank_top_k = int(rerank_top_k) if rerank_top_k.isdigit() else 10
                
                orchestrator_params = {
                    "server_top_k": server_top_k,
                    "tool_top_k": tool_top_k,
                    "similarity_threshold": sim_threshold,
                    "query_expansion_count": query_exp_count,
                    "rerank_top_k": rerank_top_k
                }
                
            elif args.orchestrator == 'REACT':
                print("Default REACT params: num_attempts=3")
                
                num_attempts = input("  num_attempts (default 3): ").strip()
                num_attempts = int(num_attempts) if num_attempts.isdigit() else 3
                
                orchestrator_params = {
                    "num_attempts": num_attempts
                }
                
            elif args.orchestrator == 'MCP0':
                print("Default MCP0 params: server_top_k=3, tool_top_k=5")
                
                server_top_k = input("  server_top_k (default 3): ").strip()
                server_top_k = int(server_top_k) if server_top_k.isdigit() else 3
                
                tool_top_k = input("  tool_top_k (default 5): ").strip()
                tool_top_k = int(tool_top_k) if tool_top_k.isdigit() else 5
                
                orchestrator_params = {
                    "server_top_k": server_top_k,
                    "tool_top_k": tool_top_k
                }
        
        print()
    
    # Validate arguments
    if args.level not in [1, 2, 3, 4, 5]:
        print(f"❌ Invalid level: {args.level}. Must be 1, 2, 3, 4, or 5.")
        return
    
    if args.orchestrator.upper() not in ['TS', 'HB', 'MCP0', 'REACT']:
        print(f"❌ Invalid orchestrator: {args.orchestrator}. Must be TS or HB or MCP0 or REACT.")
        return
    
    # Determine LLM provider from args or environment
    llm_provider = args.llm_provider
    if llm_provider is None:
        # Auto-detect from environment variables
        if os.getenv('AZURE_OPENAI_ENDPOINT') and os.getenv('AZURE_OPENAI_API_KEY'):
            # If Azure credentials are set and LOCAL_URL is not set, use Azure
            if not os.getenv('LOCAL_URL'):
                llm_provider = 'azure_openai'
            else:
                # Both are set, default to local
                llm_provider = 'local_openai'
        else:
            # Default to local if no Azure credentials
            llm_provider = 'local_openai'
    
    # Create configuration
    config = EvaluationConfig(
        level=args.level,
        orchestrator=args.orchestrator.upper(),
        mode=args.mode,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        llm_provider=llm_provider,
        orchestrator_params=orchestrator_params
    )
    
    # Confirm configuration
    print(f"📋 Configuration:")
    print(f"   Level: {config.level}")
    print(f"   Orchestrator: {config.orchestrator}")
    print(f"   LLM Provider: {config.llm_provider}")
    print(f"   Mode: {config.mode}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Max workers: {config.max_workers}")
    if config.orchestrator_params:
        print(f"   Orchestrator params: {config.orchestrator_params}")
    
    confirm = input("\nProceed with evaluation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Evaluation cancelled.")
        return
    
    # Run evaluation
    run_evaluation(config)

if __name__ == "__main__":
    main()
