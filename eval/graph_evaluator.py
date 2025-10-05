"""
Graph Evaluator for Multi-Level Tool Sequence Evaluation

This evaluator implements a global relative order validation approach that:
1. Treats the entire predicted sequence as a final plan
2. Independently checks each tool's relative position against GT dependencies
3. Avoids cascading failures from single tool ordering errors

Supports Level 1-4 tool sequence evaluation based on global relative order validation logic.
"""

import json
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Evaluation result data structure"""
    exact_match: bool
    f1_score: float
    precision: float
    recall: float
    metrics: Dict[str, Any]
    details: Dict[str, Any]


class GraphEvaluator:
    """
    Graph Evaluator - Implements global relative order validation logic
    
    Core concept:
    - Treats the predicted sequence as a complete plan
    - Independently checks if each tool's relative position satisfies GT dependency conditions
    - Avoids cascading failures from single tool ordering errors
    """
    
    def __init__(self, level: int):
        """
        Initialize evaluator
        
        Args:
            level: Evaluation level (1-4)
        """
        self.level = level
        
    def parse_ground_truth(self, ground_truth_tools: List[List[Dict]]) -> Dict[int, Dict]:
        """
        Parse Ground Truth data structure
        
        Args:
            ground_truth_tools: GT tool list, format [step][equivalent_tool_set][tool_properties]
            
        Returns:
            Dict[step_id, {
                'efs_tools': Set[Tuple[server, tool_name]],  # Equivalent Function Set
                'dependencies': Set[int]  # Dependent step IDs
            }]
        """
        gt_graph = {}
        
        for step_idx, tool_step in enumerate(ground_truth_tools):
            if not tool_step:
                continue
                
            # Collect Equivalent Function Set (EFS)
            efs_tools = set()
            dependencies = set()
            
            for efs_tool in tool_step:
                server = efs_tool.get('server_name', efs_tool.get('server', ''))
                tool_name = efs_tool.get('tool_name', '')
                efs_tools.add((server, tool_name))
                
                # Collect dependencies (all equivalent tools should have the same dependencies)
                for dep_idx in efs_tool.get('dependencies', []):
                    dependencies.add(dep_idx)
            
            gt_graph[step_idx] = {
                'efs_tools': efs_tools,
                'dependencies': dependencies
            }
        
        return gt_graph
    
    def parse_prediction(self, raw_prediction_list: List[str]) -> List[Tuple[str, str]]:
        """
        Parse prediction results
        
        Args:
            raw_prediction_list: JSON string list
            
        Returns:
            List[Tuple[server, tool_name]]: Predicted tool sequence
        """
        predicted_sequence = []
        
        for step_str in raw_prediction_list:
            try:
                step_dict = json.loads(step_str)
                server = step_dict.get('server_name', step_dict.get('server', ''))
                tool_name = step_dict.get('tool_name', '')
                predicted_sequence.append((server, tool_name))
            except json.JSONDecodeError:
                # Skip this tool if parsing fails
                continue
                
        return predicted_sequence
    
    def find_tool_matches(self, predicted_sequence: List[Tuple[str, str]], 
                         gt_graph: Dict[int, Dict]) -> Dict[int, int]:
        """
        Find matching relationships between predicted tools and GT nodes (one-time matching strategy)
        
        Important modification: Each GT node can only be matched once to avoid duplicate scoring of repeated tools
        
        Args:
            predicted_sequence: Predicted tool sequence
            gt_graph: GT graph structure
            
        Returns:
            Dict[pred_pos, gt_node_id]: Mapping from prediction position to GT node
        """
        pred_to_gt_mapping = {}
        used_gt_nodes = set()  # GT nodes that have already been matched
        
        for pred_pos, pred_tool in enumerate(predicted_sequence):
            # Find which unused GT node this predicted tool belongs to
            for gt_node_id, node_info in gt_graph.items():
                if gt_node_id not in used_gt_nodes and pred_tool in node_info['efs_tools']:
                    pred_to_gt_mapping[pred_pos] = gt_node_id
                    used_gt_nodes.add(gt_node_id)  # Mark as used
                    break  # Stop searching after finding a match
                    
        return pred_to_gt_mapping
    
    def check_dependency_satisfaction(self, pred_pos: int, gt_node_id: int,
                                      gt_to_pred_pos_map: Dict[int, int], # <--- Use new mapping
                                      gt_graph: Dict[int, Dict]) -> bool:
        """
        Revised logic: Check if specific predicted tool's dependencies are [completely satisfied]
        
        Args:
            pred_pos: Predicted tool's position in sequence
            gt_node_id: Corresponding GT node ID
            gt_to_pred_pos_map: Mapping from GT node ID to its first appearance position
            gt_graph: GT graph structure
            
        Returns:
            bool: Whether dependencies are completely satisfied
        """
        required_deps = gt_graph[gt_node_id]['dependencies']
        
        if not required_deps:
            return True  # No dependencies, directly satisfied

        for dep_node_id in required_deps:
            # Condition 1: Dependency must appear in prediction
            if dep_node_id not in gt_to_pred_pos_map:
                return False # Dependency missing, not satisfied

            # Condition 2: Dependency position must be before current tool
            dep_pos = gt_to_pred_pos_map[dep_node_id]
            if dep_pos >= pred_pos:
                return False # Dependency position incorrect, not satisfied
        
        # All dependencies satisfied
        return True

    def evaluate_sequence(self, predicted_sequence: List[Tuple[str, str]], 
                         gt_graph: Dict[int, Dict]) -> EvaluationResult:
        """
        Revised evaluation sequence method
        """
        total_predictions = len(predicted_sequence)
        total_gt_nodes = len(gt_graph)

        # Step 1: Establish mapping relationships
        # pred_to_gt_mapping: {pred_pos: gt_node_id} - which GT node each predicted tool matches
        # gt_to_pred_pos_map: {gt_node_id: pred_pos} - position where each GT node first appears
        pred_to_gt_mapping = self.find_tool_matches(predicted_sequence, gt_graph)
        gt_to_pred_pos_map = {}
        for pred_pos, gt_node_id in pred_to_gt_mapping.items():
            if gt_node_id not in gt_to_pred_pos_map:
                gt_to_pred_pos_map[gt_node_id] = pred_pos

        # Step 2: Calculate Recall (coverage)
        # Recall only cares about how many necessary GT tools were found
        satisfied_gt_for_recall = set(pred_to_gt_mapping.values())
        recall = len(satisfied_gt_for_recall) / total_gt_nodes if total_gt_nodes > 0 else 0.0

        # Step 3: Calculate Precision (placement accuracy)
        correctly_placed_predictions = 0
        prediction_details = []
        
        for pred_pos, pred_tool in enumerate(predicted_sequence):
            # Only evaluate predictions that successfully match GT tools
            if pred_pos in pred_to_gt_mapping:
                gt_node_id = pred_to_gt_mapping[pred_pos]
                
                # Check if dependencies are satisfied
                is_correctly_placed = self.check_dependency_satisfaction(
                    pred_pos, gt_node_id, gt_to_pred_pos_map, gt_graph
                )
                
                if is_correctly_placed:
                    correctly_placed_predictions += 1
                
                prediction_details.append({
                    'position': pred_pos, 'tool': pred_tool, 'matched_gt_node': gt_node_id,
                    'is_correctly_placed': is_correctly_placed
                })
            else:
                # This predicted tool does not exist in GT, or corresponding GT node has been matched
                prediction_details.append({
                    'position': pred_pos, 'tool': pred_tool, 'matched_gt_node': None,
                    'is_correctly_placed': False
                })

        precision = correctly_placed_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Step 4: Calculate F1 and Exact Match
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact Match: Only when F1 score = 1.0
        # This is equivalent to precision = 1.0 and recall = 1.0
        exact_match = (f1_score == 1.0)
        
        # 4. Assemble results
        tp = correctly_placed_predictions

        # FP (False Positives): Parts of all predictions that do not belong to TP
        fp = total_predictions - tp

        # FN (False Negatives): Number of nodes required in GT but never matched
        fn = total_gt_nodes - len(satisfied_gt_for_recall)

        # 4.1 Assemble metrics dict
        metrics = {
            # Basic counts
            'total_predictions': total_predictions,
            'total_gt_nodes': total_gt_nodes,
            
            # Core evaluation results
            'correctly_placed_predictions': correctly_placed_predictions, # Same as TP, but semantically clearer
            'matched_gt_nodes_for_recall': len(satisfied_gt_for_recall), # Numerator for calculating Recall
            
            # Standard metrics for machine learning
            'tp': tp,
            'fp': fp,
            'fn': fn,
            
            # Perfect match indicator
            'exact_match': exact_match
        }

        # 4.2 Assemble details dict, providing detailed diagnostic information
        details = {
            # GT node IDs that were successfully covered (for Recall)
            'matched_gt_node_ids': sorted(list(satisfied_gt_for_recall)),
            
            # Detailed analysis of each predicted tool
            'prediction_analysis': prediction_details, # prediction_details should contain matching and placement status for each tool
            
            # Mapping relationships for debugging
            'pred_pos_to_gt_id_mapping': pred_to_gt_mapping
        }

        # Return standardized EvaluationResult object
        return EvaluationResult(
            exact_match=exact_match,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            metrics=metrics,
            details=details)
        

    def evaluate(self, ground_truth_tools: List[List[Dict]], 
                raw_prediction_list: List[str]) -> EvaluationResult:
        """
        Main evaluation method
        
        Args:
            ground_truth_tools: GT tool structure
            raw_prediction_list: Prediction result JSON string list
            
        Returns:
            EvaluationResult: Evaluation result
        """
        # Handle empty input
        if not raw_prediction_list and not ground_truth_tools:
            # Both empty, perfect match
            return EvaluationResult(
                exact_match=True,
                f1_score=1.0,
                precision=1.0,
                recall=1.0,
                metrics={'tp': 0, 'fp': 0, 'fn': 0, 'total_predictions': 0, 'total_gt_nodes': 0, 'exact_match': True},
                details={'matched_gt_node_ids': [], 'prediction_analysis': []}
            )
        
        if not raw_prediction_list:
            # Prediction empty but GT not empty
            total_gt_nodes = len(ground_truth_tools)
            return EvaluationResult(
                exact_match=False,
                f1_score=0.0,
                precision=0.0,
                recall=0.0,
                metrics={'tp': 0, 'fp': 0, 'fn': total_gt_nodes, 'total_predictions': 0, 'total_gt_nodes': total_gt_nodes, 'exact_match': False},
                details={'matched_gt_node_ids': [], 'prediction_analysis': []}
            )
        
        if not ground_truth_tools:
            # GT empty but prediction not empty
            total_predictions = len(raw_prediction_list)
            return EvaluationResult(
                exact_match=False,
                f1_score=0.0,
                precision=0.0,
                recall=1.0,  # Technically recall=1.0 because no GT to recall
                metrics={'tp': 0, 'fp': total_predictions, 'fn': 0, 'total_predictions': total_predictions, 'total_gt_nodes': 0, 'exact_match': False},
                details={'matched_gt_node_ids': [], 'prediction_analysis': []}
            )
        
        # Normal evaluation process
        gt_graph = self.parse_ground_truth(ground_truth_tools)
        predicted_sequence = self.parse_prediction(raw_prediction_list)
        
        return self.evaluate_sequence(predicted_sequence, gt_graph)


def create_evaluator(level: int) -> GraphEvaluator:
    """
    Create evaluator for specified level
    
    Args:
        level: Evaluation level (1-4)
        
    Returns:
        GraphEvaluator: Evaluator instance
    """
    if level not in [1, 2, 3, 4]:
        raise ValueError(f"Unsupported level: {level}. Supported levels: 1, 2, 3, 4")
    
    return GraphEvaluator(level)


# Example usage and testing
if __name__ == "__main__":
    # Example data
    ground_truth_example = [
        [{'server': 's1', 'tool_name': 'tool_A', 'dependencies': []}],      # Step 0
        [{'server': 's1', 'tool_name': 'tool_B', 'dependencies': [0]}],      # Step 1
        [                                                                 # Step 2
            {'server': 's2', 'tool_name': 'tool_C', 'dependencies': [1]},
            {'server': 's2', 'tool_name': 'tool_D', 'dependencies': [1]}   # C and D are equivalent
        ]
    ]
    
    prediction_example = [
        json.dumps({'server': 's1', 'tool_name': 'tool_A'}),  # Position 0
        json.dumps({'server': 's2', 'tool_name': 'tool_C'}),  # Position 1
        json.dumps({'server': 's1', 'tool_name': 'tool_B'}),  # Position 2
    ]
    
    # Create evaluator and test
    evaluator = create_evaluator(level=3)
    result = evaluator.evaluate(ground_truth_example, prediction_example)
    