"""
Grid PG (Grid Pattern Generation) Processor Class

This module contains the GridPGProcessor class that handles all Grid PG-related 
operations including grid cell predictions, confidence filtering, score calculations,
and visualization. This class decouples the Grid PG logic from the LightningModule
for better maintainability and separation of concerns.
"""

import torch
from typing import Dict, Optional, Tuple, Any
from src.explainability.grid_score.grid_pg import get_localization_score
from .config import GridProcessorConfig
from .grid_pg import strided_forward_general


def _extract_logits(preds: Any) -> torch.Tensor:
    """Normalize HF / Lightning model outputs to raw logits."""
    if hasattr(preds, "logits") and preds.logits is not None:
        return preds.logits
    if isinstance(preds, tuple):
        first = preds[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError(f"Expected tensor logits in tuple output, got {type(first)!r}")
        return first
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"Expected tensor-like model output, got {type(preds)!r}")
    return preds

class GridPGProcessor:
    """
    Handles all Grid PG (Grid Pointing Game) related operations.
    
    This class provides methods for:
    - Grid dimension calculations
    - Grid cell predictions
    - High confidence sample filtering
    - Grid PG score calculations
    - Grid visualization processing
    """
    
    def __init__(self, config: GridProcessorConfig):
        """
        Initialize the GridPGProcessor.
        
        Args:
            config (GridProcessorConfig): The GridProcessorConfig object
        """
        self.config = config
        self.model = config.model
        self.explain_fn = config.explain_fn
        self.grid_pg_scale = config.grid_pg_scale
    
    def get_grid_dimensions(self, inputs: torch.Tensor) -> Tuple[int, int, int, int]:
        """
        Calculates the dimensions and scale for the Grid PG dataset.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the width, height, scale, and sub-image size.
        """
        W = H = inputs.shape[-1]
        scale = self.grid_pg_scale
        sub_image_size = int(W/scale)
        return W, H, scale, sub_image_size

    def get_grid_cell_predictions(self, inputs: torch.Tensor, scale: int, sub_image_size: int) -> torch.Tensor:
        """
        Performs predictions on each grid cell of the input images.

        Args:
            inputs (torch.Tensor): The input tensor.
            scale (int): The grid scale.
            sub_image_size (int): The size of each sub-image (grid cell).

        Returns:
            torch.Tensor: A tensor containing predictions for each grid cell.
        """
        predictions = []
        

        with torch.no_grad():
            for s in range(scale**2):
                i = s // scale
                j = s % scale
                grid_cell = inputs[:, :, i*sub_image_size : (i+1)*sub_image_size, j*sub_image_size : (j+1)*sub_image_size]
                
                # Get predictions for the grid cell
                preds = self.model(grid_cell)
                logits = _extract_logits(preds)
                predictions.append(logits.clone())
        
        predictions = torch.stack(predictions, dim=1)

        # Apply softmax or sigmoid
        predictions = self.config.prediction_fn(predictions)
        return predictions

    def filter_high_confidence_samples(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                     scale: int) -> torch.Tensor:
        """
        Filters samples based on a confidence threshold and correct predictions.

        Args:
            predictions (torch.Tensor): The predictions tensor with shape [B, K, N]
            targets (torch.Tensor): The target tensor with shape [B, K]
            scale (int): The grid scale.
            confidence_threshold (float): The confidence threshold for filtering.

        Returns:
            torch.Tensor: A boolean mask indicating high-confidence and correctly predicted samples.
        """
        # Use the configured confidence threshold
        actual_threshold = self.config.confidence_threshold
        

        # Get predicted classes
        predicted_classes = torch.argmax(predictions, dim=-1)
        
        # Check if predictions match targets (correct predictions)
        correct_predictions = (predicted_classes == targets)
        
        # Use predictions as confidence scores (assuming they are now probabilities/confidences)
        filtered_confidence = torch.gather(predictions, dim=2, index=targets.unsqueeze(-1)).squeeze(-1)

        # Create mask for high confidence predictions
        high_confidence = torch.where(filtered_confidence > actual_threshold, 1, 0)
        
        # Sample is valid only if all grid cells have high confidence AND correct predictions
        mask = ((high_confidence == 1) & (correct_predictions == 1)).all(dim=1)
        return mask

    def calculate_grid_pg_score(self, inputs: torch.Tensor, targets: torch.Tensor, 
                               scale: int, mode: str = "batch", 
                               confidence_mask: Optional[torch.Tensor] = None) -> Optional[float]:
        """
        Calculates the Grid PG score for the input samples.

        Args:
            inputs (torch.Tensor): The input tensor (full batch, not filtered).
            targets (torch.Tensor): The target tensor (full batch, not filtered).
            scale (int): The grid scale.
            mode (str): The mode for the contribution map computation.
            confidence_mask (Optional[torch.Tensor]): Mask indicating which samples pass confidence threshold.

        Returns:
            Optional[float]: The total Grid PG score, or None if no confident samples.
        """
        
        # Process ALL samples to ensure synchronization across ranks
        # Filter results afterwards based on confidence mask
        contribution_maps = []
        
        # Process each grid cell explanation with memory cleanup
        for explanation in range(scale**2):
            self.model.zero_grad()

            explanation_target = targets[:, explanation]            
                
            batch_inputs = inputs.detach()
            
            # Wrapper for explain_fn to return only the contribution map tensor
            # This allows strided_forward_general to perform simple summation
            def tensor_explain_fn(inp):
                inp = inp.detach()
                out = self.explain_fn(
                    inp, 
                    explain_class=explanation_target,
                    mode=mode, 
                    map_size=1,
                    to_numpy=False
                )
                # Ensure we return a tensor
                if hasattr(out, "contribution_map"): # ExplanationOutput
                    return out.contribution_map
                elif isinstance(out, dict):
                    return out["contribution_map"]
                return out # Fallback if it is already a tensor

            # Use strided execution
            # map_size argument in strided_forward_general controls the stride overlap/window size logic
            # It seems to correspond to self.config.map_size based on user intent
            contribution_map, _ = strided_forward_general(
                batch_inputs, 
                tensor_explain_fn, 
                map_size=self.config.map_size
            )
            
            # contribution_map is already averaged/normalized by strided_forward_general
            
            # Legacy cleanup (if needed)
            del batch_inputs
            
            contribution_maps.append(contribution_map)
            
            self.model.zero_grad()

        # Stack contribution maps and calculate PG score
        contribution_maps = torch.stack(contribution_maps, dim=1)
        
        # img_dims should be the original grid cell dimensions, not the full contribution map size
        original_cell_size = inputs.shape[-1] // scale  # e.g., 224//2 = 112 for scale=2
        pg_score = get_localization_score(contribution_maps, img_dims=(original_cell_size, original_cell_size), scale=scale)
        
        # Apply confidence filtering if mask is provided
        assert confidence_mask is not None
        
        # Only include scores from confident samples
        expanded_mask = confidence_mask.repeat(scale**2)
        
        # Check if any samples passed the confidence filter
        if expanded_mask.any():
            pg_score = pg_score[expanded_mask]
            pg_score = torch.mean(pg_score).item()
        else:
            # No confident samples - return None to avoid artificial score deflation
            pg_score = None

        # Final cleanup: delete large tensors
        del contribution_maps, expanded_mask

        return pg_score

    def process_validation_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[Optional[float], int]:
        """
        Process a validation step for Grid PG evaluation.
        
        Args:
            inputs (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor
            
        Returns:
            Tuple[Optional[float], int]: Grid PG score and number of confident samples
        """
        W, H, scale, sub_image_size = self.get_grid_dimensions(inputs)

        # Get predictions for each grid cell
        predictions = self.get_grid_cell_predictions(inputs, scale, sub_image_size)

        mask = self.filter_high_confidence_samples(predictions, targets, scale)
        
        # All ranks process the full batch to maintain synchronization, although this causes some overhead especially when the number of confident samples is low
        total_pg_score = self.calculate_grid_pg_score(inputs, targets, scale, confidence_mask=mask)
        
        # Count confident samples and apply filtering to the final score
        num_confident_samples = torch.count_nonzero(mask)
        
        if num_confident_samples > 0:
            return total_pg_score, int(num_confident_samples.item())
        else:
            # No confident samples in this batch - return None to avoid logging
            return None, 0

    def visualize_grid_images(self, batch) -> Optional[Dict[str, Any]]:
        """
        Process visualization for the Grid PG dataset.
        
        Args:
            batch: The input batch (inputs, targets)
            
        Returns:
            Optional[Dict[str, Any]]: Visualization results or None if not applicable
        """
        inputs, targets = batch
        
        W, H, scale, sub_image_size = self.get_grid_dimensions(inputs)
        
        # Get predictions for each grid cell
        predictions = self.get_grid_cell_predictions(inputs, scale, sub_image_size)
        
        mask = self.filter_high_confidence_samples(predictions, targets, scale)
        
        # Use the first sample in the batch
        single_input = inputs[0].unsqueeze(0)  # Add batch dimension
        single_targets = targets[0]  # Shape [scale**2]

        sample_contribution_maps = []
        sample_explanations = []
        
        if mask[0]:  # Check if the first sample passes the confidence filter
            # Iterate through each grid cell for the sample
            for grid_cell_idx in range(scale**2):
                grid_cell_target = single_targets[grid_cell_idx].unsqueeze(0)  # Add batch dimension for explain_class
                
                # Get the full explanation for the single grid cell
                # Use clone().detach() to ensure a fresh input tensor for each explanation
                with torch.enable_grad(), torch.inference_mode(False):
                    single_input.requires_grad = True
                    expl_out = self.explain_fn(single_input.detach(), explain_class=grid_cell_target, mode="native")
                
                contribution_map = expl_out["contribution_map"]
                if len(contribution_map.shape) == 3:  # If missing batch dimension
                    contribution_map = contribution_map.unsqueeze(0)
                if len(contribution_map.shape) == 4 and contribution_map.shape[1] != 1:  # If missing channel dimension
                    contribution_map = contribution_map.unsqueeze(1)
                
                sample_explanations.append(expl_out.get("explanation"))  # Store full explanation if available
                sample_contribution_maps.append(contribution_map)

            # Calculate grid scores for the sample
            sample_contribution_maps_stacked = torch.stack(sample_contribution_maps, dim=1)  # Shape [1, scale**2, 1, W, H]
            original_cell_size = inputs.shape[-1] // scale  # Same calculation as in calculate_grid_pg_score
            sample_pg_score = get_localization_score(sample_contribution_maps_stacked, img_dims=(original_cell_size, original_cell_size), scale=scale)  # Shape [1, scale**2]

            # Prepare return dictionary
            result = {
                "inputs": single_input,
                "targets": single_targets.unsqueeze(0),  # Add batch dimension
                "explanations": [sample_explanations],  # List of lists, outer list for samples, inner for grid cells
                "contribution_maps": torch.stack(sample_contribution_maps, dim=0).squeeze(1),  # Shape [scale**2, W, H]
                "grid_scores": sample_pg_score.squeeze(0),  # Shape [scale**2]
                "scale": scale
            }
            
            return result
        else:
            return None
