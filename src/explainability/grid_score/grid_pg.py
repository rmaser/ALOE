import torch
from . import utils
import numpy as  np

def get_localization_score_single(positive_attributions, head_idx, head_pos_idx, scale=2):
    """
    Evaluates the localization score for a set of positive attributions at a given classification head

    :param positive_attributions: Positive attributions
    :type positive_attributions: torch.Tensor of the shape (B, K, 1, H, W), where B is the batch size, K is the number of grid cells per image, H is the image height, and W is the image width
    :param head_idx: Index of the classification head in the attribution tensor
    :type head_idx: int, in the range [0, K-1]
    :param head_pos_idx: Index of the position of the grid cell in the grid
    :type head_pos_idx: int, in the range [0, N-1], for a N x N grid
    :param scale: Grid dimension N, defaults to 2
    :type scale: int, optional
    :return: Localization scores and sum of attributions inside each target grid cell
    :rtype: tuple consisting of two torch.Tensor each of the shape (B,)
    """
    original_img_dims = positive_attributions.shape[3] // scale, positive_attributions.shape[4] // scale
    row_idx = head_pos_idx // scale
    col_idx = head_pos_idx % scale
    positive_attributions_inside = positive_attributions[:, head_idx, :, row_idx * original_img_dims[0]:(
        row_idx + 1) * original_img_dims[0], col_idx * original_img_dims[1]:(col_idx + 1) * original_img_dims[1]].sum(dim=(1, 2, 3))
    positive_attributions_total = positive_attributions[:, head_idx].sum(
        dim=(1, 2, 3))
    return torch.where(positive_attributions_total > 1e-7, positive_attributions_inside * torch.reciprocal(positive_attributions_total), torch.tensor(0.0, device=positive_attributions_total.device, dtype=positive_attributions_total.dtype)), positive_attributions_inside


def get_localization_score(attributions, only_corners=False, img_dims=(224, 224), scale=2):
    """
    Evaluates the localization score from a set of attributions

    :param attributions: Attributions
    :type attributions: torch.Tensor of the shape (B, K, 1, H, W), where B is the batch size, K is the number of grid cells per image, H is the image height, and W is the image width
    :param only_corners: Flag to enable evaluating localization only on the top-left and bottom-right corners of the grid, defaults to False
    :type only_corners: bool, optional
    :param img_dims: Dimensions of each grid cell, defaults to (224, 224)
    :type img_dims: tuple, optional. Dimensions (h, w), where h is the height of the grid cell, and w is the width of the grid cell.
    :param scale: Grid dimension N, defaults to 2
    :type scale: int, optional
    :return: Localization scores and sum of attributions inside each target grid cell
    :rtype: tuple consisting of two torch.Tensor each of the shape (B*K,)
    """
    grid_img_dims = tuple([scale * dim for dim in img_dims])
    interpolated_attributions = utils.interpolate_attributions(
        attributions, img_dims=grid_img_dims)
    positive_attributions = utils.get_positive_attributions(
        interpolated_attributions)
    grid_size = scale * scale
    if only_corners:
        head_list = [0, grid_size - 1]
    else:
        head_list = np.arange(grid_size).tolist()
    localization_scores = []
    for head_idx, head_pos_idx in enumerate(head_list):
        localization_scores.append(get_localization_score_single(
            positive_attributions, head_idx, head_pos_idx, scale=scale)[0])
    return torch.cat(localization_scores)                                                        
                                                                                          
def strided_forward(x, model, map_size=2):                                                                 
    B, C, H, W = x.shape                                                              
    assert H == W, "height should be equal to width."                                 
    cell_hw = H // map_size                                                      
    stride = cell_hw // 2                                                             
                                                                                        
    # split input                                                                         
    run_op = None # running output                                                        
    counts = torch.zeros_like(x)                                                      
    for idx in range((H-cell_hw) // stride + 1):                                      
        for jdx in range((H-cell_hw) // stride + 1):                                  
            curr_inp = x[:, :, idx*stride : idx*stride + cell_hw,                     
                    jdx*stride : jdx*stride + cell_hw]                                
            counts[:, :, idx*stride : idx*stride + cell_hw,                           
                    jdx*stride : jdx*stride + cell_hw] += 1.0                          
            
            curr_out = model.explain(curr_inp)
            
            if run_op is None:
                # Initialize with correct shape (B, Channels, H, W) based on output
                run_op = torch.zeros((B, curr_out.shape[1], H, W), device=curr_out.device, dtype=curr_out.dtype)
            
            run_op[:, :, idx*stride : idx*stride + cell_hw,
                   jdx*stride : jdx*stride + cell_hw] += curr_out
                                                                                        
    # Normalize by overlap counts (using first channel of counts for broadcasting)
    run_op /= counts[:, 0:1, :, :]
                                                                                        
    return run_op, counts

def strided_forward_general(x, explain_fn, map_size=2):                                                                 
    B, C, H, W = x.shape                                                              
    assert H == W, "height should be equal to width."                                 
    cell_hw = H // map_size                                                      
    stride = cell_hw // 2                                                             
                                                                                        
    # split input                                                                         
    run_op = None # running output                                                        
    counts = torch.zeros_like(x)                                                      
    for idx in range((H-cell_hw) // stride + 1):                                      
        for jdx in range((H-cell_hw) // stride + 1):                                  
            curr_inp = x[:, :, idx*stride : idx*stride + cell_hw,                     
                    jdx*stride : jdx*stride + cell_hw]                                
            counts[:, :, idx*stride : idx*stride + cell_hw,                           
                    jdx*stride : jdx*stride + cell_hw] += 1.0                          
            
            curr_out = explain_fn(curr_inp)
            
            if run_op is None:
                # Initialize with correct shape (B, Channels, H, W) based on output
                run_op = torch.zeros((B, curr_out.shape[1], H, W), device=curr_out.device, dtype=curr_out.dtype)

            run_op[:, :, idx*stride : idx*stride + cell_hw,
                   jdx*stride : jdx*stride + cell_hw] += curr_out
                                                                                        
    # Normalize by overlap counts (using first channel of counts for broadcasting)
    run_op /= counts[:, 0:1, :, :]
                                                                                        
    return run_op, counts