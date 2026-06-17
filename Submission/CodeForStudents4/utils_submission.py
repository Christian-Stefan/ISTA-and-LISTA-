from CodeForStudents4.imports_submission import *


### ----------------------- Start --------------- 4.3 MRI Helper Functions ### --------------- Start ----------------------- ###
@staticmethod
def get_partial_k_space(k_space: torch.Tensor) -> torch.Tensor:

    # 1. Determine the general size, meaning shape and number of columns
    matrix_shape = k_space.shape 
    num_columns = matrix_shape[-1]
    
    # 2. The definition of a dense center block becomes of necessity since 
    # a good deal of the image content/signal has its place of origin there

    center_width: int = 15  # Width of the fully sampled center
    center_start = (num_columns - center_width) // 2
    center_end = center_start + center_width
    center_indices = torch.arange(center_start, center_end, device=k_space.device)
    
    # 3.  Gather remaining edge indices and sample randomly
    edge_indices = torch.cat([
        torch.arange(0, center_start, device=k_space.device),
        torch.arange(center_end, num_columns, device=k_space.device)
    ])
    num_edge_cols: int = 10  # Number of random columns to pick from edges
    random_edge_selection = torch.randperm(len(edge_indices), device=k_space.device)[:num_edge_cols]
    sampled_edge_indices = edge_indices[random_edge_selection]
    
    # 4. Combine both steps
    active_col_indices = torch.cat([center_indices, sampled_edge_indices])
    
    # 5. Construct the mask using the combined indices
    mask_1d = torch.zeros(num_columns, dtype=k_space.dtype, device=k_space.device)
    mask_1d[active_col_indices] = 1.0
    
    # 6. Apply via broadcasting
    measurement_matrix_M = mask_1d.view(1, 1, -1)
    partial_k_space = k_space * measurement_matrix_M

    return partial_k_space, measurement_matrix_M

@staticmethod
def get_k_space(MRI_imgs: torch.Tensor) -> torch.Tensor:
    # 1. Transform straight from standard image to frequency domain
    k_space = torch.fft.fft2(MRI_imgs)
    k_space = torch.fft.fftshift(k_space, dim=(-2, -1))
    return k_space

@staticmethod
def get_accelerated_MRI(partial_k_space: torch.Tensor) -> torch.Tensor:
    # 1. Un-center the k-space back to standard FFT format
    ift = torch.fft.ifftshift(partial_k_space, dim=(-2, -1))
    # 2. Transform back to the spatial image domain
    accelerated_MRI = torch.fft.ifft2(ift)
    
    return accelerated_MRI

### ----------------------- End --------------- 4.3 MRI Helper Functions ### --------------- End ----------------------- ###

