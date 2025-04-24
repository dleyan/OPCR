import numpy as np
import torch
from tsl import logger


def add_missing(X: torch.Tensor, missing_type: str, missing_rate: float, seed: int):
    
    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    N, T, F = X.shape     # N, T, F
    X[np.isnan(X)] = 0

    rand = random.random
    logger.info(f'Generating mask with base missing rate={missing_rate}')

    # structual missing    
    if missing_type == 'spatial':

        p, q = missing_rate, 0.25

        # point missing
        mask = rand((N, T, F)) < q
        # mask p*N nodes
        mask_node = rand((N))
        mask[mask_node<p, ] = True

    # point missing
    elif missing_type == 'point':
        p = missing_rate

        # mask p*N*T*F points
        mask = rand((N, T, F)) < missing_rate
    
    else:
        raise ValueError("Invalid missing type.")
    
    mask_X = np.where(mask, np.full_like(X, fill_value=np.nan), X)

    eval_mask = mask.astype('uint8')

    return mask_X, eval_mask
