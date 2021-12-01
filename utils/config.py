import torch
import np as np

use_camb = False

if torch.__version__ == "parrots":
    from parrots.base import use_camb

int_dtype = torch.long
np_int = np.int64
if use_camb:
    int_dtype = torch.int
    np_int = np.int32
