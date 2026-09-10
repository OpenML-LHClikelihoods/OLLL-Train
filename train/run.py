# ====== ADD THIS AT THE VERY START OF run.py ======
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Makes CUDA errors show immediately
import torch
import gc

def safe_cuda_init():
    """Attempt to establish a fresh CUDA context."""
    if torch.cuda.is_available():
        print(f"Initializing CUDA on device: {torch.cuda.current_device()}")
        torch.cuda.empty_cache()  # Clear any leftover cache from imports
        gc.collect()
        # Test the context with a tiny operation
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            print("CUDA context test passed.")
        except RuntimeError as e:
            print(f"WARNING: CUDA context appears unstable: {e}")
            # Optionally, you could fall back to CPU here
            # device = torch.device('cpu')

safe_cuda_init()
# ====== END OF ADDED CODE ======
import sys, os
print("Starting run.py", flush=True)
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
import hydra
import torch
from experiment import nLLsExperiment
from misc import get_device


@hydra.main(config_path="config", config_name="nLLs", version_base=None)
def main(cfg):
    print("Entered main()")
    device = get_device()
    print(f"Using device: {device}")

    if cfg.exp_type == "nLLs":
        exp = nLLsExperiment(cfg, device=device)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    match cfg.training.dtype:
        case 'float16':
            torch.set_default_dtype(torch.float16)
        case 'float64':
            torch.set_default_dtype(torch.float64)
        case 'float32':
            torch.set_default_dtype(torch.float32)
        case _:
            raise ValueError(f"dtype {cfg.dtype} not implemented")
        
    exp()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

# ====== ADD THIS AT THE VERY END OF run.py ======
import torch
import gc

# Force cleanup of CUDA resources
if torch.cuda.is_available():
    print("Cleaning up CUDA context before exit...")
    gc.collect()  # Collect Python garbage
    torch.cuda.empty_cache()  # Clear PyTorch's CUDA memory cache
    # torch.cuda.synchronize()  # Uncomment if you need to ensure all ops are done
    print("Cleanup complete.")
# ====== END OF ADDED CODE ======