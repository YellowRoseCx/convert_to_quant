import torch

def explain_numerical_divergence():
    # 1. PRECISION DRIFT
    x = torch.randn(1000, 1000, dtype=torch.float32)
    norm_blas = torch.linalg.norm(x)       # Optimized
    norm_naive = x.pow(2).sum().sqrt()     # Naive
    diff = torch.abs(norm_blas - norm_naive)

    # 2. OVERFLOW RISK
    y = torch.tensor([1e20], dtype=torch.float32)
    safe = torch.linalg.norm(y)            # Handles scaling
    unsafe = y.pow(2).sum().sqrt()         # Squares first -> Overflow

    print(f"Precision Diff (drift):   {diff.item():.10f}")
    print(f"Safe Result (linalg):     {safe.item()}")
    print(f"Unsafe Result (naive):    {unsafe.item()}")

if __name__ == "__main__":
    explain_numerical_divergence()
