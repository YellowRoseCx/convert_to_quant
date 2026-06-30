import torch
x = torch.randn(4096, 4096, device="cuda")
try:
    print("Testing torch.linalg.norm...")
    norm = torch.linalg.norm(x)
    print("Norm successful:", norm)
except Exception as e:
    print("Exception in norm:", e)

try:
    print("\nTesting U @ error @ Vh...")
    U = torch.randn(800, 4096, device="cuda")
    V = torch.randn(4096, 800, device="cuda")
    projected = U @ x @ V
    print("Matmul successful. Shape:", projected.shape)
except Exception as e:
    print("Exception in matmul:", e)
