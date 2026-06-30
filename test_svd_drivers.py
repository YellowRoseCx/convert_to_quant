import torch

M, N = 1024, 1024
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(M, N, device=device)

drivers = [None, 'gesvd', 'gesvdj', 'gesvda']
for driver in drivers:
    print(f"\n--- Testing torch.linalg.svd with driver={driver} ---")
    try:
        U, S, V = torch.linalg.svd(x, full_matrices=False, driver=driver)
        print(f"Success with driver={driver}")
    except Exception as e:
        print(f"Failed with driver={driver}: {e}")
