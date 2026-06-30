import torch
import math

def fsum_frobenius_norm(x):
    x_flat = x.flatten().to(torch.float64)
    sq_tensor = x_flat.pow(2)
    sq_list = sq_tensor.tolist()
    return math.sqrt(math.fsum(sq_list))

x = torch.randn(10, 10, requires_grad=True)

try:
    print("Testing pow().sum().sqrt().backward()...")
    loss1 = x.pow(2).sum().sqrt()
    loss1.backward()
    print("Success! Gradient computed.")
except Exception as e:
    print("Failed:", e)

# Reset gradients
x.grad = None

try:
    print("\nTesting fsum_frobenius_norm().backward()...")
    # Wrap in tensor so we can call backward
    loss2 = torch.tensor(fsum_frobenius_norm(x), requires_grad=True)
    loss2.backward()
    print("Success! Gradient computed.")
    print("Gradient values:", x.grad)
except Exception as e:
    print("Failed:", e)
