"""Quick sanity check that the container environment works."""
import sys
import torch

print(f"Python:  {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
print(f"Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Quick tensor op to confirm things work
x = torch.randn(3, 3)
print(f"Tensor:  {x.shape} on {x.device}")
print("Environment OK")
