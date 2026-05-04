import torch
import svd_utils

# Create dummy inputs that mimic our layers
class DummyProj:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.randn(4096, 4096, device='cuda'))

proj = DummyProj()
U_slice = torch.randn(4096, 4096)
V = torch.randn(4096, 4096)
S_grad_info = torch.randn(4096)

svd_info = {
    'U': U_slice,
    'V': V,
    'S': torch.randn(4096)
}

print("Running apply_weight_threshold...")
sparsity = svd_utils.apply_weight_threshold(proj, 10, grad_info=S_grad_info, svd_info=svd_info)
print(f"Sparsity achieved: {sparsity}%")
