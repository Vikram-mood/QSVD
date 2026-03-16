import torch
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration, LlavaNextConfig

config = LlavaNextConfig()
model = LlavaNextForConditionalGeneration(config)

class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.proj = module
    def forward(self, *args, **kwargs):
        raise ValueError("Caught")

print("Before override:")
print("type(model.model.multi_modal_projector):", type(model.model.multi_modal_projector))

projector_catcher = Catcher(model.model.multi_modal_projector)
model.model.multi_modal_projector = projector_catcher

print("After override:")
print("type(model.model.multi_modal_projector):", type(model.model.multi_modal_projector))
print("Is Catcher?", isinstance(model.model.multi_modal_projector, Catcher))

# Test forward pass override
try:
    model.model.multi_modal_projector(torch.randn(1, 10, config.vision_config.hidden_size))
except ValueError as e:
    print("Error caught successfully:", e)
except Exception as e:
    print("Other error:", type(e).__name__)
