import torch.nn as nn


class PyTorchClassifierWrapper(nn.Module):
    """Wrapper for PyTorch models to provide consistent interface with HuggingFace/ONNX models."""
    
    def __init__(self, model: nn.Module, expected_height: int = 32, expected_width: int = 32):
        super().__init__()
        self.model = model
        self.expected_height = expected_height
        self.expected_width = expected_width
    
    def forward(self, x):
        """Forward pass that returns object with .logits attribute."""
        logits = self.model(x)
        # Wrap in a simple object with .logits attribute for consistency
        class LogitsOutput:
            def __init__(self, logits):
                self.logits = logits
        return LogitsOutput(logits)
    
    def eval(self):
        self.model.eval()
        return self
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
