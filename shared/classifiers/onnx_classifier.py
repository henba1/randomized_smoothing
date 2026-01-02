"""
ONNX Classifier wrapper for integration with diffusion denoised smoothing experiments for CIFAR-10 dataset.
This module provides a PyTorch-compatible interface for ONNX models.
"""
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort

from ..utils.utils import print_onnx_device_status

import torch
import torch.nn as nn

class ONNXClassifier(nn.Module):
    """
    A PyTorch-compatible wrapper for ONNX models that provides the same interface
    as Hugging Face's AutoModelForImageClassification.
    """
    
    def __init__(self, onnx_path: str, device: str = "cuda", input_name: str | None = None, output_name: str | None = None):
        """
        Initialize the ONNX classifier wrapper.
        
        Args:
            onnx_path: Path to the ONNX model file
            device: Device to run inference on ("cuda" or "cpu")
            input_name: Name of the input tensor (auto-detected if None)
            output_name: Name of the output tensor (auto-detected if None)
        """
        super().__init__()
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        # Check available providers before selecting (some runtimes omit this helper)
        if hasattr(ort, "get_available_providers"):
            available_providers = ort.get_available_providers()
            print(f"Available providers: {available_providers}")
        else:
            available_providers = []
            print(
                "Warning: onnxruntime lacks `get_available_providers()`. "
                "Assuming CPU execution provider only."
            )
        
        # Only use CUDA if requested AND available
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            if device == "cuda":
                print(f"Warning: CUDAExecutionProvider not available. Available providers: {available_providers}. Falling back to CPU.")
        
        # Create session options if available, otherwise use None
        try:
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1
        except AttributeError:
            print("Warning: SessionOptions not available in onnxruntime, using default session options")
            session_options = None
        
        self.session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
        
        actual_providers = self.session.get_providers()
        print(f"ONNX Runtime execution providers: {actual_providers}")
        print_onnx_device_status(actual_providers, device)
        
        # Get input/output information
        self.input_details = self.session.get_inputs()
        self.output_details = self.session.get_outputs()
        
        # Auto-detect input/output names if not provided
        self.input_name = input_name or self.input_details[0].name
        self.output_name = output_name or self.output_details[0].name
        
        # Store model metadata
        self.input_shape = self.input_details[0].shape
        self.output_shape = self.output_details[0].shape
        
        # Extract expected image dimensions from input shape
        if len(self.input_shape) == 4:
            self.expected_height = self.input_shape[2] if isinstance(self.input_shape[2], int) else 32
            self.expected_width = self.input_shape[3] if isinstance(self.input_shape[3], int) else 32
        else:
            raise ValueError(f"Input shape {self.input_shape} is not supported")
        
        # Determine number of classes from output shape
        if len(self.output_shape) == 2:
            self.num_classes = self.output_shape[1] if isinstance(self.output_shape[1], int) else 10
        else:
            self.num_classes = self.output_shape[-1] if isinstance(self.output_shape[-1], int) else 10
        
        print(f"Loaded ONNX model: {os.path.basename(onnx_path)}")
        print(f"Input shape: {self.input_shape}, Output shape: {self.output_shape}")
        print(f"Expected input size: {self.expected_height}x{self.expected_width}")
        print(f"Number of classes: {self.num_classes}")
        
        # Create a dummy logits attribute for compatibility
        self.logits = None
        
    def forward(self, x: torch.Tensor | np.ndarray) -> 'ONNXOutput':
        """
        Forward pass through the ONNX model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            ONNXOutput object with logits attribute for compatibility
        """
        # Convert PyTorch tensor to numpy
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Ensure input is float32
        if x_np.dtype != np.float32:
            x_np = x_np.astype(np.float32)
        
        # Run inference
        try:
            outputs = self.session.run([self.output_name], {self.input_name: x_np})
            logits = outputs[0]
        except Exception as e:
            print(f"ONNX inference error: {e}")
            # Return zeros as fallback
            logits = np.zeros((x_np.shape[0], self.num_classes), dtype=np.float32)
        
        # Convert back to PyTorch tensor
        logits_tensor = torch.from_numpy(logits).to(x.device)
        
        # Return wrapped output for compatibility
        return ONNXOutput(logits_tensor)
    
    def eval(self):
        """Set model to evaluation mode (no-op for ONNX models)."""
        return self
    
    def cuda(self):
        """Move model to CUDA (no-op for ONNX models, handled by session)."""
        return self
    
    def to(self, device):
        """Move model to device (no-op for ONNX models, handled by session)."""
        return self
    
    def get_expected_input_size(self):
        """Get the expected input image size as a tuple (height, width)."""
        return (self.expected_height, self.expected_width)


class ONNXOutput:
    """
    Output wrapper to mimic Hugging Face model output format.
    """
    
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


def load_onnx_classifier(model_name: str, models_dir: str, device: str = "cuda") -> ONNXClassifier:
    """
    Load an ONNX classifier from the models directory.
    
    Args:
        model_name: Name of the model file (with or without .onnx extension)
        models_dir: Directory containing ONNX models
        device: Device to run inference on ("cuda" or "cpu"), defaults to "cuda"
        
    Returns:
        ONNXClassifier instance
    """
    if not model_name.endswith('.onnx'):
        model_name += '.onnx'
    
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        available_models = []
        if os.path.exists(models_dir):
            available_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
        raise FileNotFoundError(
            f"Model {model_name} not found in {models_dir}. "
            f"Available models: {available_models}"
        )
    
    return ONNXClassifier(model_path, device=device)


def list_available_models(models_dir: str) -> list:
    """
    List all available ONNX models in the models directory.
    
    Args:
        models_dir: Directory containing ONNX models
        
    Returns:
        List of available model names
    """
    if not os.path.exists(models_dir):
        return []
    
    return [f for f in os.listdir(models_dir) if f.endswith('.onnx')]