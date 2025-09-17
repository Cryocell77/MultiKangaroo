import os
import json
import torch
import torch.nn as nn

from fastchat.utils import str_to_torch_dtype
from transformers.models.llama import LlamaConfig

from kangaroo.adapter import AdapterModel
from kangaroo.multi_earlyexit import EarlyExitLlamaForCausalLM

class KangarooModel(nn.Module):

    def __init__(
            self,
            base_model_name_or_path,
            adapter_model_path,
            args,
            EARLY_STOP_LAYERS = [2, 3, 4],
    ):
        super().__init__()
        # CUDA mode: Use float16 for GPU efficiency
        self.base_model = EarlyExitLlamaForCausalLM.from_pretrained(
            base_model_name_or_path, 
            torch_dtype=str_to_torch_dtype(args.dtype),  # Use specified dtype (float16)
            device_map="cuda:0",  # Force single GPU to avoid device conflicts
            EARLY_STOP_LAYERS=EARLY_STOP_LAYERS
        )
        self.base_model = self.base_model.eval()
        self.early_exit_layers = EARLY_STOP_LAYERS

        # Load shared configuration once
        config = LlamaConfig.from_pretrained(base_model_name_or_path)
        
        # Initialize multiple adapter models
        self.adapter_models = nn.ModuleDict()
        adapter_paths = self._resolve_adapter_paths(adapter_model_path)
        
        for layer_idx in self.early_exit_layers:
            adapter_path = adapter_paths[layer_idx]
            adapter = self._load_single_adapter(adapter_path, config, args.dtype)
            self.adapter_models[str(layer_idx)] = adapter

        # Initialize head model
        self.head_model = self._load_head_model(base_model_name_or_path, config, args.dtype)

    def _resolve_adapter_paths(self, adapter_model_path):
        """Resolve adapter paths for all layers."""
        # Handle single JSON file passed as list (from command line nargs='+')
        if isinstance(adapter_model_path, list) and len(adapter_model_path) == 1 and adapter_model_path[0].endswith('.json'):
            adapter_model_path = adapter_model_path[0]
        
        if isinstance(adapter_model_path, str) and adapter_model_path.endswith('.json'):
            # JSON configuration file
            with open(adapter_model_path, 'r') as f:
                adapter_paths_config = json.load(f)
            return {int(layer): path for layer, path in adapter_paths_config.items()}
        
        elif isinstance(adapter_model_path, list):
            # List of paths
            if len(adapter_model_path) != len(self.early_exit_layers):
                raise ValueError(f"Number of adapter paths ({len(adapter_model_path)}) must match number of early exit layers ({len(self.early_exit_layers)})")
            return dict(zip(self.early_exit_layers, adapter_model_path))
        
        else:
            # Directory-based approach
            return {layer: os.path.join(adapter_model_path, f"layer_{layer}") for layer in self.early_exit_layers}

    def _load_single_adapter(self, adapter_path, config, dtype):
        """Load a single adapter model."""
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")
        
        # Use base config, but allow adapter-specific config if available
        adapter_config_path = os.path.join(adapter_path, "config.json")
        if os.path.exists(adapter_config_path):
            adapter_config = LlamaConfig.from_pretrained(adapter_config_path)
        else:
            adapter_config = config
            
        adapter = AdapterModel(adapter_config)
        adapter.load_state_dict(
            torch.load(os.path.join(adapter_path, "pytorch_model.bin"), map_location="cpu"), 
            strict=False
        )
        adapter = adapter.eval().to("cuda:0")  # Explicit device for consistency
        
        # CUDA mode: Use half precision if specified
        if dtype == "float16":
            adapter = adapter.half()
            
        return adapter

    def _load_head_model(self, base_model_name_or_path, config, dtype):
        """Load the language model head."""
        with open(os.path.join(base_model_name_or_path, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        
        weights = torch.load(os.path.join(base_model_name_or_path, head_path), map_location='cpu')
        head_model = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        head_model.weight.data = weights["lm_head.weight"].float()
        head_model = head_model.eval().to("cuda:0")  # Explicit device for consistency
        
        # CUDA mode: Use half precision if specified
        if dtype == "float16":
            head_model = head_model.half()
            
        return head_model

    def forward(self):
        raise NotImplementedError






