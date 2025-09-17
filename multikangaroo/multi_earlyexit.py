import torch
from typing import List, Optional, Tuple, Union
from transformers.models.llama import LlamaForCausalLM


class EarlyExitLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, EARLY_STOP_LAYERS):
        super().__init__(config)
        self.past_key_values = None
        self.early_exit_layers = EARLY_STOP_LAYERS

    @torch.no_grad()
    def forward_from(self,
                     inputs_embeds: Optional[torch.FloatTensor] = None,
                     input_ids: Optional[torch.LongTensor] = None,
                     start_layer: int = 0,
                     end_layer: int = None,
                     position_ids: Optional[torch.LongTensor] = None):

        # 1. Input handling
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 2. KV-cache and position_ids preparation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        device = inputs_embeds.device

        # KV-cache length calculation - use the starting layer's cache length
        if self.past_key_values is not None:
            # Use start_layer's KV-cache length (critical for verification phase)
            past_key_values_length = self.past_key_values[start_layer][0].shape[2] if self.past_key_values[start_layer] else 0

        seq_length_with_past = seq_length + past_key_values_length


        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, 
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # 3. Attention Mask preparation
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=device
        )
        attention_mask = self.model._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # 4. Core logic: execute the specified range of layers
        hidden_states = inputs_embeds
        total_layers = len(self.model.layers)
        
        # Determine the actual end layer
        if end_layer is None:
            end_layer = total_layers - 1
        
        # Validate layer indices
        if start_layer < 0 or start_layer >= total_layers:
            raise ValueError(f"start_layer ({start_layer}) must be between 0 and {total_layers-1}")
        if end_layer < start_layer or end_layer >= total_layers:
            raise ValueError(f"end_layer ({end_layer}) must be between {start_layer} and {total_layers-1}")
            
        # Initialize KV-cache if needed
        if self.past_key_values is None:
            self.past_key_values = [None] * total_layers
        
        # Process layers from start_layer to end_layer (inclusive)
        for layer_idx in range(start_layer, end_layer + 1):
            decoder_layer = self.model.layers[layer_idx]
            # Safe KV-cache access
            past_key_value = None
            if (self.past_key_values is not None and 
                layer_idx < len(self.past_key_values) and 
                self.past_key_values[layer_idx] is not None):
                past_key_value = self.past_key_values[layer_idx]
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = layer_outputs[0]
            new_kv = layer_outputs[1]
            self.past_key_values[layer_idx] = new_kv

        # 5. Origin-style: NO automatic KV-cache extension across layers
        # Origin version doesn't do cross-layer KV-cache synchronization
        # This aggressive extension may be causing position encoding overflow


        # 6. Output handling  
        if end_layer == total_layers - 1:
            # Apply final norm if we ran to the end of the model
            hidden_states_final = self.model.norm(hidden_states)
            return hidden_states, hidden_states_final
        else:
            # Return the raw hidden states for intermediate exits
            return hidden_states