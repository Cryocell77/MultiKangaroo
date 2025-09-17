"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import torch
import os

from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from kangaroo.multi_kangaroo_model import KangarooModel

def kangaroo_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, max_length=2048, EARLY_STOP_LAYERS=[2, 3, 4], SPECULATIVE_DECODING_STEPS=6, thresholds=[0.6, 0.55, 0.5]):
    # Origin-style device handling
    context_tokens = inputs.input_ids
    device = context_tokens.device
    
    # Ensure context_tokens has correct 2D shape [batch_size, seq_len]
    if context_tokens.dim() == 1:
        context_tokens = context_tokens.unsqueeze(0)  # Add batch dimension
    
    # Critical: Validate all tokens to prevent CUDA errors
    vocab_size = getattr(tokenizer, 'vocab_size', 32000)
    
    # Enhanced input validation with explicit bounds checking
    try:
        # Convert to CPU for safe validation
        context_tokens_cpu = context_tokens.cpu()
        if torch.any(context_tokens_cpu >= vocab_size) or torch.any(context_tokens_cpu < 0):
            # Clamp invalid input tokens to valid range
            context_tokens_cpu = torch.clamp(context_tokens_cpu, 0, vocab_size - 1)
            context_tokens = context_tokens_cpu.to(device)
    except Exception as e:
        # Fallback: create safe tokens if validation fails
        batch_size, seq_len = context_tokens.shape
        context_tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    
    token_eos = tokenizer.eos_token_id
    batch_size, context_length = context_tokens.shape
    
    # Validate EOS token ID
    if token_eos is None or token_eos >= vocab_size or token_eos < 0:
        token_eos = vocab_size - 1  # Use last token in vocab as fallback EOS
    
    # Simple initialization like origin version - no position encoding checks!
    
    global_tokens = torch.ones((batch_size, max_length), dtype=torch.long, device=device) * token_eos
        # ORIGIN APPROACH: Simple position_ids creation like origin version
    # Trust that the model can handle max_length positions (remove all safety checks)
    global_position_ids = torch.LongTensor([[i for i in range(max_length)]]).to(device)
    accept_length_list = [1]

    # Multi-layer early exit configuration
    assert len(EARLY_STOP_LAYERS) == len(thresholds), f"Layers and thresholds must match: {len(EARLY_STOP_LAYERS)} != {len(thresholds)}"
    early_exit_config = list(zip(EARLY_STOP_LAYERS, thresholds))
    lowest_exit_layer = min(EARLY_STOP_LAYERS)

    start_index = context_length
    global_tokens[:, :start_index] = context_tokens

    # Initialize KV-cache and generate first token
    with torch.no_grad():
        position_ids = global_position_ids[:, :start_index]
        output = model.base_model(context_tokens, position_ids=position_ids, output_hidden_states=True)
        model.base_model.past_key_values = list(output.past_key_values)
        
        # Generate first token using full model with CUDA safety
        logits = output.logits
        first_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        # Validate token ID is within vocabulary bounds
        vocab_size = logits.shape[-1]
        if first_token >= vocab_size or first_token < 0:
            first_token = token_eos
        global_tokens[:, start_index] = first_token
        
        # Initialize adapter KV-cache with lowest layer
        hidden_state_early = output.hidden_states[lowest_exit_layer]
        adapter_model = model.adapter_models[str(lowest_exit_layer)]
        _, adapter_past_key_values = adapter_model.forward_early_stop(
            inputs_embeds=hidden_state_early, 
            position_ids=global_position_ids[:, :context_length], 
            use_cache=True
        ) 

    total_inference_steps = 0

    with torch.no_grad():
        max_infer_steps = min(max_length, start_index + max_new_tokens)
        stop = False
        
        # Origin-style loop condition with safety checks

        while start_index < max_infer_steps - 1 - SPECULATIVE_DECODING_STEPS:
            start_index_copy = start_index
            
            # Origin-style natural termination - no artificial limits needed
            # Early exit mechanism provides natural sequence length control
                
            # KV-cache synchronization - ensure ALL layers have EXACT same length
            current_kv_length = model.base_model.past_key_values[0][0].shape[2] if model.base_model.past_key_values and model.base_model.past_key_values[0] else 0
            
            # Check if ANY layer has inconsistent length
            needs_sync = False
            if model.base_model.past_key_values:
                for i, layer_kv in enumerate(model.base_model.past_key_values):
                    if layer_kv is not None:
                        layer_length = layer_kv[0].shape[2]
                        if layer_length != start_index:
                            needs_sync = True
                            break
            
            if needs_sync or current_kv_length > start_index:
                past_key_values_large_ = []
                for i, layer_kv in enumerate(model.base_model.past_key_values):
                    if layer_kv is not None:
                        k, v = layer_kv
                        # Ensure exact length match
                        if k.shape[2] > start_index:
                            past_key_values_large_.append((k[:,:,:start_index,:], v[:,:,:start_index,:]))
                        elif k.shape[2] < start_index:
                            # This shouldn't happen, but handle gracefully
                            past_key_values_large_.append((k, v))
                        else:
                            past_key_values_large_.append((k, v))
                    else:
                        past_key_values_large_.append(None)
                model.base_model.past_key_values = past_key_values_large_
                del past_key_values_large_
            
            if (adapter_past_key_values and 
                len(adapter_past_key_values) > 0 and
                adapter_past_key_values[0][0].shape[2] > start_index):
                
                adapter_past_key_values_ = []
                for layer_kv in adapter_past_key_values:
                    if layer_kv is not None:
                        k, v = layer_kv
                        adapter_past_key_values_.append((k[:,:,:start_index,:], v[:,:,:start_index,:]))
                    else:
                        adapter_past_key_values_.append(None)
                adapter_past_key_values = tuple(adapter_past_key_values_)
                del adapter_past_key_values_
            
            end_index = start_index + 1
            
            # STEP 1: Multi-layer draft decoding
            for step in range(1 + SPECULATIVE_DECODING_STEPS):
                assert adapter_past_key_values[0][0].shape[2] <= end_index-1
                in_tokens_small = global_tokens[:, end_index-1:end_index]
                
                # KV-cache synchronization (origin-style)
                if adapter_past_key_values[0][0].shape[2] < end_index-1:
                    position_ids = global_position_ids[:, start_index-1:end_index]
                    hidden_state_early_last = exited_hidden_states[:,-1:,:] if 'exited_hidden_states' in locals() and exited_hidden_states is not None else None
                else:
                    position_ids = global_position_ids[:, end_index-1:end_index]
                    hidden_state_early_last = None
                
                # Draft decoding: use inputs_embeds for steps > 0 to avoid KV-cache duplication
                if step == 0:
                    # Sync KV-cache state before first call in this loop iteration
                    pass
                    
                    # First step: use input_ids (match origin: process UP TO but not including lowest_exit_layer)
                    hidden_state_early = model.base_model.forward_from(
                        input_ids=in_tokens_small[:,-1:], 
                        position_ids=position_ids[:,-1:],
                        start_layer=0, 
                        end_layer=lowest_exit_layer - 1  # Origin version: layers[:early_exit_layer] excludes the layer
                    )
                else:
                    # Subsequent steps: use embeddings (match origin: process UP TO but not including lowest_exit_layer)
                    embeddings = model.base_model.get_input_embeddings()(in_tokens_small[:,-1:])
                    hidden_state_early = model.base_model.forward_from(
                        inputs_embeds=embeddings,
                        position_ids=position_ids[:,-1:],
                        start_layer=0, 
                        end_layer=lowest_exit_layer - 1  # Origin version: layers[:early_exit_layer] excludes the layer
                    )
                
                # Prepare adapter input (origin-style concatenation)
                if hidden_state_early_last is not None:
                    adapter_input = torch.cat([hidden_state_early_last, hidden_state_early], dim=1)
                else:
                    adapter_input = hidden_state_early
                
                # Try multiple adapters with the same hidden state
                accepted_token = None
                verification_hidden_state = None
                final_exit_layer = None
                
                for exit_layer, threshold in early_exit_config:
                    # Use the corresponding adapter for this layer
                    adapter_model = model.adapter_models[str(exit_layer)]
                    hidden_state, new_adapter_kv = adapter_model.forward_early_stop(
                        inputs_embeds=adapter_input, 
                        position_ids=position_ids, 
                        past_key_values=adapter_past_key_values, 
                        use_cache=True
                    )

                    predict_logits = model.head_model(hidden_state[:,-1:,:]).float() 
                    predict_score = predict_logits.softmax(dim=-1).max().item()
                    
                    # Track confidence scores for early exit evaluation
                    
                    # Check if confidence is high enough for this layer
                    if predict_score >= threshold:
                        accepted_token = torch.argmax(predict_logits[:, -1, :], dim=-1)
                        vocab_size = predict_logits.shape[-1]
                        
                        # Enhanced token validation
                        if accepted_token.item() >= vocab_size or accepted_token.item() < 0:
                            accepted_token = torch.tensor([token_eos], device=predict_logits.device, dtype=torch.long)
                        else:
                            # Ensure correct device and dtype
                            accepted_token = accepted_token.to(device=predict_logits.device, dtype=torch.long)
                            
                        verification_hidden_state = hidden_state_early
                        final_exit_layer = exit_layer
                        adapter_past_key_values = new_adapter_kv
                        
                        # Token accepted from early exit layer
                        break
                    
                    # Origin-style early termination: if confidence too low after trying all layers, stop speculation
                    elif exit_layer == EARLY_STOP_LAYERS[-1]:  # Last layer in our multi-layer setup
                        # If even the last early exit layer doesn't meet threshold, stop speculating
                        verification_hidden_state = hidden_state_early  
                        final_exit_layer = exit_layer
                        adapter_past_key_values = new_adapter_kv
                        break
                
                # Update states (origin-style: always update exited_hidden_states)
                if step == 0:
                    exited_hidden_states = None

                # Always append hidden_state_early to exited_hidden_states (like origin)
                exited_hidden_states = hidden_state_early if exited_hidden_states is None else torch.cat([exited_hidden_states, hidden_state_early], dim=1)

                if accepted_token is not None:
                    global_tokens[:, end_index] = accepted_token
                else:
                    break  # Exit early exit loop

                # Early exit condition (origin-style)
                if step == SPECULATIVE_DECODING_STEPS or (step > 0 and predict_score < min(thresholds)):
                    break
                
                end_index += 1

            # STEP2: Big model verification (origin-style)
            position_ids = global_position_ids[:, start_index:end_index]
            
            # Always use lowest_exit_layer for verification (like origin version)
            verification_start_layer = lowest_exit_layer
            
            # KV-cache consistency check and auto-correction
            actual_kv_length = model.base_model.past_key_values[verification_start_layer][0].shape[2]
            if actual_kv_length != start_index:
                # Auto-trim KV-cache to correct length (preventive measure)
                past_key_values_corrected = []
                for k, v in model.base_model.past_key_values:
                    past_key_values_corrected.append((k[:,:,:start_index,:], v[:,:,:start_index,:]))
                model.base_model.past_key_values = past_key_values_corrected
            
            # Run verification from the exit layer to the end
            if exited_hidden_states is not None:
                # Origin-style assertion: ensure KV-cache is properly aligned
                kv_length = model.base_model.past_key_values[verification_start_layer][0].shape[2]
                assert kv_length == start_index, f"KV-cache mismatch: {kv_length} != {start_index}"
                assert exited_hidden_states.shape[1] == position_ids.shape[1]
                
                # Continue from the verification layer to the end (similar to origin's large model mode)
                hidden_state_, hidden_state = model.base_model.forward_from(
                    inputs_embeds=exited_hidden_states, 
                    start_layer=verification_start_layer, 
                    end_layer=None,
                    position_ids=position_ids
                )
            else:
                # Fallback: full model inference
                current_token = global_tokens[:, start_index:start_index+1]
                position_ids = global_position_ids[:, start_index:start_index+1]
                hidden_state_, hidden_state = model.base_model.forward_from(
                    input_ids=current_token, 
                    start_layer=0, 
                    end_layer=None,
                    position_ids=position_ids
                )
            
            logits = model.head_model(hidden_state).float()
            output_tokens = torch.argmax(logits[:, :, :], dim=-1)
            
            # Enhanced token validation for verification phase
            vocab_size = logits.shape[-1]
            output_tokens = torch.clamp(output_tokens, 0, vocab_size - 1)

            # Verification for greedy decoding
            output_length = end_index - start_index
            for i in range(output_length):
                # Additional safety check for token validity
                verified_token = output_tokens[0, i].item()
                if verified_token >= vocab_size or verified_token < 0:
                    verified_token = token_eos
                
                if i == output_length-1 or verified_token == token_eos or verified_token != global_tokens[0, start_index+1+i]:
                    global_tokens[0, start_index+1+i] = verified_token
                    start_index = start_index+1+i
                    if verified_token == token_eos:
                        stop = True
                    break

            accept_length_list.append(start_index - start_index_copy)

            # STEP 3: Post-process KV-cache (trim to current end_index)
            current_end = end_index  # This will be the start_index for next iteration
            if model.base_model.past_key_values[0][0].shape[2] > current_end:
                past_key_values_large_ = []
                for k, v in model.base_model.past_key_values:
                    past_key_values_large_.append((k[:,:,:current_end,:], v[:,:,:current_end,:]))
                model.base_model.past_key_values = past_key_values_large_

            if adapter_past_key_values[0][0].shape[2] > current_end:
                adapter_past_key_values_ = []
                for k, v in adapter_past_key_values:
                    adapter_past_key_values_.append((k[:,:,:current_end,:], v[:,:,:current_end,:]))
                adapter_past_key_values = tuple(adapter_past_key_values_)
            
            total_inference_steps += 1

            if stop:
                break

    # Final safety check: ensure all output tokens are valid
    output_tokens_tensor = global_tokens[0, :start_index+1]
    vocab_size = getattr(tokenizer, 'vocab_size', 32000)  # Default to LLaMA vocab size
    
    # Clamp all tokens to valid range
    output_tokens_tensor = torch.clamp(output_tokens_tensor, 0, vocab_size - 1)
    output_ids = output_tokens_tensor.tolist()
    
    new_token = start_index - context_length + 1
    idx = len(accept_length_list) - 1
    
    return [output_ids], new_token, idx, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--adapter-paths",
        type=str,
        nargs='+',
        required=True,
        help="The paths to the adapter models."
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs='+',
        default=[0.6, 0.55, 0.5],
        help="The confidence thresholds for cascaded early exit layers.",
    )
    parser.add_argument(
        "--exitlayers",
        type=int,
        nargs='+',
        default=[2, 3, 4],
        help="The early exit layers for cascaded inference.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="The number of GPUs per model.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    question_file = f"data/question.jsonl"

    model = KangarooModel(args.model_path, args.adapter_paths, args, EARLY_STOP_LAYERS = args.exitlayers)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    do_sample = False

    assert not args.answer_file
    os.makedirs(f"data/{args.bench_name}/{args.model_id}", exist_ok=True)

    for run in range(3):
        answer_file = f"data/{args.bench_name}/{args.model_id}/{run}.jsonl"
        print(f"Output to {answer_file}")

        run_eval(
            model=model,
            tokenizer=tokenizer,
            forward_func=kangaroo_forward,
            model_id=args.model_id,
            question_file=question_file,
            question_begin=args.question_begin,
            question_end=args.question_end,
            answer_file=answer_file,
            max_new_tokens=args.max_new_tokens,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            do_sample=do_sample,
            thresholds=args.thresholds,
            SPECULATIVE_DECODING_STEPS=args.steps,
            EARLY_STOP_LAYERS=args.exitlayers
        )

        reorg_answer_file(answer_file)