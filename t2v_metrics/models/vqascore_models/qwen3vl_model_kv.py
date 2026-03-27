import os
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Dict
from transformers import DynamicCache
from .qwen3vl_model import Qwen3VLModel, QWEN3_VL_MODELS

class Qwen3VLModelKV(Qwen3VLModel):
    """
    Production-ready wrapper for Qwen3-VL-8B with KV cache support.
    
    This class extends Qwen3VLModel to provide:
    1. prefill_video_kv: Encodes a video once and caches the KV states.
    2. forward_kv: Computes alignment scores (P(Yes)) using the cached KV states,
       providing significant speedup for multiple questions on the same video.
    """
    
    def __init__(self,
                 model_name='qwen3-vl-8b',
                 device='cuda',
                 cache_dir=None,
                 checkpoint=None,
                 torch_dtype=torch.float16,
                 attn_implementation='flash_attention_2'):
        """
        Initialize the KV-cached Qwen3-VL model.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
            cache_dir: Directory to cache the model
            checkpoint: Optional path to model checkpoint
            torch_dtype: Torch data type (defaults to float16 for KV optimization)
            attn_implementation: Attention implementation (defaults to flash_attention_2)
        """
        if cache_dir is None:
             cache_dir = os.environ.get("HF_HOME")
        
        # Override model info for KV optimization
        if model_name in QWEN3_VL_MODELS:
            QWEN3_VL_MODELS[model_name]['model']['torch_dtype'] = torch_dtype
            QWEN3_VL_MODELS[model_name]['model']['attn_implementation'] = attn_implementation
        
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir, checkpoint=checkpoint)
        
        self.past_key_values = None
        self.video_seq_len = 0
        self.video_data = None

    def prefill_video_kv(self, video_data: dict, first_question: str = "Is there a person?"):
        """
        Encode only the video part of the prompt and cache the KV states.
        
        Args:
            video_data: Processed video data (from load_images)
            first_question: A representative question used to set up the prompt template
        """
        self.model.eval()
        self.video_data = video_data
        
        q_formatted = "{} Please answer with only Yes or No.".format(first_question)
        messages = [
            {
                "role": "user",
                "content": [video_data, {"type": "text", "text": q_formatted}]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Identify the end of vision tokens
        ref_ids = inputs.input_ids[0]
        try:
            # 151653 is usually the <|vision_end|> token
            boundary_idx = (ref_ids == 151653).nonzero(as_tuple=True)[0][-1].item() + 1
        except:
             boundary_idx = 0
             
        self.video_seq_len = boundary_idx
        
        # Prefill up to boundary
        forward_kwargs = {
            "input_ids": inputs.input_ids[:, :boundary_idx],
            "attention_mask": inputs.attention_mask[:, :boundary_idx],
            "use_cache": True,
            "return_dict": True
        }
        
        for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if key in inputs:
                forward_kwargs[key] = inputs[key]
        
        # Calculate and set rope_deltas (critical for subsequent generation)
        with torch.inference_mode():
            _, rope_deltas = self.model.model.get_rope_index(
                inputs.input_ids[:, :boundary_idx],
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=inputs.attention_mask[:, :boundary_idx]
            )
            self.model.model.rope_deltas = rope_deltas
                
        with torch.inference_mode():
            outputs = self.model(**forward_kwargs)
            self.past_key_values = outputs.past_key_values

    def forward_kv(self,
        texts: List[str],
        question_template: str = "{} Please answer with only Yes or No.",
        answer_template: str = "Yes") -> torch.Tensor:
        """
        Compute alignment scores for multiple questions using the cached video KV states.
        
        Args:
            texts: List of text descriptions to check alignment with
            question_template: Template for formatting the question
            answer_template: Expected answer (default "Yes")
            
        Returns:
            Tensor of joint probabilities for the answer token(s)
        """
        if self.past_key_values is None:
             raise ValueError("KV Cache not pre-filled! Call prefill_video_kv first.")
        
        lm_probs = []
        answer_token_ids = self.processor.tokenizer.encode(answer_template, add_special_tokens=False)
        n_answer_tokens = len(answer_token_ids)

        for question_text in texts:
            q_formatted = question_template.format(question_text)
            messages = [
                {
                    "role": "user",
                    "content": [self.video_data, {"type": "text", "text": q_formatted}]
                }
            ]
            
            full_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Memory optimization: Discard pixel values
            for key in ["pixel_values", "pixel_values_videos"]:
                if key in full_inputs:
                     del full_inputs[key]
            
            full_inputs = full_inputs.to(self.device)
            
            # Gold Position IDs (Match baseline)
            with torch.inference_mode():
                vision_pos, _ = self.model.model.get_rope_index(
                    full_inputs.input_ids,
                    image_grid_thw=full_inputs.get("image_grid_thw"),
                    video_grid_thw=full_inputs.get("video_grid_thw"),
                    attention_mask=full_inputs.attention_mask
                )
                text_pos = full_inputs.attention_mask.long().cumsum(-1) - 1
                full_pos_ids = torch.cat([text_pos[None, ...], vision_pos], dim=0)
            
            suffix_pos_ids = full_pos_ids[:, :, self.video_seq_len:]
            
            # Prepare suffix inputs
            suffix_inputs = {k: v for k, v in full_inputs.items() if k not in ["pixel_values", "pixel_values_videos"]}
            suffix_inputs["input_ids"] = full_inputs.input_ids[:, self.video_seq_len:]
            suffix_inputs["attention_mask"] = full_inputs.attention_mask 
            total_len = full_inputs.input_ids.shape[1]
            
            current_kv = DynamicCache()
            
            # Robust copy logic using iteration
            if isinstance(self.past_key_values, (tuple, list)): 
                 for i, (k, v) in enumerate(self.past_key_values):
                     current_kv.update(k.clone(), v.clone(), layer_idx=i)
            elif hasattr(self.past_key_values, "key_cache"): 
                 for i, (k, v) in enumerate(zip(self.past_key_values.key_cache, self.past_key_values.value_cache)):
                     current_kv.update(k.clone(), v.clone(), layer_idx=i)
            elif hasattr(self.past_key_values, "layers"):
                 for i, layer in enumerate(self.past_key_values.layers):
                     current_kv.update(layer.keys.clone(), layer.values.clone(), layer_idx=i)
            else:
                 try:
                     for i, (k, v) in enumerate(self.past_key_values):
                         current_kv.update(k.clone(), v.clone(), layer_idx=i)
                 except Exception as e:
                     raise ValueError(f"Unknown cache type: {type(self.past_key_values)} - {str(e)}")
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **suffix_inputs,
                    past_key_values=current_kv,
                    position_ids=suffix_pos_ids,
                    cache_position=torch.arange(self.video_seq_len, total_len, device=self.device),
                    max_new_tokens=n_answer_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            generated_ids = outputs.sequences[0][suffix_inputs["input_ids"].shape[1]:]
            last_token_id = generated_ids[-1].item()
            special_ids = [
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.bos_token_id,
                self.processor.tokenizer.pad_token_id
            ]
            offset = 1 if last_token_id in special_ids else 0
            
            joint_prob = 1.0
            actual_n = min(n_answer_tokens, len(outputs.scores) - offset) if offset > 0 else n_answer_tokens
                
            for i in range(actual_n):
                pos = -(actual_n - i + offset)
                token_logits = outputs.scores[pos][0]
                probs = torch.nn.functional.softmax(token_logits, dim=-1)
                joint_prob *= probs[answer_token_ids[i]].item()
                
            lm_probs.append(joint_prob ** (1.0 / actual_n))
            
        return torch.tensor(lm_probs)

    def generate_answer_kv(self,
        question: str,
        max_new_tokens: int = 10) -> str:
        """
        Generate a text response using the cached video KV states.
        
        Args:
            question: Text prompt/question
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        if self.past_key_values is None:
             raise ValueError("KV Cache not pre-filled! Call prefill_video_kv first.")

        q_formatted = "{} Please answer with only Yes or No.".format(question)
        messages = [
            {
                "role": "user",
                "content": [self.video_data, {"type": "text", "text": q_formatted}]
            }
        ]
        
        full_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        suffix_input_ids = full_inputs.input_ids[:, self.video_seq_len:]
        total_len = self.video_seq_len + suffix_input_ids.shape[1]
        attention_mask = torch.ones((1, total_len), device=self.device)
        cache_position = torch.arange(self.video_seq_len, total_len, device=self.device)

        # Robust clone for generation
        current_kv = DynamicCache()
        if hasattr(self.past_key_values, "key_cache"):
            for i, (k, v) in enumerate(zip(self.past_key_values.key_cache, self.past_key_values.value_cache)):
                current_kv.update(k.clone(), v.clone(), layer_idx=i)
        elif isinstance(self.past_key_values, (tuple, list)):
            for i, (k, v) in enumerate(self.past_key_values):
                current_kv.update(k.clone(), v.clone(), layer_idx=i)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=suffix_input_ids,
                past_key_values=current_kv,
                attention_mask=attention_mask,
                cache_position=cache_position,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        return self.processor.decode(outputs[0], skip_special_tokens=True)
