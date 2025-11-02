import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Dict
from transformers import AutoModelForImageTextToText, AutoProcessor
from .vqa_model import VQAScoreModel

QWEN3_VL_MODELS = {
    'qwen3-vl-235b-a22b': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-235B-A22B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-235B-A22B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-235b-a22b-thinking': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-235B-A22B-Thinking',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-235B-A22B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-30b-a3b': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-30b-a3b-thinking': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-30B-A3B-Thinking',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-30B-A3B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-32b': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-32B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-32B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-32b-thinking': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-32B-Thinking',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-32B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-8b': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-8B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-8B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-8b-thinking': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-8B-Thinking',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-8B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-4b': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-4B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-4B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-4b-thinking': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-4B-Thinking',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-4B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-2b': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-2B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-2B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-vl-2b-thinking': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-VL-2B-Thinking',
        },
        'model': {
            'path': 'Qwen/Qwen3-VL-2B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
}


class Qwen3VLModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    
    def __init__(self,
                 model_name='qwen3-vl-235b-a22b',
                 device='cuda',
                 cache_dir=None,
                 checkpoint=None):
        assert model_name in QWEN3_VL_MODELS, f"Model {model_name} not found in QWEN3_VL_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN3_VL_MODELS[model_name]
        self.checkpoint = checkpoint if checkpoint else self.model_info['model']['path']
        self.load_model()

    def load_model(self):
        model_path = self.checkpoint
        
        # Load model using AutoModelForImageTextToText for Qwen3-VL
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=self.model_info['model']['torch_dtype'],
            attn_implementation=self.model_info['model']['attn_implementation'],
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_info['tokenizer']['path'])
        self.model.eval()

        self.device = next(self.model.parameters()).device

    def load_images(self, paths: List[str]) -> List[dict]:
        """Load images or videos and return them in Qwen3-VL format"""
        processed_data = []
        
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                processed_data.append({"type": "video", "video": path})
            elif path.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    processed_data.append({"type": "image", "image": image})
                elif np_array.ndim == 4:  # Multiple frames (treat as video)
                    # For Qwen3-VL, we need to save frames as a temporary video or process differently
                    # For simplicity, we'll process the first frame as an image
                    # In production, you might want to save this as a temporary video file
                    frames = [Image.fromarray(frame.astype('uint8'), 'RGB') for frame in np_array]
                    # Use first frame for now - consider video handling improvement
                    processed_data.append({"type": "image", "image": frames[0]})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                processed_data.append({"type": "image", "image": image})
                
        return processed_data

    def forward(self,
        images: List[str],
        texts: List[str],
        fps=None,
        question_template: str = "Does this image show \"{}\"?",
        answer_template: str = "Yes",
        max_new_tokens: int = 1) -> torch.Tensor:
        """
        Calculate alignment scores using the probability of the answer token(s).
        
        Args:
            images: List of image/video file paths
            texts: List of text descriptions to check alignment with
            fps: Frames per second for video processing (unused in Qwen3-VL)
            question_template: Template for formatting the question
            answer_template: Expected answer (default "Yes")
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Tensor of joint probabilities for the answer token(s)
        """
        assert len(images) == len(texts), "Number of images/videos and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(images)
        
        lm_probs = []
        
        for idx, (data, question, answer) in enumerate(zip(processed_data, questions, answers)):
            # print(f"\n{'='*60}")
            # print(f"Sample {idx + 1}/{len(images)}")
            # print(f"Path: {images[idx]}")
            # print(f"Text: {texts[idx]}")
            
            messages = [
                {
                    "role": "user",
                    "content": [data, {"type": "text", "text": question}]
                }
            ]
            
            # Prepare inputs using processor
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Tokenize the answer template
            answer_token_ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            n_answer_tokens = len(answer_token_ids)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Extract generated tokens
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # print(f"\nGenerated output:")
            # print(f"  {generated_text}")
            
            # CHECK: Make sure last generated token is not a special token
            last_token_id = generated_ids[-1].item()
            special_token_ids = [
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.bos_token_id,
                self.processor.tokenizer.pad_token_id
            ]
            
            offset = 0
            if last_token_id in special_token_ids:
                special_name = "EOS" if last_token_id == self.processor.tokenizer.eos_token_id else \
                            "BOS" if last_token_id == self.processor.tokenizer.bos_token_id else "PAD"
                # print(f"  Note: Last token is {special_name}, adjusting scoring")
                # Remove the special token from consideration
                n_answer_tokens = min(n_answer_tokens, len(outputs.scores) - 1)
                offset = 1
                if n_answer_tokens <= 0:
                    raise ValueError("No content tokens to score after removing special tokens")
            
            # Check if we have enough tokens to score
            if len(outputs.scores) < n_answer_tokens:
                print(f"  Warning: Generated {len(outputs.scores)} tokens but need {n_answer_tokens}, adjusting")
                n_answer_tokens = len(outputs.scores)
                answer_token_ids = answer_token_ids[:n_answer_tokens]
            
            # Get the indices and text of the scored tokens
            if offset > 0:
                scored_token_ids = generated_ids[-(n_answer_tokens + offset):-offset].tolist()
            else:
                scored_token_ids = generated_ids[-(n_answer_tokens):].tolist()
            
            scored_indices = list(range(len(generated_ids) - n_answer_tokens - offset, len(generated_ids) - offset))
            scored_tokens_text = self.processor.tokenizer.decode(scored_token_ids, skip_special_tokens=True)
            
            # print(f"\nScoring token(s): '{scored_tokens_text}'")
            # print(f"  Token indices in generated sequence: {scored_indices}")
            
            # Extract probability for last n_answer_tokens
            joint_prob = 1.0
            for i in range(n_answer_tokens):
                position = -(n_answer_tokens - i + offset)
                token_logits = outputs.scores[position][0]
                token_probs_dist = torch.nn.functional.softmax(token_logits, dim=-1)
                
                expected_token_id = answer_token_ids[i]
                token_prob = token_probs_dist[expected_token_id].item()
                joint_prob *= token_prob
                
                # Show top 5 alternatives at this position
                # print(f"\n  Position {position} in outputs.scores (token index {scored_indices[i]} in sequence):")
                # print(f"    Answer Template token ID: {expected_token_id}")
                # print(f"    Answer Template token text: '{self.processor.tokenizer.decode([expected_token_id])}'")
                # print(f"    P(answer_template): {token_prob:.6f}")
                # print(f"\n    Top 5 alternatives:")
                
                top_probs, top_indices = torch.topk(token_probs_dist, 5)
                for rank, (prob, token_id) in enumerate(zip(top_probs, top_indices), 1):
                    token_id_int = token_id.item()
                    token_text = self.processor.tokenizer.decode([token_id_int])
                    is_expected = "✓" if token_id_int == expected_token_id else " "
                    print(f"      {rank}. ID={token_id_int:6d} | P={prob.item():.6f} | Text='{token_text}' {is_expected}")
            
            # print(f"\nJoint probability: {joint_prob:.6f}")
            lm_probs.append(joint_prob)
        
        # print(f"\n{'='*60}")
        # print(f"Final scores: {lm_probs}")
        
        return torch.tensor(lm_probs)


    def forward_with_trace(self,
                    images: List[str],
                    texts: List[str],
                    fps=None,
                    question_template: str = "Does this image show \"{}\"?",
                    answer_template: str = "Yes",
                    max_new_tokens: int = 1) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Calculate alignment scores with detailed trace information for debugging.
        
        Args:
            images: List of image/video file paths
            texts: List of text descriptions to check alignment with
            fps: Frames per second for video processing (unused in Qwen3-VL)
            question_template: Template for formatting the question
            answer_template: Expected answer (default "Yes")
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Tuple of (scores tensor, list of trace dictionaries)
        """
        assert len(images) == len(texts), "Number of images/videos and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(images)
        
        lm_probs = []
        traces = []
        
        for idx, (data, question, answer) in enumerate(zip(processed_data, questions, answers)):
            # print(f"\n{'='*60}")
            # print(f"Sample {idx + 1}/{len(images)}")
            # print(f"Path: {images[idx]}")
            # print(f"Text: {texts[idx]}")
            
            messages = [
                {
                    "role": "user",
                    "content": [data, {"type": "text", "text": question}]
                }
            ]
            
            # Prepare inputs using processor
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Tokenize the answer template
            answer_token_ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            n_answer_tokens = len(answer_token_ids)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Extract generated tokens
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # print(f"\nGenerated output:")
            # print(f"  {generated_text}")
            
            # CHECK: Make sure last generated token is not a special token
            last_token_id = generated_ids[-1].item()
            special_token_ids = [
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.bos_token_id,
                self.processor.tokenizer.pad_token_id
            ]
            
            if last_token_id in special_token_ids:
                special_name = "EOS" if last_token_id == self.processor.tokenizer.eos_token_id else \
                            "BOS" if last_token_id == self.processor.tokenizer.bos_token_id else "PAD"
                # print(f"  Note: Last token is {special_name}, adjusting scoring")
                # Remove the special token from consideration
                n_answer_tokens = min(n_answer_tokens, len(outputs.scores) - 1)
                if n_answer_tokens <= 0:
                    raise ValueError("No content tokens to score after removing special tokens")
            
            # Check if we have enough tokens to score
            if len(outputs.scores) < n_answer_tokens:
                print(f"  Warning: Generated {len(outputs.scores)} tokens but need {n_answer_tokens}, adjusting")
                n_answer_tokens = len(outputs.scores)
                answer_token_ids = answer_token_ids[:n_answer_tokens]
            
            # Calculate offset to exclude special tokens if needed
            offset = 1 if last_token_id in special_token_ids else 0

            # Get the indices and text of the scored tokens (excluding special token if present)
            if offset > 0:
                scored_token_ids = generated_ids[-(n_answer_tokens + offset):-offset].tolist()
            else:
                scored_token_ids = generated_ids[-(n_answer_tokens):].tolist()

            scored_indices = list(range(len(generated_ids) - n_answer_tokens - offset, len(generated_ids) - offset))
            scored_tokens_text = self.processor.tokenizer.decode(scored_token_ids, skip_special_tokens=True)
            
            # print(f"\nScoring token(s): '{scored_tokens_text}'")
            # print(f"  Token indices in generated sequence: {scored_indices}")
            
            # Extract probability for last n_answer_tokens
            joint_prob = 1.0
            for i in range(n_answer_tokens):
                position = -(n_answer_tokens - i + offset)  # Adjusted for offset
                token_logits = outputs.scores[position][0]
                token_probs_dist = torch.nn.functional.softmax(token_logits, dim=-1)
                
                expected_token_id = answer_token_ids[i]
                token_prob = token_probs_dist[expected_token_id].item()
                joint_prob *= token_prob
                
                # Show top 5 alternatives at this position
                # print(f"\n  Position {position} in outputs.scores (token index {scored_indices[i]} in sequence):")
                # print(f"    Answer Template token ID: {expected_token_id}")
                # print(f"    Answer Template token text: '{self.processor.tokenizer.decode([expected_token_id])}'")
                # print(f"    P(Answer Template): {token_prob:.6f}")
                # print(f"\n    Top 5 alternatives:")
                
                top_probs, top_indices = torch.topk(token_probs_dist, 5)
                for rank, (prob, token_id) in enumerate(zip(top_probs, top_indices), 1):
                    token_id_int = token_id.item()
                    token_text = self.processor.tokenizer.decode([token_id_int])
                    is_expected = "✓" if token_id_int == expected_token_id else " "
                    # print(f"      {rank}. ID={token_id_int:6d} | P={prob.item():.6f} | Text='{token_text}' {is_expected}")
            
            # print(f"\nJoint probability: {joint_prob:.6f}")
            
            # Store minimal trace info
            trace = {
                'generated_text': generated_text,
                'generated_length': len(generated_ids),
                'scored_indices': scored_indices,
                'scored_tokens_text': scored_tokens_text,
                'probability': joint_prob
            }
            
            lm_probs.append(joint_prob)
            traces.append(trace)
        
        # print(f"\n{'='*60}")
        # print(f"Final scores: {lm_probs}")
        
        return torch.tensor(lm_probs), traces
    
    def generate(self,
                images: List[str],
                texts: List[str],
                fps=None,
                max_new_tokens: int = 256) -> List[str]:
        """
        Generate text responses for given images and text prompts.
        
        Args:
            images: List of image/video file paths
            texts: List of text prompts/questions
            fps: Frames per second (unused in Qwen3-VL)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of generated text responses
        """
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        processed_data = self.load_images(images)
        
        generated_texts = []
        for data, text in zip(processed_data, texts):
            messages = [
                {
                    "role": "user",
                    "content": [data, {"type": "text", "text": text}]
                }
            ]
            
            # Prepare inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0].strip()
                generated_texts.append(output_text)
                
        return generated_texts