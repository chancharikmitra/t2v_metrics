import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Dict
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
from .vqa_model import VQAScoreModel

QWEN2_VL_MODELS = {
    # Qwen2_VL
    'qwen2-vl-2b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-2B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2-VL-2B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },
    'qwen2-vl-7b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },
    'qwen2-vl-72b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-72B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2-VL-72B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },

    # Qwen2.5_VL:
    'qwen2.5-vl-3b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-3B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },
    'qwen2.5-vl-7b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },
    'qwen2.5-vl-32b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-32B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-32B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },
    'qwen2.5-vl-72b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-72B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-72B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    }
}

class Qwen2VLModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='qwen2.5-vl-7b',
                 device='cuda',
                 cache_dir=None,
                 checkpoint=None):
        assert model_name in QWEN2_VL_MODELS, f"Model {model_name} not found in QWEN2_VL_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN2_VL_MODELS[model_name]
        self.checkpoint = checkpoint if checkpoint else self.model_info['model']['path']
        self.load_model()

    def load_model(self):
        # Switch from model dictionary to checkpoint argument
        # model_path = self.model_info['model']['path']
        print('When loading a qwen model, ensure that your model_name or checkpoint contains "qwen2.5". Otherwise, it will be loaded using the "qwen2" config and architecture.')
        model_path = self.checkpoint
        if 'qwen2.5' in model_path or 'qwen2.5' in self.model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info['model']['torch_dtype'],
                attn_implementation=self.model_info['model']['attn_implementation'],
                device_map="auto"
            )    
        
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info['model']['torch_dtype'],
                attn_implementation=self.model_info['model']['attn_implementation'],
                device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(self.model_info['tokenizer']['path'])
        self.model.eval()

        self.device = next(self.model.parameters()).device # If there are multiple GPUs put the model on the first parameters GPU

    def load_images(self, paths: List[str], fps: float = None) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        fps = fps if fps is not None else self.model_info.get('fps', 8.0)
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file path
                # video_frames = self.load_video(path, num_frames)
                if fps == "dynamic":
                    processed_data.append({"type": "video", "video": path, "max_pixels": 360*420})
                else:
                    processed_data.append({"type": "video", "video": path, "max_pixels": 360*420, "fps":fps})
            elif path.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    processed_data.append({"type": "image", "image": image})
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [Image.fromarray(frame.astype('uint8'), 'RGB') for frame in np_array]
                    processed_data.append({"type": "video", "video": frames})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                processed_data.append({"type": "image", "image": image})
        return processed_data
    # Qwen2.5-vl forward method
    def forward(self,
            images: List[str],
            texts: List[str],
            fps=None,
            question_template: str = "Does this image show \"{}\"? Answer the question with Yes or No",
            answer_template: str = "Yes",
            max_new_tokens: int = 1) -> torch.Tensor:
    
        assert len(images) == len(texts), "Number of images/videos and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(images, fps)
        
        lm_probs = []
        for idx, (data, question, answer) in enumerate(zip(processed_data, questions, answers)):
            # print(f"\n{'='*60}")
            # print(f"Sample {idx + 1}/{len(images)}")
            # print(f"Path: {images[idx]}")
            # print(f"Text: {texts[idx]}")
            
            messages = [{"role": "user", "content": [data, {"type": "text", "text": question}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
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
                       question_template: str = "Does this image show \"{}\"? Answer the question with Yes or No",
                       answer_template: str = "Yes",
                       max_new_tokens: int = 1) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Calculate alignment scores with detailed trace information for debugging.
        
        Args:
            images: List of image/video file paths
            texts: List of text descriptions to check alignment with
            fps: Frames per second for video processing
            question_template: Template for formatting the question
            answer_template: Expected answer (default "Yes")
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Tuple of (scores tensor, list of trace dictionaries)
        """
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(images, fps)
        
        lm_probs = []
        traces = []
        
        for idx, (data, question, answer) in enumerate(zip(processed_data, questions, answers)):
            # print(f"\n{'='*60}")
            # print(f"Sample {idx + 1}/{len(images)}")
            # print(f"Path: {images[idx]}")
            # print(f"Text: {texts[idx]}")
            
            messages = [{"role": "user", "content": [data, {"type": "text", "text": question}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
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
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        processed_data = self.load_images(images, fps)
        
        generated_texts = []
        for data, text in zip(processed_data, texts):
            messages = [{"role": "user", "content": [data, {"type": "text", "text": text}]}]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                generated_texts.append(text)
                
        return generated_texts