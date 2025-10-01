import torch
import numpy as np
from PIL import Image
from typing import List, Union
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
    # Add other Qwen3-VL model variants here as they become available
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
                paths: List[str],
                texts: List[str],
                fps=None,
                question_template: str = "Does this image show \"{}\"?",
                answer_template: str = "Yes") -> torch.Tensor:
        """
        Calculate alignment scores using the probability of the answer token.
        
        Args:
            paths: List of image/video file paths
            texts: List of text descriptions to check alignment with
            fps: Frames per second for video processing (unused in Qwen3-VL)
            question_template: Template for formatting the question
            answer_template: Expected answer (default "Yes")
            
        Returns:
            Tensor of probabilities for the answer token
        """
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(paths)
        
        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
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
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            # Extract probability of the answer token
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            
            # Get token ID for the answer
            ans_token_id = self.processor.tokenizer.encode(answer, add_special_tokens=False)[0]
            lm_prob = probs[0, ans_token_id].item()
            lm_probs.append(lm_prob)
            
        return torch.tensor(lm_probs)
    
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