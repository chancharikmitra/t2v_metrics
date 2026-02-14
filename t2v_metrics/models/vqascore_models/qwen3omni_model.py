import torch
import numpy as np
from PIL import Image
from typing import  List, Union, Tuple, Dict, Optional
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from .vqa_model import VQAScoreModel

QWEN3_OMNI_MODELS = {
    'qwen3-omni-30b-a3b-captioner': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Captioner',
        },
        'model': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Captioner',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-omni-30b-a3b': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen3-omni-30b-a3b-thinking': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Thinking',
        },
        'model': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
}

class Qwen3OmniModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    allows_audio = True
    
    def __init__(self,
                 model_name='qwen3-omni-30b-a3b-instruct',
                 device='cuda',
                 cache_dir=None,
                 checkpoint=None,
                 use_audio_in_video=True):
        assert model_name in QWEN3_OMNI_MODELS, f"Model {model_name} not found in QWEN3_OMNI_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN3_OMNI_MODELS[model_name]
        self.checkpoint = checkpoint if checkpoint else self.model_info['model']['path']
        self.use_audio_in_video = use_audio_in_video
        self.load_model()

    def load_model(self):
        model_path = self.checkpoint
        
        # Load model using Qwen3OmniMoeForConditionalGeneration
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.model_info['model']['torch_dtype'],
            attn_implementation=self.model_info['model']['attn_implementation'],
            device_map="auto"
        )
        
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_info['tokenizer']['path'])
        self.model.eval()

        self.device = next(self.model.parameters()).device

    def load_media(self, paths: List[str], audio_paths: Optional[List[str]] = None) -> List[dict]:
        """Load images, videos, and optionally audio files"""
        processed_data = []
        
        for i, path in enumerate(paths):
            content = []
            
            # Add image/video
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                content.append({"type": "video", "video": path})
            elif path.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    content.append({"type": "image", "image": image})
                elif np_array.ndim == 4:  # Multiple frames
                    # Use first frame as image
                    image = Image.fromarray(np_array[0].astype('uint8'), 'RGB')
                    content.append({"type": "image", "image": image})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                content.append({"type": "image", "image": image})
            
            # Add audio if provided
            if audio_paths and i < len(audio_paths) and audio_paths[i]:
                content.append({"type": "audio", "audio": audio_paths[i]})
            
            processed_data.append(content)
                
        return processed_data

    def forward(self,
            paths: List[str],
            texts: List[str],
            audio_paths: Optional[List[str]] = None,
            fps=None,
            question_template: str = "Does this show \"{}\"?",
            answer_template: str = "Yes",
            max_new_tokens: int = 1) -> torch.Tensor:
        """
        Calculate alignment scores using the probability of the answer token(s).
        
        Args:
            paths: List of image/video file paths
            texts: List of text descriptions to check alignment with
            audio_paths: Optional list of audio file paths
            fps: Frames per second (unused in Qwen3-Omni)
            question_template: Template for formatting the question
            answer_template: Expected answer (default "Yes")
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Tensor of probabilities for the answer token(s)
        """
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_media(paths, audio_paths)
        
        lm_probs = []
        for content, question, answer in zip(processed_data, questions, answers):
            conversation = [
                {
                    "role": "user",
                    "content": content + [{"type": "text", "text": question}]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=self.use_audio_in_video
            )
            inputs = inputs.to(self.device).to(self.model.dtype)
            
            # Tokenize the answer template
            answer_token_ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            n_answer_tokens = len(answer_token_ids)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video
                )
            
            # Extract generated tokens
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            
            # CHECK: Make sure last generated token is not a special token
            last_token_id = generated_ids[-1].item()
            special_token_ids = [
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.bos_token_id,
                self.processor.tokenizer.pad_token_id
            ]
            
            offset = 0
            if last_token_id in special_token_ids:
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
            
            # Extract probability for last n_answer_tokens (geometric mean)
            joint_prob = 1.0
            for i in range(n_answer_tokens):
                position = -(n_answer_tokens - i + offset)
                token_logits = outputs.scores[position][0]
                token_probs_dist = torch.nn.functional.softmax(token_logits, dim=-1)
                
                expected_token_id = answer_token_ids[i]
                token_prob = token_probs_dist[expected_token_id].item()
                joint_prob *= token_prob
            
            geometric_mean_prob = joint_prob ** (1.0 / n_answer_tokens)
            lm_probs.append(geometric_mean_prob)
            
        return torch.tensor(lm_probs)


    def forward_with_trace(self,
                        paths: List[str],
                        texts: List[str],
                        audio_paths: Optional[List[str]] = None,
                        fps=None,
                        question_template: str = "Does this show \"{}\"?",
                        answer_template: str = "Yes",
                        max_new_tokens: int = 1) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Calculate alignment scores with detailed trace information for debugging.
        
        Args:
            paths: List of image/video file paths
            texts: List of text descriptions to check alignment with
            audio_paths: Optional list of audio file paths
            fps: Frames per second (unused in Qwen3-Omni)
            question_template: Template for formatting the question
            answer_template: Expected answer (default "Yes")
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Tuple of (scores tensor, list of trace dictionaries)
        """
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_media(paths, audio_paths)
        
        lm_probs = []
        traces = []
        
        for idx, (content, question, answer) in enumerate(zip(processed_data, questions, answers)):
            print(f"\n{'='*60}")
            print(f"Sample {idx + 1}/{len(paths)}")
            print(f"Path: {paths[idx]}")
            print(f"Text: {texts[idx]}")
            
            conversation = [
                {
                    "role": "user",
                    "content": content + [{"type": "text", "text": question}]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=self.use_audio_in_video
            )
            inputs = inputs.to(self.device).to(self.model.dtype)
            
            # Tokenize the answer template
            answer_token_ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            n_answer_tokens = len(answer_token_ids)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video
                )
            
            # Extract generated tokens
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nGenerated output:")
            print(f"  {generated_text}")
            
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
                print(f"  Note: Last token is {special_name}, adjusting scoring")
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
                scored_token_ids = generated_ids[-n_answer_tokens:].tolist()
            
            scored_indices = list(range(len(generated_ids) - n_answer_tokens - offset, len(generated_ids) - offset))
            scored_tokens_text = self.processor.tokenizer.decode(scored_token_ids, skip_special_tokens=True)
            
            print(f"\nScoring token(s): '{scored_tokens_text}'")
            print(f"  Token indices in generated sequence: {scored_indices}")
            
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
                print(f"\n  Position {position} in outputs.scores (token index {scored_indices[i]} in sequence):")
                print(f"    Answer Template token ID: {expected_token_id}")
                print(f"    Answer Template token text: '{self.processor.tokenizer.decode([expected_token_id])}'")
                print(f"    P(Answer Template): {token_prob:.6f}")
                print(f"\n    Top 5 alternatives:")
                
                top_probs, top_indices = torch.topk(token_probs_dist, 5)
                for rank, (prob, token_id) in enumerate(zip(top_probs, top_indices), 1):
                    token_id_int = token_id.item()
                    token_text = self.processor.tokenizer.decode([token_id_int])
                    is_expected = "✓" if token_id_int == expected_token_id else " "
                    print(f"      {rank}. ID={token_id_int:6d} | P={prob.item():.6f} | Text='{token_text}' {is_expected}")
            
            geometric_mean_prob = joint_prob ** (1.0 / n_answer_tokens)
            print(f"\nJoint probability: {joint_prob:.6f}")
            print(f"Geometric mean probability: {geometric_mean_prob:.6f}")
            
            # Store minimal trace info
            trace = {
                'generated_text': generated_text,
                'generated_length': len(generated_ids),
                'scored_indices': scored_indices,
                'scored_tokens_text': scored_tokens_text,
                'joint_probability': joint_prob,
                'geometric_mean_probability': geometric_mean_prob
            }
            
            lm_probs.append(geometric_mean_prob)
            traces.append(trace)
        
        print(f"\n{'='*60}")
        print(f"Final scores: {lm_probs}")
        
        return torch.tensor(lm_probs), traces
    
    def generate(self,
                images: List[str],
                texts: List[str],
                audio_paths: Optional[List[str]] = None,
                fps=None,
                max_new_tokens: int = 256,
                speaker: str = "Ethan",
                save_audio_path: Optional[str] = None) -> Union[List[str], tuple]:
        """
        Generate text (and optionally audio) responses for given inputs.
        
        Args:
            images: List of image/video file paths
            texts: List of text prompts/questions
            audio_paths: Optional list of audio file paths
            fps: Frames per second (unused)
            max_new_tokens: Maximum number of tokens to generate
            speaker: Speaker voice for audio generation (default "Ethan")
            save_audio_path: If provided, save generated audio to this path
            
        Returns:
            List of generated text responses, or tuple of (texts, audios) if audio generated
        """
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        processed_data = self.load_media(images, audio_paths)
        
        generated_texts = []
        generated_audios = []
        
        for content, text in zip(processed_data, texts):
            conversation = [
                {
                    "role": "user",
                    "content": content + [{"type": "text", "text": text}]
                }
            ]
            
            # Prepare inputs
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, imgs, videos = process_mm_info(conversation, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text_prompt,
                audio=audios,
                images=imgs,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=self.use_audio_in_video
            )
            inputs = inputs.to(self.device).to(self.model.dtype)
            
            with torch.inference_mode():
                text_ids, audio = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    speaker=speaker,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video
                )
                
                output_text = self.processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()
                
                generated_texts.append(output_text)
                
                if audio is not None:
                    audio_data = audio.reshape(-1).detach().cpu().numpy()
                    generated_audios.append(audio_data)
                    
                    # Save audio if path provided
                    if save_audio_path:
                        sf.write(save_audio_path, audio_data, samplerate=24000)
        
        # Return texts only if no audio generated, otherwise return both
        if generated_audios and any(a is not None for a in generated_audios):
            return generated_texts, generated_audios
        return generated_texts