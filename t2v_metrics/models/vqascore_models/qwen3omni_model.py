import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
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
                answer_template: str = "Yes") -> torch.Tensor:
        """
        Calculate alignment scores using the probability of the answer token.
        
        Args:
            paths: List of image/video file paths
            texts: List of text descriptions to check alignment with
            audio_paths: Optional list of audio file paths
            fps: Frames per second (unused in Qwen3-Omni)
            question_template: Template for formatting the question
            answer_template: Expected answer (default "Yes")
            
        Returns:
            Tensor of probabilities for the answer token
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
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video
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