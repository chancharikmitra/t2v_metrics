"""
Test script for Gemini VQAScore and generation via Vertex AI.
Also verifies whether temperature has any effect on logprob-based scores.

Usage:
    python test_gemini.py
"""

import traceback
from t2v_metrics.models.vqascore_models.gemini_model import GeminiModel

VIDEO  = "videos/baby.mp4"
TEXTS  = [
    "a baby crying",       # should score high
    "a dog playing fetch", # should score low
]
PROJECT_ID = "PROJECT_ID" # Just use global for region.
MODELS = ["gemini-2.5-flash", "gemini-3-pro-preview"]
TEMPERATURES = [0.0, 0.5, 1.0]

# Explicit yes/no question template to ensure the model answers Yes or No
QUESTION_TEMPLATE = 'Does this video show "{}"? Please answer yes or no.'


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def test_vqascore(model: GeminiModel):
    print(f"\n--- VQAScore ---")
    try:
        # Single pair
        score = model.forward(
            images=[VIDEO],
            texts=[TEXTS[0]],
            question_template=QUESTION_TEMPLATE,
        )
        print(f"Single pair | '{TEXTS[0]}' => {score.item():.6f}")
    except Exception:
        traceback.print_exc()

    try:
        # 1 video x N texts — duplicate video to match length
        scores = model.forward(
            images=[VIDEO] * len(TEXTS),
            texts=TEXTS,
            question_template=QUESTION_TEMPLATE,
        )
        print(f"\n1 video x {len(TEXTS)} texts:")
        for text, s in zip(TEXTS, scores):
            print(f"  '{text}': {s.item():.6f}")
    except Exception:
        traceback.print_exc()


def test_temperature(model: GeminiModel):
    print(f"\n--- Temperature effect on P(Yes) ---")
    print(f"  (Identical scores = logprobs are temperature-agnostic on Google's end)")
    for temp in TEMPERATURES:
        try:
            score = model.forward(
                images=[VIDEO],
                texts=[TEXTS[0]],
                question_template=QUESTION_TEMPLATE,
                temperature=temp,
            )
            print(f"  temperature={temp:.1f}  =>  P(Yes) = {score.item():.6f}")
        except Exception:
            traceback.print_exc()


def test_generate(model: GeminiModel):
    print(f"\n--- Generation ---")
    prompts = [
        "Describe what is happening in this video in one sentence.",
        "What is the emotional tone of this video?",
    ]
    try:
        responses = model.generate(
            images=[VIDEO, VIDEO],
            texts=prompts,
            temperature=0.7,
        )
        for prompt, response in zip(prompts, responses):
            print(f"\n  Prompt  : {prompt}")
            print(f"  Response: {response}")
    except Exception:
        traceback.print_exc()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    import os
    assert os.path.exists(VIDEO), f"Video not found: {VIDEO!r}."

    for model_name in MODELS:
        print(f"\n{'#'*60}")
        print(f"# {model_name}")
        print(f"{'#'*60}")

        try:
            model = GeminiModel(
                model_name=model_name,
                project_id=PROJECT_ID,
                location="global",
                logprobs=5,
            )
        except Exception:
            print("FAILED to initialize model:")
            traceback.print_exc()
            continue

        test_vqascore(model)
        test_temperature(model)
        test_generate(model)


if __name__ == "__main__":
    main()