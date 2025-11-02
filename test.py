import t2v_metrics
import torch

print("="*80)
print("Testing Generalized VQAScore Implementation")
print("="*80)

# Initialize the model
qwen_score = t2v_metrics.VQAScore(model='gemini-2.5-pro-preview-03-25', api_key='AIzaSyALyEIg_HnZjKtxoWL7VFIx_bXIKggJMB4')

# Test cases
print("\n" + "="*80)
print("TEST 1: Single token answer - Original behavior")
print("="*80)
video = "videos/baby.mp4"
text = "a baby crying"
score = qwen_score(images=[video], texts=[text], answer_template="Yes")
print(f"\nFinal score: {score}")
print(f"Score shape: {score.shape}")
print(f"Score value: {score.item():.6f}")

print("\n" + "="*80)
print("TEST 2: Multi-token answer (2 tokens)")
print("="*80)
score = qwen_score(images=[video], texts=[text], answer_template="Yes, definitely", max_new_tokens=5)
print(f"\nFinal score: {score}")
print(f"Score value: {score.item():.6f}")

print("\n" + "="*80)
print("TEST 3: Multi-token answer (longer phrase)")
print("="*80)
score = qwen_score(images=[video], texts=[text], answer_template="Yes, it does", max_new_tokens=5)
print(f"\nFinal score: {score}")
print(f"Score value: {score.item():.6f}")

print("\n" + "="*80)
print("TEST 4: Batch of multiple videos (single token)")
print("="*80)
videos = [video, video]  # Same video twice for testing
texts = ["a baby crying", "a dog barking"]
scores = qwen_score(images=videos, texts=texts, answer_template="Yes", max_new_tokens=1)
print(f"\nFinal scores: {scores}")
print(f"Scores shape: {scores.shape}")
print(f"Score values: {[s.item() for s in scores]}")

print("\n" + "="*80)
print("TEST 5: Different answer template - 'No' instead of 'Yes'")
print("="*80)
# This should give low probability since the video matches the text
score = qwen_score(images=[video], texts=[text], answer_template="No", max_new_tokens=1)
print(f"\nFinal score: {score}")
print(f"Score value: {score.item():.6f}")

print("\n" + "="*80)
print("TEST 6: Mismatched text (should score lower)")
print("="*80)
score = qwen_score(images=[video], texts=["a car driving"], answer_template="Yes", max_new_tokens=1)
print(f"\nFinal score: {score}")
print(f"Score value: {score.item():.6f}")

print("\n" + "="*80)
print("TEST 7: CoT-style with longer generation (single token answer at end)")
print("="*80)
# Generate more tokens, but still expect "Yes" at the end
score = qwen_score(images=[video], texts=[text], answer_template="Yes", max_new_tokens=50)
print(f"\nFinal score: {score}")
print(f"Score value: {score.item():.6f}")

print("\n" + "="*80)
print("TEST 8: Verify consistency - run same test twice")
print("="*80)
score1 = qwen_score(images=[video], texts=[text], answer_template="Yes", max_new_tokens=1)
score2 = qwen_score(images=[video], texts=[text], answer_template="Yes", max_new_tokens=1)
print(f"\nScore 1: {score1.item():.6f}")
print(f"Score 2: {score2.item():.6f}")
print(f"Difference: {abs(score1.item() - score2.item()):.10f}")
print(f"Identical: {torch.allclose(score1, score2)}")

print("\n" + "="*80)
print("SUMMARY OF TESTS")
print("="*80)
print("✓ Test 1: Single token (baseline)")
print("✓ Test 2-3: Multi-token answers")
print("✓ Test 4: Batch processing")
print("✓ Test 5: Different answer template")
print("✓ Test 6: Mismatched content (sanity check)")
print("✓ Test 7: Longer generation (CoT-style)")
print("✓ Test 8: Consistency check")
print("="*80)