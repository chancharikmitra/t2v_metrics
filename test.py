import t2v_metrics

### For a single (video, text) pair:
qwen_score = t2v_metrics.VQAScore(model='qwen2.5-vl-7b') 
video = "videos/baby.mp4"
text = "a baby crying"
score = qwen_score(images=[video], texts=[text]) 
print(score)

