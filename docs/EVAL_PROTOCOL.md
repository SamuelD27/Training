# Evaluation Protocol

> Source: DeepResearchReport.md Section 7

Systematic evaluation ensures your LoRA performs well across different conditions.

---

## Quick Evaluation Checklist

- [ ] Identity recognizable in samples?
- [ ] Works at different LoRA strengths?
- [ ] Works with varied prompts?
- [ ] No sameface/carbon copy?
- [ ] Good diversity across seeds?

---

## Fixed Prompt Set

Test your LoRA with these standardized prompts. Replace `<token>` with your trigger token.

### 1. Neutral Portrait
```
<token>, a portrait photo, neutral background, soft lighting, looking at camera
```
**Expected**: Clear identity, no artifacts

### 2. New Environment
```
<token>, standing in a modern office, professional setting, natural lighting
```
**Expected**: Identity preserved in new context

### 3. New Outfit
```
<token>, wearing a blue suit jacket, white shirt, professional attire
```
**Expected**: Identity preserved with different clothing

### 4. New Style (Cinematic)
```
<token>, cinematic portrait, dramatic lighting, shallow depth of field, film grain
```
**Expected**: Identity preserved with style transfer

### 5. Control (No Token)
```
a person, portrait photo, neutral background, soft lighting
```
**Expected**: Generic person, NOT your subject

---

## LoRA Strength Sweep

Test each prompt at multiple LoRA strengths:

| Strength | Expected Behavior |
|----------|-------------------|
| **0.5** | Subtle influence, identity may be weak |
| **0.8** | Good balance, identity present but flexible |
| **1.0** | Full strength, strong identity |
| **1.2** | Overstrength, may show artifacts |

### What to Look For

**0.5 Strength**
- Identity should be noticeable but not dominant
- Good for stylistic flexibility
- If identity invisible → LoRA too weak

**0.8 Strength**
- Should clearly represent subject
- Good balance of identity and prompt adherence
- Recommended starting point for use

**1.0 Strength**
- Strong identity representation
- May reduce prompt flexibility
- Good for portraits where identity is priority

**1.2 Strength**
- Tests robustness
- May show overfit artifacts (sameface, pose lock)
- If artifacts appear → consider more training diversity

---

## Identity Metrics

### Human Inspection (Primary)

Ask yourself:
1. Is the person recognizable?
2. Does it look like the same person across prompts?
3. Are there unnatural artifacts?
4. Is there pose/expression variety?

### Face Embedding Similarity (Optional)

Use ArcFace or similar for quantitative comparison:

```python
# Example with insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0)

# Get embeddings
ref_face = app.get(reference_image)[0]
gen_face = app.get(generated_image)[0]

# Calculate similarity
similarity = np.dot(ref_face.embedding, gen_face.embedding)
print(f"Similarity: {similarity:.3f}")  # Higher = more similar
```

**Targets**:
- > 0.6: Good identity match
- > 0.7: Strong identity match
- < 0.5: Weak identity

### Diversity Score

Generate 4+ images with same prompt, different seeds:
- Should show variation in pose/expression
- If all identical → overfit

---

## Overfit Detection

### Symptoms

| Symptom | Severity | Action |
|---------|----------|--------|
| Same pose every time | High | Reduce steps, add dropout |
| Carbon-copy images | High | Stop early, use regularization |
| Only works with specific prompts | Medium | Freeze TE, more caption variety |
| Loss drops but visuals don't improve | Medium | You've hit saturation, stop |
| Background elements from training appear | Low | More regularization |

### Quick Overfit Test

Generate 4 images with identical prompt but different seeds:

```
<token>, portrait photo, neutral background
Seeds: 42, 123, 456, 789
```

**Healthy LoRA**:
- Same person
- Different angles/expressions
- Natural variation

**Overfit LoRA**:
- Identical pose
- Same expression
- Carbon-copy look

---

## Evaluation Workflow

### After Fast Iteration (1500 steps)

1. Generate samples with fixed prompt set
2. Check identity presence at 0.8 and 1.0 strength
3. Run overfit test
4. **Decision**:
   - Identity good + no overfit → proceed to final
   - Identity weak → more images or higher rank
   - Already overfit → stop, use this checkpoint

### After Final Training (2500 steps)

1. Full fixed prompt set at all 4 strengths
2. Diversity test (4 seeds)
3. Compare checkpoints (every 500 steps)
4. Select best checkpoint (may not be final)

---

## Sample Generation During Training

Training scripts automatically generate samples every 250 steps using prompts from `configs/sample_prompts.txt`.

### Reviewing Training Samples

```bash
# List samples
ls output/samples/RUN_NAME/

# View in order
ls -la output/samples/RUN_NAME/ | sort -k9
```

### What to Watch For

**Early steps (0-500)**:
- Identity may be weak
- This is normal

**Mid steps (500-1500)**:
- Identity should emerge
- Watch for overfit signs

**Late steps (1500+)**:
- Identity should be strong
- If identical to mid-steps → saturation

---

## Recommended Tools

### Image Viewer
```bash
# On pod with X forwarding
feh output/samples/RUN_NAME/

# Download and view locally
rsync -avz root@POD:/workspace/lora_training/output/samples/ ./samples/
```

### Quick Comparison Grid
```bash
# Create montage (requires imagemagick)
montage output/samples/RUN_NAME/*.png -geometry 512x512+2+2 comparison.png
```

---

## Recording Results

Keep a log of your evaluations:

```markdown
## Run: flux_final_20241215

### Identity Strength
- 0.5: Weak but present
- 0.8: Good - clear recognition
- 1.0: Strong - primary use case
- 1.2: Slight overfit artifacts

### Prompt Adherence
- Neutral: Excellent
- New Environment: Good
- New Outfit: Good
- Cinematic: Good
- Control: Pass (shows generic person)

### Overfit Test
- 4 seeds: Good variation
- Slight pose preference for ¾ view

### Recommendation
Use checkpoint at step 2000 for best balance
```
