# Regularization Set Guide

> Source: DeepResearchReport.md Section 6

Regularization images help prevent **concept collapse** (also called "sameface" or "overfit to subject"). They teach the model what a generic person looks like, so it doesn't forget and make everyone look like your subject.

---

## Why Use Regularization?

| Problem | Without Reg | With Reg |
|---------|-------------|----------|
| Identity strength | May be weak | Strong |
| Concept collapse | High risk | Low risk |
| Prompt flexibility | Poor | Good |
| Training time | Shorter | Longer |

**Recommendation from report**: Use regularization for the final/production profile, especially with higher rank LoRAs.

---

## Directory Structure

Place regularization images in:

```
/workspace/lora_training/data/reg/
├── images/
│   ├── person001.jpg
│   ├── person002.jpg
│   ├── person003.png
│   └── ...
└── captions/
    ├── person001.txt
    ├── person002.txt
    ├── person003.txt
    └── ...
```

Or alongside each other:
```
/workspace/lora_training/data/reg/
├── person001.jpg
├── person001.txt
├── person002.jpg
├── person002.txt
└── ...
```

---

## Image Requirements

### Count
- **Minimum**: 20 images
- **Recommended**: 30-50 images

### Content
- Generic people (not your subject)
- Diverse demographics
- Various ages, ethnicities, genders
- Different angles (front, profile, ¾)
- Various backgrounds

### Quality
- High resolution (512px minimum)
- Good lighting
- Clear faces
- No watermarks or text

---

## Caption Rules

**CRITICAL**: Regularization captions must NOT contain your trigger token.

### Good Captions
```
a person
a man
a woman
a person standing outdoors
a portrait of a person
a man wearing casual clothes
a woman with neutral expression
```

### Bad Captions (DO NOT USE)
```
<trigger_token>, a person          # NO! Contains trigger token
sks person                          # NO! Contains trigger token
john_doe, standing outdoors         # NO! Contains trigger token
```

---

## Where to Get Regularization Images

### Option 1: Public Datasets
- FFHQ (Flickr-Faces-HQ)
- CelebA-HQ
- UTKFace

### Option 2: Generate Them
Use the base FLUX model to generate generic people:
```
a portrait of a person, neutral background
a man, professional headshot
a woman, casual portrait
```

### Option 3: Stock Photos
Use royalty-free stock photos (check licenses).

---

## Enabling Regularization

Set the environment variable before training:

```bash
USE_REG=1 bash scripts/train_flux_final.sh
```

Or set in your environment:
```bash
export USE_REG=1
export REG_DIR=/workspace/lora_training/data/reg
```

---

## Caption Script Example

Quick script to create generic captions:

```bash
#!/bin/bash
# create_reg_captions.sh
# Run in your reg/images directory

for img in *.jpg *.png *.jpeg; do
    [ -f "$img" ] || continue
    base="${img%.*}"
    echo "a person" > "${base}.txt"
done
```

Or with variety:
```bash
#!/bin/bash
CAPTIONS=(
    "a person"
    "a portrait of a person"
    "a man"
    "a woman"
    "a person, neutral expression"
)

i=0
for img in *.jpg *.png *.jpeg; do
    [ -f "$img" ] || continue
    base="${img%.*}"
    echo "${CAPTIONS[$((i % ${#CAPTIONS[@]}))]}" > "${base}.txt"
    ((i++))
done
```

---

## Troubleshooting

### Identity too weak
- Reduce regularization count
- Lower `prior_loss_weight` (default 1.0, try 0.5)

### Still getting sameface
- Increase regularization count
- Check that reg captions don't contain trigger token
- Add more diverse reg images

### Training too slow
- Regularization does increase training time
- Consider using reg only for final profile, not fast iteration
