#!/usr/bin/env python3
"""
Dataset Analysis Script for Identity LoRA Training
==================================================

Analyzes a dataset and provides recommendations for training parameters.
Does NOT modify images - analysis only.

Usage:
    python scripts/analyze_dataset.py [--data-dir PATH] [--output PATH]
    python scripts/analyze_dataset.py --help

Output:
    - Console summary
    - JSON report at logs/dataset_report.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# Optional imports with graceful fallback
try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not installed. Some features disabled.")

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Constants
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
SUPPORTED_CAPTION_EXTENSIONS = {'.txt', '.caption'}

# Aspect ratio buckets (64px steps per sd-scripts)
BUCKET_STEP = 64


def find_images(data_dir: Path) -> list[Path]:
    """Find all supported image files in directory."""
    images = []

    # Check direct files
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        images.extend(data_dir.glob(f'*{ext}'))
        images.extend(data_dir.glob(f'*{ext.upper()}'))

    # Check images/ subdirectory
    images_subdir = data_dir / 'images'
    if images_subdir.exists():
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            images.extend(images_subdir.glob(f'*{ext}'))
            images.extend(images_subdir.glob(f'*{ext.upper()}'))

    return sorted(set(images))


def find_caption(image_path: Path, data_dir: Path) -> Optional[Path]:
    """Find caption file for an image."""
    stem = image_path.stem

    # Check alongside image
    for ext in SUPPORTED_CAPTION_EXTENSIONS:
        caption_path = image_path.with_suffix(ext)
        if caption_path.exists():
            return caption_path

    # Check captions/ subdirectory
    captions_dir = data_dir / 'captions'
    if captions_dir.exists():
        for ext in SUPPORTED_CAPTION_EXTENSIONS:
            caption_path = captions_dir / f'{stem}{ext}'
            if caption_path.exists():
                return caption_path

    return None


def get_image_info(image_path: Path) -> dict:
    """Extract information from an image file."""
    info = {
        'path': str(image_path),
        'name': image_path.name,
        'size_bytes': image_path.stat().st_size,
    }

    if not HAS_PIL:
        return info

    try:
        with Image.open(image_path) as img:
            info['width'] = img.width
            info['height'] = img.height
            info['aspect_ratio'] = round(img.width / img.height, 3)
            info['mode'] = img.mode
            info['format'] = img.format

            # Check EXIF for rotation
            info['rotation'] = 0
            try:
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        if tag == 'Orientation':
                            # Orientation values: 3=180, 6=270, 8=90
                            rotation_map = {3: 180, 6: 270, 8: 90}
                            info['rotation'] = rotation_map.get(value, 0)
                            break
            except (AttributeError, KeyError):
                pass

    except Exception as e:
        info['error'] = str(e)

    return info


def calculate_blur_score(image_path: Path) -> Optional[float]:
    """Calculate blur score using Laplacian variance (higher = sharper)."""
    if not HAS_CV2:
        return None

    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Resize for consistent scoring
        img = cv2.resize(img, (512, 512))

        # Laplacian variance
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return round(laplacian_var, 2)
    except Exception:
        return None


def calculate_perceptual_hash(image_path: Path) -> Optional[str]:
    """Calculate perceptual hash for duplicate detection."""
    if not HAS_IMAGEHASH or not HAS_PIL:
        return None

    try:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def find_duplicates(hashes: dict[str, list[str]], threshold: int = 5) -> list[list[str]]:
    """Find potential duplicates based on perceptual hash similarity."""
    if not HAS_IMAGEHASH:
        return []

    duplicates = []
    hash_list = list(hashes.items())

    for i, (hash1, paths1) in enumerate(hash_list):
        # Same hash = exact duplicates
        if len(paths1) > 1:
            duplicates.append(paths1)
            continue

        # Similar hashes (within threshold)
        for hash2, paths2 in hash_list[i+1:]:
            try:
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)
                if h1 - h2 <= threshold:
                    duplicates.append(paths1 + paths2)
            except Exception:
                pass

    return duplicates


def calculate_bucket_resolution(width: int, height: int, base_reso: int = 512) -> tuple[int, int]:
    """Calculate the bucket resolution for an image."""
    aspect = width / height

    # Calculate bucket dimensions maintaining aspect ratio
    if aspect >= 1:
        bucket_w = base_reso
        bucket_h = int(base_reso / aspect)
    else:
        bucket_h = base_reso
        bucket_w = int(base_reso * aspect)

    # Round to nearest bucket step
    bucket_w = max(BUCKET_STEP, (bucket_w // BUCKET_STEP) * BUCKET_STEP)
    bucket_h = max(BUCKET_STEP, (bucket_h // BUCKET_STEP) * BUCKET_STEP)

    return bucket_w, bucket_h


def analyze_captions(caption_paths: list[Path]) -> dict:
    """Analyze caption files for patterns and issues."""
    analysis = {
        'total': len(caption_paths),
        'missing': 0,
        'empty': 0,
        'has_trigger_token': 0,
        'avg_length': 0,
        'unique_captions': 0,
        'warnings': []
    }

    caption_texts = []
    word_counts = Counter()

    for path in caption_paths:
        if path is None:
            analysis['missing'] += 1
            continue

        try:
            text = path.read_text().strip()
            if not text:
                analysis['empty'] += 1
                continue

            caption_texts.append(text)
            words = text.lower().split()
            word_counts.update(words)

        except Exception:
            analysis['missing'] += 1

    if caption_texts:
        analysis['avg_length'] = round(sum(len(c) for c in caption_texts) / len(caption_texts), 1)
        analysis['unique_captions'] = len(set(caption_texts))

        # Check for repeated identical captions
        if analysis['unique_captions'] < len(caption_texts) * 0.5:
            analysis['warnings'].append("Many duplicate captions detected - add variety")

        # Check for potential trigger tokens (uncommon words appearing in most captions)
        total_captions = len(caption_texts)
        for word, count in word_counts.most_common(20):
            if count >= total_captions * 0.8 and len(word) > 3:
                analysis['has_trigger_token'] += 1
                break

    return analysis


def generate_recommendations(
    image_count: int,
    resolutions: list[tuple[int, int]],
    aspect_ratios: list[float],
    blur_scores: list[float],
    caption_analysis: dict
) -> dict:
    """Generate training recommendations based on dataset analysis."""

    recommendations = {
        'base_resolution': 512,
        'min_bucket_reso': 256,
        'max_bucket_reso': 768,
        'profile': 'fast',
        'warnings': [],
        'suggestions': []
    }

    # Image count recommendations
    if image_count < 10:
        recommendations['warnings'].append(f"Very few images ({image_count}). Minimum recommended: 15-30")
        recommendations['profile'] = 'fast'
    elif image_count < 15:
        recommendations['warnings'].append(f"Low image count ({image_count}). Consider adding more images")
    elif image_count >= 30:
        recommendations['suggestions'].append(f"Good dataset size ({image_count} images)")
        recommendations['profile'] = 'final'

    # Resolution recommendations
    if resolutions:
        min_width = min(w for w, h in resolutions)
        min_height = min(h for w, h in resolutions)
        max_width = max(w for w, h in resolutions)
        max_height = max(h for w, h in resolutions)

        min_dim = min(min_width, min_height)
        max_dim = max(max_width, max_height)

        # Base resolution
        if min_dim >= 1024:
            recommendations['base_resolution'] = 768
            recommendations['min_bucket_reso'] = 384
            recommendations['max_bucket_reso'] = 1024
            recommendations['suggestions'].append("High-res dataset - using 768 base resolution")
        elif min_dim >= 768:
            recommendations['base_resolution'] = 768
            recommendations['min_bucket_reso'] = 384
            recommendations['max_bucket_reso'] = 1024
        elif min_dim >= 512:
            recommendations['base_resolution'] = 512
            recommendations['min_bucket_reso'] = 256
            recommendations['max_bucket_reso'] = 768
        else:
            recommendations['base_resolution'] = 512
            recommendations['min_bucket_reso'] = 256
            recommendations['max_bucket_reso'] = 512
            recommendations['warnings'].append(f"Some images are low resolution ({min_dim}px min)")

    # Aspect ratio diversity
    if aspect_ratios:
        unique_ratios = len(set(round(ar, 1) for ar in aspect_ratios))
        if unique_ratios < 3:
            recommendations['warnings'].append("Low aspect ratio diversity - consider varied crops")

        # Check for extreme ratios
        extreme = [ar for ar in aspect_ratios if ar < 0.5 or ar > 2.0]
        if extreme:
            recommendations['warnings'].append(f"{len(extreme)} images have extreme aspect ratios")

    # Blur analysis
    if blur_scores:
        avg_blur = sum(blur_scores) / len(blur_scores)
        low_quality = [s for s in blur_scores if s < 50]

        if low_quality:
            recommendations['warnings'].append(f"{len(low_quality)} images may be blurry (low sharpness score)")

        if avg_blur < 100:
            recommendations['suggestions'].append("Consider adding sharper images for better detail")

    # Caption analysis
    if caption_analysis['missing'] > 0:
        recommendations['warnings'].append(f"{caption_analysis['missing']} images missing captions")

    if caption_analysis['empty'] > 0:
        recommendations['warnings'].append(f"{caption_analysis['empty']} captions are empty")

    if not caption_analysis.get('has_trigger_token'):
        recommendations['warnings'].append("No consistent trigger token detected in captions")

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description='Analyze dataset for Identity LoRA training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_dataset.py
  python scripts/analyze_dataset.py --data-dir /path/to/dataset
  python scripts/analyze_dataset.py --output custom_report.json
        """
    )

    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=Path('/workspace/lora_training/data/subject'),
        help='Path to dataset directory (default: /workspace/lora_training/data/subject)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('/workspace/lora_training/logs/dataset_report.json'),
        help='Path for output JSON report'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed per-image information'
    )

    args = parser.parse_args()

    # Normalize paths
    data_dir = args.data_dir.resolve() if args.data_dir.exists() else args.data_dir
    output_path = args.output

    print("=" * 60)
    print("Dataset Analysis for Identity LoRA Training")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")

    # Check if directory exists
    if not data_dir.exists():
        print(f"\n[ERROR] Dataset directory not found: {data_dir}")
        print("\nExpected structure:")
        print("  data/subject/")
        print("    images/")
        print("      image1.jpg")
        print("      image2.png")
        print("    captions/")
        print("      image1.txt")
        print("      image2.txt")
        print("\nOr:")
        print("  data/subject/")
        print("    image1.jpg")
        print("    image1.txt")
        print("    image2.png")
        print("    image2.txt")

        # Create minimal report
        report = {
            'status': 'error',
            'error': 'Dataset directory not found',
            'data_dir': str(data_dir)
        }

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        sys.exit(1)

    # Find images
    images = find_images(data_dir)

    if not images:
        print(f"\n[ERROR] No images found in {data_dir}")
        print("Supported formats:", ', '.join(SUPPORTED_IMAGE_EXTENSIONS))

        report = {
            'status': 'error',
            'error': 'No images found',
            'data_dir': str(data_dir)
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        sys.exit(1)

    print(f"\nFound {len(images)} images")
    print("-" * 40)

    # Analyze images
    image_infos = []
    resolutions = []
    aspect_ratios = []
    blur_scores = []
    hashes = defaultdict(list)
    rotated_images = []
    caption_paths = []

    for img_path in images:
        info = get_image_info(img_path)
        image_infos.append(info)

        if 'width' in info and 'height' in info:
            resolutions.append((info['width'], info['height']))
            aspect_ratios.append(info['aspect_ratio'])

        if info.get('rotation', 0) != 0:
            rotated_images.append(info['name'])

        # Blur score
        blur = calculate_blur_score(img_path)
        if blur is not None:
            info['blur_score'] = blur
            blur_scores.append(blur)

        # Perceptual hash
        phash = calculate_perceptual_hash(img_path)
        if phash:
            info['phash'] = phash
            hashes[phash].append(info['name'])

        # Caption
        caption_path = find_caption(img_path, data_dir)
        caption_paths.append(caption_path)
        info['has_caption'] = caption_path is not None

        if args.verbose:
            status = "+" if info.get('has_caption') else "-"
            blur_str = f" blur={info.get('blur_score', 'N/A')}" if 'blur_score' in info else ""
            print(f"  [{status}] {info['name']} {info.get('width', '?')}x{info.get('height', '?')}{blur_str}")

    # Find duplicates
    duplicates = find_duplicates(hashes)

    # Analyze captions
    caption_analysis = analyze_captions(caption_paths)

    # Resolution statistics
    if resolutions:
        widths = [w for w, h in resolutions]
        heights = [h for w, h in resolutions]

        print("\nResolution Statistics:")
        print(f"  Min: {min(widths)}x{min(heights)}")
        print(f"  Max: {max(widths)}x{max(heights)}")
        print(f"  Avg: {int(sum(widths)/len(widths))}x{int(sum(heights)/len(heights))}")

    # Aspect ratio distribution
    if aspect_ratios:
        ar_buckets = Counter(round(ar, 1) for ar in aspect_ratios)
        print("\nAspect Ratio Distribution:")
        for ar, count in sorted(ar_buckets.items()):
            bar = "#" * min(count, 20)
            label = "landscape" if ar > 1.1 else ("portrait" if ar < 0.9 else "square")
            print(f"  {ar:.1f} ({label}): {bar} ({count})")

    # Blur analysis
    if blur_scores:
        avg_blur = sum(blur_scores) / len(blur_scores)
        print(f"\nSharpness (blur score, higher=sharper):")
        print(f"  Min: {min(blur_scores):.1f}")
        print(f"  Max: {max(blur_scores):.1f}")
        print(f"  Avg: {avg_blur:.1f}")

    # Rotated images
    if rotated_images:
        print(f"\nImages with EXIF rotation: {len(rotated_images)}")
        for name in rotated_images[:5]:
            print(f"  - {name}")
        if len(rotated_images) > 5:
            print(f"  ... and {len(rotated_images) - 5} more")

    # Duplicates
    if duplicates:
        print(f"\nPotential duplicates detected: {len(duplicates)} groups")
        for group in duplicates[:3]:
            print(f"  - {', '.join(group)}")

    # Caption analysis
    print(f"\nCaption Analysis:")
    print(f"  Total images: {len(images)}")
    print(f"  With captions: {caption_analysis['total'] - caption_analysis['missing']}")
    print(f"  Missing captions: {caption_analysis['missing']}")
    print(f"  Empty captions: {caption_analysis['empty']}")
    print(f"  Unique captions: {caption_analysis['unique_captions']}")
    print(f"  Avg caption length: {caption_analysis['avg_length']} chars")

    # Generate recommendations
    recommendations = generate_recommendations(
        len(images),
        resolutions,
        aspect_ratios,
        blur_scores,
        caption_analysis
    )

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    print(f"\nSuggested Profile: {recommendations['profile'].upper()}")
    print(f"Base Resolution: {recommendations['base_resolution']}")
    print(f"Bucket Range: {recommendations['min_bucket_reso']} - {recommendations['max_bucket_reso']}")

    if recommendations['warnings']:
        print("\nWarnings:")
        for warn in recommendations['warnings']:
            print(f"  [!] {warn}")

    if recommendations['suggestions']:
        print("\nSuggestions:")
        for sug in recommendations['suggestions']:
            print(f"  [+] {sug}")

    # Build report
    report = {
        'status': 'success',
        'data_dir': str(data_dir),
        'image_count': len(images),
        'resolution_stats': {
            'min_width': min(w for w, h in resolutions) if resolutions else None,
            'min_height': min(h for w, h in resolutions) if resolutions else None,
            'max_width': max(w for w, h in resolutions) if resolutions else None,
            'max_height': max(h for w, h in resolutions) if resolutions else None,
        },
        'aspect_ratio_distribution': dict(Counter(round(ar, 1) for ar in aspect_ratios)) if aspect_ratios else {},
        'blur_stats': {
            'min': min(blur_scores) if blur_scores else None,
            'max': max(blur_scores) if blur_scores else None,
            'avg': round(sum(blur_scores) / len(blur_scores), 2) if blur_scores else None,
        },
        'rotated_images': rotated_images,
        'potential_duplicates': duplicates,
        'caption_analysis': caption_analysis,
        'recommendations': recommendations,
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
