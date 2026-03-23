import os
import random
import shutil
import subprocess
import argparse
import re
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import torch
import glob
from transformers import AutoProcessor, AutoModelForCausalLM
import toml

# Global variables for storing model and processor
florence_model = None
florence_processor = None

def init_florence_model():
    """Initialize Florence model, only needs to be called once"""
    global florence_model, florence_processor
    
    if florence_model is not None and florence_processor is not None:
        return  # Model already loaded, no need to reload
    
    print("Loading Florence model, please wait...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model and processor
    florence_model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    
    florence_processor = AutoProcessor.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True
    )
    print("Florence model loaded successfully")

def generate_caption(image_path, concept_prefix=""):
    """Generate caption for image using Florence model"""
    global florence_model, florence_processor
    
    if florence_model is None or florence_processor is None:
        init_florence_model()  # Ensure model is loaded
    
    device = next(florence_model.parameters()).device
    torch_dtype = next(florence_model.parameters()).dtype
    
    # Read image
    image = Image.open(image_path).convert("RGB")
    prompt = "<DETAILED_CAPTION>"

    # Construct input
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # Generate caption
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-process
    parsed_answer = florence_processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
    
    # Add concept prefix
    if concept_prefix:
        caption_text = f"{concept_prefix} {caption_text}"

    return caption_text

def apply_grayscale_to_mask_region(input_image_path, mask_image_path, output_path):
    """Apply grayscale to regions where mask is white (255) in the input image"""
    # Read input image
    input_img = Image.open(input_image_path).convert('RGB')
    
    # Read mask image
    mask_img = Image.open(mask_image_path).convert('L')
    
    # Resize mask if dimensions don't match input image
    if mask_img.size != input_img.size:
        mask_img = mask_img.resize(input_img.size, Image.LANCZOS)
    
    # Convert to numpy arrays
    input_array = np.array(input_img)
    mask_array = np.array(mask_img)
    
    # Create copy of output array
    output_array = input_array.copy()
    
    # Find white pixels in mask (threshold set to 200 to capture all white regions)
    white_pixels = mask_array > 200
    
    # Set these positions to gray (128) in the input image
    output_array[white_pixels] = 128
    
    # Convert numpy array back to PIL image
    output_img = Image.fromarray(output_array)
    
    # Save output image
    output_img.save(output_path)

def create_white_image(size=(512, 512)):
    """Create a white image"""
    # Create white image (255, 255, 255)
    white_array = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    return Image.fromarray(white_array)

def find_matching_files(video_dir):
    """Find edited images and corresponding original images and masks in video directory"""
    source_frames_dir = os.path.join(video_dir, 'source_frames')
    additional_edited_dir = os.path.join(video_dir, 'additional_edited_frames')
    source_masks_dir = os.path.join(video_dir, 'source_masks')
    
    # Check if necessary directories exist
    if not os.path.exists(source_frames_dir):
        print(f"Error: source_frames directory does not exist: {source_frames_dir}")
        return []
    
    if not os.path.exists(additional_edited_dir):
        print(f"Error: additional_edited_frames directory does not exist: {additional_edited_dir}")
        return []
        
    if not os.path.exists(source_masks_dir):
        print(f"Error: source_masks directory does not exist: {source_masks_dir}")
        return []
    
    # Get edited image files
    edited_files = []
    for file in os.listdir(additional_edited_dir):
        if file.endswith('.png') or file.endswith('.jpg'):
            edited_files.append(file)
    
    if not edited_files:
        print(f"No edited images found in {additional_edited_dir}")
        return []
    
    results = []
    
    # Find corresponding original image and mask for each edited image
    for edited_file in edited_files:
        # Extract numeric part from filename (e.g., 00048.png -> 00048)
        base_name = os.path.splitext(edited_file)[0]
        
        # Find corresponding original image
        original_path = os.path.join(source_frames_dir, f"{base_name}.png")
        if not os.path.exists(original_path):
            print(f"Warning: Corresponding original image not found: {original_path}")
            continue
        
        # Find corresponding mask
        mask_path = os.path.join(source_masks_dir, f"{base_name}.png")
        if not os.path.exists(mask_path):
            print(f"Warning: Corresponding mask not found: {mask_path}")
            continue
        
        results.append({
            'edited': os.path.join(additional_edited_dir, edited_file),
            'original': original_path,
            'mask': mask_path,
            'base_name': base_name
        })
    
    return results

def save_image_as_png(input_path, output_path):
    """Convert image of any format to PNG format and save"""
    try:
        # Open image and convert to RGB mode
        img = Image.open(input_path).convert('RGB')
        # Save as PNG format
        img.save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error converting image {input_path}: {e}")
        return False

def generate_sequence(original_path, edited_path, mask_path, output_root, temp_dir, sequence_id, base_name, concept_prefix=""):
    """Generate a video sequence with structure: original frame, grayed original frame, edited frame, mask"""
    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate video filename - use base_name directly, without .png
    output_video_name = f'edit_seq_{base_name}.mp4'
    output_video = os.path.join(output_root, output_video_name)
    
    # 1. First frame: original frame
    # shutil.copy(original_path, os.path.join(temp_dir, f'frame_000.png'))
    save_image_as_png(original_path, os.path.join(temp_dir, f'frame_000.png'))
    
    # 2. Second frame: original frame with grayscale applied
    # Apply grayscale processing first, then save as PNG
    grayed_path = os.path.join(temp_dir, f'frame_001.png')
    apply_grayscale_to_mask_region(original_path, mask_path, grayed_path)
    
    # 3. Third frame: edited frame
    # shutil.copy(edited_path, os.path.join(temp_dir, f'frame_002.png'))
    save_image_as_png(edited_path, os.path.join(temp_dir, f'frame_002.png'))
    
    # 4. Fourth frame: corresponding mask
    # shutil.copy(mask_path, os.path.join(temp_dir, f'frame_003.png'))
    save_image_as_png(mask_path, os.path.join(temp_dir, f'frame_003.png'))
    
    # Check if all frames were created successfully
    frames_count = sum(1 for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.png'))
    if frames_count != 4:
        print(f"Warning: Expected 4 frames but only {frames_count} frames generated")
    
    # Generate mp4 using ffmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite existing files
        '-framerate', '5',  # Set framerate to 5fps
        '-i', os.path.join(temp_dir, 'frame_%03d.png'),  # Input frames
        '-c:v', 'libx264',  # Use h264 encoding
        '-pix_fmt', 'yuv420p',  # Set pixel format
        '-crf', '23',  # Set video quality
        output_video
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    # Generate caption for edited frame
    caption_text = generate_caption(edited_path, concept_prefix)
    
    return output_video_name, caption_text

def process_directory(video_dir, output_root, temp_root, concept_prefix=""):
    """Process all edited images in a video directory"""
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Find matching files
    matching_files = find_matching_files(video_dir)
    
    if not matching_files:
        print(f"No matching image sets found in directory {video_dir}")
        return []
    
    sequences_info = []
    
    # Process each set of matching files
    for i, files in enumerate(matching_files):
        temp_dir = os.path.join(temp_root, f'temp_{i:02d}')
        
        print(f'Processing image {i+1}/{len(matching_files)}...')
        print(f'Using images: original={os.path.basename(files["original"])}, edited={os.path.basename(files["edited"])}, mask={os.path.basename(files["mask"])}')
        
        # Generate sequence
        output_video_name, caption = generate_sequence(
            files['original'], 
            files['edited'], 
            files['mask'], 
            output_root, 
            temp_dir, 
            i, 
            files['base_name'], 
            concept_prefix
        )
        
        # Generate corresponding txt file
        txt_filename = output_video_name.replace('.mp4', '.txt')
        txt_path = os.path.join(output_root, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)
        
        sequences_info.append({
            'video_name': output_video_name,
            'txt_name': txt_filename,
            'caption': caption,
            'original': os.path.basename(files['original']),
            'edited': os.path.basename(files['edited']),
            'mask': os.path.basename(files['mask'])
        })
        
        print(f'Generated sequence {i+1}/{len(matching_files)}: {output_video_name} and {txt_filename}')
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    return sequences_info

def process_video_directory(video_dir, output_dir, temp_dir, concept_prefix=""):
    """Process a single video directory"""
    print(f"\nProcessing video directory: {video_dir}")
    
    # Check if video directory exists
    if not os.path.exists(video_dir):
        print(f"Error: Video directory does not exist: {video_dir}")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process directory
    sequences = process_directory(video_dir, output_dir, temp_dir, concept_prefix)
    
    if sequences:
        print(f"Generated {len(sequences)} training sequences, each sequence contains video file and corresponding txt file")
    
    return sequences

def find_latest_lora_checkpoint(lora_dir):
    """Find the latest checkpoint in lora directory"""
    if not os.path.exists(lora_dir):
        return None
    
    # Look for epoch directories
    epoch_dirs = []
    for item in os.listdir(lora_dir):
        item_path = os.path.join(lora_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a timestamp directory or direct epoch directory
            if item.startswith('epoch'):
                epoch_dirs.append((item, item_path))
            else:
                # Check subdirectories for epoch folders
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and subitem.startswith('epoch'):
                        epoch_dirs.append((subitem, subitem_path))
    
    if not epoch_dirs:
        return None
    
    # Sort by epoch number and return the latest
    def extract_epoch_num(epoch_name):
        match = re.search(r'epoch(\d+)', epoch_name)
        return int(match.group(1)) if match else 0
    
    epoch_dirs.sort(key=lambda x: extract_epoch_num(x[0]), reverse=True)
    return epoch_dirs[0][1]

def create_additional_configs(data_dir):
    """Create training_additional.toml and dataset_additional.toml based on existing configs"""
    configs_dir = os.path.join(data_dir, 'configs')
    
    # Check if configs directory exists
    if not os.path.exists(configs_dir):
        print(f"Warning: configs directory does not exist: {configs_dir}")
        return
    
    # Paths for existing configs
    training_config_path = os.path.join(configs_dir, 'training.toml')
    dataset_config_path = os.path.join(configs_dir, 'dataset.toml')
    
    # Check if existing configs exist
    if not os.path.exists(training_config_path):
        print(f"Warning: training.toml not found: {training_config_path}")
        return
    
    if not os.path.exists(dataset_config_path):
        print(f"Warning: dataset.toml not found: {dataset_config_path}")
        return
    
    try:
        # Read existing configs
        with open(training_config_path, 'r', encoding='utf-8') as f:
            training_config = toml.load(f)
        
        with open(dataset_config_path, 'r', encoding='utf-8') as f:
            dataset_config = toml.load(f)
        
        # Create additional_dataset.toml
        additional_dataset_config = dataset_config.copy()
        
        # Add additional_traindata directory to dataset config
        additional_traindata_path = os.path.join(data_dir, 'additional_traindata').replace(os.sep, '/')
        additional_directory = {
            'path': additional_traindata_path,
            'num_repeats': 9
        }
        
        # Ensure directory list exists
        if 'directory' not in additional_dataset_config:
            additional_dataset_config['directory'] = []
        
        # Add the additional directory
        additional_dataset_config['directory'].append(additional_directory)
        
        # Save additional dataset config
        additional_dataset_path = os.path.join(configs_dir, 'dataset_additional.toml')
        with open(additional_dataset_path, 'w', encoding='utf-8') as f:
            toml.dump(additional_dataset_config, f)
        
        print(f"Created dataset_additional.toml: {additional_dataset_path}")
        
        # Create additional_training.toml
        additional_training_config = training_config.copy()
        
        # Modify paths and settings for additional training
        original_output_dir = additional_training_config.get('output_dir', './lora')
        additional_output_dir = original_output_dir.replace('/lora', '/lora_additional')
        additional_training_config['output_dir'] = additional_output_dir
        
        # Point to additional dataset config
        additional_training_config['dataset'] = additional_dataset_path.replace(os.sep, '/')
        
        # Reduce epochs for additional training (assuming it's fine-tuning)
        additional_training_config['epochs'] = 10
        
        # More frequent saves for shorter training
        if 'save_every_n_epochs' in additional_training_config:
            additional_training_config['save_every_n_epochs'] = 5
        
        # Find the latest checkpoint from original lora training
        lora_dir = os.path.join(data_dir, 'lora')
        latest_checkpoint = find_latest_lora_checkpoint(lora_dir)
        
        if latest_checkpoint:
            # Add init_from_existing to adapter section
            if 'adapter' not in additional_training_config:
                additional_training_config['adapter'] = {}
            additional_training_config['adapter']['init_from_existing'] = latest_checkpoint.replace(os.sep, '/')
            print(f"Will initialize additional training from: {latest_checkpoint}")
        else:
            print("Warning: No existing lora checkpoint found for initialization")
        
        # Save additional training config
        additional_training_path = os.path.join(configs_dir, 'training_additional.toml')
        with open(additional_training_path, 'w', encoding='utf-8') as f:
            toml.dump(additional_training_config, f)
        
        print(f"Created training_additional.toml: {additional_training_path}")
        
        return additional_training_path, additional_dataset_path
        
    except Exception as e:
        print(f"Error creating additional configs: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate training sequences for edited images')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory path (containing source_frames, additional_edited_frames, source_masks, prefix.txt)')
    args = parser.parse_args()
    
    # Derive other paths from data_dir
    output_dir = os.path.join(args.data_dir, 'additional_traindata')
    temp_dir = os.path.join(args.data_dir, 'temp_edited')
    prefix_file = os.path.join(args.data_dir, 'prefix.txt')
    
    # Read concept prefix
    concept_prefix = ""
    if os.path.exists(prefix_file):
        try:
            with open(prefix_file, 'r', encoding='utf-8') as f:
                concept_prefix = f.read().strip()
            print(f"Read concept prefix from {prefix_file}: {concept_prefix}")
        except Exception as e:
            print(f"Failed to read prefix.txt file: {e}")
            concept_prefix = "p3rs0n,"  # Use default value
    else:
        print(f"prefix.txt file not found: {prefix_file}, using default prefix")
        concept_prefix = "p3rs0n,"
    
    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Pre-load Florence model
    init_florence_model()
    
    # Process video directory
    print(f"Starting to process data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Temporary directory: {temp_dir}")
    print(f"Concept prefix: {concept_prefix}")
    
    sequences = process_video_directory(
        args.data_dir, 
        output_dir, 
        temp_dir,
        concept_prefix
    )
    
    # Print processing results
    print(f"\nSuccessfully processed {len(sequences)} sequences")
    
    if sequences:
        print("\nGenerated sequences:")
        for i, seq in enumerate(sequences, 1):
            print(f"{i}. {seq['video_name']} - {seq['caption'][:50]}...")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\nProcessing completed, all data saved to: {output_dir}")
    
    # Create additional training configs
    print("\nCreating additional training configurations...")
    config_result = create_additional_configs(args.data_dir)
    
    if config_result:
        print("Additional configuration files created successfully!")
        print(f"You can now run additional training with:")
        print(f"python train.py --config {config_result[0]}")
    else:
        print("Failed to create additional configuration files.")

if __name__ == '__main__':
    main() 