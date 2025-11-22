#!/bin/bash

# Default values
DATASET_BASE_PATH="./Ditto-1M/videos"
DATASET_METADATA_PATH="./Ditto-1M/csvs_for_DiffSynth/metadata.csv"
OUTPUT_PATH="./exps/ditto"
MODEL_ID="Wan-AI/Wan2.1-VACE-1.3B"
NUM_EPOCHS=5
LEARNING_RATE="1e-4"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_base_path) DATASET_BASE_PATH="$2"; shift ;;
        --dataset_metadata_path) DATASET_METADATA_PATH="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --model_id) MODEL_ID="$2"; shift ;;
        --num_epochs) NUM_EPOCHS="$2"; shift ;;
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Construct model_id_with_origin_paths based on MODEL_ID
MODEL_ID_WITH_PATHS="${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:models_t5_umt5-xxl-enc-bf16.pth,${MODEL_ID}:Wan2.1_VAE.pth"

echo "Starting training with:"
echo "  Dataset Base Path: $DATASET_BASE_PATH"
echo "  Metadata Path: $DATASET_METADATA_PATH"
echo "  Output Path: $OUTPUT_PATH"
echo "  Model ID: $MODEL_ID"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  --data_file_keys "video,vace_video" \
  --height 480 \
  --width 832 \
  --num_frames 73 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "$MODEL_ID_WITH_PATHS" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs "$NUM_EPOCHS" \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "$OUTPUT_PATH" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 128 \
  --extra_inputs "vace_video" \
  --use_gradient_checkpointing_offload \
  --save_steps 1000
