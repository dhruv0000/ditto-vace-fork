---
language:
- en
license: cc-by-nc-sa-4.0
size_categories:
- n>1T
task_categories:
- video-to-video
---

# Ditto-1M: A High-Quality Synthetic Dataset for Instruction-Based Video Editing

> **Ditto: Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset** <br>
> Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, Yinghao Xu, Yujun Shen, Qifeng Chen

<div align=center>
<img src="./assets/data_teaser.jpg" width=850px>
</div>

**Figure:** Our proposed synthetic data generation pipeline can automatically produce high-quality and highly diverse video editing data, encompassing both global and local editing tasks.

<div align=center>

## 🔗 **Links & Resources**

[**[**📄 Paper**](https://arxiv.org/abs/2510.15742)**]
[**[**🌐 Project Page**](https://ezioby.github.io/Ditto_page/)**]
[**[**💻 Github Code**](https://github.com/EzioBy/Ditto)**]
[**[**📦 Model Weights**](https://huggingface.co/QingyanBai/Ditto_models/tree/main)**]


</div>

## Updating List
#### - [√] 10/22/2025 - We have uploaded the csvs that can be directly used for model training with DiffSynth-Studio, as well as the metadata json for sim2real setting.
#### - [√] 10/22/2025 - We finish uploading all the videos of the dataset!

## Dataset Overview

Ditto-1M is a comprehensive dataset of one million high-fidelity video editing triplets designed to tackle the fundamental challenge of instruction-based video editing. This dataset was generated using our novel data generation pipeline that fuses the creative diversity of a leading image editor with an in-context video generator, overcoming the limited scope of existing models.

The dataset contains diverse video editing scenarios including:
- **Global style transfer**: Artistic style changes, color grading, and visual effects
- **Global freeform editing**: Complex scene modifications, environment changes, and creative transformations
- **Local editing**: Precise object modifications, attribute changes, and local transformations


## Dataset Structure

The dataset is organized as follows:

```
Ditto-1M/
├── mini_test_videos/         # 30+ video cases for testing
├── videos/                   # Main video data
│   ├── source/               # Source videos (original videos)
│   ├── local/                # Local editing results
│   ├── global_style1/        # Global style editing
│   ├── global_style2/        # Global style editing
│   ├── global_freeform1/     # Freeform editing
│   ├── global_freeform2/     # Freeform editing
│   └── global_freeform3/     # Freeform editing (relatively hard)
├── source_video_captions/    # QwenVL generated captions for source videos
├── training_metadata/        # Training metadata including video paths and editing instructions
└── csvs_for_DiffSynth/       # CSVs for model training with DiffSynth-Studio
```

### Data Categories

- **Source Videos (~180G)**: Original videos before editing
- **Global Style (~230+120G)**: Artistic style transformations and color grading
- **Global Freeform (~370+430+270G)**: Complex scene modifications and creative editing
- **Local Editing (~530G)**: Precise modifications to specific objects or regions


### Training Metadata

Each metadata json file contains triplet items of:
- `source_path`: Path to the source video
- `instruction`: Editing instruction
- `edited_path`: Path to the corresponding edited video

## Downloading and Extracting the Dataset

### Full Dataset Download

```python
from datasets import load_dataset

# Download the entire dataset
dataset = load_dataset("QingyanBai/Ditto-1M")
```

### Selective Download

Due to the large size of the videos folder (~2TB), you can only download the specific subsets if you only need to train on specific tasks:

```python
from huggingface_hub import snapshot_download

# Download the metadata and source captions
snapshot_download(
    repo_id="QingyanBai/Ditto-1M",
    repo_type="dataset",
    local_dir="./Ditto-1M",
    allow_patterns=["source_video_captions/*", "training_metadata/*"]
)

# Download only the mini test videos
snapshot_download(
    repo_id="QingyanBai/Ditto-1M",
    repo_type="dataset",
    local_dir="./Ditto-1M",
    allow_patterns=["mini_test_videos/*"]
)

# Download the local editing data
snapshot_download(
    repo_id="QingyanBai/Ditto-1M",
    repo_type="dataset", 
    local_dir="./Ditto-1M",
    allow_patterns=["videos/source/*", "videos/local/*"]
)

# Download the global editing videos
snapshot_download(
    repo_id="QingyanBai/Ditto-1M",
    repo_type="dataset",
    local_dir="./Ditto-1M",
    allow_patterns=["videos/source/*", "videos/global_style1/*", "videos/global_style2/*", "videos/global_freeform1/*", "videos/global_freeform2/*"]
)

# Download only the style editing videos
snapshot_download(
    repo_id="QingyanBai/Ditto-1M",
    repo_type="dataset",
    local_dir="./Ditto-1M", 
    allow_patterns=["videos/source/*", "videos/global_style1/*", "videos/global_style2/*"]
)

```

### Extracting the Video Data
On Linux/macOS or Windows (with Git Bash/WSL):
```bash
# Navigate to the directory containing the split files
cd path/to/your/dataset/part

# For example, to extract the global_style1 videos:
cat global_style1.tar.gz.* | tar -zxv
```
This command concatenates all the split parts and pipes the output directly to tar for extraction, saving both disk space (by not creating an intermediate merged file) and time (as you can start previewing videos immediately without waiting for the entire tar merging process to complete).



## Dataset Statistics

- **Total Examples**: 1,000,000+ video editing triplets
- **Video Resolution**: Various resolutions (1280\*720 / 720\*1280)
- **Video Length**: 101 frames per video
- **Categories**: Global style, Global freeform, Local editing
- **Instructions**: Captions and editing instructions generated by intelligent agents
- **Quality Control**: Processed with the data filtering pipeline and enhanced with the denoising enhancer

## Citation

If you find this dataset useful, please consider citing our paper:

```bibtex
@article{bai2025ditto,
  title={Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset},
  author={Bai, Qingyan and Wang, Qiuyu and Ouyang, Hao and Yu, Yue and Wang, Hanlin and Wang, Wen and Cheng, Ka Leong and Ma, Shuailei and Zeng, Yanhong and Liu, Zichen and Xu, Yinghao and Shen, Yujun and Chen, Qifeng},
  journal={arXiv preprint arXiv:2510.15742},
  year={2025}
}
```