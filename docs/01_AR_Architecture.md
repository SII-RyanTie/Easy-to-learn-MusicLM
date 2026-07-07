# 📖 Module 1: Autoregressive (AR) Architecture

This module contains the core codebase for training and inferencing a MusicLM-style autoregressive model. In this architecture, we use text features extracted by FLAN-T5 as conditions to autoregressively predict discrete audio tokens from MuCodec using a Transformer Decoder.

Below is the standardized step-by-step guide to running the entire pipeline.

---

## 🛠️ 1. Preparation: Workspace & Data Preprocessing
First, ensure you have completed the model and dataset downloads outlined in `00_Preparation.md`. 

To get the `.pt` feature files (tokens and text embeddings) required for training, you have two options:

**Option A: Download Pre-processed Data (Recommended)**
To save time, we have already processed and uploaded the ready-to-use token datasets to Hugging Face. You can directly download them and place them in your data directory:
🔗 [**SII-RyanTie/MTG_MuCodec_Token_Datasets**](https://huggingface.co/datasets/SII-RyanTie/MTG_MuCodec_Token_Datasets)

**Option B: Process from Scratch**
If you want to process your own raw audio and text data, navigate to the dedicated directory for this module and run the data preprocessing script:

```bash
# Enter the AR architecture directory
cd 01_AR_Architecture

# Execute data preprocessing (Extracts tokens and text embeddings)
python preprocess_data.py
```

---

## 🚀 2. Start Training: FSDP Multi-GPU Acceleration
Once the data is ready, you can launch the training process. Our training script natively integrates **FSDP (Fully Sharded Data Parallel)**, enabling highly efficient large-model training across multiple GPUs.

You can specify the `--yamlPath` to load different model scale configurations (e.g., the 0.5B parameter setup):

```bash
python main_mucodec_w_label.py \
    --yamlPath ./config/music_trasformer_0.5B_mucodec_w_label.yaml
```

> **💡 Tip:** Once training starts, logs, TensorBoard records, and model checkpoints will be automatically saved in the `./log/` directory (named with the current timestamp).

---

## 🎵 3. Inference: Generate Your Music
After training is complete (or by using downloaded pre-trained weights), you can use the inference script for batch audio generation. The script supports multi-GPU parallel generation and features built-in **CFG (Classifier-Free Guidance)** to significantly enhance text-audio alignment.

Please replace the `--config` and `--ckpt_path` below with your actual training log paths:

```bash
python infer_mucodec_w_label.py \
    --config ./log/YOUR_TRAINING_LOG_DIR/music_trasformer_0.5B_mucodec_w_label.yaml \
    --ckpt_path ./log/YOUR_TRAINING_LOG_DIR/checkpoints/step_43342.pth \
    --input_dir ../data/MTG_dataset/mtg_tokens_mucodec_val_w_label \
    --output_dir ./output/MusicTransformer_0.5B \
    --num_samples 100 \
    --batch_size 8 \
    --prompt_sec 0.04 \
    --gen_sec 40.92 \
    --temperature 1.0 \
    --cfg_scale 3.0 \
    --seed 42
```

### ⚙️ Core Inference Parameters Guide
To achieve the best generation results, feel free to tweak the following configuration arguments:

| **Parameter** | **Description** | **Recommended Value** |
| :--- | :--- | :--- |
| `--cfg_scale` | Classifier-Free Guidance scale. Higher values force the music to stick closer to the text prompt, but overly high values may degrade audio quality. | `3.0` ~ `5.0` |
| `--temperature` | Sampling temperature. Controls generation diversity. Lower values produce conservative/repetitive music, while higher values increase richness/randomness. | `1.0` |
| `--gen_sec` | Target duration of the generated audio (in seconds). | `10.0` ~ `40.0` |
| `--batch_size` | Inference batch size per GPU. Decrease this value if you encounter Out-Of-Memory (OOM) errors. | `4` or `8` |

Once the generation process finishes, all the `.wav` audio files will be saved in your specified `--output_dir`. Go take a listen to what your model has composed!
