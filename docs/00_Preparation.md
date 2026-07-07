# 📖 Module 0: Preparation & Data Pipeline

Before diving into complex generative architectures, we need to prepare our "ingredients" and "tools." For an audio foundation model like MusicLM, high-quality audio datasets, a powerful Text Encoder, and an efficient Audio Tokenizer are the bedrock of success. 

Here is a complete guide to gathering all the core dependencies and data needed to run this project. Let's get our environment fully equipped.

---

## 📦 Core Pre-trained Models
In text-to-music generation, we rarely train all components from scratch. Instead, we stand on the shoulders of giants by leveraging powerful open-source foundation models.

### 1. Text Encoder: FLAN-T5
FLAN-T5 is responsible for "understanding" human prompts. It translates natural language (e.g., "a cheerful pop guitar riff") into dense semantic vectors that the model can process. We recommend the `large` version, which strikes an excellent balance between VRAM consumption and semantic comprehension.
- **Download Command:**
  ```bash
  huggingface-cli download google/flan-t5-large --local-dir /your/path/flan-t5-large
  ```

### 2. Audio Tokenizer: MuCodec
Continuous audio waveforms cannot be directly fed into a language model. We need MuCodec to "compress" and "discretize" complex audio signals into individual tokens—just like splitting a sentence into words—and later reconstruct them back into high-fidelity audio during generation.
- **How to get it:** Please visit the official repository by Tencent AI Lab to clone the code and download the pre-trained weights.
- **Repository:** [tencent-ailab/MuCodec](https://github.com/tencent-ailab/MuCodec)

---

## 🎵 Training Datasets Acquisition
You can't make bricks without straw. We need massive amounts of music data to teach the model about "melody," "rhythm," and "arrangement."

### 1. Base Audio Data: MTG Dataset
MTG is a classic open-source music dataset containing a rich variety of audio tracks. Given the large size of audio files, we highly recommend enabling multi-threading (`--max-workers`) to speed up the download.
- **Download Command:**
  ```bash
  huggingface-cli download m-a-p/MTG --repo-type dataset --local-dir /your/path/MTG_dataset --max-workers 4
  ```

### 2. Text-Audio Alignment Data: MOSS-Music
Audio alone is not enough. To teach the model the mapping between "text" and "music," we need high-quality descriptive labels (captions). MOSS-Music provides excellent open-source multi-modal music data support.
- **How to get it:** Please refer to the official guide by the OpenMOSS team for data pulling and preprocessing.
- **Repository:** [OpenMOSS/MOSS-Music](https://github.com/OpenMOSS/MOSS-Music)

---

## 💡 Resource Quick Reference
For your convenience, here is a summary checklist of all the preparatory resources:

| **Resource Name** | **Type** | **Core Function** | **Source** |
| :--- | :--- | :--- | :--- |
| **FLAN-T5-Large** | Pre-trained Model | Extracts semantic features from text prompts | Hugging Face |
| **MuCodec** | Pre-trained Model | Converts between audio waveforms and discrete tokens | GitHub |
| **MTG** | Dataset | Provides fundamental raw music audio files | Hugging Face |
| **MOSS-Music** | Dataset / Tools | Provides high-quality music text descriptions and alignments | GitHub |

> **🚀 Acceleration Tip:** 
> If you experience network instability while downloading from Hugging Face, you can prepend a mirror endpoint to your command, for example:
> `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...`

Once you have all these models and datasets securely in place, we are ready to move on to the next chapter and start building the exciting model architectures!
