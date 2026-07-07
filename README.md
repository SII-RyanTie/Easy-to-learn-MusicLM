# 🎵Easy-to-learn-MusicLM
![这里写图片的描述，比如：MusicLM架构图](assets/cover.png)

A comprehensive, end-to-end tutorial for implementing MusicLM with the FSDP framework, covering everything from **data preprocessing** and **model building** to **training and evaluation**.

> **🎯 Who is this for?**
> This guide is designed for beginners who have a basic understanding of deep learning concepts and are eager to explore the field of AI music generation. It provides a hands-on, easy-to-follow approach to scaling models.

## 🛠️ Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage your Python environment. This ensures that all dependencies and versions are strictly isolated and reproducible. Create a new environment named `musiclm` with Python 3.10.20:

```bash
conda create -n musiclm python=3.10.20 -y
conda activate musiclm
pip install -r requirements.txt
```

<!-- ## Directory Structure
Easy-to-learn-MusicLM/
├── README.md                  # 主页（简介、环境配置、目录导航）
├── requirements.txt
├── docs/                      # 详细教程文件夹
│   ├── 00_Preparation.md      # 数据处理与特殊环境配置
│   ├── 01_AR_Architecture.md  # AR 架构详细讲解
│   └── 02_NAR_Flow_Matching.md# NAR 架构详细讲解
└── src/                       # 核心代码文件 -->

## 🚀 Let's Start Learning

This tutorial is divided into structured modules. You can follow them step-by-step or jump directly to the architecture you are most interested in. Currently, the guide covers two mainstream approaches to AI music generation:

### 📖 Module 0: Preparation & Data Pipeline
Before diving into model architectures, we need high-quality data and specific environment configurations. This section covers audio data cleaning, tokenization (extracting discrete acoustic tokens), and setting up the distributed training environment.
👉 **[Read the Preparation Tutorial here](./docs/00_Preparation.md)**

### 📖 Module 1: Autoregressive (AR) Architecture
Learn how to build a Text-to-Music model using the classic Autoregressive approach. This section covers the fundamental concepts of token prediction and sequential generation.
👉 **[Read the AR Architecture Tutorial here](./docs/01_AR_Architecture.md)**

### 📖 Module 2: Non-Autoregressive (NAR) with Flow Matching
Explore the cutting-edge Non-Autoregressive architecture based on Flow Matching. This section dives into how to achieve faster, parallelized audio generation with higher efficiency.
👉 **[Read the NAR Flow Matching Tutorial here](./docs/02_NAR_Flow_Matching.md)**

*(More advanced topics and FSDP scaling techniques will be updated soon!)*
