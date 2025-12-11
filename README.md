# TextTo3D

A versatile framework for generating 3D content from text prompts, built on top of [threestudio](https://github.com/threestudio-project/threestudio).

## ⚡ Quick Start

Try it out instantly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b8o-Wfz5xI_xS7ZnZ7STGp8dq9qanxc0?usp=sharing)

## 🎨 Results

| Prompt: "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" |
| :---: |
| <img src="img/panda.jpg" width="300"/>

## 🚀 Usage

### 1. Training (Text-to-3D)
Run the `launch.py` script with your desired configuration:

```bash
# Example: Generate a 3D model from text
python launch.py --config configs/prolificdreamer.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
