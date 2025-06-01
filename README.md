<div align="center">

<h1>SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion Models</h1>
<h4><b>CVPR 2025</b></h4>
<p>Previously <em>StreamMultiDiffusion: Real-Time Interactive Generation</br>with Region-Based Semantic Control</em></p>

| ![mask](./assets/demo_app_nostream.gif) | ![result](./assets/demo_app.gif) |
| :----------------------------: | :----------------------------: |
| Draw multiple prompt-masks in a large canvas | Real-time creation |

[**Jaerin Lee**](http://jaerinlee.com/) · [**Daniel Sungho Jung**](https://dqj5182.github.io/) · [**Kanggeon Lee**](https://github.com/dlrkdrjs97/) · [**Kyoung Mu Lee**](https://cv.snu.ac.kr/index.php/~kmlee/)

<p align="center">
    <img src="assets/logo_cvlab.png" height=60>
</p>

[![Project](https://img.shields.io/badge/Project-Page-green)](https://jaerinlee.com/research/semantic-draw)
[![ArXiv](https://img.shields.io/badge/Arxiv-2403.09055-red)](https://arxiv.org/abs/2403.09055)
[![Github](https://img.shields.io/github/stars/ironjr/semantic-draw)](https://github.com/ironjr/semantic-draw)
[![X](https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_)](https://twitter.com/_ironjr_)
[![LICENSE](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/ironjr/semantic-draw/blob/main/LICENSE)
[![HFPaper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2403.09055)

[![HFDemoMain](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Main-yellow)](https://huggingface.co/spaces/ironjr/semantic-draw)
[![HFDemo1](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-CanvasSD1.5-yellow)](https://huggingface.co/spaces/ironjr/semantic-draw-canvas-sd15)
[![HFDemo2](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-CanvasSDXL-yellow)](https://huggingface.co/spaces/ironjr/semantic-draw-canvas-sdxl)
[![HFDemo3](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-CanvasSD3-yellow)](https://huggingface.co/spaces/ironjr/semantic-draw-canvas-sd3)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/SemanticPalette-jupyter/blob/main/SemanticPalette_jupyter.ipynb)

</div>

**SemanticDraw** is a real-time interactive text-to-image generation framework that allows you to **draw with meanings** 🧠 using semantic brushes 🖌️.

<p align="center">
  <img src="./assets/figure_one.png" width=100%>
</p>

---

## 🚀 Quick Start

```bash
# Install
conda create -n semdraw python=3.12 && conda activate semdraw
git clone https://github.com/ironjr/semantic-draw
cd semantic-draw
pip install -r requirements.txt

# Run streaming demo
cd demo/stream
python app.py --model "runwayml/stable-diffusion-v1-5" --port 8000

# Open http://localhost:8000 in your browser
```

For SD3 support, additionally run:
```bash
pip install git+https://github.com/initml/diffusers.git@clement/feature/flash_sd3
```

Note: this is default in requirements.txt

---

## 📚 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Demo Applications](#-demo-applications)
- [Usage Examples](#-usage-examples)
- [Documentation](#-documentation)
- [FAQ](#-faq)
- [Citation](#-citation)

---

## ⭐ Features

| Interactive Drawing | Prompt Separation | Real-time Editing |
| :---: | :---: | :---: |
| ![usage1](./assets/feature1.gif) | ![usage2](./assets/feature3.gif) | ![usage3](./assets/feature2.gif) |
| Paint with semantic brushes | No unwanted content mixing | Edit photos in real-time |

---

## 🔧 Installation

### Basic Installation

```bash
conda create -n smd python=3.12 && conda activate smd
git clone https://github.com/ironjr/StreamMultiDiffusion
cd StreamMultiDiffusion
pip install -r requirements.txt
```

### Stable Diffusion 3 Support

```bash
pip install git+https://github.com/initml/diffusers.git@clement/feature/flash_sd3
```

---

## 🎨 Demo Applications

We provide several demo applications with different features and model support:

### 1. StreamMultiDiffusion (Main Demo)

Real-time streaming interface with semantic drawing capabilities.

```bash
cd demo/stream
python app.py --model "your-model" --height 512 --width 512 --port 8000
```

<details>
<summary><b>Options</b></summary>

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to SD1.5 checkpoint (HF or local .safetensors) | None |
| `--height` | Canvas height | 768 |
| `--width` | Canvas width | 1920 |
| `--bootstrap_steps` | Semantic region separation (1-3 recommended) | 1 |
| `--seed` | Random seed | 2024 |
| `--device` | GPU device number | 0 |
| `--port` | Web server port | 8000 |

</details>

### 2. Semantic Palette

Simplified interface for different SD versions:

#### SD 1.5 Version
```bash
cd demo/semantic_palette
python app.py --model "runwayml/stable-diffusion-v1-5" --port 8000
```

#### SDXL Version
```bash
cd demo/semantic_palette_sdxl
python app.py --model "your-sdxl-model" --port 8000
```

#### SD3 Version
```bash
cd demo/semantic_palette_sd3
python app.py --port 8000
```

<details>
<summary><b>Using Custom Models (.safetensors)</b></summary>

1. Place your `.safetensors` file in the demo's `checkpoints` folder
2. Run with: `python app.py --model "your-model.safetensors"`

</details>

---

## 💻 Usage Examples

### Python API

<details>
<summary><b>Basic Generation</b></summary>

```python
import torch
from model import StableMultiDiffusionPipeline

# Initialize
device = torch.device('cuda:0')
smd = StableMultiDiffusionPipeline(device, hf_key='runwayml/stable-diffusion-v1-5')

# Generate
image = smd.sample('A photo of the dolomites')
image.save('output.png')
```

</details>

<details>
<summary><b>Region-Based Generation</b></summary>

```python
import torch
from model import StableMultiDiffusionPipeline
from util import seed_everything

# Setup
seed_everything(2024)
device = torch.device('cuda:0')
smd = StableMultiDiffusionPipeline(device)

# Define prompts and masks
prompts = ['background: city', 'foreground: a cat', 'foreground: a dog']
masks = load_masks()  # Your mask loading logic

# Generate
image = smd(prompts, masks=masks, height=768, width=768)
image.save('output.png')
```

</details>

<details>
<summary><b>Streaming Generation</b></summary>

```python
from model import StreamMultiDiffusion

# Initialize streaming pipeline
smd = StreamMultiDiffusion(device, height=512, width=512)

# Register layers
smd.update_single_layer(idx=0, prompt='background', mask=bg_mask)
smd.update_single_layer(idx=1, prompt='object', mask=obj_mask)

# Stream generation
while True:
    image = smd()
    display(image)
```

</details>

### Jupyter Notebooks

Explore our [notebooks](./notebooks) directory for interactive examples:
- Basic usage tutorial
- Advanced region control
- SD3 examples
- Custom model integration

---

## 📖 Documentation

### Detailed Guides

- [Old README](./README_old.md)
- [Notebooks](./notebooks)

### Paper

For technical details, see our [paper](https://arxiv.org/abs/2403.09055) and [project page](https://jaerinlee.com/research/semantic-draw).

---

## 🙋 FAQ

<details>
<summary><b>What is Semantic Palette?</b></summary>

Semantic Palette lets you paint with text prompts instead of colors. Each brush carries a meaning (prompt) that generates appropriate content in real-time.

</details>

<details>
<summary><b>Which models are supported?</b></summary>

- ✅ Stable Diffusion 1.5 and variants
- ✅ SDXL and variants (with Lightning LoRA)
- ✅ Stable Diffusion 3
- ✅ Custom .safetensors checkpoints

</details>

<details>
<summary><b>Hardware requirements?</b></summary>

- Minimum: GPU with 8GB VRAM (for 512x512)
- Recommended: GPU with 11GB VRAM (for larger resolutions) (Tested with 1080 ti).

</details>

---

## 🚩 Recent Updates

- 🔥 **June 2025**: Presented at CVPR 2025
- ✅ **June 2024**: SD3 support with Flash Diffusion
- ✅ **April 2024**: StreamMultiDiffusion v2 with responsive UI
- ✅ **March 2024**: SDXL support with Lightning LoRA
- ✅ **March 2024**: First version released

See [README_old.md](./README_old.md) for full history.

---

## 🌏 Citation

```bibtex
@inproceedings{lee2025semanticdraw,
    title="{SemanticDraw:} Towards Real-Time Interactive Content Creation from Image Diffusion Models",
    author={Lee, Jaerin and Jung, Daniel Sungho and Lee, Kanggeon and Lee, Kyoung Mu},
    booktitle={CVPR},
    year={2025}
}
```

---

## 🤗 Acknowledgements

Built upon [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [MultiDiffusion](https://multidiffusion.github.io/), and [LCM](https://latent-consistency-models.github.io/). Special thanks to the Hugging Face team and the model contributors.

---

## 📧 Contact

Please email `jarin.lee@gmail.com` or [open an issue](https://github.com/ironjr/StreamMultiDiffusion/issues).
