# 🧠 AURA: A Multi-Modal Medical Agent for Understanding, Reasoning & Annotation

[![arXiv](https://img.shields.io/badge/arXiv-2507.16940-b31b1b.svg)](https://arxiv.org/abs/2507.16940)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AURA** represents a paradigm shift from static prediction systems to agentic AI agents capable of reasoning, interacting with tools, and adapting to complex medical imaging tasks.

## 🌟 Overview

AURA is the first visual linguistic explainability agent designed specifically for comprehensive analysis, explanation, and evaluation of medical images. By enabling dynamic interactions, contextual explanations, and hypothesis testing, AURA represents a significant advancement toward more transparent, adaptable, and clinically aligned AI systems.

### Key Features

- 🔍 **Comprehensive Medical Image Analysis**: Advanced chest X-ray interpretation with clinical reasoning
- 🎨 **Counterfactual Generation**: Create "what-if" scenarios to explain AI decisions
- 👥 **Demographic Analysis**: Analyze age, sex, and race variations in medical images
- 🧩 **Multi-Tool Integration**: Seamless integration of segmentation, VQA, and generation tools
- 📊 **Visual Explainability**: Generate difference maps and overlay findings
- 🤖 **Agentic Reasoning**: Dynamic tool selection and parameter optimization
- 💬 **Interactive Interface**: User-friendly Gradio-based chat interface

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Hugging Face account with access to MAIRA-2

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/AURA.git
   cd AURA
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face token
   ```

5. **Launch AURA**
   ```bash
   python main_aura.py
   ```

## 🛠️ Architecture

AURA leverages a modular architecture with the following components:

### Core Agent
- **LLM Backbone**: Qwen-32B for reasoning and tool orchestration
- **Medical Model**: MAIRA-2 for chest X-ray report generation
- **Tool Integration**: Seamless connection to specialized medical tools

### Tool Suite
- **Segmentation Tools**: MedSAM for anatomical and pathological segmentation
- **Counterfactual Generation**: RadEdit for image editing and variation
- **Visual Question Answering**: CheXAgent for clinical question answering
- **Analysis Tools**: TorchXrayVision for pathology detection
- **Visualization**: Difference maps and overlay generation

## 📋 Usage

### Basic Analysis
```python
# Upload a chest X-ray image and ask:
"Analyze this chest X-ray and provide comprehensive visual explainability analysis"
```

### Counterfactual Generation
```python
# Generate counterfactuals for specific findings:
"Remove the pleural effusion and show me the difference"
```

### Demographic Analysis
```python
# Analyze demographic variations:
"Generate counterfactuals showing age and sex variations"
```

## 🔧 Configuration

### Environment Variables
- `HF_TOKEN`: Your Hugging Face token for model access
- `MODEL_NAME`: Model name (default: microsoft/maira-2)
- `CACHE_DIR`: Cache directory for models and data

### Tool Configuration
Each tool can be configured through the `prompt.yml` file, allowing for:
- Parameter optimization
- Tool selection strategies
- Output formatting preferences

## 📊 Performance

AURA demonstrates state-of-the-art performance in:
- **Medical Image Understanding**: Comprehensive analysis of chest X-rays
- **Counterfactual Quality**: High-fidelity image generation with clinical relevance
- **Explainability**: Clear visualization of AI decision processes
- **Tool Integration**: Seamless orchestration of multiple specialized tools

## 🚧 Important Note: PRISM Integration

**PRISM (Precise counterfactual Image generation using language guided Stable diffusion Model)** is currently closed-source and not included in this repository. The processing pipeline is prepared for PRISM integration, and we will update the code as soon as PRISM becomes open-weight.

For more information about PRISM, visit: [PRISM on Hugging Face](https://huggingface.co/amar-kr/PRISM)

## 📝 TODO

- [ ] **Integrate PRISM**: Add PRISM counterfactual generation when available
- [ ] **Hugging Face Model**: Publish AURA on Hugging Face Model Hub

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use AURA in your research, please cite our paper:

```bibtex
@misc{fathi2025aura,
  title={AURA: A Multi-Modal Medical Agent for Understanding, Reasoning & Annotation},
  author={Fathi, Nima and Kumar, Amar and Arbel, Tal},
  eprint={2507.16940},
  url={https://arxiv.org/abs/2507.16940},
  year={2025}
}
```

## 🙏 Acknowledgments

- **MAIRA-2**: Microsoft's medical AI model for report generation
- **MedSAM**: Medical segmentation model
- **RadEdit**: Counterfactual image generation
- **PRISM**: StableDiffusion fine-tuned for chest X-ray image generation
- **CheXAgent**: Visual question answering for medical images
- **TorchXrayVision**: Chest X-ray pathology detection
- **Qwen**: Large language model backbone

<div align="center">
  <p><strong>🌟 Star this repository if you find it useful!</strong></p>
  <p>Made with ❤️ by the AURA team</p>
</div>