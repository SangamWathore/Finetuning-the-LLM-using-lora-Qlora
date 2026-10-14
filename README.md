# Llama 2 QLoRA Fine-Tuning

This project demonstrates parameter-efficient fine-tuning of **Llama 2 7B Chat** using **QLoRA**, **LoRA**, Hugging Face Transformers, PEFT, TRL, and BitsAndBytes.

The training notebook uses the `mlabonne/guanaco-llama2-1k` instruction dataset and fine-tunes an adapter rather than updating every parameter of the base model.

## What this project covers

- Loading an instruction-tuning dataset
- 4-bit quantization with BitsAndBytes
- LoRA adapter configuration with PEFT
- Supervised fine-tuning with TRL's `SFTTrainer`
- Saving and loading the fine-tuned adapter
- Merging adapter weights with the base model
- Uploading the resulting model to Hugging Face Hub

## Repository structure

```text
.
├── Untitled31.ipynb       # End-to-end fine-tuning walkthrough
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes secrets and large model artifacts
└── LICENSE                # License for the project materials
```

## Requirements

- Python 3.10 or newer
- NVIDIA GPU with CUDA support recommended
- A Hugging Face account with access to the Llama 2 model
- Google Colab or another environment with enough GPU memory

QLoRA reduces memory usage, but Llama 2 7B fine-tuning still requires a suitable GPU. Actual memory requirements depend on sequence length, batch size, and the training configuration.

## Setup

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Open `Untitled31.ipynb` in Jupyter or upload it to Google Colab, then run the cells from top to bottom.

## Hugging Face authentication

The notebook requires authentication to download Llama 2 and to upload a model. Accept the relevant model license on Hugging Face first, then authenticate with:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Never commit Hugging Face tokens or other credentials to this repository.

## Configuration used in the notebook

The notebook uses:

- Base model: `meta-llama/Llama-2-7b-chat-hf`
- Dataset: `mlabonne/guanaco-llama2-1k`
- Quantization: 4-bit loading
- LoRA rank: 64
- LoRA dropout: 0.1
- Training epochs: 1

These values are educational defaults. For a production experiment, record the final hyperparameters, GPU type, training time, and evaluation results in this README.

## Results and evaluation

The notebook trains an adapter and saves it to the configured output directory. To make results reproducible and easy to compare, evaluate the base and fine-tuned models with the same prompts and record:

- Training loss
- Evaluation prompts and generated answers
- Inference settings such as temperature and max tokens
- GPU type and approximate training time

This repository is currently a learning demonstration; benchmark results should be added after a dedicated evaluation run.

## Limitations and responsible use

The dataset and base model have their own terms, licenses, and limitations. Review the Llama 2 license and dataset terms before using the resulting model commercially. Fine-tuning can reproduce biases or errors from the base model and training data, so validate outputs for your intended use case.

## License

The original project materials are released under the MIT License. The Llama 2 base model and training dataset remain subject to their respective licenses.
