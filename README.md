# Gemma LoRA and QLoRA Fine-tuning

This project implements the CS13104 lab exercise: fine-tuning a Gemma model on IMDB Reviews with LoRA and QLoRA, then comparing trainable parameters, memory use, training time, validation metrics, and generated outputs.

## Important limitation

Gemma 7B fine-tuning is not practical on a CPU-only laptop or within a short submission window. The notebook is therefore configured as a safe dry run by default. It contains the complete training and evaluation workflow, but it does not invent benchmark values when the experiments have not been executed.

Run the full experiment on a Linux CUDA machine with a compatible NVIDIA GPU, or on a TPU environment configured for Keras. QLoRA specifically requires CUDA-compatible BitsAndBytes support.

## Files

- `fine_tuning_llm.ipynb`: implementation of the assignment.
- `requirements.txt`: focused notebook and fine-tuning dependencies.
- `.gitignore`: excludes the assignment PDF, caches, datasets, checkpoints, adapters, and generated results.

## Recommended environment

Use Python 3.10 or 3.11. The neighboring workspace projects mention Python 3.14, but the TensorFlow, PyTorch, Transformers, and BitsAndBytes combination is more consistently supported on Python 3.10/3.11.

A CUDA GPU with substantial VRAM is recommended. A 4-bit Gemma model can still require several gigabytes of VRAM before optimizer states, activations, and dataset batches are counted. Confirm the model license and authenticate with the relevant model provider before downloading Gemma.

## Setup

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gemma-lora --display-name "Python (Gemma LoRA)"
jupyter notebook fine_tuning_llm.ipynb
```

For QLoRA, install a PyTorch build matching the CUDA version on the target machine. BitsAndBytes is not a CPU fallback.

## Notebook execution

1. Run the configuration and capability-check cells.
2. Authenticate for the selected Gemma model if required.
3. Keep `RUN_TRAINING = False` for a CPU-only inspection or smoke run.
4. Set `RUN_TRAINING = True` only on the accelerator machine.
5. Remove or increase the sample limits when running the full experiment.
6. Execute the LoRA and QLoRA sections with the same dataset split and training budget.
7. Record the generated comparison table and rank-sweep plot only after the runs finish.
8. Run the final qualitative generation section with the same prompt for all three model variants.

The notebook defaults to small deterministic samples. Set `USE_SAMPLE_DATA = False` for the full dataset. Results are saved below `artifacts/`, which is ignored by Git.

## Assignment coverage

The notebook covers model and tokenizer setup, IMDB preparation, dynamic padding, Keras LoRA configuration, frozen base weights, optimizer and checkpoint configuration, validation metrics, adapter save/reload, PEFT QLoRA with NF4 and double quantization, memory/time measurement, LoRA rank values 2/4/8, comparison plotting, and qualitative generation.

The Keras and Hugging Face APIs can change between releases. If a model preset or adapter method has moved, the notebook raises an explicit error and the installed library documentation should be consulted rather than silently reporting a result.

## Submission note

`GenAI.pdf` is intentionally ignored as requested. Do not commit model weights, downloaded datasets, access tokens, checkpoints, or fabricated metrics.
