
+++
disableToc = false
title = "Fine-tuning LLMs for text generation"
weight = 22
+++

This section covers how to fine-tune a language model for text generation and use it with LocalAI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mudler/LocalAI/blob/master/examples/e2e-fine-tuning/notebook.ipynb)

## Overview

Fine-tuning adapts a pre-trained language model to your specific use case by training it on a curated dataset. Instead of training from scratch, you update the model's weights with domain-specific data, which is dramatically cheaper and faster.

### When to Fine-tune

Fine-tuning is appropriate when:

- **Prompt engineering is insufficient** — the model consistently fails at a task even with good prompts
- **You need a specific style or tone** — e.g., matching your brand voice or a particular writing style
- **Domain specialization** — the model needs to understand domain-specific terminology (legal, medical, scientific)
- **Instruction following** — you want the model to follow a particular instruction format consistently
- **Cost reduction** — a smaller fine-tuned model can replace a larger general-purpose one

Fine-tuning is **not** the best approach when:

- The model already performs well with prompt engineering or few-shot examples
- You need the model to learn entirely new factual knowledge (consider RAG instead)
- Your dataset is very small (fewer than ~50 examples)

### Fine-tuning Approaches

| Approach | Tool | Best For | GPU Memory |
|----------|------|----------|------------|
| QLoRA | Axolotl, Llama-Factory | Most use cases, memory-efficient | 12-24 GB |
| LoRA | Axolotl, Llama-Factory | Higher quality than QLoRA, more memory | 24-48 GB |
| Full fine-tune | Axolotl, Llama-Factory | Maximum quality, requires significant resources | 48+ GB |

**QLoRA** (Quantized Low-Rank Adaptation) is the recommended starting point — it quantizes the base model to 4-bit and trains small adapter layers, allowing you to fine-tune 7B models on a single consumer GPU.

### Supported Tools

This guide covers two popular fine-tuning frameworks:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — streamlined configuration-driven fine-tuning with broad model support
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — web UI and CLI-based fine-tuning with built-in dataset management

Both produce models that can be converted to GGUF format for use with LocalAI.

## The Fine-tuning Pipeline

Regardless of which tool you use, the pipeline is:

1. **Prepare a dataset** in the appropriate format
2. **Configure and run** the fine-tuning process
3. **Merge LoRA adapters** with the base model (if using LoRA/QLoRA)
4. **Convert to GGUF** format for LocalAI
5. **Deploy** the model with a LocalAI YAML configuration

## Dataset Preparation

### Dataset Formats

#### Completion Format

The simplest format — each example is the full text the model should learn to produce. This is what the e2e example uses.

```json
[
  {
    "text": "As an AI language model you are trained to reply to an instruction. Try to be as much polite as possible\n\n## Instruction\n\nWrite a poem about a tree.\n\n## Response\n\nTrees are beautiful, ..."
  },
  {
    "text": "As an AI language model you are trained to reply to an instruction. Try to be as much polite as possible\n\n## Instruction\n\nExplain photosynthesis.\n\n## Response\n\nPhotosynthesis is the process by which ..."
  }
]
```

The structure within each `text` field follows whatever instruction format you want the model to learn:

```
<System prompt>

## Instruction

<Question or instruction>

## Response

<Expected response from the LLM>
```

At inference time, you provide the text up to the `## Response` marker, and the model completes the rest.

#### Alpaca Format

A structured format with explicit fields for instruction, optional input, and output:

```json
[
  {
    "instruction": "Summarize the following text.",
    "input": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "output": "A sentence demonstrating all alphabet letters using a fox and dog scenario."
  },
  {
    "instruction": "Translate to French.",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous ?"
  }
]
```

#### ChatML / Conversational Format

For chat models, use multi-turn conversations:

```json
[
  {
    "conversations": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to reverse a string."},
      {"role": "assistant", "content": "def reverse_string(s):\n    return s[::-1]"}
    ]
  },
  {
    "conversations": [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "The capital of France is Paris."}
    ]
  }
]
```

#### ShareGPT Format

Similar to ChatML but uses `from`/`value` keys, common in community datasets:

```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful assistant."},
      {"from": "human", "value": "Explain quantum computing."},
      {"from": "gpt", "value": "Quantum computing uses quantum-mechanical phenomena..."}
    ]
  }
]
```

### Dataset Quality Guidelines

- **Minimum size**: 50-100 examples for basic behavior changes, 500-1000+ for reliable results
- **Quality over quantity**: 200 high-quality examples outperform 2000 noisy ones
- **Consistency**: Use consistent formatting across all examples
- **Diversity**: Cover the range of inputs the model will see in production
- **Balance**: Avoid heavy class imbalance if training for classification
- **Deduplication**: Remove exact or near-duplicate examples
- **Validation split**: Hold out 5-10% of your data for evaluation

### Building a Dataset from Scratch

If you don't have existing data, consider these approaches:

1. **Manual creation**: Write examples by hand — highest quality but slow
2. **Seed + expand**: Write 20-50 seed examples, then use a larger model (e.g., via the LocalAI API) to generate variations
3. **Existing datasets**: Browse [Hugging Face Datasets](https://huggingface.co/datasets) for datasets in your domain
4. **Data extraction**: Convert existing documentation, FAQs, or support tickets into training examples

## Method 1: Fine-tuning with Axolotl

Axolotl is configuration-driven — you define a YAML file and it handles training, LoRA, quantization, and more.

There is an e2e example of fine-tuning a LLM model to use with [LocalAI](https://github.com/mudler/LocalAI) written by [@mudler](https://github.com/mudler) available [here](https://github.com/mudler/LocalAI/tree/master/examples/e2e-fine-tuning/).

### Requirements

- Linux with a CUDA-capable GPU (12 GB+ VRAM for QLoRA with 7B models)
- Python 3.10+
- CUDA toolkit installed

### Install Axolotl

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
cd axolotl
pip install packaging
pip install -e '.[flash-attn,deepspeed]'
cd ..
```

Configure the distributed training launcher:

```bash
accelerate config default
```

### Axolotl Configuration

Create an `axolotl.yaml` file. Here is an example QLoRA configuration for a 7B model:

```yaml
base_model: mistralai/Mistral-7B-v0.1
model_type: MistralForCausalLM
tokenizer_type: LlamaTokenizer
load_in_8bit: false
load_in_4bit: true

adapter: qlora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

datasets:
  - path: dataset.json
    ds_type: json
    type: completion
    field: text

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

output_dir: ./qlora-out
gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 4
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.0002
train_on_inputs: false
group_by_length: false

bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
flash_attention: true

warmup_steps: 10
eval_steps: 20
save_strategy: epoch
save_total_limit: 3
```

Key parameters to adjust:

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `base_model` | The Hugging Face model to fine-tune | Choose based on your use case and available VRAM |
| `lora_r` | LoRA rank — controls adapter capacity | 8-64; higher = more capacity but more memory |
| `lora_alpha` | LoRA scaling factor | Typically 2x `lora_r` or equal to it |
| `sequence_len` | Maximum token length per example | Match your use case; longer = more memory |
| `micro_batch_size` | Batch size per GPU | Reduce if you hit OOM errors |
| `num_epochs` | Number of training passes | 3-5 for most tasks; watch for overfitting |
| `learning_rate` | Step size for weight updates | 1e-4 to 3e-4 for QLoRA |

### Run Fine-tuning

Optionally pre-tokenize for faster training:

```bash
python -m axolotl.cli.preprocess axolotl.yaml
```

Launch training:

```bash
accelerate launch -m axolotl.cli.train axolotl.yaml
```

### Merge LoRA Adapters

After training completes, merge the LoRA adapters back into the base model:

```bash
python3 -m axolotl.cli.merge_lora axolotl.yaml \
  --lora_model_dir="./qlora-out" \
  --load_in_8bit=False \
  --load_in_4bit=False
```

This produces a full merged model in `./qlora-out/merged/`.

## Method 2: Fine-tuning with LLaMA-Factory

LLaMA-Factory provides both a web UI and CLI for fine-tuning, with built-in dataset management.

### Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

### Prepare Your Dataset

Register your dataset in `LLaMA-Factory/data/dataset_info.json`:

```json
{
  "my_dataset": {
    "file_name": "my_dataset.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

Place your dataset file at `LLaMA-Factory/data/my_dataset.json`.

### CLI Fine-tuning

```bash
llamafactory-cli train \
  --stage sft \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --dataset my_dataset \
  --template default \
  --finetuning_type lora \
  --lora_rank 32 \
  --lora_alpha 16 \
  --output_dir ./lora-output \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --quantization_bit 4 \
  --fp16 true
```

### Web UI Fine-tuning

LLaMA-Factory includes a Gradio-based web interface:

```bash
llamafactory-cli webui
```

This opens a browser UI where you can select models, datasets, and hyperparameters visually.

### Export Merged Model

After training, merge and export the model:

```bash
llamafactory-cli export \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --adapter_name_or_path ./lora-output \
  --template default \
  --finetuning_type lora \
  --export_dir ./merged-model \
  --export_size 2 \
  --export_legacy_format false
```

## Converting to GGUF

Both methods produce a Hugging Face format model that needs to be converted to GGUF for LocalAI's llama.cpp backend.

### Build llama.cpp Tools

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
cd ..
```

### Convert to GGUF

```bash
python3 llama.cpp/convert_hf_to_gguf.py ./qlora-out/merged
```

This produces an F16 (16-bit float) GGUF file.

### Quantize the Model

Quantization reduces model size and speeds up inference at the cost of some quality:

```bash
./llama.cpp/build/bin/llama-quantize \
  ./qlora-out/merged/model-F16.gguf \
  ./custom-model-q4_0.gguf q4_0
```

Common quantization levels:

| Format | Size (7B model) | Quality | Speed | Use Case |
|--------|-----------------|---------|-------|----------|
| `Q2_K` | ~2.7 GB | Low | Fastest | Testing, low-resource environments |
| `Q4_0` | ~3.8 GB | Good | Fast | General use, good balance |
| `Q4_K_M` | ~4.1 GB | Better | Fast | Recommended default |
| `Q5_K_M` | ~4.8 GB | Very Good | Moderate | When quality matters more than speed |
| `Q6_K` | ~5.5 GB | Excellent | Moderate | Near-lossless quality |
| `Q8_0` | ~7.2 GB | Near-perfect | Slower | Maximum quality |
| `F16` | ~14 GB | Lossless | Slowest | Reference/evaluation only |

`Q4_K_M` is the recommended starting point for production use.

## Using the Fine-tuned Model with LocalAI

### Copy the Model

Place the quantized GGUF file in your LocalAI models directory:

```bash
cp custom-model-q4_0.gguf /path/to/localai/models/
```

### Create a Model Configuration

Create a YAML configuration file at `models/custom-model.yaml`:

```yaml
name: custom-model
parameters:
  model: custom-model-q4_0.gguf
  temperature: 0.7
  top_p: 0.9
  top_k: 40

context_size: 4096
threads: 4
gpu_layers: 35
mmap: true

template:
  chat: |
    {{.Input}}
    ## Response

  completion: |
    {{.Input}}
    ## Response
```

{{% notice tip %}}
The template must match the format used during fine-tuning. If you trained with the Alpaca format, use an Alpaca-style template. If you used ChatML, use a ChatML template. A mismatch between training and inference templates is one of the most common causes of poor results.
{{% /notice %}}

### Example: ChatML Template

If you fine-tuned using the ChatML format:

```yaml
name: custom-chatml-model
parameters:
  model: custom-model-q4_0.gguf
  temperature: 0.7

context_size: 4096
gpu_layers: 35

template:
  chat_message: |
    <|im_start|>{{.RoleName}}
    {{.Content}}<|im_end|>
  chat: |
    {{.Input}}
    <|im_start|>assistant
```

### Test the Model

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-model",
    "messages": [{"role": "user", "content": "Hello, test the fine-tuned model!"}],
    "temperature": 0.7
  }'
```

## Hyperparameter Tuning Guidelines

### Learning Rate

The most impactful hyperparameter. Start with these ranges:

- **QLoRA**: `1e-4` to `3e-4`
- **LoRA**: `1e-4` to `2e-4`
- **Full fine-tune**: `1e-5` to `5e-5`

If the model outputs gibberish after training, the learning rate was likely too high. If the model doesn't change behavior, it was too low.

### Epochs

- **1-2 epochs**: For large datasets (10k+ examples)
- **3-5 epochs**: For medium datasets (500-5000 examples)
- **5-10 epochs**: For small datasets (50-500 examples), but watch for overfitting

Signs of overfitting: training loss continues to drop while validation loss increases, or the model starts reproducing training examples verbatim.

### LoRA Rank (`lora_r`)

Controls the capacity of the adapter:

- **8**: Minimal changes, style adjustments
- **16-32**: General fine-tuning, recommended starting point
- **64-128**: Complex tasks, significant behavior changes

Higher ranks use more memory and are more prone to overfitting with small datasets.

### Batch Size and Gradient Accumulation

The effective batch size is `micro_batch_size × gradient_accumulation_steps × num_gpus`. Larger effective batch sizes produce more stable training:

- **Effective batch size 8-16**: Good for small datasets
- **Effective batch size 32-64**: Good for larger datasets

If you hit out-of-memory errors, reduce `micro_batch_size` and increase `gradient_accumulation_steps` to maintain the same effective batch size.

## Resource Requirements

### GPU Memory (VRAM) Requirements

| Model Size | QLoRA (4-bit) | LoRA (16-bit) | Full Fine-tune |
|-----------|---------------|---------------|----------------|
| 1-3B | 6-8 GB | 12-16 GB | 24-32 GB |
| 7B | 12-16 GB | 24-32 GB | 48-64 GB |
| 13B | 20-24 GB | 40-48 GB | 80-128 GB |
| 30-34B | 36-48 GB | 80+ GB | 160+ GB |
| 70B | 48-80 GB | 160+ GB | 320+ GB |

These estimates include memory for the model, optimizer states, and gradients. Actual requirements vary with sequence length, batch size, and whether gradient checkpointing is enabled.

### Recommended Hardware

| Budget | GPU | VRAM | Models You Can Fine-tune |
|--------|-----|------|--------------------------|
| Consumer | RTX 3090 / 4090 | 24 GB | QLoRA up to 13B |
| Prosumer | RTX A6000 | 48 GB | QLoRA up to 34B, LoRA up to 13B |
| Cloud | A100 (80 GB) | 80 GB | QLoRA up to 70B, LoRA up to 34B |
| Multi-GPU | 2-4x A100 | 160-320 GB | Full fine-tune up to 70B |

### Training Time Estimates

Training time depends heavily on dataset size, model size, sequence length, and GPU:

| Dataset Size | Model Size | GPU | Approximate Time (QLoRA, 3 epochs) |
|-------------|-----------|-----|--------------------------------------|
| 1,000 examples | 7B | RTX 4090 | 30-60 minutes |
| 5,000 examples | 7B | RTX 4090 | 2-4 hours |
| 10,000 examples | 7B | RTX 4090 | 4-8 hours |
| 1,000 examples | 13B | A100 80GB | 1-2 hours |
| 10,000 examples | 13B | A100 80GB | 6-12 hours |

### Cloud GPU Options

If you don't have a local GPU, you can use:

- **Google Colab** (free tier provides T4 with 16 GB — sufficient for QLoRA on 7B models)
- **RunPod, Vast.ai, Lambda** — on-demand GPU rental
- **AWS, GCP, Azure** — cloud GPU instances

## Common Pitfalls and Solutions

### Training Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| CUDA Out of Memory | Model + training state exceeds VRAM | Reduce `micro_batch_size`, enable `gradient_checkpointing`, reduce `sequence_len`, use QLoRA |
| Loss not decreasing | Learning rate too low, data issues | Increase learning rate, check dataset quality, verify data format |
| Loss goes to NaN | Learning rate too high, data corruption | Reduce learning rate, check for empty or malformed examples |
| Model outputs gibberish | Overtrained, wrong merge, bad data | Reduce epochs, verify merge step, check dataset for issues |
| Model ignores fine-tuning | Too few epochs, learning rate too low | Increase epochs or learning rate, increase `lora_r` |

### Conversion Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `convert_hf_to_gguf.py` fails | Missing model architecture support | Update llama.cpp to latest version, check if your model architecture is supported |
| Quantized model produces bad output | Over-quantization | Try a higher quantization level (e.g., `Q5_K_M` instead of `Q2_K`) |
| Large model file after merge | F16 weights | This is expected; quantize to reduce size |

### Inference Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Model doesn't follow instructions | Template mismatch | Ensure the LocalAI template matches the format used during training |
| Repetitive output | Temperature too low, or overfitting | Increase temperature, add `repeat_penalty: 1.1` in config |
| Output cuts off mid-sentence | `context_size` too small or `max_tokens` too low | Increase `context_size` in YAML config, set higher `max_tokens` in API request |

## Worked Example: Customer Support Bot

This end-to-end example fine-tunes a model to answer questions about a fictional product.

### Step 1: Create the Dataset

Create `dataset.json`:

```json
[
  {
    "instruction": "You are a helpful customer support agent for Acme Widgets.",
    "input": "How do I reset my widget?",
    "output": "To reset your Acme Widget, hold the power button for 10 seconds until the LED flashes blue. Release the button and wait 30 seconds for the device to restart. Your settings will be preserved."
  },
  {
    "instruction": "You are a helpful customer support agent for Acme Widgets.",
    "input": "What is the warranty period?",
    "output": "All Acme Widgets come with a 2-year manufacturer warranty covering defects in materials and workmanship. Extended warranty options are available at purchase time. Contact support@acme.example for warranty claims."
  },
  {
    "instruction": "You are a helpful customer support agent for Acme Widgets.",
    "input": "My widget won't turn on.",
    "output": "If your Acme Widget won't power on, try these steps:\n1. Ensure the charging cable is securely connected\n2. Try a different power outlet\n3. Hold the power button for 15 seconds to force restart\n4. If none of these work, the battery may need replacement — contact support@acme.example"
  }
]
```

In practice, you would need at least 50-100 such examples for useful results.

### Step 2: Configure Axolotl

Create `axolotl.yaml`:

```yaml
base_model: mistralai/Mistral-7B-v0.1
model_type: MistralForCausalLM
tokenizer_type: LlamaTokenizer
load_in_4bit: true

adapter: qlora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

datasets:
  - path: dataset.json
    ds_type: json
    type: alpaca

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

output_dir: ./support-bot-out
gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 5
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.0002
gradient_checkpointing: true
flash_attention: true
bf16: auto

warmup_steps: 10
eval_steps: 20
save_strategy: epoch
```

### Step 3: Train

```bash
accelerate launch -m axolotl.cli.train axolotl.yaml
```

### Step 4: Merge and Convert

```bash
# Merge LoRA
python3 -m axolotl.cli.merge_lora axolotl.yaml \
  --lora_model_dir="./support-bot-out" \
  --load_in_8bit=False --load_in_4bit=False

# Convert to GGUF
python3 llama.cpp/convert_hf_to_gguf.py ./support-bot-out/merged

# Quantize
./llama.cpp/build/bin/llama-quantize \
  ./support-bot-out/merged/model-F16.gguf \
  ./support-bot-q4km.gguf q4_k_m
```

### Step 5: Deploy with LocalAI

Copy the model and create `models/support-bot.yaml`:

```yaml
name: support-bot
parameters:
  model: support-bot-q4km.gguf
  temperature: 0.3
  top_p: 0.9

context_size: 2048
gpu_layers: 35

template:
  chat: |
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {{.Input}}

    ### Response:
```

Test it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support-bot",
    "messages": [
      {"role": "system", "content": "You are a helpful customer support agent for Acme Widgets."},
      {"role": "user", "content": "How long does the battery last?"}
    ],
    "temperature": 0.3
  }'
```

## Post-Fine-tuning Evaluation

### Manual Evaluation

The most important step — test the model with representative prompts:

1. **In-distribution**: Test with examples similar to your training data
2. **Edge cases**: Test with unusual or ambiguous inputs
3. **Out-of-distribution**: Test with inputs outside the training domain to check for overfitting
4. **Adversarial**: Try to make the model produce incorrect or inappropriate output

### Automated Metrics

If you held out a validation set, you can measure:

- **Perplexity**: Lower is better; measures how well the model predicts the validation text
- **BLEU/ROUGE**: For tasks with expected reference outputs (summarization, translation)
- **Task-specific accuracy**: For classification or extraction tasks

### A/B Comparison

Compare the fine-tuned model against the base model on the same prompts:

```bash
# Base model
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "base-model", "messages": [{"role": "user", "content": "test prompt"}]}'

# Fine-tuned model
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "support-bot", "messages": [{"role": "user", "content": "test prompt"}]}'
```

### Iterating

If results are unsatisfactory:

1. **Add more data** — especially for cases where the model fails
2. **Adjust hyperparameters** — learning rate, epochs, LoRA rank
3. **Clean the dataset** — remove low-quality or contradictory examples
4. **Try a different base model** — some models fine-tune better for certain tasks
5. **Change the template** — ensure the instruction format is clear and consistent

## Advanced Topics

### Multi-GPU Training

For models that don't fit on a single GPU, use DeepSpeed with Axolotl:

```yaml
# Add to axolotl.yaml
deepspeed: deepspeed_configs/zero2.json
```

Launch with multiple GPUs:

```bash
accelerate launch --multi_gpu --num_processes 2 -m axolotl.cli.train axolotl.yaml
```

### Training on Conversation Data

For multi-turn chat models, use the `sharegpt` dataset type in Axolotl:

```yaml
datasets:
  - path: conversations.json
    ds_type: json
    type: sharegpt
    conversation: chatml
```

### Using Hugging Face Datasets Directly

Both Axolotl and LLaMA-Factory can pull datasets from Hugging Face:

```yaml
# Axolotl
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
```

```bash
# LLaMA-Factory
llamafactory-cli train --dataset alpaca_en ...
```

### LoRA Without Quantization

For higher quality at the cost of more VRAM, use LoRA without quantization:

```yaml
# In axolotl.yaml
load_in_4bit: false
load_in_8bit: false
adapter: lora
```

This typically produces slightly better results than QLoRA but requires roughly 2-4x the VRAM.
