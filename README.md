# NVIDIA NeMo Workflow

This repository contains the workflow for finetuning and pretraining large language models using the NVIDIA NeMo framework. The workflow is designed to be run on the Kempner AI cluster, which is equipped with NVIDIA A100 and H100 GPUs.

## NeMo AI Framework

NVIDIA NeMo is an scalable AI framework built for researchers and PyTorch developers working on Large Language Models (LLMs), Multimodal Models (MM), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV). It streamlines the development of cutting-edge AI systems by offering a modular, flexible, and easy-to-extend architecture.

Below are quick links to the official NeMo documentation and GitHub repository:

- [NeMo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)


## Prerequisites

- Partition with compatible NVIDIA GPU (A100 or H100)
- Containerized environment with NVIDIA NeMo installed
- Pretrained model access (e.g., Llama3, GPT2)
- SLURM script for job submission


> [!NOTE]  
> Some models on HuggingFace are hosted under Gated Repositories, which require special access. To use these models, you must not only obtain an access token but also accept the model's terms and conditions. To verify your access, visit the [Gated Repositories](https://huggingface.co/settings/gated-repos) page and check whether the model is listed and access is granted. If not, go to the model’s page and submit an access request. Once your request is approved, you’ll be able to use the model in your code. Sufficient checks are in place to ensure that the model is accessible before running the code. 

At each workflow step, we will provide a SLURM script that can be used to run the code on the Kempner AI cluster. The scripts are designed to be easily modifiable for different models, datasets, and configurations.


## Available Workflows

NeMo provides a variety of workflows for different tasks, including: Data Curation, Training, Alignment among others. In this repository, the objective is to replicate all available workflows in the NeMo framework for the Kempner AI cluster. If there is a specific workflow you would like to see implemented, please let us know by opening an [issue](https://github.com/KempnerInstitute/nvidia-nemo-workflows/issues) in the GitHub repository. 

> [!TIP]  
> Click on the workflow name to navigate to the corresponding SLURM script and related files and discussions.

### Pretraining 

| Workflow Name                   | Model                                                              | Dataset     |
|---------------------------------|--------------------------------------------------------------------|-------------|
| [gpt2_pretraining_codeparrot](pretraining/gpt2_pretraining_codeparrot) | [Megatron GPT2](https://huggingface.co/nvidia/megatron-gpt2-345m)  | [CodeParrot](https://huggingface.co/codeparrot)  |


### Finetuning

| Type       | Workflow Name                                                                  | Model       | Dataset   |
|------------|--------------------------------------------------------------------------------|-------------|-----------|
| Full       | [sft_full_llama3-70b_dolly15k](finetuning/full/sft_full_llama3-70b_dolly15k)   | LLaMA3-70B  | dolly15k  |
| Full       | [sft_full_llama3-70b_pubmedqa](finetuning/full/sft_full_llama3-70b_pubmedqa)   | LLaMA3-70B  | pubmedqa  |
| Full       | [sft_full_llama3-8b_dolly15k](finetuning/full/sft_full_llama3-8b_dolly15k)     | LLaMA3-8B   | dolly15k  |
| Full       | [sft_full_llama3-8b_pubmedqa](finetuning/full/sft_full_llama3-8b_pubmedqa)     | LLaMA3-8B   | pubmedqa  |
| LoRA       | [sft_lora_llama3-70b_dolly15k](finetuning/lora/sft_lora_llama3-70b_dolly15k)   | LLaMA3-70B  | dolly15k  |
| LoRA       | [sft_lora_llama3-70b_pubmedqa](finetuning/lora/sft_lora_llama3-70b_pubmedqa)   | LLaMA3-70B  | pubmedqa  |
| LoRA       | [sft_lora_llama3-8b_dolly15k](finetuning/lora/sft_lora_llama3-8b_dolly15k)     | LLaMA3-8B   | dolly15k  |
| LoRA       | [sft_lora_llama3-8b_pubmedqa](finetuning/lora/sft_lora_llama3-8b_pubmedqa)     | LLaMA3-8B   | pubmedqa  |
| P-Tuning   | [ptuning_llama3-70b_dolly15k](finetuning/p-tuning/ptuning_llama3-70b_dolly15k) | LLaMA3-70B  | dolly15k  |
| P-Tuning   | [ptuning_llama3-8b_dolly15k](finetuning/p-tuning/ptuning_llama3-8b_dolly15k)   | LLaMA3-8B   | dolly15k  |

### RL

| Type       | Workflow Name                                                                  | Model       | Dataset   |
|------------|--------------------------------------------------------------------------------|-------------|-----------|
| Full       | [dpo_llama3-8b](RL/DPO/llama3-8b)   | LLaMA3-8B  | email response  |

## Available Singularity Images

Path to the Singularity images on the Kempner cluster:

```
/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking
```

| Image Name            | Build Date | MD5 Hash                           | Size (GB) |
|-----------------------|------------|------------------------------------|-----------|
| `nemo-25.02.rc0.simg` | 2025-02-11 | `bdcf489f8706e9af1748421bfc07a6c5` | 24G       |


## Known Issues and Checks

We document common issues and checks that can be performed to ensure a successful workflow run. These include SSL certificate errors, HuggingFace authentication issues, and gated repository access problems. Please refer to the following section for more details:
-  [Checks](checks/README.md)


