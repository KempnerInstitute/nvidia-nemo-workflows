# Pruning

This directory contains the SLURM job setup for pruning LLaMA3.1-8B models on the wikitext datasets using NVIDIA NeMo.

The pruning process runs on a SLURM-managed GPU cluster using a Singularity container.

---

## Objective

Pruning involes reducing the size of a model by removing neurons deemed less important based on importance estimation metrics.

Four pruning strategies are provided: drop layers, depth pruning, width pruning, and width-depth pruning.

This directory includes SLURM job scripts for data preprocessing and model format conversion for the pruning process and visualization of the importance estimation metrics during pruning.

This setup uses NVIDIA NeMo’s native support for pruning of large language models.

---

##  Requirements

  - Pretrained NeMo LLaMA3.1-8B model (`llama3.1-8b.nemo`) or HuggingFace LLaMA3.1-8B model
  - Wikitext dataset in JSONL format
  - NeMo Singularity image (`nemo-25.04.sif`)

---

## SLURM Job Script 


Launch the SLURM job from the respective directories. For example, to launch the pruning of LLaMa3.1-8B model using depth pruning 

```bash
cd pruning-strategies
sbatch depth-pruning.slrm
```

---

##  Notes
- Be sure to set correct `--partition` and `--account` in the SLURM header.
- Be sure to set model, output, and/or data paths in the SLURM job scripts.
---
