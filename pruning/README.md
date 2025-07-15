# Pruning

This directory contains the SLURM job setup for pruning LLaMA3.1-8B models on the wikitext datasets using NVIDIA NeMo.

The fine-tuning process runs on a SLURM-managed multi-node GPU cluster using a Singularity container.

---

## Objective

Pruning involes reducing the size of a model by removing neurons deemed less important based on importance estimation metrics.

Four pruning strategies are provided: drop layers, depth pruning, width pruning, and width-depth pruning.

This directory includes Slurm job scripts for data preprocessing and model format conversion for the pruning process and visualization of the importance estimation metrics during pruning.

This setup uses NVIDIA NeMo’s native support for pruning of large language models.

---

##  Requirements

  - Pretrained NeMo LLaMA3.1 model (`llama3.1-8b.nemo`)
  - Wikitext datasets in JSONL format
  - NeMo Singularity image (`nemo-25.04.sif`)

---

## SLURM Job Script 


Launch the Slurm job from the respective directories. For example, to launch the pruning of LLaMa3 70B model using depth pruning 

```bash
cd pruning-strategies
sbatch depth-pruning.slrm
```

---

##  Notes
- Be sure to set correct `--partition` and `--account` in the SLURM header.

---
