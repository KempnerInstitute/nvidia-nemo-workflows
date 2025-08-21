# Knowledge Distillation

This directory contains the SLURM job setup for distilling knowledge from the original LLaMA3.1-8B model to their pruned models on the wikitext dataset using NVIDIA NeMo.

The distillation process runs on a SLURM-managed GPU cluster using a Singularity container.

---

## Objective

Knowledge distillation involves the process of having a smaller "student" model mimic the predictive behaviors of a larger "teacher" models. It can be used to recover the accuracy of models after pruning with less resources compared to regular fine-tuning. 

This setup uses NVIDIA NeMo’s native support for distillation of large language models.

---

##  Requirements

  - Pretrained NeMo 2.0 LLaMA3.1-8B model (`llama3.1-8b`)
  - Pruned NeMo LLaMA3.1-8B model
  - Tokenized wikitext dataset
  - NeMo Singularity image (`nemo-25.04.sif`)

---

## SLURM Job Script 


Launch the SLURM job from the respective directories. For example, to launch the distillation of LLaMa3.1-8B model using logit distillation

```bash
cd distillation
sbatch logit-distillation.slrm
```

---

##  Notes
- Be sure to set correct `--partition` and `--account` in the SLURM header.
- Be sure to set teacher model, student model, output, and/or data paths in the SLURM job scripts.
---