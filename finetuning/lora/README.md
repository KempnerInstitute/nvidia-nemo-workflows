# LoRA Finetuning  

This directory contains the Slurm job setup for fine-tuning the **LLaMA3-70B** and **LLaMA3-8B** models using ** LoRA finetuning** on the **Databricks Dolly-15k** and **pubmedqa** datasets with NVIDIA NeMo.

Fine-tuning is executed on a SLURM-managed multi-node GPU cluster using a Singularity container.

---

## Objective

[LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) is a parameter-efficient fine-tuning (PEFT) technique that injects trainable low-rank matrices into transformer layers. This enables adapting large models using a small number of parameters, reducing memory and compute costs.

LoRA decomposes weight updates into low-rank matrices (A, B) while freezing the base model.


NeMo implements this approach using the `model.peft.peft_scheme=ptuning` option. For more, see the [NeMo PEFT documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/peft/intro.html).

---

##  Requirements

  - Pretrained NeMo LLaMA3 model (`llama3-70b.nemo` and `llama3-8b.nemo`)
  - Dolly-15K + PubMedQA datasets in JSONL format
  - NeMo Singularity image (nemo-25.02.rc0.simg or newer)

---

## SLURM Job Script 


Launch the Slurm job from the respective directories. For example, to launch the finetuning of LLaMa3 70B model on pubmedqa with LoRA 

```bash
cd sft_lora_llama3-70b_pubmedqa
sbatch sft_lora_llama3-70b_pubmedqa.slrm
```

---

##  Notes
- Be sure to set correct `--partition` and `--account` in the SLURM header.
- Adjust `max_steps` or `global_batch_size` to fit your hardware.

---

