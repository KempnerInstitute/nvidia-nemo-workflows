# Full Finetuning  

This directory contains the SLURM job setup for full fine-tuning of the LLaMA3-70B and LLaMA3-8B models on the Databricks Dolly-15k and PubMedQA datasets using NVIDIA NeMo.

The fine-tuning process runs on a SLURM-managed multi-node GPU cluster using a Singularity container.

---

## Objective

Full fine-tuning involves updating all parameters of the pretrained model, offering the highest flexibility and performance gains for downstream tasks—at the cost of increased compute and memory usage.

Unlike parameter-efficient methods such as LoRA, full fine-tuning retrains the entire model, which can lead to better performance when ample resources and labeled data are available.

This setup uses NVIDIA NeMo’s native support for full fine-tuning of large language models.

---

##  Requirements

  - Pretrained NeMo LLaMA3 model (`llama3-70b.nemo` and `llama3-8b.nemo`)
  - Dolly-15K + PubMedQA datasets in JSONL format
  - NeMo Singularity image (nemo-25.02.rc0.simg or newer)

---

## SLURM Job Script 


Launch the Slurm job from the respective directories. For example, to launch the finetuning of LLaMa3 70B model on pubmedqa with LoRA 

```bash
cd sft_full_llama3-70b_pubmedqa
sbatch sft_full_llama3-70b_pubmedqa.slrm
```

---

##  Notes
- Be sure to set correct `--partition` and `--account` in the SLURM header.
- Adjust `max_steps` or `global_batch_size` to fit your hardware.

---

