# P-Tuning of LLaMA3-8B and LLaMA3-70B on Dolly15K (NeMo Workflow)

This directory contains a reproducible setup for fine-tuning the **LLaMA3-70B** and **LLaMA3-8B** models using **P-Tuning v2** on the **Databricks Dolly-15k** dataset with NVIDIA NeMo.

Fine-tuning is executed on a SLURM-managed multi-node GPU cluster using a Singularity container.

---

## Objective

[P-Tuning v2 (Li et al., 2021)](https://arxiv.org/abs/2103.10385) is a parameter-efficient fine-tuning (PEFT) technique that prepends learnable prefix tokens to the input embeddings of a pretrained language model. Instead of updating all model weights, it only tunes a small set of prefix parameters, enabling faster training and reduced memory usage.

![P-Tuning Diagram](https://nvidia.github.io/NeMo/_images/peft_prefix_tuning.png)

*Illustration from NVIDIA NeMo: P-Tuning applies learnable tokens to condition model behavior with minimal weight updates.*

NeMo implements this approach using the `model.peft.peft_scheme=ptuning` option. For more, see the [NeMo PEFT documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/peft/intro.html).

---

##  Requirements
  - Pretrained NeMo LLaMA3 model (`llama3-70b.nemo`)
  - Dolly-15K JSONL dataset: `training.jsonl`, `validation.jsonl`, `test.jsonl`
  - NeMo Singularity image: `nemo-25.02.rc0.simg`

---

---

## SLURM Job Script 


Launch the p-tuning of Llama3 70B, 

```bash
cd ptuning_llama3-70b_dolly15k
sbatch ptuning_llama3-70b_dolly15k.slrm 
```


---

##  Notes
- Be sure to set correct `--partition` and `--account` in the SLURM header.
- Adjust `max_steps` or `global_batch_size` to fit your hardware.

---



