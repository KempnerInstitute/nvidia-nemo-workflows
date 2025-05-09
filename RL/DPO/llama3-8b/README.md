 Preference Optimization (DPO) Workflow

This directory contains a SLURM job script (`dpo_llama3-8b.slrm`) for fine-tuning the Llama3 8B Instruction model using **Direct Preference Optimization (DPO)** with the **NVIDIA NeMo framework**. DPO offers a reinforcement learning-free approach to aligning language models with human preferences.

---

## Overview

The `dpo_llama3-8b.slrm` script is designed to be submitted to a SLURM workload manager. It configures the environment and executes the DPO training process using NeMo’s model alignment tools.

---

## Prerequisites

- Access to a SLURM-managed GPU cluster.
- NVIDIA NeMo installed and available in the environment.
- Datasets formatted for DPO fine-tuning (preference pair data).
- Model weights (e.g., pretrained Llama3 8B) accessible locally or via HuggingFace.

---

## Usage

1. **Edit the SLURM script**:

   Open `dpo_llama3-8b.slrm` and update the following:
   - Paths for datasets, output directory, and pretrained model.
   - SLURM parameters such as job name, partition, number of nodes/GPUs, time, and account.

2. **Submit the job to SLURM**:

   ```bash
   sbatch dpo_llama3-8b.slrm
   ```

   The fine-tuned model will be saved to the specified output directory (currently set it to be `OUTPUT_DIR=./dpo-output`).

---

## Notes

- Monitor job progress with standard SLURM commands such as `squeue`, `sacct`, or `tail -f slurm-<jobid>.out`.

---

## References

-  [NVIDIA NeMo GitHub](https://github.com/NVIDIA/NeMo)
-  [DPO in NeMo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/dpo.html)

---

For additional NeMo fine-tuning workflows, including PEFT and LoRA, refer to other directories in this repository.

