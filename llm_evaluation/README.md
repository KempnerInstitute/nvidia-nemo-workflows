# Evaluation

This directory contains the SLURM job setup for evaluating LLMs in the NeMo model format.

The evaluation process runs on a SLURM-managed GPU cluster using a Singularity container.

---

## Objective

Evaluation involves assessing a model's performance based on benchmark tasks and datasets.

The directory comes with a deployment SLURM job script to deploy NeMo models through PyTriton to allow evaluation on completion and chat benchmarks.

This directory includes SLURM job scripts for the following evaluation tools:
- **MMLU (through NeMo)**
- **MMLU (through LM Evaluation Harness)**
- **Post-training LM validation loss implementation**

<br/>

The MMLU evaluation benchmark through NeMo supports can be configured to support the following tasks:
```python
COMPLETIONS_TASKS = (
    "gsm8k",
    "mgsm",
    "mmlu",
    "mmlu_pro",
    "mmlu_redux",
)
CHAT_TASKS = (
    "gpqa_diamond_cot",
    "gsm8k_cot_instruct",
    "ifeval",
    "mgsm_cot",
    "mmlu_instruct",
    "mmlu_pro_instruct",
    "mmlu_redux_instruct",
    "wikilingua",
)
```

The MMLU benchmark through LM Evaluation Harness can be extended to support additional tasks and datasets available through the LM Evaluation Harness (see [here](https://github.com/EleutherAI/lm-evaluation-harness)), allowing for more flexible evaluation.

Some parts of this setup uses NVIDIA NeMo’s native support for evaluating large language models, while others depend on external benchmark suites and custom evaluation pipelines.

---

##  Requirements

  - Pretrained NeMo 2.0 model (eg. `llama3.1-8b`)
  - NeMo Singularity image (`nemo-25.04.sif`)

---

## SLURM Job Script 

### Evaluation through NeMo

1. Deploy the NeMo model.
    ```bash
    cd deploy-model
    sbatch deploy-nemo.slrm
    ```
2. Obtain the node on which the model was deployed in the cluster.

3. Use the node as the `triton_http_address` and run the evaluation.
 
    For example, to evaluate using MMLU,
    ```bash
    cd mmlu-str
    sbatch mmlu_str-by-nemo.slrm
    ```


### Evaluation through LM Evaluation Harness

1. Deploy the NeMo model.
    ```bash
    cd deploy-model
    sbatch deploy-nemo.slrm
    ```
2. Obtain the node on which the model was deployed in the cluster.

3. Use the node as the `triton_http_address` and run the evaluation.
 
    For example, to evaluate using MMLU,
    ```bash
    cd mmlu-str
    sbatch mmlu_str.slrm
    ```

### Post-training LM Validation Loss


Launch the SLURM job from the respective directory.
```bash
cd lm-validation-loss
sbatch validation-loss.slrm
```

---

##  Notes
- Be sure to set correct `--partition` and `--account` in the SLURM header.
- Be sure to set model, output, and/or data paths in the SLURM job scripts.
---