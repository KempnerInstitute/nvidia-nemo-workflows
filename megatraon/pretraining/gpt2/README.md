
# GPT Model Training

The Generative Pre-trained Transformer (GPT) is a decoder-only Transformer model widely used in Large Language Models (LLMs). 
This document outlines the steps for pretraining a GPT model.

The file `gpt2_pretraining_model_parallel.slrm` contains details about the data, model, and compute resources used for the pretraining job. 
Please review and modify this file as needed to suit your training requirements.

The main differnce with the [GPT2 via NeMo wrapper](https://github.com/KempnerInstitute/nvidia-nemo-workflows/tree/main/pretraining/gpt2_pretraining_codeparrot) is that the job here uses torchrun to distribute the job, and the config files are specific to Megatron format. It is important to make sure that the number of tasks per node is just one since the tasks are handled by torchrun. 

# Data Preparation

GPT model pretraining requires vocabulary files and a training dataset. A pre-built dataset is provided for training. 
First, define the main directory containing the required data:

```bash
data_pth="<absolute-path-to-data-including-vocab-and-documents>"
```

Then, specify the paths for the vocabulary and training data:

```bash
VOCAB_FILE=$data_pth/gpt2-vocab.json
MERGE_FILE=$data_pth/gpt2-merges.txt
DATA_PATH=$data_pth/codeparrot_content_document
```

# Model Parallelism

To efficiently distribute computation across multiple GPUs, tensor and pipeline parallelism sizes must be defined. 
Since Kempner nodes have multiple GPUs per node, the tensor parallel size is set to four. 
Pipeline parallelism corresponds to the number of nodes.

For example, to run the job on four nodes, each using four GPUs:

```bash
export TP_SIZE=4
export PP_SIZE=4
```

# Model Parameters

Model parameters can be customized by modifying the settings in the Slurm file. 
For example, to set the number of attention heads to 12, update the following parameter:

```bash
model.num_layers=12 \
```

Additional model parameters such as batch size, hidden size, and sequence length can also be modified accordingly.

# Job Parameters

Edit job parameters to match your requirements. Ensure that the model parallelization parameters align with the tensor and pipeline parallelism settings.

```bash
#SBATCH --job-name="gpt-meg"     # Job name
#SBATCH --partition=<partition-name>
#SBATCH --account=<account-name>
#SBATCH --nodes=2                 # Number of nodes
#SBATCH --ntasks-per-node=1        # Total number of tasks per node
#SBATCH --gpus-per-node=4          # Number of GPUs per node
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=00:35:00            # Total runtime limit (HH:MM:SS)
```

# Job Submission

After configuring the model parameters, batch sizes, optimizers, and job settings, submit the job with:

```bash
sbatch gpt2_pretraining_model_parallel.slrm
```

