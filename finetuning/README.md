
# Fine-Tuning

We will fine-tune **Llama3 8B** and **Llama3 70B** models using **full tuning, LoRA, and p-tuning** approaches. For these examples, we will use the `dolly-15k` and `pubmedqa` datasets.  

Each fine-tuning job is defined in a **Slurm job script** located within the respective directory. For example:  

- `full/sft_full_llama3-70b_dolly15k.slrm` is the Slurm script for **supervised fine-tuning (SFT) of the Llama3 70B model** on the `dolly-15k` dataset.  

Below are key variables that should be customized when running the job.  



## PEFT Scheme

The PEFT (Parameter Efficient Fine Tuning) need to be defined. For example, SCHEME="lora". 

## Data Preparation

 A pre-built dataset is provided in shared directory on the cluster.   First, define the main directory containing the required data:

```bash
data_pth="<absolute-path-to-data-including-vocab-and-documents>"
```

Then specify the dataset json files for the train, valid, and test datasets.

```
TRAIN_DS="[$data_pth/training.jsonl]"
VALID_DS="[$data_pth/validation.jsonl]"
TEST_DS="[$data_pth/test.jsonl]"
```
## Pretrained model path

The pretrained model should be in the nemo format. Here we provide the locations of 
Llama3-8B and Llama3-70B models. 

## Model Parallelism

To efficiently distribute computation across multiple GPUs, tensor and pipeline parallelism sizes must be defined. 
Since Kempner nodes have multiple GPUs per node, the tensor parallel size is set to four. 
Pipeline parallelism corresponds to the number of nodes.

For example, to run the job on four nodes, each using four GPUs:

```bash
export TP_SIZE=4
export PP_SIZE=4
```

## Job Parameters

Edit job parameters to match your requirements. Ensure that the model parallelization parameters align with the tensor and pipeline parallelism settings.

```bash
#SBATCH --job-name="gpt-meg"     # Job name
#SBATCH --partition=<partition-name>
#SBATCH --account=<account-name>
#SBATCH --nodes=8                 # Number of nodes
#SBATCH --ntasks-per-node=4        # Total number of tasks per node
#SBATCH --gpus-per-node=4          # Number of GPUs per node
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=00:35:00            # Total runtime limit (HH:MM:SS)
```

## Job Submission

After configuring the model parameters, batch sizes, optimizers, and job settings, submit the job with:

```bash
sbatch <slrm-job-file.slrm>
```

