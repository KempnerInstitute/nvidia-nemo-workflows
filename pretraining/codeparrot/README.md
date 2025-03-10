# GPT Model Training

The Generative Pre-trained Transformer (GPT) is a decoder-only Transformer model widely used in Large Language Models (LLMs). 
Here we outline the steps to do pretraining of GPT model. 

The file `gpt2_pretraining_model_parallel.slrm` contains information about the data, model, and compute resources used for a pretraining job. 
Please review and modify this file as needed to suit your training requirements.

# Data Preparation 

GPT model pretraining requires vocabulary files and a training dataset. Here, we provide a pre-built dataset for training. 
First, define the main directory containing the required data:

```bash
data_pth="<absolute-path-to-data-including-vocab-and-documents>"
```

Then, set the paths for the vocabulary and training data:

```bash
VOCAB_FILE=$data_pth/gpt2-vocab.json
MERGE_FILE=$data_pth/gpt2-merges.txt
DATA_PATH=$data_pth/codeparrot_content_document
```

# Model Parallelism

To efficiently distribute computation across multiple GPUs, we define tensor and pipeline parallelism sizes. 
Since Kempner nodes have multiple GPUs per node, we set the tensor parallel size to four. 
Pipeline parallelism corresponds to the number of nodes. 

For example, to run the job on four nodes, each using four GPUs:

```bash
export TP_SIZE=4
export PP_SIZE=4
```

# Model Parameters

You can customize the model parameters by modifying the settings in the Slurm file. 
For example, to set the number of attention heads to 12, update the following parameter:

```bash
model.num_layers=12 \
```
Likewise, you may edit any of the lines related to model 

```
# GPT Model Training

The Generative Pre-trained Transformer (GPT) is a decoder-only Transformer model widely used in Large Language Models (LLMs). 
All job and model parameters are defined in the file `gpt2_pretraining_model_parallel.slrm`. 
Please review and modify this file as needed to suit your training requirements.

# Data Preparation 

GPT model pretraining requires vocabulary files and a training dataset. Here, we provide a pre-built dataset for training. 
First, define the main directory containing the required data:

```bash
data_pth="<absolute-path-to-data-including-vocab-and-documents>"
```

Then, set the paths for the vocabulary and training data:

```bash
VOCAB_FILE=$data_pth/gpt2-vocab.json
MERGE_FILE=$data_pth/gpt2-merges.txt
DATA_PATH=$data_pth/codeparrot_content_document
```

# Model Parallelism

To efficiently distribute computation across multiple GPUs, we define tensor and pipeline parallelism sizes. 
Since Kempner nodes have multiple GPUs per node, we set the tensor parallel size to four. 
Pipeline parallelism corresponds to the number of nodes. 

For example, to run the job on eight nodes, each using four GPUs:

```bash
export TP_SIZE=4
export PP_SIZE=8
```

# Model Parameters

You can customize the model parameters by modifying the settings in the Slurm file. 
For example, to set the number of attention heads to 12, update the following parameter:

```bash
model.num_layers=12 \
```

```bash
    model.micro_batch_size=2 \
    model.global_batch_size=4 \
    model.max_position_embeddings=1024 \
    model.encoder_seq_length=1024 \
    model.hidden_size=768 \
    model.ffn_hidden_size=3072 \
    model.num_layers=12 \
    model.num_attention_heads=12 \
    model.init_method_std=0.021 \
    model.hidden_dropout=0.1 \
    model.layernorm_epsilon=1e-5 \
    model.data.seq_length=1024 \
```

# Job Submission

After configuring the model parameters, including batch sizes and optimizers, submit the job with:

```bash
sbatch gpt2_pretraining_model_parallel.slrm
```

# Job Submission

After configuring the model parameters, including batch sizes and optimizers, submit the job with:

```bash
sbatch gpt2_pretraining_model_parallel.slrm
```


