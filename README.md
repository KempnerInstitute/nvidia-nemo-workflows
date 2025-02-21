# nvidia-nemo-workflows

## Finetuning

We will describe distributted finetuning of Llama3 (8B and 70B) models with NeMo framework. We will explore how to do full finetuning and pefermance efficient finetuning using the Tensor and Pipeline parallel approaches. Later we will also see the use of fully sharded data parallel applications for the finetuning large models. 

1. LoRA 
2. QLoRA
3. Full Finetuning

Our examples cover the following datasets, models, and parallism approaches. 
  * Data: databricks-dolly-15k
  * Models: Llama3-8B, Llama3-70B
  * Parallelism: Tensor, Pipeline, and Fully Sharded Data Parallel
   
## Pretraining

Here we will start with GPT2 model and try up 3B models. 

Our examples cover the following datasets, models, and parallism approaches. 
 * Data: codeparrot, wikitext
 * Models: GPT2, Nemotron
 * Parallelism: Tensor, Pipeline, and Fully Sharded Data Parallel


