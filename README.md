# nvidia-nemo-workflows

## NeMo framework setup

NeMo can be installed using either Conda or the NGC container, with the latter being the preferred method. On the Kempner AI cluster, we will use Singularity instead of Docker. The Singularity image is available at:
```
/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo-25.02.rc0.simg
```

It is recommended to mount volumes before launching a job. This can be done by specifying Singularity bind options by passing bind options directly when launching the container or predefining them using the SINGULARITY_BIND environment variable. To ensure the necessary bindings are set, export the SINGULARITY_BIND variable either in your shell or within the Slurm job script.

```
export SINGULARITY_BIND="/etc/nsswitch.conf,/etc/slurm,/lib64/libnss_sss.so.2:/lib/libnss_sss.so.2,/var/run/munge:/run/munge,/slurm,/usr/bin/sacct,/usr/bin/salloc,/usr/bin/sbatch,/usr/bin/scancel,/usr/bin/scontrol,/usr/bin/scrontab,/usr/bin/seff,/usr/bin/sinfo,/usr/bin/squeue,/usr/bin/srun,/usr/bin/sshare,/usr/bin/sstat,/usr/bin/strace,/usr/lib64/libmunge.so.2,/usr/lib64/slurm,/var/lib/sss,/etc/pki/ca-trust/extracted/pem/,/sys,/etc/ssl/certs/,/usr/lib64/pmix,"

```

## Finetuning

We will describe distributted finetuning of Llama3 (8B and 70B) models with NeMo framework. We will explore how to do full finetuning and pefermance efficient finetuning using the Tensor and Pipeline parallel approaches. Later we will also see the use of fully sharded data parallel applications for the finetuning large models. 

1. Full Finetuning
2. Low Rank Adaptation (LoRA)
3. p-tuning (prompt-tuning)

## Pretraining

Here we will work with GPT2 model.

Our examples cover the following datasets, models, and parallism approaches. 
 * Data: codeparrot
 * Models: GPT2, Nemotron
 * Parallelism: Tensor and Pipeline Parallelism


