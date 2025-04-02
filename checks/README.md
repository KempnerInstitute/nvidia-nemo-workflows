# Checks

A successful workflow run depends on a wide range of configuration parameters, including authentication tokens, dataset paths, container bindings, SLURM resource requests, and environment variables. Misconfigurations in any of these can cause the job to fail or behave unexpectedly. This section outlines common pitfalls and provides lightweight shell scripts and SLURM job examples to help verify the setup. These preflight checks are designed to catch basic issues early through dry runs or minimal resource submissions, ensuring the environment is correctly configured before launching full-scale jobs.


## SSL Certificate Errors

- **Problem**: The container cannot establish a secure connection to Hugging Face due to a missing TLS CA certificate bundle (`/etc/ssl/certs/ca-bundle.crt` not found).
- **Cause**: Singularity container lacked the host’s certificate bundle, and the default path differed on Rocky 8.
- **Resolution**: 

  - **Mount the Host Certificate Bundle**: Ensure the host’s certificate bundle is mounted into the container. This can be done by adding the following line to your SLURM script, note that there are other paths that are also mounted:
  
    ```bash
    export SINGULARITY_BIND="/etc/nsswitch.conf,/etc/slurm,/lib64/libnss_sss.so.2:/lib/libnss_sss.so.2,/var/run/munge:/run/munge,/slurm,/usr/bin/sacct,/usr/bin/salloc,/usr/bin/sbatch,/usr/bin/scancel,/usr/bin/scontrol,/usr/bin/scrontab,/usr/bin/seff,/usr/bin/sinfo,/usr/bin/squeue,/usr/bin/srun,/usr/bin/sshare,/usr/bin/sstat,/usr/bin/strace,/usr/lib64/libmunge.so.2,/usr/lib64/slurm,/var/lib/sss,/etc/pki/ca-trust/extracted/pem/,/sys,/usr/lib64/pmix,/etc/pki/tls/certs/:/etc/ssl/certs/"
    ``` 
  - **Verify**: After launching an [interactive job](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/kempner_cluster/accessing_gpu_by_fasrc_users.html#interactive-jobs), check if the certificate bundle is correctly mounted by running:
  
    ```bash
    SINGULARITY_IMAGE="<path-to-singularity-image>"
    singularity exec --nv $SINGULARITY_IMAGE curl https://huggingface.co
    ```
    This should return a valid html page scritp.  

## HuggingFace Authentication (401 Unauthorized)

- **Problem**: The job fails with a 401 Unauthorized error when trying to access a HuggingFace model or dataset.
- **Cause**: The model is gated, requiring a valid token, and the token wasn't initially provided or was incorrect. 
- **Resolution**: 

  - Verify HuggingFace Token and Access
    - Log in to https://huggingface.co, navigate to `Settings > Access Tokens`, and create a new **Read** token and store it in secure note. It should look like `hf_xxx...`.
    - To check if the token is valid, run the following command in an [interactive shell](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/kempner_cluster/accessing_gpu_by_fasrc_users.html#interactive-jobs):
    
    ```bash
    # First export singularity bind, then run the following commands
    SINGULARITY_IMAGE="<path-to-singularity-image>"
    singularity exec --nv $SINGULARITY_IMAGE python3 -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('gpt2', token='your_token_here'); print('Success')"
    ```
    This should return `Success`. If it fails, check the token and try again.

## Gated Repository Access (403 Forbidden)

- **Problem**: Even with a valid token, access to meta-llama/Meta-Llama-3-8B was denied with a 403 error.
- **Cause**: The model is gated and requires explicit access permissions.
- **Resolution**: 
  - Request access to the model by visiting the model page on HuggingFace and clicking on the "Request Access" button.
  - A list of repo names with their Request Status can be found at https://huggingface.co/settings/gated-repos .
  - After receiving access, verify it by running the following command in an [interactive shell](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/kempner_cluster/accessing_gpu_by_fasrc_users.html#interactive-jobs):
  
    ```bash
    # First export singularity bind, then run the following commands
    SINGULARITY_IMAGE="<path-to-singularity-image>"
    singularity exec --nv $SINGULARITY_IMAGE python3 -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', token='your_token_here'); print('Success')"
    ```
    This should return `Success`. If it fails, check the token, check the status of your request to the gated repository and try again.

## Singularity Image Verification

The shared folder for singularity image can have multiple images. To ensure that you are using the correct image, you can verify the image by checking its MD5 hash.

```bash
SINGULARITY_IMAGE="<path-to-singularity-image>"
md5sum $SINGULARITY_IMAGE
```
This should return the MD5 hash of the image. You can compare this hash with the one provided in the table to ensure you are using the correct image. 

Please note that the MD5 hash is not a security feature, but rather a way to ensure that the image has not been tampered with. If you want to verify the integrity of the image, you can use the `singularity verify` command. This command checks the signature of the image which is not available in the current image.

> [!NOTE]
> Please add any other common errors you encounter to this section. This will help others troubleshoot similar issues in the future. Feel free to open and [issue](https://github.com/KempnerInstitute/nvidia-nemo-workflows/issues) in the GitHub repository if you need assistance or have questions.