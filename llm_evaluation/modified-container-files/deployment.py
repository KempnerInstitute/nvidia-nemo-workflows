# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: This script is only an example of using NeMo with NeMo-Run's APIs and is subject to change without notice.
# This script is used for evaluation on local and slurm executors using NeMo-Run.
# It uses deploy method from nemo/llm/collections/api.py to deploy nemo2.0 ckpt on PyTriton server and uses evaluate
# method from nemo/llm/collections/api.py to run evaluation on it.
# (https://github.com/NVIDIA/NeMo-Run) to configure and execute the runs.

import argparse
from typing import Optional

import nemo_run as run

from nemo.collections.llm import deploy, evaluate
from nemo.collections.llm.evaluation.api import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Evaluation")
    parser.add_argument(
        "--nemo_checkpoint",
        type=str,
        required=True,
        help="NeMo 2.0 checkpoint to be evaluated",
    )
    parser.add_argument(
        "--triton_http_address", 
        type=str, 
        default="0.0.0.0", 
        help="IP address at which PyTriton server is created"
    )
    parser.add_argument(
        "--fastapi_port", 
        type=int, 
        default=8080, 
        help="Port at which FastAPI server is created"
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=4096,
        help="Max input length of the model",
    )
    parser.add_argument(
        "--tensor_parallelism_size",
        type=int,
        default=1,
        help="Tensor parallelism size to deploy the model",
    )
    parser.add_argument(
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="Pipeline parallelism size to deploy the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for deployment and evaluation",
    )
    parser.add_argument(
        '--nodes', 
        type=int, 
        default=2, 
        help="Num nodes for the executor"
    )
    parser.add_argument(
        '--devices', 
        type=int, 
        default=8, 
        help="Num devices per node for the executor"
    )
    return parser



def main():
    args = get_parser().parse_args()

    print("Start deployment.")

    deploy(
        nemo_checkpoint=args.nemo_checkpoint,
        fastapi_port=args.fastapi_port,
        triton_http_address=args.triton_http_address,
        max_input_len=args.max_input_len,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
        max_batch_size=args.batch_size,
        num_nodes=args.nodes,
        num_gpus=args.devices
    )


if __name__ == "__main__":
    main()
