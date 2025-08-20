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


ENDPOINT_TYPES = {"chat": "chat/completions/", "completions": "completions/"}

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

EVAL_TASKS = COMPLETIONS_TASKS + CHAT_TASKS


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Evaluation")
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
        "--endpoint_type",
        type=str,
        default="completions",
        help="Whether to use completions or chat endpoint",
        choices=list(ENDPOINT_TYPES),
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="mmlu",
        help="Evaluation benchmark to run.",
        choices=EVAL_TASKS,
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit evaluation to `limit` samples. Default: use all samples."
    )
    parser.add_argument(
        "--parallel_requests",
        type=int,
        default=1,
        help="Number of parallel requests to send to server. Default: use default for the task.",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=None,
        help="Request timeout for querying the server. Default: use default for the task.",
    )
    return parser


def main():
    args = get_parser().parse_args()

    api_endpoint=ApiEndpoint(
        url=f"http://{args.triton_http_address}:{args.fastapi_port}/v1/{ENDPOINT_TYPES[args.endpoint_type]}",
        type=args.endpoint_type,
    )
    eval_target=EvaluationTarget(api_endpoint=api_endpoint)
    eval_params=ConfigParams(
        limit_samples=args.limit,
        parallelism=args.parallel_requests,
        request_timeout=args.request_timeout,
    )
    eval_config=EvaluationConfig(
        type=args.eval_task, 
        params=eval_params
    )

    print("Start evaluation.")
    print("API endpoint: ", api_endpoint)

    evaluate(target_cfg=eval_target, eval_cfg=eval_config)

    print("Finished evaluation.")


if __name__ == "__main__":
    main()
