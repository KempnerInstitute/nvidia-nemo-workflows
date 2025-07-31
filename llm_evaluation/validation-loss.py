from argparse import ArgumentParser

from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec


def get_args():
    parser = ArgumentParser(description="""LM Validation Loss Evaluation""")

    parser.add_argument("--model_path", 
                        type=str, 
                        required=True, 
                        help="NeMo 2.0 model path")
    parser.add_argument("--val_data", 
                        type=str, 
                        required=True, 
                        help="Validation data path for loss evaluation")
    parser.add_argument("--gbs",
                        type=int, 
                        default=4, 
                        help="Global batch size")
    parser.add_argument("--mbs", 
                        type=int, 
                        default=4, 
                        help="Micro batch size")
    parser.add_argument("--seq_length", 
                        type=int, 
                        default=1024, 
                        help="Sequence length")
    parser.add_argument("--tp_size", 
                        type=int, 
                        default=1, 
                        help="Tensor model parallel size")
    parser.add_argument("--pp_size", 
                        type=int, 
                        default=1, 
                        help="Pipeline model parallel size")
    parser.add_argument("--devices", 
                        type=int, 
                        default=1, 
                        help="Number of GPUs per node")
    parser.add_argument("--num_nodes", 
                        type=int, 
                        default=1, 
                        help="Number of nodes")

    return parser.parse_args()


def main():
    args = get_args()
    
    data_path = args.val_data

    data =  PreTrainingDataModule(
        paths={"train": [data_path],
               "validation": [data_path],
               "test": [data_path]},
        global_batch_size=args.gbs,
        micro_batch_size=args.mbs,
        seq_length=args.seq_length,
    )

    model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
        model_path=args.model_path,
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        devices=args.devices,
        num_nodes=args.num_nodes,
        inference_only=True,
        legacy_ckpt=False,
        strategy_kwargs={"sequence_parallel": False, "replace_progress_bar": False},
        trainer_kwargs={"max_steps": 1, "limit_val_batches": 1.0},
        model_config_overrides={"sequence_parallel": False},
    )

    trainer.validate(model, data)


if __name__ == "__main__":
    main()
