import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser")
    
    # 模型初始化部分
    parser.add_argument("--model_init_path", type=str, default="Qwen/Qwen1.5-0.5B", help="Model initialization path")
    parser.add_argument("--model_save_dir", type=str, default="./model_save/pre/", help="Model save directory")
    parser.add_argument("--logs_dir", type=str, default="./logs/", help="Logs directory")
    parser.add_argument("--train_dirs", nargs="+", type=str, default=['data/wiki_chinese/'],
                        help="List of directories that contains train files; will loop each and find all inner parquets")
    parser.add_argument("--eval_files", nargs="+", type=str,
                        default=['data/skypile/parquet/dev10K_2020-40_zh_head_0009.parquet'],
                        help="List of train files")
    
    # Trainer 部分
    parser.add_argument("--iterable_train_dataset", action="store_true",
                        help="The data set is too big, whether to use itervble dataset, see datasets.load_dataset func")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Using fp16 as the half precison calculation; if not, use bf16")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_accumulation_steps", type=int, default=100)  # Avoid OOV
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_steps", type=int, default=20000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--wandb_cache_path", type=str, default="/wandb")
    
    args = parser.parse_args()
    args.cuda_available = torch.cuda.is_available()
    
    return args
