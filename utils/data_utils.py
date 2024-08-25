from datasets import Dataset, load_dataset
from typing import List, Union, Dict, Any, Mapping
import random
import tqdm
import numpy as np
import pyarrow.parquet as pq
import os

from torch.utils.data import IterableDataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch


class DataIdProcessor():
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.get_map_dtype()
        self.args = args
        self.get_parquet_files(args.train_dirs)
    
    def get_map_dtype(self):
        vocab_size = len(self.tokenizer)
        if vocab_size % 64 != 0:
            vocab_size = (vocab_size // 64 + 1) * 64
            # ## token to id缓存到文件，使用的时候不用再次tokenize
        # 如果词表大小小于 65535 用uint16存储，节省磁盘空间，否则用uint32存储
        map_dtype = np.uint16 if vocab_size < 65535 else np.uint32
        self.map_dtype = map_dtype
        print(f"final vocab size: {vocab_size} ; use map dtye {map_dtype}")
    
    def __get_size_of_praquet(self, file_name: str) -> int:
        '''  获取一个parquet文件的行数  '''
        parquet_data = pq.read_table(file_name)
        
        return parquet_data.num_rows
    
    def token_to_id(self, samples: dict) -> dict:
        batch_txt = samples["text"]
        outputs = self.tokenizer(
            batch_txt,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        input_ids = [np.array(item, dtype=self.map_dtype) for item in outputs["input_ids"]]
        return {"input_ids": input_ids}  # 这里没有加特殊token
    
    def get_parquet_files(self, eval_dirs):
        # Read files
        parquet_list = []
        for directory in eval_dirs:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_list.append(os.path.join(root, file))
        random.shuffle(parquet_list)
        self.parquet_list = parquet_list
        
        # Get instance num
        self.instance_num = 0
        for p_name in tqdm.tqdm(self.parquet_list, desc="Loop each parquet"):
            self.instance_num += self.__get_size_of_praquet(p_name)
        
        # Log
        print("will load all parquet data from \n", "\n\t".join(parquet_list), "Of total length", self.instance_num)
        total_batches = self.args.num_train_epochs * self.instance_num
        per_step_batches = self.args.per_device_train_batch_size * int(os.environ.get('WORLD_SIZE', 1))
        per_step_batches *= self.args.gradient_accumulation_steps
        self.args.MAX_STEPS = 1 + int(total_batches / per_step_batches)  # 这里额外注意，值是真实更新step数，非pass数
        print(
            f"\n\nTotal data set length: {self.instance_num} | total_batches: {total_batches}\n"
            f"word size is {int(os.environ.get('WORLD_SIZE')) if os.environ.get('WORLD_SIZE') else -1} | "
            f"per_update_step_batches (including accumulate steps): {per_step_batches} | "
            f"per_acc_token_num: {per_step_batches * 512 / 1e6:.4f} \n"
            f"Training step num is {self.args.MAX_STEPS}.\n\n"
            f"Save model each {self.args.save_steps} updating steps ; eval model each {self.args.eval_steps} steps ; log each {self.args.log_steps} steps")
    
    def get_iter_dataset(self):
        eval_dataset = None
        
        # self.args.save_steps = 1 + self.args.save_steps // per_step_batches
        # self.args.eval_steps = self.args.save_steps
        
        print("Using Iterable Dataset")
        
        dataset = load_dataset(
            path="parquet", data_files=self.parquet_list,
            streaming=True, keep_in_memory=False,
            split="train",
            # cache_dir=".cache", # 是否缓存数据，占用空间较大
        )
        
        dataset = dataset.shuffle(buffer_size=5_000)
        # dataset = dataset.with_format("torch") # candidate Solution Option1
        # dataset = IterableWrapper(dataset)  # candidate Solution Option2
        
        maped_dataset = dataset.map(self.token_to_id, batched=True)
        
        if self.args.eval_files:
            eval_dataset = load_dataset(
                path="parquet",
                data_files=self.args.eval_files,
                split="train",  # 这里不重要，只是需要传入这个参数让返回值为 arrow dataset
                # cache_dir=".cache", # 是否缓存数据，占用空间较大
                keep_in_memory=False,
            )
            
            eval_dataset = eval_dataset.to_iterable_dataset()
            eval_dataset = eval_dataset.map(self.token_to_id, batched=True)
        
        return maped_dataset, eval_dataset
    
    def get_maped_dataset(self) -> Dataset:
        eval_dataset = None
        
        # self.args.save_steps = 1 + self.args.save_steps // per_step_batches
        # self.args.eval_steps = self.args.save_steps
        
        dataset = load_dataset(
            path="parquet",
            data_files=self.parquet_list,
            split="train",
            # cache_dir=".cache", # 是否缓存数据，占用空间较大
            keep_in_memory=False,
        )
        dataset = dataset.shuffle(seed=42, keep_in_memory=False)
        
        maped_dataset = dataset.map(
            self.token_to_id,
            batched=True,
            batch_size=10000,
            remove_columns=dataset.column_names,
            num_proc=24,  # 不兼容 iterable dataset
            keep_in_memory=False,
        )
        
        if self.args.eval_files:
            eval_dataset = load_dataset(
                path="parquet",
                data_files=self.args.eval_files,
                split="train",  # 这里不重要，只是需要传入这个参数让返回值为 arrow dataset
                # cache_dir=".cache", # 是否缓存数据，占用空间较大
                keep_in_memory=False,
            )
            print("Load eval data set from", self.args.eval_files, "of length", len(eval_dataset))
            
            eval_dataset = eval_dataset.map(
                self.token_to_id,
                batched=True,
                batch_size=10000,
                remove_columns=dataset.column_names,
                num_proc=24,  # 不兼容 iterable dataset
                keep_in_memory=False,
            )
        
        return maped_dataset, eval_dataset


def my_torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
    # Handle dict or lists with proper padding and conversion to tensor.
    if isinstance(examples[0], Mapping):
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        )
    else:
        batch = {
            "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        }
    
    # If special token mask has been preprocessed, pop it from the dict.
    # special_tokens_mask = batch.pop("special_tokens_mask", None)
    labels = batch["input_ids"].clone()
    # if self.tokenizer.pad_token_id is not None:
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch
