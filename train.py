import os
import time
import torch
import wandb

from transformers import (
    Qwen2Tokenizer,
    Qwen2ForCausalLM,
    Qwen2Config,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.args import parse_args
from utils.data_utils import DataIdProcessor, my_torch_call
from utils.trainer_utils import my_trainer_callback, print_parameters_in_billions

pretrain_args = parse_args()
print("=====" * 20 + "\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")  # For logging

# ## 第一次训练，直接注释掉就行
# if pretrain_args.report_to == "wandb":
#     # wandb.login() # 如果么有配置过，则需要先手动配置下，推荐命令行的方法，paste 一个 api token 就行
#     # turn off watch to log faster
#     os.environ["WANDB_WATCH"] = "false"
#     os.environ["WANDB_MODE"]="offline"
#     os.environ["WANDB_CACHE_DIR"] = pretrain_args.wandb_cache_path
#     # wandb.init(id="76wd7a71",  # 只有 resume run 需要 ； 初始化时应该是不需要的
#     #            project="huggingface",
#     #            resume="allow")
#     id = wandb.util.generate_id()
#     print(f"启用 wandb 记录 Log; 本地保存在 {pretrain_args.wandb_cache_path} ; run id 为 {id}\n\n")

# 初始化配置模型 ----------------------------------------------------------------------------------------------------
tokenizer = Qwen2Tokenizer.from_pretrained(pretrain_args.model_init_path)
config = Qwen2Config.from_pretrained(pretrain_args.model_init_path)

# 自定义修改模型大小
# config.num_hidden_layers, config.num_key_value_heads, \
# config.num_key_value_heads, config.intermediate_size, config.hidden_size = 1, 8, 8, 8, 16

try:  # 配置 FlashAttention
    # from transformers.utils.import_utils import is_flash_attn_available
    # if is_flash_attn_available: # == 4.37.2 ; 4.39 竟然不行
    from flash_attn import flash_attn_func
    
    print("\nUsing flash_attn 2 from package flash_attn. \n\n")
    config.attn_implementation = "flash_attention_2" if pretrain_args.cuda_available else "eager"
    torch.set_default_dtype(torch.float16 if pretrain_args.use_fp16 else torch.bfloat16)

except Exception as e:
    if torch.cuda.is_available():
        print("\nUsing sdpa Attention. \n\n")
        config.attn_implementation = "sdpa"
        torch.set_default_dtype(torch.float16 if pretrain_args.use_fp16 else torch.bfloat16)
    else:
        print("\nUsing Eager Attention. \n\n")
        attn_implementation = "eager"

model = Qwen2ForCausalLM(config)  # 直接随机初始化 # 从`config`定义，不是`from_pretrained`。
# model = Qwen2ForCausalLM.from_pretrained(pretrain_args.model_init_path)  # 但是可以用于预测最终的理想 loss

# if accelerate.utils.is_main_process():
print_parameters_in_billions(model)

# 获取数据 -----------------------------------------------------------------------------------------------------------
data_id_processor = DataIdProcessor(tokenizer, pretrain_args)
train_dataset, eval_dataset = data_id_processor.get_iter_dataset() \
    if pretrain_args.iterable_train_dataset \
    else data_id_processor.get_maped_dataset()

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# data_collator.torch_call = my_torch_call
data_collator.torch_call = my_torch_call.__get__(data_collator, DataCollatorForLanguageModeling)

# Trainer 训练 -------------------------------------------------------------------------------------------------------
args = TrainingArguments(
    use_cpu=not pretrain_args.cuda_available,  # for Mac's M chip debug
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=pretrain_args.per_device_train_batch_size,
    # per_device_eval_batch_size=pretrain_args.per_device_eval_batch_size,
    gradient_accumulation_steps=pretrain_args.gradient_accumulation_steps,
    num_train_epochs=pretrain_args.num_train_epochs,
    max_steps=pretrain_args.MAX_STEPS,
    warmup_ratio=pretrain_args.warmup_ratio,
    learning_rate=pretrain_args.learning_rate,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.1,
    bf16=pretrain_args.cuda_available and not pretrain_args.use_fp16,
    fp16=pretrain_args.cuda_available and pretrain_args.use_fp16,
    
    load_best_model_at_end=False,
    auto_find_batch_size=True,  # 防止OOM ; 会自动找到不爆显存的 batch size，覆盖 per_device_train_batch_size
    # group_by_length=True,
    # deepspeed='./ds_config_one_gpu.json',
    ## eval -------------------------------------------------------------------------------------------
    evaluation_strategy="steps",  # 不 eval 了，只看 loss 了
    eval_steps=pretrain_args.eval_steps,
    eval_accumulation_steps=pretrain_args.eval_accumulation_steps,
    ## save -------------------------------------------------------------------------------------------
    save_steps=pretrain_args.save_steps,
    save_strategy="steps",
    save_total_limit=pretrain_args.save_total_limit,
    save_only_model=False,  # Not saving Optimier
    save_safetensors=False,  # 解决  missing keys in checkpoint loaded: ['lm_head.weight']. （ 版本问题？
    ## report ------------------------------------------------------------------------------------------
    report_to=pretrain_args.report_to,  # wandb
    logging_steps=pretrain_args.log_steps,
    log_level="info",
    # logging_first_step=True,
    ## 单机多卡配置 --------------------------------------------------------------------------------------
    # local_rank=int(os.environ.get('LOCAL_RANK', -1)),
    # dispatch_batches=False  # 多卡独立进行 batch 准备，而不是主卡分发
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[my_trainer_callback],
    # eval_dataset=eval_dataset,
)

trainer.evaluate()  # 看下最终的测试效果

# 支持断点续训
trainer.train(
    resume_from_checkpoint=False  # or just pass the checkpoint path
)
