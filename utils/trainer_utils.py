from transformers import TrainerCallback, TrainingArguments, TrainerControl, TrainerState
import torch


class MyTrainerCallback(TrainerCallback):
    log_cnt = 0
    
    def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        """
        在打印 n 次日志后清除cuda缓存，适合低显存设备，能防止OOM
        """
        self.log_cnt += 1
        
        if torch.cuda.is_available():
            if self.log_cnt % 10 == 0:
                torch.cuda.empty_cache()
    
    def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        """
        在on_epoch_end时保存一次模型。
        TrainingArguments的 save_strategy 中 epoch 和 steps 不兼容。要实现每隔 save_steps 步保存一次检查点，考虑到磁盘空间大小，最多只保存最近3个检查点。
        """
        # 设置should_save=True并返回即可
        control.should_save = True
        return control


def print_parameters_in_billions(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_in_billions = num_params / 1e9
    print(f"Model has {num_params_in_billions:.2f} billion parameters.")


my_trainer_callback = MyTrainerCallback()