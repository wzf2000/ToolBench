# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import math
import pathlib
import typing
import os
import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from accelerate.utils import DistributedType
import transformers
from transformers import Trainer, deepspeed

from toolbench.train.train_llama3 import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module,
)

from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
replace_llama_attn_with_flash_attn()

import wandb

wandb.login(
    anonymous="must",
    key = "79614754350ba92f82c0502f5e8c9625676cf1b9",
    relogin=True,
)
wandb.init(project="new-SFT", name="llama3.1-8B")

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    
    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }
    
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False
    
    # llama3.1 has a max_position_embeddings of 131072 (2^17)
    # orig_ctx_len = getattr(config, "max_position_embeddings", None)
    # if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
    #     scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
    #     config.rope_scaling = {
    #         "type": "linear",
    #         "factor": scaling_factor,
    #     }

    model: transformers.LlamaForCausalLM = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        **model_load_kwargs,
    )
    # model.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if training_args.deepspeed is not None and local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    template_id = model_args.template_id if hasattr(model_args, "template_id") else model_args.model_name_or_path

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        template_id=template_id,
        train_ratio=0.98,
        data_args=data_args
    )

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    with torch.autocast("cuda"):
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    trainer.save_state()

    # Save states. Weights might be a placeholder in zero3 and need a gather
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), lora_args.lora_bias
    )
    if local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()