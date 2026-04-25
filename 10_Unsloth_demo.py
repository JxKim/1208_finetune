# 需要把Unsloth的import 放在最前面，这样，Unsloth才能对trl相关trainer打补丁
from unsloth import FastLanguageModel
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

model_path = "Qwen/Qwen3-0.6B"
model_path = "model/Qwen3-0.6B"
# 1、加载量化模型
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    # 量化配置
    load_in_4bit=True,
      use_exact_model_name=True,
    local_files_only=True,
)

# 2、配置Lora
peft_model = FastLanguageModel.get_peft_model(
    model=model,
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj",
      "up_proj",
      "down_proj"
],
    lora_dropout=0.05,
    # bias="none"
)

import os

os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/Qwen3-0.6B-TRL-SFT"
# 1、处理数据，处理成SFTTrainer能够接收的数据类型
from datasets import load_dataset
dataset = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})
sampled_train = dataset["train"].shuffle(seed=42).select(range(1000))
sampled_test = dataset["test"].shuffle(seed=42).select(range(200))
dataset["train"]=sampled_train
dataset["test"]=sampled_test
from typing import List,Dict
# 使用Unsloth时，map方法需要调整
def convert_to_messages_format(examples:Dict[str,List]):
    conversations:List[List[Dict]] = examples["conversation"]
    result = []
    for conversation in conversations:
        # 遍历一个batch当中的一条数据
        all_messages:Dict = conversation[0]
        new_message_list = []
        new_message_list.append({"role":"user","content":all_messages["human"]})
        new_message_list.append({"role":"assistant","content":all_messages["assistant"]})
        res = tokenizer.apply_chat_template(new_message_list,tokenize=False,add_generation_prompt = False)
        result.append(res)
    
    return {"messages":result}


mapped_dataset = dataset.map(convert_to_messages_format,batched=True,remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])

# 2、构造SFTConfig实例
config = SFTConfig(
    output_dir="./finetuned/Qwen3-0.6B-TRL-SFT",
    per_device_train_batch_size=3,
    gradient_accumulation_steps=4,
    dataset_text_field="messages",
    #num_train_epochs=1,
    max_steps=1000,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # warmup_steps = ,
    bf16=True,
    # gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=100,
    save_steps=100,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=100,
    max_length=2500,
    # assistant_only_loss=True,
    report_to = ["tensorboard"],
    # chat_template_path="./chat_template.jinja",
    load_best_model_at_end=True,
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    save_total_limit=3
)

from unsloth.chat_templates import train_on_responses_only

# 4、构造SFTTrainer实例
trainer = SFTTrainer(
    model=peft_model,
    processing_class=tokenizer,
    args=config,
    train_dataset=mapped_dataset["train"],
    eval_dataset=mapped_dataset["test"]
)

trainer = train_on_responses_only(
    trainer=trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"
)




# 6、调用SFTTrainer的train进行训练
trainer.train()

# 7、调用SFTTrainer实例去保存模型参数和Tokenizer
# trainer.save_model("./finetuned/Qwen3-0.6B-TRL-SFT-Unsloth")

# 将适配器模型的权重和原始模型的权重直接做一个合并
peft_model.save_pretrained_merged("./finetuned/Qwen3-0.6B-TRL-SFT-Unsloth-Merged",tokenizer,save_method = "merged_16bit")