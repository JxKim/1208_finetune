from peft import LoraConfig

# LoraConfig()

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B/")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

from peft import get_peft_model

lora_model = get_peft_model(model=model,peft_config=lora_config)

from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/Qwen3-0.6B-TRL-LoRA"
# 1、处理数据，处理成SFTTrainer能够接收的数据类型
from datasets import load_dataset
dataset = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})

from typing import List,Dict
def convert_to_messages_format(examples:Dict[str,List]):
    conversations:List[List[Dict]] = examples["conversation"]
    result = []
    for conversation in conversations:
        # 遍历一个batch当中的一条数据
        all_messages:Dict = conversation[0]
        new_message_list = []
        new_message_list.append({"role":"user","content":all_messages["human"]})
        new_message_list.append({"role":"assistant","content":all_messages["assistant"]})
        result.append(new_message_list)
    
    return {"messages":result}


mapped_dataset = dataset.map(convert_to_messages_format,batched=True,remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])

# 2、构造SFTConfig实例
config = SFTConfig(
    output_dir="./finetuned/Qwen3-0.6B-TRL-LoRA",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=12,
    #num_train_epochs=1,
    max_steps=1000,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # warmup_steps = ,
    bf16=True,
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=100,
    save_steps=100,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=100,
    max_length=2500,
    assistant_only_loss=True,
    report_to = ["tensorboard"],
    chat_template_path="./chat_template.jinja",
    load_best_model_at_end=True,
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    save_total_limit=3
)

from transformers import AutoModelForCausalLM, AutoTokenizer
# 3、加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(r"model/Qwen3-0.6B/")

# 4、构造SFTTrainer实例
trainer = SFTTrainer(
    model=lora_model,
    processing_class=tokenizer,
    args=config,
    train_dataset=mapped_dataset["train"],
    eval_dataset=mapped_dataset["test"]
)


# 6、调用SFTTrainer的train进行训练
trainer.train()

# 7、调用SFTTrainer实例去保存模型参数和Tokenizer
trainer.save_model("./finetuned/Qwen3-0.6B-TRL-LoRA")
