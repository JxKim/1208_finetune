from dataclasses import dataclass
import datasets
from typing import List
@dataclass
class SFTConfig:
    """SFT过程当中，所有的相关配置"""

    train_data_size:int = 5000
    device:str = "cuda"
    max_lr_rate:float = 2e-5
    min_lr_rate:float = 2e-6
    batch_size :int = 4
    warmup_ratio:float = 0.1
    logging_steps:int = 100
    output_dir:str ="./finetuned/Qwen3-0.6B-SFT"

from transformers import AutoTokenizer,AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B-Base")
# 1、加载数据
def get_data_ultrachat_200k(config:SFTConfig):
    # 2、下载并处理数据集
    # 2.1 加载UltraChat 200k数据集
    ultrachat_200k_data:datasets.DatasetDict = datasets.load_dataset("./data/ultrachat_200k")
    
    train_data = [] # 该方法，最终返回的需要训练的数据列表
    i = 0
    # 2.2 对数据进行tokenize处理
    while True:
        data:List = ultrachat_200k_data["train_sft"][i]["messages"]
        
        data.insert(0,{"role":"system","content":"You are a helpful assistant."})
        input_ids = tokenizer.apply_chat_template(data,tokenize=True,add_generation_prompt=False,truncation=True,max_length=2500)
        
        train_data.append(input_ids)
        i += 1
        if i % 1000 == 0:
            print(f"已经处理了 {i} 条数据")
        # 仅训练train_data_size条数据
        if i == config.train_data_size:
            break
    
    return train_data

from transformers import PreTrainedTokenizerFast
import torch
def create_answer_mask(input_ids,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从input_ids当中找出assistant回答的部分，然后输出一个与input_ids相同shape的mask，
    后续将其与pad_mask进行逻辑与操作，得到最终的mask，用以计算损失
    """

    
    # 构建answer mask，输入的input_ids为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和input_ids相同shape的全0矩阵
    answer_mask = torch.zeros_like(input_ids)

    # 2. 遍历input_ids中的每一条数据，查找assistant回答的部分，将其设置为1
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    for idx,ids in enumerate(input_ids):
        # 获取到所有的eos_position
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()

        # 排除第一个eos_position: 第一个对应的是system prompt
        eos_position = eos_position[1:]
        # 解析获得user_ends和assistant_ends
        user_ends,assistant_ends = _parse_conversation_turns(eos_position)
        # 设置answer mask
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)   
    
    # 结果返回:
    return answer_mask

def _parse_conversation_turns(eos_positions:List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.2<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取（具体索引位置需要加一，因为eos_token_id后面还有一个\n换行符），每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[11,21]
        assistant_ends:[16,26]
    """

    use_ends = [pos+1 for pos in eos_positions[::2]]
    assistant_ends = [pos+1 for pos in eos_positions[1::2]]

    return use_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么user_turns:[11,21]，assistant_ends:[16,26]

    user_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    assistant_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    要想获取到assistant的回答的起始位置，就需要再跳过\n,<|im_start|>,assistant 这三个token，所以需要加3.
    要想获取到assistant的回答的结束位置，就需要往前跳一个<|im_end|>，所以需要减1.
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    if num_user_turns == num_assistant_turns:
        for user_end,assistant_end in zip(user_ends,assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

    elif num_user_turns == num_assistant_turns + 1:
        for user_end,assistant_end in zip(user_ends[:-1],assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1] 
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1

def compute_loss(model_logits,target_labels,assistant_answer_mask):
    """
    model_logits: 模型前向传播之后得到的结果
    target_labels: 答案
    assistant_answer_mask: 计算损失时使用的掩码, 0的位置，表示不需要计算损失，1的位置表示需要计算损失
    """
    # model_logits: [batch_size, num_tokens, vocab_size]
    token_probs = torch.log_softmax(model_logits,dim=-1)
    # 需要得到，token输出答案当中对应的token的概率大小
    # target_labels:[batch_size, num_tokens]
    # target_labels.unsqueeze(-1)： [batch_size, num_tokens,1]
    answer_probs = torch.gather(
        token_probs,
        dim=-1,
        index=target_labels.unsqueeze(-1)
    )

    masked_answer_probs = torch.mul(answer_probs.squeeze(-1),assistant_answer_mask)

    negative_log_probs = -1 * masked_answer_probs

    total_loss = negative_log_probs.sum()

    token_nums = assistant_answer_mask.sum()

    average_loss = total_loss / token_nums

    return average_loss

def cosine_decay(total_steps,current_step,warmup_ratio,max_learning_rate, min_learning_rate):
    """
    学习率调度器：
    1000，0.1，
    学习率在warmup_ratio*total_steps这么step范围内，从0开始，先线性增长到max_learning_rate
    """
    import numpy as np
    warmup_steps = total_steps * warmup_ratio

    if current_step<warmup_steps:
        return max_learning_rate*current_step / warmup_steps
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        decay = 0.5 * (1+np.cos(np.pi * progress))
        return min_learning_rate + (max_learning_rate-min_learning_rate) * decay
    

from torch.utils.tensorboard import SummaryWriter
def train(model,tokenizer,config:SFTConfig):
    
    model.to(config.device)
    model.train()

    # 构造相关的实例
    # 1、优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.max_lr_rate)
    # 2、获取数据集
    train_data = get_data_ultrachat_200k(config)

    # 3、算出总共有多少step
    total_steps = (len(train_data) + config.batch_size -1)// config.batch_size

    # 4、SummaryWriter
    writer = SummaryWriter(log_dir="logs/Qwen3-0.6B-SFT")
    # 5、引入progress bar
    import tqdm
    progress_bar = tqdm.tqdm(total=total_steps,desc="step")
    total_loss_list = []
    for step in range(total_steps):
        # 1、从train_data中得到当前batch的数据
        batch_data = train_data[step * config.batch_size:(step+1) * config.batch_size]

        # 2、padding
        max_len = max([len(seq["input_ids"]) for seq in batch_data])

        padded_seqs = []
        for seq in batch_data:
            current_seq_length = len(seq["input_ids"])
            padding_length = max_len - current_seq_length
            padded_seq = torch.nn.functional.pad(torch.tensor(seq["input_ids"],dtype=torch.long),(0,padding_length),value=tokenizer.pad_token_id)
            padded_seqs.append(padded_seq.tolist())
        
        # 3、构造input_ids和labels
        padded_seq_tensor = torch.tensor(padded_seqs,dtype=torch.long).to(config.device)

        input_ids = padded_seq_tensor[:,:-1]
        labels = padded_seq_tensor[:,1:]
        

        # 4、构建掩码
        assistant_answer_mask = create_answer_mask(input_ids,tokenizer)
        # padding_mask：是pad_token_id的地方为0，不是pad_token_id为1
        padding_mask = torch.where(labels==tokenizer.pad_token_id,0,1)
        final_mask = assistant_answer_mask * padding_mask

        if final_mask.sum() == 0:
            print("当前batch当中，没有需要计算损失的token，忽略")
            progress_bar.update(1)
            continue

        # 5、前向传播

        output_logits = model(input_ids).logits

        # 6、计算损失
        loss = compute_loss(output_logits,labels,final_mask)

        total_loss_list.append(loss.item())
        # 7、loss反向传播
        loss.backward()

        current_learning_rate = cosine_decay(
            total_steps = total_steps,
            current_step = step,
            warmup_ratio = config.warmup_ratio,
            max_learning_rate=config.max_lr_rate,
            min_learning_rate=config.min_lr_rate
        )

        for param_group in optimizer.param_groups:
            param_group["lr"]=current_learning_rate

        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{total_loss_list[-1]:.4f}", lr=f"{current_learning_rate:.2e}")


        should_log = (step+1) % config.logging_steps == 0 or (step+1) == total_steps
        if should_log:
            # 获取到logging_steps个loss
            loss_list = total_loss_list[-config.logging_steps:]
            average_loss = sum(loss_list) / config.logging_steps

            writer.add_scalar("Loss",scalar_value=average_loss,global_step=step)

def save_model_tokenizer(model,tokenizer,output_dir):
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型和tokenizer已经保存至:{output_dir}")


def main():

    sft_config = SFTConfig()
    train(model=model,tokenizer=tokenizer,config=sft_config)
    save_model_tokenizer(model,tokenizer,sft_config.output_dir)

if __name__ =="__main__":
    main()