from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--base_model",type=str,help="基座模型的路径")
parser.add_argument("--adapter_model",type=str,help="适配器的路径")
parser.add_argument("--merge_path",type=str,help="合并后的模型保存的路径")
args = parser.parse_args()
base_model_str = args.base_model
adapter = args.adapter_model
merge_path = args.merge_path
print("当前从外部拿到的参数为,",base_model_str,adapter)
# 1、加载BaseModel和tokenizer
base_model = AutoModelForCausalLM.from_pretrained(base_model_str)
tokenizer = AutoTokenizer.from_pretrained(base_model_str)
# 2、加载PeftModel
peft_model = PeftModel.from_pretrained(base_model,model_id=adapter)

# 3、将base model和peft model merge 在一起
merged_model = peft_model.merge_and_unload()

# 4、将merged_model保存，并且将原模型的tokenizer保存到相同的路径当中去
merged_model.save_pretrained(merge_path)
tokenizer.save_pretrained(merge_path)

