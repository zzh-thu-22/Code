import json
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from utils import create_multi_options_prompt, read_json_objects
import argparse

parser = argparse.ArgumentParser(description="Run inference with LLM")
parser.add_argument('--LLM_path', type=str, required=True, help='Path to the LLM')
parser.add_argument('--cuda_device', type=str, default='0', help='GPU device ID to use (default: "0")')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the results')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = "cuda" if torch.cuda.is_available() else "cpu"
LLM_path = args.LLM_path
tokenizer = AutoTokenizer.from_pretrained(LLM_path, trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(LLM_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
model = model.eval()

dataset = ["arc", 'physics', 'multiemo']


def get_token_pr(prompt, len):
    input_ids = tokenizer(prompt, return_tensors="pt")
    input_ids.to(device)
    with torch.no_grad():
        if len == 4:
           option_ids = ['A', 'B', 'C', 'D']
        elif len == 3:
           option_ids = ['A', 'B', 'C']
        elif len == 2:
            option_ids = ['A', 'B']
        option_indices = [tokenizer(e).input_ids[-1] for e in option_ids] 

        logits = model(
            **input_ids
        ).logits[:, -1]

        logits = F.softmax(logits, dim=-1)
        prob = logits[0, option_indices].detach().cpu().to(torch.float32).numpy()
       
        return prob

if __name__ == "__main__":
    LLM_name = LLM_path.split('/')[-1]
    for d in dataset:
        path = f'/dataset/four_options/{d}.json'
        data = read_json_objects(path)
        for i in range(0, len(data)):
            case = data[i]
            question = case['question']
            option_ids = ['A', 'B', 'C', 'D']
            options = case['choices']

            user_prompt = create_multi_options_prompt(question, options)
            log_prob = np.log(get_token_pr(user_prompt, len(options)) + 1e-10)
            avg_log_prob = np.mean(log_prob)

            index = np.where(log_prob < avg_log_prob)[0]
            remain_index = np.where(log_prob >= avg_log_prob)[0]
            if len(index) == 3:
                answer = option_ids[np.argmax(log_prob)]
            else:
                new_options = []
                for j in range(4):
                    if j not in index:
                        new_options.append(options[j])
                user_prompt = create_multi_options_prompt(question, new_options)
                prob = get_token_pr(user_prompt, len(new_options))
                
                index = remain_index[np.argmax(prob)]
                answer = option_ids[index]

            
            path = args.output_path + f'/{d}/{LLM_name}/Log.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')

        

