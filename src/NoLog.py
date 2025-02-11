import json
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from utils import create_multi_options_prompt, read_json_objects, get_top_two_indices, get_top_three_indices
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


def get_token_pr(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")
    input_ids.to(device)
    with torch.no_grad():
        option_ids = ['A', 'B', 'C', 'D']
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


            ### elimate 1 option at a time
            prompt = create_multi_options_prompt(question, options)
            prob = get_token_pr(prompt)

            index = get_top_three_indices(prob)
            options1 = options.copy()
            for j in range(4):
                if j not in index:
                    options1[j] = '[MASK]'

            prompt = create_multi_options_prompt(question, options1)
            prob1 = get_token_pr(prompt)
            for j in range(4):
                if j not in index:
                    prob1[j] = -np.inf

            index = np.argmax(prob1)
            index = options.index(options1[index])
            answer = option_ids[index]

            path = args.output_path + f'/{d}/{LLM_name}/NoLog_1.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')

            
            ### elimate 2 options at a time
            index1 = get_top_two_indices(prob)
            options2 = options.copy()
            for j in range(4):
                if j not in index1:
                    options2[j] = '[MASK]'

            prompt = create_multi_options_prompt(question, options2)
            prob2 = get_token_pr(prompt)
            for j in range(4):
                if j not in index1:
                    prob2[j] = -np.inf

            index = np.argmax(prob2)
            index = options.index(options2[index])
            answer = option_ids[index]
       
            path = args.output_path + f'/{d}/{LLM_name}/NoLog_2.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### sequential elimation
            index2 = get_top_two_indices(prob1)
            options3 = options1.copy()
            for j in range(4):
                if j not in index2:
                    options3[j] = '[MASK]'
    
            prompt = create_multi_options_prompt(question, options3)
            prob3 = get_token_pr(prompt)
            for j in range(4):
                if j not in index2:
                    prob3[j] = -np.inf

            index = np.argmax(prob3)
            index = options.index(options3[index])
            answer = option_ids[index]

            path = args.output_path + f'/{d}/{LLM_name}/NoLog_seq.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')





