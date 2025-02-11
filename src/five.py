import json
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from utils import create_multi_options_prompt, read_json_objects, get_top_two_indices, get_top_three_indices, get_top_four_indices
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

dataset = ['LD', 'riddle_sense']


def get_token_pr(prompt, len):
    input_ids = tokenizer(prompt, return_tensors="pt")
    input_ids.to(device)
    with torch.no_grad():
        if len == 5:
            option_ids = ['A', 'B', 'C', 'D', 'E']
        elif len == 4:
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
        path = f'/dataset/five_options/{d}.json'
        data = read_json_objects(path)
        for i in range(0, len(data)):
            case = data[i]
            question = case['question']
            option_id = ['A', 'B', 'C', 'D', 'E']
            options = case['choices']

            
            ### elimate 0 option
            prompt = create_multi_options_prompt(question, options)
            prob1 = get_token_pr(prompt, len(options))
            index = np.argmax(prob1)
            answer = option_id[index]


            path = args.output_path + f'/{d}/{LLM_name}/0.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### elimate 1 option
            index1 = get_top_four_indices(prob1)
            index1.sort()
            options1 = [options[j] for j in index1]
            prompt = create_multi_options_prompt(question, options1)

            prob2 = get_token_pr(prompt, len(options1))
            index = np.argmax(prob2)
            index = options.index(options1[index])
            answer = option_id[index]

            path = args.output_path + f'/{d}/{LLM_name}/1.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### elimate 2 options
            index2 = get_top_three_indices(prob2)
            index2.sort()
            index3 = [index1[j] for j in index2]
            options2 = [options[j] for j in index3]
            prompt = create_multi_options_prompt(question, options2)

            prob3 = get_token_pr(prompt, len(options2))
            index = np.argmax(prob3)
            index = options.index(options2[index])
            answer = option_id[index]

            path = args.output_path + f'/{d}/{LLM_name}/2.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### elimate 3 options
            index4 = get_top_two_indices(prob3)
            index4.sort()
            index5 = [index3[j] for j in index4]
            options3 = [options[j] for j in index5]
            prompt = create_multi_options_prompt(question, options3)

            prob4 = get_token_pr(prompt, len(options3))
            index = np.argmax(prob4)
            index = options.index(options3[index])
            answer = option_id[index]

            path = args.output_path + f'/{d}/{LLM_name}/3.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')
