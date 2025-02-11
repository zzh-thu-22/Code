import json
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from utils import create_multi_options_prompt, create_eli_prompt, create_eli_two_prompt, read_json_objects
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

dataset = ["arc", 'openbookqa', 'metaphor_understand']


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


            ### let the LLM select one incorrect option
            prompt1 = create_eli_prompt(question, options)
            prob1 = get_token_pr(prompt1, len(options))

            index = np.argmax(prob1)
            options1 = options.copy()
            options1.pop(index)

            prompt1 = create_multi_options_prompt(question, options1)
            prob1 = get_token_pr(prompt1, len(options1))

            index = np.argmax(prob1)
            index = options.index(options1[index])
            answer = option_ids[index]

            path = args.output_path + f'/{d}/{LLM_name}/EE_new_one.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### let the LLM sequentially eliminate two options
            prompt2 = create_eli_prompt(question, options1)
            prob2 = get_token_pr(prompt2, len(options1))

            index = np.argmax(prob2)
            options2 = options1.copy()
            options2.pop(index)

            prompt2 = create_multi_options_prompt(question, options2)
            prob2 = get_token_pr(prompt2, len(options2))

            index = np.argmax(prob2)
            index = options.index(options2[index])
            answer = option_ids[index]

            path = args.output_path + f'/{d}/{LLM_name}/EE_new_seq.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### let the LLM select two incorrect options
            prompt3 = create_eli_two_prompt(question, options)
            prob3 = get_token_pr(prompt3, len(options))
            options3 = options.copy()

            index = []
            index1 = np.argmax(prob3)
            index.append(index1)
            prob3[index1] = -np.inf 
            index1 = np.argmax(prob3)
            index.append(index1)
            index.sort()

            options3.pop(index[1])
            options3.pop(index[0])

            prompt3 = create_multi_options_prompt(question, options3)
            prob3 = get_token_pr(prompt3, len(options3))

            index = np.argmax(prob3)
            index = options.index(options3[index])
            answer = option_ids[index]
       

            path = args.output_path + f'/{d}/{LLM_name}/EE_new_two.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')





