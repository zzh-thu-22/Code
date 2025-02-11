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

dataset = ["arc", 'openbookqa']


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

def create_permutation_options(options):
    default_options = options

    if len(default_options) == 4:
        permutation = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
    elif len(default_options) == 3:
        permutation = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    elif len(default_options) == 2:
        permutation = [[0, 1], [1, 0]]

    shuffled_options = []
    for perm in permutation:
        shuffled_option = [default_options[idx] for idx in perm]
        shuffled_options.append(shuffled_option)

    return shuffled_options

def calculate_prob(question, options):
    if len(options) == 4:
        permutation = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        probs = np.zeros(4)
        for j, option in enumerate(options):
            prompt = create_multi_options_prompt(question, option)
            prob = get_token_pr(prompt, len(option))

            opt1 = permutation[j].index(0)
            opt2 = permutation[j].index(1)
            opt3 = permutation[j].index(2)
            opt4 = permutation[j].index(3)

            probs[0] += prob[opt1]
            probs[1] += prob[opt2]
            probs[2] += prob[opt3]
            probs[3] += prob[opt4]   
        probs = probs / 4
    elif len(options) == 3:
        permutation = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        probs = np.zeros(3)
        for j, option in enumerate(options):
            prompt = create_multi_options_prompt(question, option)
            prob = get_token_pr(prompt, len(option))

            opt1 = permutation[j].index(0)
            opt2 = permutation[j].index(1)
            opt3 = permutation[j].index(2)

            probs[0] += prob[opt1]
            probs[1] += prob[opt2]
            probs[2] += prob[opt3]
        probs = probs / 3
    elif len(options) == 2:
        permutation = [[0, 1], [1, 0]]
        probs = np.zeros(2)
        for j, option in enumerate(options):
            prompt = create_multi_options_prompt(question, option)
            prob = get_token_pr(prompt, len(option))

            opt1 = permutation[j].index(0)
            opt2 = permutation[j].index(1)

            probs[0] += prob[opt1]
            probs[1] += prob[opt2]
        probs = probs / 2

    return probs



if __name__ == "__main__":
    LLM_name = LLM_path.split('/')[-1]
    for d in dataset:
        path = f'/dataset/four_options/{d}.json'
        data = read_json_objects(path)
        for i in range(0, len(data)):
            case = data[i]
            question = case['question']
            option_id = ['A', 'B', 'C', 'D']
            options = case['choices']

            ### cyclic
            options1 = create_permutation_options(options)
            probs1 = calculate_prob(question, options1)
            index = np.argmax(probs1)
            answer = option_id[index]

            path = args.output_path + f'/{d}/{LLM_name}/cyclic.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### our_cyclic
            index1 = get_top_three_indices(probs1)
            index1.sort()
            options2 = [options[j] for j in index1]
            options2 = create_permutation_options(options2)
            probs2 = calculate_prob(question, options2)

            index2 = get_top_two_indices(probs2)
            index2.sort()
            index3 = [index1[j] for j in index2]
            options3 = [options[j] for j in index3]
            options3 = create_permutation_options(options3)
            
            probs3 = calculate_prob(question, options3)
            index = np.argmax(probs3)
            index = options.index(options3[0][index])
            answer = option_id[index]

            path = args.output_path + f'/{d}/{LLM_name}/our_cyclic.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')
