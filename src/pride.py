import json
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from utils import create_multi_options_prompt, read_json_objects, get_top_two_indices, get_top_three_indices, Softmax
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
        ).logits[:, -1].view(-1)
        
        prob = F.softmax(
            logits[..., option_indices], dim=-1
        ).detach().cpu().to(torch.float32).numpy()
       
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

def calculate_prob(all_prior, probs):
    if len(probs) == 4:
        probs = probs / all_prior
    elif len(probs) == 3:
        probs = probs / all_prior[:3]
    elif len(probs) == 2:
        probs = probs / all_prior[:2]

    return probs

if __name__ == "__main__":
    # calculate prior
    LLM_name = LLM_path.split('/')[-1]

    for d in dataset:
        path = f'/dataset/prior/{d}_prior.json'
        data = read_json_objects(path)

        all_prior = np.zeros(4, dtype=np.float32)
        for i in range(len(data)):
            case = data[i]
            question = case['question']
            option_id = ['A', 'B', 'C', 'D']
            options = case['choices']

            options1 = create_permutation_options(options)
            prior = np.zeros(4, dtype=np.float32)
            for j, option in enumerate(options1):
                prompt = create_multi_options_prompt(question, option)
                prob = get_token_pr(prompt, len(option))
                prob = np.log(prob)
                prior += prob

            prior = prior / 4
            prior = Softmax(prior)
            all_prior += prior

        all_prior = all_prior / len(data)


        path = f'/dataset/four_options/{d}.json'
        data = read_json_objects(path)
        for i in range(0, len(data)):
            case = data[i]
            question = case['question']
            option_id = ['A', 'B', 'C', 'D']
            options = case['choices']

            
            ### pride
            prompt = create_multi_options_prompt(question, options)
            prob = get_token_pr(prompt, len(options))
            prob = calculate_prob(all_prior, prob)
            
            index = np.argmax(prob)
            answer = option_id[index]

            path = args.output_path + f'/{d}/{LLM_name}/pride.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')

            
            ### our_pride
            index1 = get_top_three_indices(prob)
            index1.sort()
            options1 = [options[j] for j in index1]
            prompt = create_multi_options_prompt(question, options1)

            prob1 = get_token_pr(prompt, len(options1))
            prob1 = calculate_prob(all_prior, prob1)
            

            index3 = get_top_two_indices(prob1)
            index3.sort()
            index4 = [index1[j] for j in index3]
            options3 = [options[j] for j in index4]

            prompt = create_multi_options_prompt(question, options3)
            prob3 = get_token_pr(prompt, len(options3))
            prob3 = calculate_prob(all_prior, prob3)

            index = np.argmax(prob3)
            index = options.index(options3[index])
            answer = option_id[index]

            path = args.output_path + f'/{d}/{LLM_name}/our_pride.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')
