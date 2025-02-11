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

dataset = ["arc", 'CC', 'openbookqa', 'physics',  'metaphor_understand', 'multiemo', 'phrase_relatedness']


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


            ### MCP
            prompt = create_multi_options_prompt(question, options)
            prob = get_token_pr(prompt, len(options))

            index = np.argmax(prob)
            answer = option_ids[index]
   
            path = args.output_path + f'/{d}/{LLM_name}/MCP.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')

    
            ### eliminate two options at a time
            two_index = get_top_two_indices(prob)
            two_index.sort()
            two_options = [options[idx] for idx in two_index]

            two_prompt = create_multi_options_prompt(question, two_options)
            two_prob = get_token_pr(two_prompt, len(two_options))

            index = np.argmax(two_prob)
            index1 = options.index(two_options[index])
            answer = option_ids[index1]
    
            path = args.output_path + f'/{d}/{LLM_name}/our_two.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### eliminate one option at a time
            one_index = get_top_three_indices(prob)
            one_index.sort()
            one_options = [options[idx] for idx in one_index]

            one_prompt = create_multi_options_prompt(question, one_options)
            one_prob = get_token_pr(one_prompt, len(one_options))

            index = np.argmax(one_prob)
            index1 = options.index(one_options[index])
            answer = option_ids[index1]

            path = args.output_path + f'/{d}/{LLM_name}/our_one.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### sequential elimination
            seq_index = get_top_two_indices(one_prob)
            seq_index.sort()
            seq_options = [one_options[idx] for idx in seq_index]

            seq_prompt = create_multi_options_prompt(question, seq_options)
            seq_prob = get_token_pr(seq_prompt, len(seq_options))

            index = np.argmax(seq_prob)
            index1 = options.index(seq_options[index])
            answer = option_ids[index1]
                  
            path = args.output_path + f'/{d}/{LLM_name}/our_seq.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')