import json
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from utils import create_no_option_prompt, read_json_objects
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


def cal_score(prompt, option):
    input_text = prompt + option
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_length = prompt_ids.shape[1]
   
    option_ids = input_ids[:, prompt_length:]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    log_probs = 0
    for i in range(option_ids.shape[1]):
        token_logits = logits[0, prompt_length + i -1, :]
        token_log_probs = F.log_softmax(token_logits, dim=-1)
        token_log_prob = token_log_probs[option_ids[0, i]]
        log_probs += token_log_prob.item()
    
    if option_ids.shape[1] == 0:
        return -np.inf
    avg_log_prob = log_probs / option_ids.shape[1]

    return avg_log_prob


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

            scores = np.zeros(len(options))
            for j, option in enumerate(options):
                prompt = create_no_option_prompt(question)
                score = cal_score(prompt, option)
                scores[j] = score
            
            index = np.argmax(scores)
            answer = option_id[index]

            path = args.output_path + f'/{d}/{LLM_name}/AVG.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')
