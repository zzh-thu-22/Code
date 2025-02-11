import json
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import random
from utils import create_shots_prompt, create_shot_prompt, read_json_objects
import argparse

parser = argparse.ArgumentParser(description="Run inference with LLM")
parser.add_argument('--LLM_path', type=str, required=True, help='Path to the LLM')
parser.add_argument('--cuda_device', type=str, default='0', help='GPU device ID to use (default: "0")')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the results')
parser.add_argument('--seed', type=int, required=True, help='random seed')
parser.add_argument('--shots_number', type=int, required=True, help='number of shots')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = "cuda" if torch.cuda.is_available() else "cpu"
LLM_path = args.LLM_path
tokenizer = AutoTokenizer.from_pretrained(LLM_path, trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(LLM_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
model = model.eval()

dataset = ['arc', 'physics']


def make_shots(number, seed_number, path):
    prompts = ''
    shot_data = read_json_objects(path)

    seed = seed_number
    random.seed(seed)

    shots = random.sample(shot_data, number)

    for shot in shots:
        question = shot['question']
        options = shot['choices']
        answer = shot['answer']
        prompt = create_shot_prompt(question, options, answer)
        prompts = prompts + prompt + '\n\n'

    return prompts

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
        token_logits = logits[0, i + prompt_length - 1, :]
        token_log_probs = F.log_softmax(token_logits, dim=-1)
        token_log_prob = token_log_probs[option_ids[0, i]]
        log_probs += token_log_prob.item()
    
    if option_ids.shape[1] == 0:
        return -np.inf    
    return log_probs


if __name__ == "__main__":
    LLM_name = LLM_path.split('/')[-1]
    for d in dataset:
        path = f'/dataset/four_options/{d}.json'
        shot_path = f'/dataset/few_shot/{d}.json'

        data = read_json_objects(path)
        shots = make_shots(args.shots_number, args.seed, shot_path)
        for i in range(0, len(data)):
            case = data[i]
            question = case['question']
            option_ids = ['A', 'B', 'C', 'D']
            options = case['choices']

            origin_scores = np.zeros(len(options))
            for j, option in enumerate(options):
                prompt = create_shots_prompt(question, options, shots)
                score = cal_score(prompt, option)
                origin_scores[j] = score
            
            avg_score = np.mean(origin_scores)
            mask_index = np.where(origin_scores < avg_score)[0]
            
            if len(mask_index) == 3:
                answer = option_ids[np.argmax(origin_scores)]
            else:
                new_options = options.copy()
                new_scores = np.zeros(len(new_options))
                for index in mask_index:
                    new_options[index] = '[MASK]'
                for j, option in enumerate(new_options):
                    new_prompt = create_shots_prompt(question, new_options, shots)
                    score = cal_score(new_prompt, option)
                    new_scores[j] = score
                new_scores[mask_index] = -np.inf
                answer = option_ids[np.argmax(new_scores)]

            path = args.output_path + f'/{args.shots_number}/{args.seed}/{d}/{LLM_name}/IE.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')
