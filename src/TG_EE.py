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

dataset = ["arc", 'CC', 'openbookqa', 'physics', 'metaphor_understand', 'multiemo', 'phrase_relatedness']


def base_answer(prompt, len):
    if len == 4:
        option_ids = ['A', 'B', 'C', 'D']
    elif len == 3:
        option_ids = ['A', 'B', 'C']
    elif len == 2:
        option_ids = ['A', 'B']

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, do_sample=True, temperature=0.1, max_new_tokens=250)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)  
       
    text = text.split('Answer:')[-1].strip()
    if text == '':
       return 'none'
    else:
        text = text[0]
    
    if text not in option_ids:
        return 'none'
    else:
        index = option_ids.index(text)
        return index

def eli_answer(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, do_sample=True, temperature=0.1, max_new_tokens=250)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
       
    text = text.split('Incorrect Answer:')[-1].strip()
    if text != '':
       text = text[0]
    if 'A' in text:
        return 0
    elif 'B' in text:
        return 1
    elif 'C' in text:
        return 2
    elif 'D' in text:
        return 3
    else:
        return 'none'

def eli_two_answer(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, do_sample=True, temperature=0.1, max_new_tokens=250)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
      
    text = text.split('Two Incorrect Answers:')[-1]
    text = text.split('Correct Answer:')[0].strip()
    if 'A' in text and 'B' in text:
        return [1,0]
    elif 'A' in text and 'C' in text:
        return [2,0]
    elif 'A' in text and 'D' in text:
        return [3,0]
    elif 'B' in text and 'C' in text:
        return [2,1]
    elif 'B' in text and 'D' in text:
        return [3,1]
    elif 'C' in text and 'D' in text:
        return [3,2]
    else:
        return 'none'

           
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
                
            ### TG
            user_prompt = create_multi_options_prompt(question, options)
            index = base_answer(user_prompt, len(options))

            if index != 'none':
                answer = option_ids[index]
            else:
                answer = 'none'
            
            path = args.output_path + f'/{d}/{LLM_name}/TG.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### let the LLM select one incorrect option
            prompt1 = create_eli_prompt(question, options)
            index1 = eli_answer(prompt1)
            options1 = options.copy()

            if index1 != 'none':
                options1.pop(index1)

            prompt1 = create_multi_options_prompt(question, options1)
            index = base_answer(prompt1, len(options1))

            if index != 'none':
                index = options.index(options1[index])
                answer = option_ids[index]
            else:
                answer = 'none'

            path = args.output_path + f'/{d}/{LLM_name}/EE_one.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')

            
            ### let the LLM select two incorrect options
            prompt2 = create_eli_two_prompt(question, options)
            index2 = eli_two_answer(prompt2)
            options2 = options.copy()

            if index2 != 'none':
                for j in index2:
                  options2.pop(j)

            prompt2 = create_multi_options_prompt(question, options2)
            index = base_answer(prompt2, len(options2))

            if index != 'none':
                index = options.index(options2[index])
                answer = option_ids[index]
            else:
                answer = 'none'

            path = args.output_path + f'/{d}/{LLM_name}/EE_two.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            ### let the LLM sequentially eliminate two options
            prompt3 = create_eli_prompt(question, options1)
            index3 = eli_answer(prompt3)
            options3 = options1.copy()

            if index3 != 'none' and index3 != 3:
                options3.pop(index3)

            prompt3 = create_multi_options_prompt(question, options3)
            index = base_answer(prompt3, len(options3))

            if index != 'none':
                index = options.index(options3[index])
                answer = option_ids[index]
            else:
                answer = 'none'

            path = args.output_path + f'/{d}/{LLM_name}/EE_seq.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')
