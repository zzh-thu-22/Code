from typing import List
import numpy as np
import json

def read_json_objects(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        obj = json.loads(content)
    return obj

def Softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x

def create_multi_options_prompt(question: str, options: List[str]):
        option_ids = list('ABCDE')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        prompt = f"""Question: {question.strip()}
Choices:
{options_str}
Answer: """  
        return prompt

def create_no_option_prompt(question: str):
        prompt = f"""Question: {question.strip()}
Answer: """ 
        return prompt

def create_eli_prompt(question: str, options: List[str]):
        option_ids = list('ABCD')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        user_prompt = f"""Question: {question.strip()}
Choices:
{options_str}
Incorrect Answer: """ 
        return user_prompt

def create_eli_two_prompt(question: str, options: List[str]):
        option_ids = list('ABCD')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        user_prompt = f"""Question: {question.strip()}
Choices:
{options_str}
Two Incorrect Answers: """ 
        return user_prompt

def create_shots_prompt(question: str, options: List[str], shots):
        option_ids = list('ABCD')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        user_prompt = f"""{shots}Question: {question.strip()}
Choices:
{options_str}
Answer: """  
        return user_prompt

def create_shot_prompt(question: str, options: List[str], answer: str):
        option_ids = list('ABCD')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        user_prompt = f"""Question: {question.strip()}
Choices:
{options_str}
Answer:{answer}"""  
        return user_prompt

def get_top_two_indices(confidence):
    max_value = np.max(confidence)
    max_indices = np.where(confidence == max_value)[0].tolist()
    if len(max_indices) > 1:
        return max_indices
    remaining_conf = np.delete(confidence, max_indices)
    second_max_value = np.max(remaining_conf)
    second_max_indices = np.where(confidence == second_max_value)[0].tolist()
    return max_indices + second_max_indices

def get_top_three_indices(confidence):
    max_value = np.max(confidence)
    max_indices = np.where(confidence == max_value)[0].tolist()
    if len(max_indices) >= 3:
        return max_indices[:3]
    remaining_conf = np.delete(confidence, max_indices)
    second_max_value = np.max(remaining_conf)
    second_max_indices = np.where(confidence == second_max_value)[0].tolist()
    all_indices = max_indices + second_max_indices
    if len(all_indices) >= 3:
        return all_indices[:3]
    remaining_conf = np.delete(remaining_conf, np.where(remaining_conf == second_max_value))
    if len(remaining_conf) == 0:  
        return all_indices
    third_max_value = np.max(remaining_conf)
    third_max_indices = np.where(confidence == third_max_value)[0].tolist()
    all_indices += third_max_indices
    return all_indices[:3]

def get_top_four_indices(confidence):
    origin_indices = [0, 1, 2, 3, 4]
    min_value = np.min(confidence)
    min_indices = np.where(confidence == min_value)[0].tolist()
    if len(min_indices) > 1:
        return origin_indices
    indices = np.delete(origin_indices, min_indices)
    return indices