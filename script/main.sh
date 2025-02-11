#!/bin/bash

conda activate env_name

LLM_path=(
    "/LLM/llama3-8b" 
    "/LLM/Llama-3.1-8B"
    "/LLM/Mistral-7B-Instruct-v0.2"
    "/LLM/Mistral-7B-Instruct-v0.3"
    "/LLM/Qwen2-7B"
    "/LLM/Phi-3-medium-128k-instruct"
    "/LLM/Falcon3-10B-Base"
    "/LLM/Falcon3-7B-Base"
    "/LLM/Falcon3-1B-Base"
    "/LLM/OLMo-2-1124-7B"
)

task=(
    'our_MCP'
    'AVG'
    'LM'
    'channel'
    'PMI'
    'AOLP'
    'IE'
    'TG_EE'
)

for m in "${LLM_path[@]}"; do
    for t in "${task[@]}"; do

        nohup python /src/$t.py --LLM_path="$m" --cuda_device="0" --output_path="/main_result"
        wait
    done
done
