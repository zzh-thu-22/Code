#!/bin/bash

conda activate env_name

LLM_path=(
    "/LLM/Llama-3.1-8B"
    "/LLM/Phi-3-medium-128k-instruct"
    "/LLM/Falcon3-10B-Base"
    "/LLM/Falcon3-7B-Base"
)

task=(
    'cyclic'
    'pride'
)

for m in "${LLM_path[@]}"; do
    for t in "${task[@]}"; do

        nohup python /src/$t.py --LLM_path="$m" --cuda_device="0" --output_path="/debias_result"
        wait
    done
done
