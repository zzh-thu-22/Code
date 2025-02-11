#!/bin/bash

conda activate env_name

LLM_path=(
    "/LLM/Llama-3.1-8B"
    "/LLM/Phi-3-medium-128k-instruct"
    "/LLM/OLMo-2-1124-7B"
)

task=(
    'five'
)

for m in "${LLM_path[@]}"; do
    for t in "${task[@]}"; do

        nohup python /src/$t.py --LLM_path="$m" --cuda_device="0" --output_path="/five_result"
        wait
    done
done
