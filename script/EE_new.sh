#!/bin/bash

conda activate env_name

LLM_path=(
    "/LLM/Falcon3-10B-Base"
    "/LLM/Falcon3-7B-Base"
    "/LLM/OLMo-2-1124-7B"
)

task=(
    'EE_new'
)

for m in "${LLM_path[@]}"; do
    for t in "${task[@]}"; do

        nohup python /src/$t.py --LLM_path="$m" --cuda_device="0" --output_path="/EE_new_result"
        wait
    done
done
