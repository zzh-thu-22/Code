#!/bin/bash

conda activate env_name

LLM_path=(
    "/LLM/Phi-3-medium-128k-instruct"
    "/LLM/Falcon3-7B-Base"
    "/LLM/OLMo-2-1124-7B"
)

task=(
    'IE_new'
    'Log'
    'NoLog'
)

for m in "${LLM_path[@]}"; do
    for t in "${task[@]}"; do

        nohup python /src/$t.py --LLM_path="$m" --cuda_device="0" --output_path="/strategies_result"
        wait
    done
done
