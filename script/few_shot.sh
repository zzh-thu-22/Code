#!/bin/bash

conda activate env_name

LLM_path=(
    "/LLM/Llama-3.1-8B"
    "/LLM/Phi-3-medium-128k-instruct"
    "/LLM/Falcon3-10B-Base"
)

task=(
    'few_shot_our_MCP'
    'few_shot_IE'
)

seeds=(
    0
    1
    2
)

shot_number=(
    1
    5
    10
)

for seed in "${seeds[@]}"; do
    for shots in "${shot_number[@]}"; do
        for m in "${LLM_path[@]}"; do
            for t in "${task[@]}"; do

                nohup python /src/$t.py --LLM_path="$m" --cuda_device="0" --output_path="/few_shot_result" --seed="$seed" --shots_number="$shots"
                wait
            done
        done
    done
done
