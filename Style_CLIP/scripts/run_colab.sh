#!/usr/bin/env bash
set -e

pip install -r requirements.txt

if [ ! -d "stylegan2-pytorch" ]; then
  git clone https://github.com/rosinality/stylegan2-pytorch.git
fi

python src/optimize.py \
  --stylegan-dir stylegan2-pytorch \
  --stylegan-weights stylegan2-pytorch/stylegan2-ffhq-config-f.pt \
  --arcface-weights stylegan2-pytorch/model_ir_se50.pth \
  --prompt "a person with bright green hair" \
  --num-steps 200 \
  --id-lambda 0.06 \
  --l2-lambda 0.0045 \
  --output-dir results
