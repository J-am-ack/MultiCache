#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
wan_python_bin="${WAN_PYTHON_BIN:-/home/yangjm/miniconda3/envs/mpa/bin/python}"
wan_gpu="${WAN_GPU:-7}"

cd "${script_dir}"
CUDA_VISIBLE_DEVICES="${wan_gpu}" \
PYTHONPATH="${project_root}/third_party/Wan2.1" \
"${wan_python_bin}" sencache.py \
  --ckpt_dir "${project_root}/checkpoints/Wan2.1-T2V-1.3B" \
  --task t2v-1.3B \
  --size '832*480' \
  --offload_model True \
  --t5_cpu \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --prompt_file test_prompt.json \
  --output_dir "${project_root}/outputs/spatial_smoke" \
  --frame_num 17 \
  --sample_steps 20 \
  --base_seed 42 \
  --spatial_cache \
  --spatial_cache_ratio 0.25 \
  --spatial_cache_block 1,4,4 \
  --measure_flops False \
  --measure_latency True
