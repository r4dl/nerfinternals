#!/bin/bash

helpFunction_launch_train()
{
   echo "Usage: $0 [-s] [<gpu_list>]"
   echo -e "\t-s: Launch a single training job per gpu."
   echo -e "\t<gpu_list> [OPTIONAL] list of space-separated gpu numbers to launch train on (e.g. 0 2 4 5)"
   exit 1 # Exit program after printing help
}

vis="tensorboard"
method_name="activation-nerfacto"
single=false
while getopts "s:h" opt; do
    case "$opt" in
        s ) single=true ;;
        ? ) helpFunction_launch_train
          exit;;
    esac
done

shift $((OPTIND-1))

# Deal with gpu's. If passed in, use those.
GPU_IDX=("$@")
if [ -z "${GPU_IDX[0]+x}" ]; then
    echo "no gpus set... finding available gpus"
    # Find available devices
    num_device=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    START=0
    END=${num_device}-1
    GPU_IDX=()

    for (( id=START; id<=END; id++ )); do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo '[0-9]+')
        if [[ $free_mem -gt 10000 ]]; then
            GPU_IDX+=( "$id" )
        fi
    done
fi
echo "available gpus... ${GPU_IDX[*]}"

dataparser_opts=(
--downscale_factor 4)

method_opts=(
  --pipeline.datamanager.camera-optimizer.mode off
  --pipeline.model.use-average-appearance-embedding False
  --pipeline.model.distortion-loss-mult 0
  --pipeline.model.proposal-initial-sampler piecewise
  --pipeline.model.disable-scene-contraction False
  --pipeline.datamanager.train-num-rays-per-batch 1024
  --pipeline.datamanager.eval-num-rays-per-batch 2048
  --pipeline.model.eval-num-rays-per-chunk 2048)

DATASETS=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
date
tag=$(date +'%Y-%m-%d')
idx=0
len=${#GPU_IDX[@]}
GPU_PID=()
timestamp=$(date "+%Y-%m-%d_%H%M%S")
# kill all the background jobs if terminated:
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

dataparser="nerfstudio-data"

for dataset in "${DATASETS[@]}"; do
    if "$single" && [ -n "${GPU_PID[$idx]+x}" ]; then
        echo "Waiting for GPU ${GPU_IDX[$idx]}"
        wait "${GPU_PID[$idx]}"
        echo "GPU ${GPU_IDX[$idx]} is available"
    fi
    export CUDA_VISIBLE_DEVICES="${GPU_IDX[$idx]}"
    ns-train "${method_name}" "${method_opts[@]}" \
             --data="../../nerfstudio/data/nerf_llff_data/${dataset}${trans_file}" \
             --output-dir="../outputs" \
             --experiment-name="${dataset}" \
             --relative-model-dir=nerfstudio_models/ \
             --viewer.quit-on-train-completion=True \
             --vis "${vis}" \
             --timestamp "$timestamp" \
             ${dataparser} "${dataparser_opts[@]}" & GPU_PID[$idx]=$!
    echo "Launched ${method_name} ${dataset} on  ${tag}"
done
wait
