#!/bin/bash
#SBATCH --job-name=TSB_CNNNormalize
#SBATCH --account=F202500002ALVLABDEUCALIONG
#SBATCH --partition=dev-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00

source /share/env/module_select.sh

module load Miniconda3
ml CUDA/12.1


source /eb/x86_64/software/Miniconda3/23.5.2-0/bin/activate

conda activate torch

output_dir="runs/dev/CNNNormalize/"
mkdir -p "$output_dir"

dataset_dir="Datasets/TSB-AD-M/"
datasets=("$dataset_dir"*) # Store the list of datasets in an array
num_datasets="${#datasets[@]}"
gpu_index=0
max_parallel=4 # Set the maximum number of parallel jobs (equal to the number of GPUs)
running_jobs=()

echo "Processing up to $max_parallel files in parallel..."

for ((i=0; i<num_datasets; i++)); do
  dataset="${datasets[$i]}"

  if [ -f "$dataset" ]; then
    # Extract the base name of the file
    filename=$(basename "$dataset")
    # Remove the extension (if any) - optional
    filename_without_ext="${filename%.*}"
    output_file="$output_dir/${filename_without_ext}.out" # Or just "$output_dir/$filename"

    echo "Launching job for: $dataset on GPU $gpu_index, output to: $output_file"
    CUDA_VISIBLE_DEVICES="$gpu_index" python -m benchmark_exp.Run_Detector_M --AD_Name CNNNormalize --save True --dataset_dir "$dataset_dir" --filename "${filename_without_ext}.csv" >> "$output_file" &

    pid="$!"
    running_pids+=("$pid")
    echo "Launched with PID: $pid"
    gpu_index=$(( (gpu_index + 1) % max_parallel ))

    if [ ${#running_pids[@]} -ge 4 ]; then
      echo "Waiting for PIDs: ${running_pids[@]}"
      wait "${running_pids[@]}"
      echo "Finished waiting for batch."
      running_pids=()
    fi
  fi
done

wait "${running_pids[@]}"

echo "Finished processing all files in Datasets/TSB-AD-M/"




