#!/bin/bash
# find a good noise

cycles=100
i=0
frame=210

while (( ${i}<${cycles} ))
do
  echo "Running inference iteration $i ..."

  python inference.py \
      --config_path configs/self_forcing_dmd.yaml \
      --output_folder videos/self_forcing_dmd/noise_init \
      --checkpoint_path checkpoints/self_forcing_dmd.pt \
      --data_path prompts/MovieGenVideoBench_extended.txt \
      --use_ema \
      --num_output_frames "${frame}"
  ((i++))
done

echo "All $cycles cycles completed."
