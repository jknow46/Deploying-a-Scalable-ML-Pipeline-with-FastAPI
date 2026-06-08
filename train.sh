#!/bin/bash
export WANDB_MODE=offline
python components/train_random_forest/run.py \
  --trainval_artifact data/trainval.csv \
  --rf_config components/train_random_forest/feature_engineering/parameters.json \
  --output_artifact model_export