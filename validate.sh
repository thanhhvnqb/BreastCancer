# soft_pos = 1 - smoothing
# --input-size 3 2048 1024  --log-wandb
# Datasets: rsna, vindr, miniddsm, cmmd, cddcesm, bmcd
# Models: "convnext_small.fb_in22k_ft_in1k_384", "mobilenetv3_large_100.ra_in1k", "mobilenetv3_small_050.lamb_in1k"
# Link timm models: https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
 # bmcd, vindr error
dataset=bmcd
fold_idx=3
model=mobilenetv3_small_050.lamb_in1k
checkpoint=output/train/${model}/${dataset}_fold_${fold_idx}_8/model_best.pth.tar
PYTHONPATH=$(pwd):$PYTHONPATH python src/exp/val.py -f src/exp/trainer.py \
    --dataset $dataset --exp-kwargs fold_idx=$fold_idx \
    --model $model  --checkpoint $checkpoint --num-classes 1 \
    --batch-size 8 --input-size 3 1024 512 \
    --crop-pct 1.0 --workers 24 --amp --amp-impl native \
    --use-ema --gp max --log-interval 100
    