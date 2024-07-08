# soft_pos = 1 - smoothing
# --input-size 3 2048 1024  --log-wandb
# Datasets: rsna, vindr, miniddsm, cmmd, cddcesm, bmcd
# Models: "convnext_small.fb_in22k_ft_in1k_384", "mobilenetv3_large_100.ra_in1k", "mobilenetv3_small_050.lamb_in1k"
# Link timm models: https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
 # bmcd, vindr error
dataset=bmcd
fold_idx=3
model=mobilenetv3_small_050.lamb_in1k
PYTHONPATH=$(pwd):$PYTHONPATH python src/exp/trainval.py -f src/exp/trainer.py \
    --dataset $dataset --experiment ${dataset}_fold_${fold_idx}  \
    --exp-kwargs fold_idx=$fold_idx \
    --model $model --pretrained --num-classes 1 \
    --batch-size 8 --validation-batch-size 16 --input-size 3 1024 512 \
    --opt sgd --lr 3e-3 --min-lr 5e-5 --sched cosine --warmup-lr 3e-5 \
    --epochs 35 --warmup-epoch 4 --cooldown-epochs 1 \
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 \
    --workers 24 --eval-metric single_pfbeta \
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
    --save-images --model-ema --model-ema-decay 0.9998 --gp max --log-interval 100
    