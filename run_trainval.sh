# soft_pos = 1 - smoothing
# --input-size 3 2048 1024  --log-wandb
PYTHONPATH=$(pwd):$PYTHONPATH python src/tools/exp/trainval.py -f src/tools/exp/trainer.py \
    --dataset rsna --experiment rsna_fold_3  \
    --exp-kwargs fold_idx=3 num_sched_epochs=10 num_epochs=35 \
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True \
    --model convnext_small.fb_in22k_ft_in1k_384 --pretrained --num-classes 1 \
    --batch-size 16 --validation-batch-size 8 --input-size 3 1024 512 \
    --opt sgd --lr 3e-3 --min-lr 5e-5 --sched cosine --warmup-lr 3e-5 \
    --epochs 30 --warmup-epoch 4 --cooldown-epochs 1 \
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 \
    --workers 24 --eval-metric gbmean_best_pfbeta \
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
    --save-images --model-ema --model-ema-decay 0.9998 --gp max --log-interval 1