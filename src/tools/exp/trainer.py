import logging
import math
import os

import albumentations as A
import cv2
import numpy as np
import sklearn
import torch
import torch.distributed as dist
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from ignite.distributed import DistributedProxySampler
import datasets
from timm import utils
from timm.data import (
    AugMixDataset,
    FastCollateMixup,
    Mixup,
    create_dataset,
    create_loader,
    resolve_data_config,
)
from timm.loss import (
    BinaryCrossEntropy,
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from timm.models import (
    create_model,
    load_checkpoint,
    model_parameters,
    resume_checkpoint,
    safe_model_name,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from torch.utils.data import Sampler, WeightedRandomSampler

from utils import augs
from utils import metrics as rsna_metrics
from utils import samplers
from utils.loss import BinaryCrossEntropyPosSmoothOnly
from utils.metrics import compute_usual_metrics, pfbeta_np

import settings


class ValAugment:

    def __init__(self):
        pass

    def __call__(self, img):
        return img


class TrainAugment:

    def __init__(self):
        self.transform_fn = A.Compose(
            [
                # crop
                augs.CustomRandomSizedCropNoResize(
                    scale=(0.5, 1.0), ratio=(0.5, 0.8), always_apply=False, p=0.4
                ),
                # flip
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # downscale
                A.OneOf(
                    [
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_AREA
                            ),
                            always_apply=False,
                            p=0.1,
                        ),
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LANCZOS4, downscale=cv2.INTER_AREA
                            ),
                            always_apply=False,
                            p=0.1,
                        ),
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_LINEAR
                            ),
                            always_apply=False,
                            p=0.8,
                        ),
                    ],
                    p=0.125,
                ),
                # contrast
                # relative dark/bright between region, like HDR
                A.OneOf(
                    [
                        A.RandomToneCurve(scale=0.3, always_apply=False, p=0.5),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.2),
                            contrast_limit=(-0.4, 0.5),
                            brightness_by_max=True,
                            always_apply=False,
                            p=0.5,
                        ),
                    ],
                    p=0.5,
                ),
                # affine
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=[-0.15, 0.15],
                            rotate_limit=[-30, 30],
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            shift_limit_x=[-0.1, 0.1],
                            shift_limit_y=[-0.2, 0.2],
                            rotate_method="largest_box",
                            always_apply=False,
                            p=0.6,
                        ),
                        # one of with other affine
                        A.ElasticTransform(
                            alpha=1,
                            sigma=20,
                            alpha_affine=10,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=None,
                            approximate=False,
                            same_dxdy=False,
                            always_apply=False,
                            p=0.2,
                        ),
                        # distort
                        A.GridDistortion(
                            num_steps=5,
                            distort_limit=0.3,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=None,
                            normalized=True,
                            always_apply=False,
                            p=0.2,
                        ),
                    ],
                    p=0.5,
                ),
                # random erase
                A.CoarseDropout(
                    max_holes=6,
                    max_height=0.15,
                    max_width=0.25,
                    min_holes=1,
                    min_height=0.05,
                    min_width=0.1,
                    fill_value=0,
                    mask_fill_value=None,
                    always_apply=False,
                    p=0.25,
                ),
            ],
            p=0.9,
        )

        print("TRAIN AUG:\n", self.transform_fn)

    def __call__(self, img):
        return self.transform_fn(image=img)["image"]


class ValTransform:

    def __init__(self, input_size, interpolation=cv2.INTER_LINEAR):
        self.input_size = input_size
        self.interpolation = interpolation
        self.max_h, self.max_w = input_size

        def _fit_resize(image, **kwargs):
            img_h, img_w = image.shape[:2]
            r = min(self.max_h / img_h, self.max_w / img_w)
            new_h, new_w = int(img_h * r), int(img_w * r)
            new_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            return new_image

        self.transform_fn = A.Compose(
            [
                A.Lambda(name="FitResize", image=_fit_resize, always_apply=True, p=1.0),
                A.PadIfNeeded(
                    min_height=self.max_h,
                    min_width=self.max_w,
                    pad_height_divisor=None,
                    pad_width_divisor=None,
                    position=A.augmentations.geometric.transforms.PadIfNeeded.PositionType.CENTER,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=None,
                    always_apply=True,
                    p=1.0,
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )

    def __call__(self, img):
        return self.transform_fn(image=img)["image"]


TrainTransform = ValTransform


class Exp:

    def __init__(self, args):
        # synchorize with change in args (alias of same obj)
        self.args = args
        self.data_config = None

        # infer num channels
        in_chans = 3
        if self.args.in_chans is not None:
            in_chans = args.in_chans
        elif self.args.input_size is not None:
            in_chans = args.input_size[0]
        self.args.in_chans = in_chans

        self.meta = {
            "fold_idx": 0,
            "num_sched_epochs": 6,
            "num_epochs": 50,
            "start_ratio": 1 / 3,
            "end_ratio": 1 / 7,
            "one_pos_mode": True,
        }
        old_meta_len = len(self.meta)
        self.meta.update(self.args.exp_kwargs)
        assert len(self.meta) == old_meta_len
        print("\n------\nEXP METADATA:\n", self.meta)
        self.output_dir = os.path.join(
            settings.MODEL_CHECKPOINT_DIR, "timm_classification"
        )

    def build_model(self):
        model = create_model(
            self.args.model,
            pretrained=self.args.pretrained,
            in_chans=self.args.in_chans,
            num_classes=self.args.num_classes,
            drop_rate=self.args.drop,
            drop_path_rate=self.args.drop_path,
            drop_block_rate=self.args.drop_block,
            global_pool=self.args.gp,
            bn_momentum=self.args.bn_momentum,
            bn_eps=self.args.bn_eps,
            scriptable=self.args.torchscript,
            checkpoint_path=self.args.initial_checkpoint,
            **self.args.model_kwargs,
        )

        # if utils.is_primary(self.args):
        #     _logger.info(
        #     f'Model {safe_model_name(self.args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

        assert self.data_config is None
        self.data_config = resolve_data_config(
            vars(self.args), model=model, verbose=utils.is_primary(self.args)
        )

        return model

    def build_train_dataset(self):
        assert self.data_config is not None
        fold_idx = self.meta["fold_idx"]
        augment_fn = TrainAugment()
        transform_fn = TrainTransform(self.data_config["input_size"][1:])

        rsna_train_dataset_info = {
            "csv_path": os.path.join(
                settings.PROCESSED_DATA_DIR,
                "classification",
                "rsna",
                "fold",
                f"train_fold_{fold_idx}.csv",
            ),
            "img_dir": os.path.join(
                settings.PROCESSED_DATA_DIR,
                "classification",
                "rsna",
                "cleaned_images",
            ),
        }
        train_datasets_info = [("rsna", rsna_train_dataset_info)]
        # EXTERNAL_DATASETS = ["bmcd", "cddcesm", "cmmd", "miniddsm", "vindr"]
        # for dataset_name in EXTERNAL_DATASETS:
        #     dataset_info = {
        #         "csv_path": os.path.join(
        #             settings.PROCESSED_DATA_DIR,
        #             "classification",
        #             dataset_name,
        #             "cleaned_label.csv",
        #         ),
        #         "img_dir": os.path.join(
        #             settings.PROCESSED_DATA_DIR,
        #             "classification",
        #             dataset_name,
        #             "cleaned_images",
        #         ),
        #     }
        #     train_datasets_info.append((dataset_name, dataset_info))

        train_dataset = datasets.RSNADataset(
            train_datasets_info,
            augment_fn,
            transform_fn,
            n_channels=self.args.input_size[0],
            subset="train",
        )
        return train_dataset

    def build_train_loader(self, collate_fn=None):
        train_dataset = self.build_train_dataset()

        # wrap dataset in AugMix helper
        if self.args.num_aug_splits > 1:
            train_dataset = AugMixDataset(
                train_dataset, num_splits=self.args.num_aug_splits
            )

        # create data loaders w/ augmentation pipeiine
        train_interpolation = self.args.train_interpolation
        if self.args.no_aug or not train_interpolation:
            train_interpolation = self.data_config["interpolation"]

        if self.args.distributed:
            assert not isinstance(train_dataset, torch.utils.data.IterableDataset)
            sampler = DistributedProxySampler(
                samplers.BalanceSamplerV2(
                    train_dataset,
                    batch_size=self.args.batch_size,
                    num_sched_epochs=self.meta["num_sched_epochs"],
                    num_epochs=self.meta["num_epochs"],
                    start_ratio=self.meta["start_ratio"],
                    end_ratio=self.meta["end_ratio"],
                    one_pos_mode=self.meta["one_pos_mode"],
                    seed=self.args.seed,
                )
            )
        else:
            sampler = samplers.BalanceSamplerV2(
                train_dataset,
                batch_size=self.args.batch_size,
                num_sched_epochs=self.meta["num_sched_epochs"],
                num_epochs=self.meta["num_epochs"],
                start_ratio=self.meta["start_ratio"],
                end_ratio=self.meta["end_ratio"],
                one_pos_mode=self.meta["one_pos_mode"],
                seed=self.args.seed,
            )

        train_loader = create_loader(
            train_dataset,
            input_size=self.data_config["input_size"],
            batch_size=self.args.batch_size,
            is_training=True,
            use_prefetcher=self.args.prefetcher,
            no_aug=self.args.no_aug,
            re_prob=self.args.reprob,
            re_mode=self.args.remode,
            re_count=self.args.recount,
            re_split=self.args.resplit,
            scale=self.args.scale,
            ratio=self.args.ratio,
            hflip=self.args.hflip,
            vflip=self.args.vflip,
            color_jitter=self.args.color_jitter,
            auto_augment=self.args.aa,
            num_aug_repeats=self.args.aug_repeats,
            num_aug_splits=self.args.num_aug_splits,
            interpolation=train_interpolation,
            mean=self.data_config["mean"],
            std=self.data_config["std"],
            num_workers=self.args.workers,
            distributed=self.args.distributed,
            collate_fn=collate_fn,
            pin_memory=self.args.pin_mem,
            device=self.args.device,
            use_multi_epochs_loader=self.args.use_multi_epochs_loader,
            worker_seeding=self.args.worker_seeding,
            sampler=sampler,
        )
        return train_loader

    def build_val_dataset(self):
        assert self.data_config is not None

        fold_idx = self.meta["fold_idx"]

        augment_fn = ValAugment()
        transform_fn = ValTransform(self.data_config["input_size"][1:])

        rsna_val_dataset_info = {
            "csv_path": os.path.join(
                settings.PROCESSED_DATA_DIR,
                "classification",
                "rsna",
                "fold",
                f"val_fold_{fold_idx}.csv",
            ),
            "img_dir": os.path.join(
                settings.PROCESSED_DATA_DIR,
                "classification",
                "rsna",
                "cleaned_images",
            ),
        }
        val_datasets_info = [("rsna", rsna_val_dataset_info)]

        val_dataset = datasets.RSNADataset(
            val_datasets_info,
            augment_fn,
            transform_fn,
            n_channels=self.args.input_size[0],
            subset="val",
        )
        return val_dataset

    def build_val_loader(self):
        val_dataset = self.build_val_dataset()

        val_workers = self.args.workers
        if self.args.distributed and (
            "tfds" in self.args.dataset or "wds" in self.args.dataset
        ):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            val_workers = min(2, self.args.workers)
        val_loader = create_loader(
            val_dataset,
            input_size=self.data_config["input_size"],
            batch_size=self.args.validation_batch_size or self.args.batch_size,
            is_training=False,
            use_prefetcher=self.args.prefetcher,
            interpolation=self.data_config["interpolation"],
            mean=self.data_config["mean"],
            std=self.data_config["std"],
            num_workers=val_workers,
            distributed=self.args.distributed,
            crop_pct=self.data_config["crop_pct"],
            pin_memory=self.args.pin_mem,
            device=self.args.device,
        )
        return val_loader

    def build_train_loss_fn(self):
        pos_weight = self.args.pos_weight
        pos_weight = None if pos_weight <= 0 else pos_weight
        # assert self.args.smoothing > 0 and self.args.bce_loss
        # setup loss function
        if self.args.jsd_loss:
            assert self.args.num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(
                num_splits=self.args.num_aug_splits, smoothing=self.args.smoothing
            )
        elif self.args.mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if self.args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    target_threshold=self.args.bce_target_thresh
                )
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif self.args.smoothing:
            if self.args.bce_loss:
                if pos_weight is not None:
                    assert self.args.num_classes == 1
                    pos_weight = torch.Tensor([pos_weight])
                print("Using pos weight:", pos_weight)
                train_loss_fn = BinaryCrossEntropyPosSmoothOnly(
                    smoothing=self.args.smoothing,
                    target_threshold=self.args.bce_target_thresh,
                    pos_weight=pos_weight,
                )
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(
                    smoothing=self.args.smoothing
                )
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device=self.args.device)
        return train_loss_fn

    def build_val_loss_fn(self):
        val_loss_fn = nn.CrossEntropyLoss().to(device=self.args.device)
        return val_loss_fn

    def build_optimizer(self, model):
        optimizer = create_optimizer_v2(
            model,
            **optimizer_kwargs(cfg=self.args),
            **self.args.opt_kwargs,
        )
        return optimizer

    def build_lr_scheduler(self, optimizer):
        lr_scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            **scheduler_kwargs(self.args),
            updates_per_epoch=self.args.updates_per_epoch,
        )
        return lr_scheduler, num_epochs

    def compute_metrics(
        self,
        df,
        plot_save_path,
        thres_range=(0, 1, 0.01),
        sort_by="pfbeta",
        additional_info=False,
    ):
        ori_df = df[
            ["site_id", "patient_id", "laterality", "cancer", "preds", "targets"]
        ]
        all_metrics = {}

        reducer_single = lambda df: df
        reducer_gbmean = lambda df: df.groupby(["patient_id", "laterality"]).mean()
        reducer_gbmax = lambda df: df.groupby(["patient_id", "laterality"]).mean()
        reducer_gbmean_site1 = (
            lambda df: df[df.site_id == 1]
            .reset_index(drop=True)
            .groupby(["patient_id", "laterality"])
            .mean()
        )
        reducer_gbmean_site2 = (
            lambda df: df[df.site_id == 2]
            .reset_index(drop=True)
            .groupby(["patient_id", "laterality"])
            .mean()
        )

        reducers = {
            "single": reducer_single,
            "gbmean": reducer_gbmean,
            "gbmean_site1": reducer_gbmean_site1,
            "gbmean_site2": reducer_gbmean_site2,
            "gbmax": reducer_gbmax,
        }

        for reducer_name, reducer in reducers.items():
            df = reducer(ori_df.copy())
            preds = df["preds"].to_numpy()
            gts = df["targets"].to_numpy()
            # mean_sample_weights = mean_df['sample_weights']
            _metrics = self._compute_metrics(gts, preds, None, thres_range, sort_by)
            all_metrics[f"{reducer_name}_best_thres"] = _metrics["best_thres"]
            all_metrics.update(
                {
                    f"{reducer_name}_best_{k}": v
                    for k, v in _metrics["best_metric"].items()
                }
            )
            all_metrics[f"{reducer_name}_pfbeta"] = _metrics["pfbeta"]
            all_metrics[f"{reducer_name}_auc"] = _metrics["auc"]
            all_metrics[f"{reducer_name}_prauc"] = _metrics["prauc"]

        # rank 0 only
        if additional_info:
            rsna_metrics.compute_all(ori_df, plot_save_path)
        return all_metrics

    def _compute_metrics(
        self,
        gts,
        preds,
        sample_weights=None,
        thres_range=(0, 1, 0.01),
        sort_by="pfbeta",
    ):
        if isinstance(gts, torch.Tensor):
            gts = gts.cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        assert isinstance(gts, np.ndarray) and isinstance(preds, np.ndarray)
        assert len(preds) == len(gts)

        # ##### METRICS FOR PROBABILISTIC PREDICTION #####
        # # log loss for pos/neg/overall
        # pos_preds = preds[gts == 1]
        # neg_preds = preds[gts == 0]
        # if len(pos_preds) > 0:
        #     pos_loss = sklearn.metrics.log_loss(np.ones_like(pos_preds),
        #                                         pos_preds,
        #                                         eps=1e-15,
        #                                         normalize=True,
        #                                         sample_weight=None,
        #                                         labels=[0, 1])
        # else:
        #     pos_loss = 99999.
        # if len(neg_preds) > 0:
        #     neg_loss = sklearn.metrics.log_loss(np.zeros_like(neg_preds),
        #                                         neg_preds,
        #                                         eps=1e-15,
        #                                         normalize=True,
        #                                         sample_weight=None,
        #                                         labels=[0, 1])
        # else:
        #     neg_loss = 99999.
        # total_loss = sklearn.metrics.log_loss(gts,
        #                                       preds,
        #                                       eps=1e-15,
        #                                       normalize=True,
        #                                       sample_weight=None,
        #                                       labels=[0, 1])

        # Probabilistic-fbeta
        pfbeta = pfbeta_np(gts, preds, beta=1.0)
        # AUC
        fpr, tpr, _thresholds = sklearn.metrics.roc_curve(gts, preds, pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)

        # PR-AUC
        precisions, recalls, _thresholds = sklearn.metrics.precision_recall_curve(
            gts, preds
        )
        pr_auc = sklearn.metrics.auc(recalls, precisions)

        ##### METRICS FOR CATEGORICAL PREDICTION #####
        # PER THRESHOLD METRIC
        per_thres_metrics = []
        for thres in np.arange(*thres_range):
            bin_preds = (preds > thres).astype(np.uint8)
            metric_at_thres = compute_usual_metrics(gts, bin_preds, beta=1.0)
            pfbeta_at_thres = pfbeta_np(gts, bin_preds, beta=1.0)
            metric_at_thres["pfbeta"] = pfbeta_at_thres

            if sample_weights is not None:
                w_metric_at_thres = compute_usual_metrics(gts, bin_preds, beta=1.0)
                w_metric_at_thres = {f"w_{k}": v for k, v in w_metric_at_thres.items()}
                metric_at_thres.update(w_metric_at_thres)
            per_thres_metrics.append((thres, metric_at_thres))

        per_thres_metrics.sort(key=lambda x: x[1][sort_by], reverse=True)

        # handle multiple thresholds with same scores
        top_score = per_thres_metrics[0][1][sort_by]
        same_scores = []
        for j, (thres, metric_at_thres) in enumerate(per_thres_metrics):
            if metric_at_thres[sort_by] == top_score:
                same_scores.append(abs(thres - 0.5))
            else:
                assert metric_at_thres[sort_by] < top_score
                break
        if len(same_scores) == 1:
            best_thres, best_metric = per_thres_metrics[0]
        else:
            # the nearer 0.5 threshold is --> better
            best_idx = np.argmin(np.array(same_scores))
            best_thres, best_metric = per_thres_metrics[best_idx]

        # best thres, best results, all results
        return {
            "best_thres": best_thres,
            "best_metric": best_metric,
            "all_metrics": per_thres_metrics,
            "pfbeta": pfbeta,
            "auc": auc,
            "prauc": pr_auc,
            # 'pos_log_loss': pos_loss,
            # 'neg_log_loss': neg_loss,
            # 'log_loss': total_loss,
        }
