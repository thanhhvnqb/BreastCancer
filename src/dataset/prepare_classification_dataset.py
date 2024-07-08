import argparse
import os
import shutil

from _prepare_classification_dataset_stage1 import *
from _prepare_classification_dataset_stage2 import *

import settings
from utils.misc import rm_and_mkdir

STAGE1_PROCESS_FUNCS = {
    "rsna": stage1_process_rsna,
    "vindr": stage1_process_vindr,
    "miniddsm": stage1_process_miniddsm,
    "cmmd": stage1_process_cmmd,
    "cddcesm": stage1_process_cddcesm,
    "bmcd": stage1_process_bmcd,
}

STAGE2_PROCESS_FUNCS = {
    "rsna": stage2_process_rsna,
    "vindr": stage2_process_vindr,
    "miniddsm": stage2_process_miniddsm,
    "cmmd": stage2_process_cmmd,
    "cddcesm": stage2_process_cddcesm,
    "bmcd": stage2_process_bmcd,
}


def parse_args():
    parser = argparse.ArgumentParser("Prepair classification dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["rsna", "vindr", "miniddsm", "cmmd", "cddcesm", "bmcd"],
        help="Path to root directory of processed dataset.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Path to root directory of processed dataset.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Stage of the dataset preparation. 0: both. 1: stage1, 2: stage2.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for dicomsdl decoding.",
    )
    args = parser.parse_args()
    return args


def main(args):
    dataset = args.dataset
    STAGES = ["stage1", "stage2"] if args.stage == 0 else ["stage" + str(args.stage)]

    print("Processing", dataset)
    raw_root_dir = args.root_dir
    cleaned_root_dir = os.path.join(
        settings.PROCESSED_DATA_DIR, "classification", dataset
    )
    stage1_images_dir = os.path.join(cleaned_root_dir, "stage1_images")
    cleaned_label_path = os.path.join(cleaned_root_dir, "cleaned_label.csv")
    cleaned_images_dir = os.path.join(cleaned_root_dir, "cleaned_images")

    if "stage1" in STAGES:
        # remove `stage1_images` directory
        if os.path.exists(stage1_images_dir):
            try:
                shutil.rmtree(stage1_images_dir)
            except OSError:
                # OSError: Cannot call rmtree on a symbolic link
                os.remove(stage1_images_dir)
        rm_and_mkdir(cleaned_root_dir)

        stage1_process_func = STAGE1_PROCESS_FUNCS[dataset]
        stage1_process_func(
            raw_root_dir, stage1_images_dir, cleaned_label_path, force_copy=False
        )

    if "stage2" in STAGES:
        rm_and_mkdir(cleaned_images_dir)
        assert os.path.exists(cleaned_label_path)

        stage2_process_func = STAGE2_PROCESS_FUNCS[dataset]
        print("Converting to 8-bits png images..")
        stage2_process_func(
            stage1_images_dir,
            cleaned_label_path,
            cleaned_images_dir,
            n_jobs=args.num_workers,
            n_chunks=args.num_workers,
        )
        print("Done!")
        print("-----------------\n\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
