import json
import os

import pandas as pd
import numpy as np
import settings


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_df_info(df):
    return {
        "num_patients": df.patient_id.nunique(),
        "num_samples": len(df),
        "pos_sample_num": df.cancer.sum(),
        "pos_sample_percent": df.cancer.mean(),
        # "neg_sample_num": len(df) - df.cancer.sum(),
        # "neg_sample_percent": 1 - df.cancer.mean(),
    }


if __name__ == "__main__":
    datasets = ["bmcd", "cddcesm", "cmmd", "miniddsm", "rsna", "vindr"]
    info = {}

    for dataset in datasets:
        d_info = {}

        DATASET_DIR = os.path.join(
            settings.PROCESSED_DATA_DIR, "classification", dataset
        )
        CSV_LABEL_PATH = os.path.join(
            settings.PROCESSED_DATA_DIR,
            "classification",
            dataset,
            "cleaned_label.csv",
        )
        df = pd.read_csv(CSV_LABEL_PATH)
        d_info["full"] = get_df_info(df)
        for i in range(4):
            d_info[f"fold_{i}"] = {}
            df = pd.read_csv(os.path.join(DATASET_DIR, "fold", f"train_fold_{i}.csv"))
            d_info[f"fold_{i}"]["train"] = get_df_info(df)
            df = pd.read_csv(os.path.join(DATASET_DIR, "fold", f"val_fold_{i}.csv"))
            d_info[f"fold_{i}"]["val"] = get_df_info(df)
        info[dataset] = d_info
    all_info = {}
    all_info["full"] = {}
    for key in ["num_patients", "num_samples", "pos_sample_num"]:
        all_info["full"][key] = sum(info[dataset]["full"][key] for dataset in datasets)
    all_info["full"]["pos_sample_percent"] = (
        all_info["full"]["pos_sample_num"] / all_info["full"]["num_samples"]
    )
    for sub in [f"fold_{i}" for i in range(4)]:
        all_info[sub] = {}
        for s in ["train", "val"]:
            all_info[sub][s] = {}
            for key in ["num_patients", "num_samples", "pos_sample_num"]:
                all_info[sub][s][key] = sum(
                    info[dataset][sub][s][key] for dataset in datasets
                )
            all_info[sub][s]["pos_sample_percent"] = (
                all_info[sub][s]["pos_sample_num"] / all_info[sub][s]["num_samples"]
            )
    info["all"] = all_info
    print(f"--result\n{json.dumps(info, indent=4, cls=NpEncoder)}")
