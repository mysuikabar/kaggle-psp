import gc
from logging import getLogger
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error

from model.model_gbdt import ModelXGB
from model.runner import Runner

logger = getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(config):
    ROOT = Path(__file__).parents[1]
    FEATURES_STORE_DIR = ROOT / "data/features/"
    TARGET_PATH = ROOT / "data/raw_data/labels.csv"

    # 特徴量・正解ラベルの読み込み
    logger.info("Loading data ...")

    features = config.features.copy()
    feature = features.pop(0)
    df = pd.read_parquet(FEATURES_STORE_DIR / f"{feature}.parquet").set_index(
        "session_id"
    )
    for feature in features:
        df_tmp = pd.read_parquet(FEATURES_STORE_DIR / f"{feature}.parquet").set_index(
            "session_id"
        )
        df = df.join(df_tmp)

    df_target = pd.read_csv(TARGET_PATH, index_col="session_id")
    df = df.join(df_target)
    target_columns = df_target.columns

    logger.info("Loading data is completed")

    del df_tmp, df_target
    gc.collect()

    # モデルの学習・予測
    X, y = df.drop(target_columns, axis=1), df[f"q{config.target}"]

    model_name = config.model.model_name
    if model_name == "xgboost":
        model_cls = ModelXGB

    params = OmegaConf.to_container(config.model.params)  # to dict
    runner = Runner(model_cls=model_cls, params=params)
    oof = runner.run_cv(X, y)
    runner.save_models(model_name=config.experiment_name)

    # 評価
    mse = mean_squared_error(y, oof)
    logger.info(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
