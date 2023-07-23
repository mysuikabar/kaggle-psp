from pathlib import Path

import polars as pl

from preprocessing.features import FEATURES_DICT


def main():
    ROOT = Path(__file__).parents[1]
    TRAIN_PATH = ROOT / "data/raw_data/train_0-4.csv"
    FEATURES_STORE_DIR = ROOT / "data/features/"

    # 生データの追加する特徴量
    cols_basic = [
        pl.col("page").cast(pl.Float32),
        pl.col("fqid").fill_null("fqid_None"),
        pl.col("text_fqid").fill_null("text_fqid_None"),
        (
            (pl.col("elapsed_time") - pl.col("elapsed_time").shift(1))
            .fill_null(0)
            .clip(0, 1e9)
            .over(["session_id", "level"])
            .alias("elapsed_time_diff")
        ),
        (
            (pl.col("screen_coor_x") - pl.col("screen_coor_x").shift(1))
            .abs()
            .over(["session_id", "level"])
            .alias("screen_coor_x_diff")
        ),
        (
            (pl.col("screen_coor_y") - pl.col("screen_coor_y").shift(1))
            .abs()
            .over(["session_id", "level"])
            .alias("screen_coor_y_diff")
        ),
    ]

    # 生データの読み込み
    df = pl.read_csv(TRAIN_PATH)
    df = df.with_columns(cols_basic)

    # 特徴量ごとにparquet形式で特徴量ストアに保存
    for label, features in FEATURES_DICT.items():
        path = FEATURES_STORE_DIR / f"{label}.parquet"
        if path.exists():
            continue
        else:
            df_aggregated = df.groupby("session_id", maintain_order=True).agg(features)
            df_aggregated.write_parquet(FEATURES_STORE_DIR / f"{label}.parquet")


if __name__ == "__main__":
    main()
