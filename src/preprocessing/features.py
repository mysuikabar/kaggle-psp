from typing import Dict, List

import polars as pl

from .consts import cols_cat, cols_num, dialogs

FEATURES_DICT: Dict[str, List[pl.Expr]] = {}

FEATURES_DICT["basic_aggs"] = [
    pl.col("index").count().alias("session_count"),
    (pl.col("elapsed_time").max() - pl.col("elapsed_time").min()).alias(
        "total_time"
    ),  # 経過時間
    (
        pl.col("index").count()
        / (pl.col("elapsed_time").max() - pl.col("elapsed_time").min())
    ).alias(
        "click_freq"
    ),  # クリックする頻度
    *[pl.col(col).drop_nulls().n_unique().alias(f"{col}_nunique") for col in cols_cat],
    *[pl.col(col).mean().alias(f"{col}_mean") for col in cols_num],
    *[pl.col(col).sum().alias(f"{col}_sum") for col in cols_num],
    *[pl.col(col).std().alias(f"{col}_std") for col in cols_num],
    *[pl.col(col).min().alias(f"{col}_min") for col in cols_num],
    *[pl.col(col).max().alias(f"{col}_max") for col in cols_num],
]

FEATURES_DICT["dialogs_aggs"] = [
    *[
        pl.col("index")
        .filter(pl.col("text").str.contains(c))
        .count()
        .alias(f"word_{c}")
        for c in dialogs
    ],
    *[
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .mean()
        .alias(f"word_mean_{c}")
        for c in dialogs
    ],
    *[
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .std()
        .alias(f"word_std_{c}")
        for c in dialogs
    ],
    *[
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .max()
        .alias(f"word_max_{c}")
        for c in dialogs
    ],
    *[
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .sum()
        .alias(f"word_sum_{c}")
        for c in dialogs
    ],
    *[
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .median()
        .alias(f"word_median_{c}")
        for c in dialogs
    ],
]
