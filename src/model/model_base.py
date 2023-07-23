from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd

Features = Union[np.ndarray, pd.DataFrame]
Label = Union[np.ndarray, pd.DataFrame]


class ModelBase(metaclass=ABCMeta):
    def __init__(self, params: dict) -> None:
        self.params = params

    @abstractmethod
    def fit(
        self,
        X_tr: Features,
        y_tr: Label,
        X_va: Optional[Features] = None,
        y_va: Optional[Label] = None,
    ) -> None:
        """モデルの学習を行う

        Args:
            X_tr (pd.DataFrame): 学習データの特徴量
            y_tr (pd.Series): 学習データの正解ラベル
            X_va (Optional[pd.DataFrame], optional): 評価データの特徴量
            y_va (Optional[pd.Series], optional): 評価データの正解ラベル
        """
        pass

    @abstractmethod
    def predict(self, X_te: Features) -> Label:
        """モデルの予測を行う

        Args:
            X_te (pd.DataFrame): テストデータの特徴量

        Returns:
            np.array: テストデータの予測値
        """
        pass

    @abstractmethod
    def save_model(self, file_path: str) -> None:
        """モデルを保存する

        Args:
            file_path (str): モデルを保存するパス
        """
        pass

    @abstractmethod
    def load_model(self, file_path: str) -> None:
        """モデルをロードする

        Args:
            file_path (str): ロードするモデルのパス
        """
        pass
