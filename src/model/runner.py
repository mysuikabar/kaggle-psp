from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .model_base import ModelBase


class Runner:
    def __init__(self, model_cls: Callable[[dict], ModelBase], params: dict) -> None:
        self.model_cls = model_cls
        self.params = params
        self.model_list: List[ModelBase] = []
        self.idx_list: List[np.ndarray] = []

    def build_model(self) -> ModelBase:
        return self.model_cls(self.params)

    def run_cv(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> np.ndarray:
        oof = np.zeros_like(y)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for idx_tr, idx_va in kf.split(X=X, y=y):
            X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]
            y_tr, y_va = y.iloc[idx_tr], y.iloc[idx_va]

            # モデルの学習・予測
            model = self.build_model()
            model.fit(X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va)
            oof[idx_va] = model.predict(X_va)

            # 学習済みモデル、対応するインデックスの格納
            self.model_list.append(model)
            self.idx_list.append(idx_va)

        return oof

    def save_models(self, model_name: str) -> None:
        for i, model in enumerate(self.model_list):
            model.save_model(f"{model_name}_fold{i}.pickle")
