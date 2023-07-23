import pickle
from abc import abstractmethod
from typing import Dict, Optional

import xgboost as xgb

from .model_base import Features, Label, ModelBase


class ModelGBDT(ModelBase):
    def save_model(self, file_path: str) -> None:
        """pickle形式でモデルを保存する

        Args:
            file_path (str): モデルを保存するパス
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, file_path: str) -> None:
        """pickle形式のモデルをロードする

        Args:
            file_path (str): ロードするモデルのパス
        """
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)

    @property
    @abstractmethod
    def feature_importance(self) -> Dict[str, float]:
        pass


class ModelXGB(ModelGBDT):
    def fit(
        self,
        X_tr: Features,
        y_tr: Label,
        X_va: Optional[Features] = None,
        y_va: Optional[Label] = None,
    ) -> None:
        validation = X_va is not None

        # データの準備
        dtrain = xgb.DMatrix(data=X_tr, label=y_tr)
        if validation:
            dvalid = xgb.DMatrix(data=X_va, label=y_va)

        # パラメータの設定
        params = self.params.copy()
        num_boost_round = params.pop("num_boost_round")

        # 学習
        if validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train")],
            )

    def predict(self, X_te: Features) -> Label:
        dtest = xgb.DMatrix(X_te)
        return self.model.predict(dtest)

    @property
    def feature_importance(
        self, importance_type: str = "total_gain"
    ) -> Dict[str, float]:
        return self.model.get_score(importance_type=importance_type)
