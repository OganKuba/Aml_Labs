from xgboost import XGBClassifier


def XGB(n):
    return XGBClassifier(
        n_estimators=n, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        use_label_encoder=False, random_state=0, n_jobs=-1
    )
