import pandas as pd
from sklearn.linear_model import ElasticNetCV
import mlflow


class ElasticNet:

    @staticmethod
    def train_elastic_net(train_data: pd.DataFrame, test_data: pd.DataFrame, features: list, random_state: int,
                          tolerance: float) -> pd.DataFrame:
        # Enable auto log
        mlflow.sklearn.autolog()

        r2_scores: dict = {}
        coeffs: dict = {}

        for feature in features:
            model = ElasticNetCV(cv=5, random_state=random_state, tol=tolerance)

            # Create y and X
            train_copy = pd.DataFrame(data=train_data.copy(), columns=features)

            y_train = train_copy[f"{feature}"]
            del train_copy[f"{feature}"]
            X_train = train_copy

            test_copy = pd.DataFrame(data=test_data.copy(), columns=features)
            y_test = test_copy[f"{feature}"]
            del test_copy[f"{feature}"]
            X_test = test_copy

            model.fit(X_train, y_train)
            r2_scores[feature] = model.score(X_test, y_test)
            coeffs[feature] = model.coef_

        scores = pd.DataFrame.from_dict(r2_scores, orient='index')
        scores["Marker"] = scores.index
        scores.rename(columns={0: "Score"}, inplace=True)
        scores.reset_index(drop=True, inplace=True)

        return scores
