import pandas as pd
from sklearn.linear_model import ElasticNetCV
import mlflow


class ElasticNet:

    @staticmethod
    def train_elastic_net(train_data: pd.DataFrame, test_data: pd.DataFrame, markers: list, random_state: int,
                          tolerance: float) -> pd.DataFrame:
        # Enable auto log
        mlflow.sklearn.autolog()

        r2_scores: dict = {}
        coeffs: dict = {}

        for marker in markers:
            model = ElasticNetCV(cv=5, random_state=random_state, tol=tolerance)

            # Create y and X
            train_copy = pd.DataFrame(data=train_data.copy(), columns=markers)

            y_train = train_copy[f"{marker}"]
            del train_copy[f"{marker}"]
            X_train = train_copy

            test_copy = pd.DataFrame(data=test_data.copy(), columns=markers)
            y_test = test_copy[f"{marker}"]
            del test_copy[f"{marker}"]
            X_test = test_copy

            model.fit(X_train, y_train)
            r2_scores[marker] = model.score(X_test, y_test)
            coeffs[marker] = model.coef_

        scores = pd.DataFrame.from_dict(r2_scores, orient='index')
        scores["Marker"] = scores.index
        scores.rename(columns={0: "Score"}, inplace=True)
        scores.reset_index(drop=True, inplace=True)

        return scores
