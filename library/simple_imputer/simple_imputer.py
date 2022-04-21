from sklearn.impute import SimpleImputer
import pandas as pd


class SimpleImputation:

    @staticmethod
    def impute(train_data: pd.DataFrame, test_data: pd.DataFrame, missing_values: any = 0):
        imp_mean = SimpleImputer(missing_values=missing_values, strategy='mean')
        imp_mean.fit(train_data)
        imputed_values = imp_mean.transform(test_data)

        return pd.DataFrame(data=imputed_values, columns=test_data.columns)
