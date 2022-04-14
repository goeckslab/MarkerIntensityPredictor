from sklearn.impute import SimpleImputer
import pandas as pd


class SimpleImputation:

    @staticmethod
    def impute(data_set: pd.DataFrame, missing_values: any = 0):
        imp_mean = SimpleImputer(missing_values=missing_values, strategy='mean')
        imputed_values = imp_mean.fit_transform(data_set)


        return pd.DataFrame(data=imputed_values, columns=data_set.columns)
