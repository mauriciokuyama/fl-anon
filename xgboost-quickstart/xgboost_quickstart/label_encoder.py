import pandas as pd
from sklearn.preprocessing import LabelEncoder

class MyLabelEncoder:
    def __init__(self):
        self._label_encoders = {}

    def encode(self, df : pd.DataFrame):
        new_df = df.copy()
        for i, column in enumerate(new_df.columns):
            if new_df[column].dtype == 'object':
                self._label_encoders[column] = LabelEncoder()
                new_df[column] = self._label_encoders[column].fit_transform(new_df[column])

        return new_df

    def decode(self, df):
        new_df = df.copy()
        for column in new_df.columns:
            if column in self._label_encoders:
                new_df[column] = self._label_encoders[column].inverse_transform(new_df[column])

        return new_df
