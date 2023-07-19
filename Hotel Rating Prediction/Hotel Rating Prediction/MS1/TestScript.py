import pickle
import pandas as pd


class TestScript:
    def __init__(self, df, path):
        self.df = df
        self.fill, self.encoder, self.scaler = {}, {}, None
        self.path = path
        self.topFeatures = []

    def load(self):
        self.fill = pickle.load(open(self.path+'fill.sav', 'rb'))
        self.encoder = pickle.load(open(self.path+'label_encoder.sav', 'rb'))
        self.scaler = pickle.load(open(self.path+'scaler.sav', 'rb'))
        self.topFeatures = pickle.load(open(self.path + 'top_features.sav', 'rb'))

    def label_encoding(self):
        for column, model in self.encoder.items():
            if column in self.df.columns:
                self.df[column] = model.transform(self.df[column])

    def Drop(self):
        columns_to_drop = []
        for c in self.df.columns:
            if c not in self.topFeatures:
                columns_to_drop.append(c)
        self.df.drop(columns=columns_to_drop, inplace=True)

    def Scale(self):
        scaled = pd.DataFrame(self.scaler.transform(self.df))
        scaled.columns = self.df.columns
        self.df = scaled

    def Fill(self):
        for c in self.df.columns:
            if c in self.fill.keys():
                self.df[c].fillna(self.fill[c], inplace=True)

    def clean(self):
        self.load()
        self.df.dropna(subset=['Reviewer_Score'], inplace=True)
        self.Drop()
        self.Fill()
        self.label_encoding()
        self.Scale()
        return self.df
