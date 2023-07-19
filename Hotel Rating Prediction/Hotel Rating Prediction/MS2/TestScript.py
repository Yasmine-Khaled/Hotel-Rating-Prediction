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
        print("features\n", self.df.columns)

    def Scale(self):
        scaled = pd.DataFrame(self.scaler.transform(self.df.drop(columns=['Reviewer_Score'])))
        scaled.columns = self.df.drop(columns=['Reviewer_Score']).columns
        self.df = pd.concat([scaled, self.df['Reviewer_Score']], axis=1)

    def Fill(self):
        for c in self.df.columns:
            if c in self.fill.keys():
                self.df[c].fillna(self.fill[c], inplace=True)

    @staticmethod
    def handle_tripType(x):
        tmp = x
        tmp = tmp.translate(str.maketrans('', '', "'[]")).lower().split(',')
        for i in range(len(tmp)):
            tmp2 = tmp[i]
            tmp2 = tmp2.split()
            if len(tmp2) > 1 and tmp2[1] == 'trip':
                return tmp2[0]
        return None

    def clean(self):
        self.load()
        self.df['type_of_trip'] = self.df['Tags'].apply(lambda t: self.handle_tripType(t))
        self.df.dropna(subset=['Reviewer_Score'], inplace=True)
        self.Drop()
        self.Fill()
        self.label_encoding()
        self.Scale()
        return self.df
