import pickle
import contractions
import numpy as np
import pandas as pd
import pycountry
from geotext import GeoText
import re
from nltk import WordNetLemmatizer, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


class TrainingClean:
    def __init__(self, df):
        self.df = df
        self.sia = SentimentIntensityAnalyzer()
        self.Fill = {}
        self.drop = []
        self.label_encoder = {
            'Hotel_Name': preprocessing.LabelEncoder(),
            'Reviewer_Nationality': preprocessing.LabelEncoder(),
            'type_of_trip': preprocessing.LabelEncoder(),
            'people': preprocessing.LabelEncoder(),
            'Room_Type': preprocessing.LabelEncoder(),
            'hotel_country': preprocessing.LabelEncoder(),
            'hotel_city': preprocessing.LabelEncoder(),
            'Reviewer_Score': preprocessing.LabelEncoder()
        }
        self.scaler = preprocessing.MinMaxScaler()

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

    @staticmethod
    def handleCount(x, n):
        if len(x) > 0 and x[0] == ' ':
            x = x[1:]
        if len(x) > 0 and x[-1] == ' ':
            x = x[:-1]
        x = x.lower()
        neg = ['no negative', 'nothing', 'nil', 'none', 'n a']
        pos = ['no positive', 'nothing', 'nil', 'none', 'n a']
        if n == 0 and x in neg:
            return 0
        elif n == 1 and x in pos:
            return 0
        else:
            return len(x.split())

    @staticmethod
    def handle_withAPet(x):
        tmp = x
        tmp = tmp.translate(str.maketrans('', '', "'[]")).lower().split(',')
        for i in tmp:
            if 'pet' in i:
                return 1
        return 0

    @staticmethod
    def handle_people(x):
        tmp = x
        tmp = tmp.translate(str.maketrans('', '', "'[]")).lower().split(',')
        keys = ['couple', 'group', 'solo', 'family', 'friends']
        for i in tmp:
            for j in keys:
                if j in i:
                    return j
        return None

    @staticmethod
    def handle_roomType(x):
        tmp = x
        tmp = tmp.translate(str.maketrans('', '', "'[]")).lower().split(',')
        keys = ['standard', 'double', 'room', 'suite', 'studio', 'deluxe', 'superior', 'apartment', 'floor',
                'king', 'special', 'classic', 'maisonette', 'loft', 'duplex', 'queen', 'king', 'classique', 'townhouse',
                'garden', 'luxury', 'cosy', 'maison', 'atrium', 'view', 'bank', 'cool', 'lafayette', 'nest', 'park']
        for i in tmp:
            for j in keys:
                if j in i:
                    return i[1:-1]
        return None

    @staticmethod
    def handle_nights(x):
        tmp = x
        tmp = tmp.translate(str.maketrans('', '', "'[]")).lower().split(',')
        for i in tmp:
            if 'night' in i:
                n = ''.join(filter(str.isdigit, i))
                if n == '':
                    return None
                return int(n)
        return None

    @staticmethod
    def handle_submittedfrom(x):
        tmp = x
        tmp = tmp.translate(str.maketrans('', '', "'[]")).lower().split(',')
        for i in tmp:
            if 'submitted' in i:
                return 1
        return 0

    def replace_negative_review(self):
        for i in range(len(self.df)):
            text = self.df.loc[i, 'Negative_Review'].lower()
            tmp = self.sia.polarity_scores(text)
            pos_score = tmp['pos']
            neg_score = tmp['neg']
            neu = tmp['neu']
            if pos_score > neg_score and pos_score > neu:
                self.df.loc[i, 'Review_Total_Negative_Word_Counts'] = 0

    def replace_positive_review(self):
        for i in range(len(self.df)):
            text = self.df.loc[i, 'Positive_Review']
            tmp = self.sia.polarity_scores(text)
            pos_score = tmp['pos']
            neg_score = tmp['neg']
            neu = tmp['neu']
            if pos_score < neg_score and neu < neg_score:
                self.df.loc[i, 'Review_Total_Positive_Word_Counts'] = 0

    def Remove_outliers(self):
        q1 = self.df['Average_Score'].quantile(0.25)
        q3 = self.df['Average_Score'].quantile(0.75)
        iqr = q3 - q1
        self.df = self.df[~((self.df['Average_Score'] < (q1 - 1.5 * iqr)) | (
                self.df['Average_Score'] > (q3 + 1.5 * iqr)))]

        q1 = self.df['Review_Total_Negative_Word_Counts'].quantile(0.25)
        q3 = self.df['Review_Total_Negative_Word_Counts'].quantile(0.75)
        iqr = q3 - q1
        self.df = self.df[~((self.df['Review_Total_Negative_Word_Counts'] < (q1 - 1.5 * iqr)) | (
                self.df['Review_Total_Negative_Word_Counts'] > (q3 + 1.5 * iqr)))]

        q1 = self.df['Review_Total_Positive_Word_Counts'].quantile(0.25)
        q3 = self.df['Review_Total_Positive_Word_Counts'].quantile(0.75)
        iqr = q3 - q1
        self.df = self.df[~((self.df['Review_Total_Positive_Word_Counts'] < (q1 - 1.5 * iqr)) | (
                self.df['Review_Total_Positive_Word_Counts'] > (q3 + 1.5 * iqr)))]

    @staticmethod
    def handle_cities(x):
        cities = []
        for i in range(len(x)):
            z = str(x[i])
            z = z.split(" ")
            c = GeoText(str(z))
            if len(c.cities) == 0:
                cities.append('')
            else:
                cities.append(c.cities[-1])

        return cities

    @staticmethod
    def handle_countries(x):
        countries = []
        country_names = list(map(lambda y: y.name.lower(), pycountry.countries))
        country_names = '|'.join(country_names)
        pattern = r'\b(?:' + country_names + ')\\b'
        for i in range(len(x)):
            z = str(x[i]).lower()
            match = re.search(pattern, z)
            if match:
                countries.append(match.group(0))
            else:
                countries.append('')
        return countries

    @staticmethod
    def cleanText(x):
        x = [text.lower() for text in x]
        lemmatizer = WordNetLemmatizer()
        cleaned = []
        for txt in x:
            txt.replace('wasn t', 'was not')
            txt.replace('weren t', 'were not')
            txt.replace('aren t', 'are not')
            txt.replace('isn t', 'is not')
            txt.replace('don t', 'do not')
            txt.replace('doesn t', 'does not')
            txt.replace('didn t', 'did not')
            txt.replace('shouldn t', 'should not')
            txt.replace('i m', 'i am')
            txt.replace('isn t', 'is not')
            txt.replace('aren t', 'are not')
            txt = lemmatizer.lemmatize(txt)
            word_tokens = word_tokenize(txt)
            sentence = ' '.join(contractions.fix(word) for word in word_tokens)
            cleaned.append(sentence)
        return cleaned

    def save(self):
        pickle.dump(self.label_encoder, open('label_encoder.sav', 'wb'))
        pickle.dump(self.Fill, open('fill.sav', 'wb'))
        pickle.dump(self.scaler, open('scaler.sav', 'wb'))

    def clean(self):
        self.df.drop_duplicates(inplace=True)

        self.df['Review_Date'] = pd.to_datetime(self.df['Review_Date'])
        self.df['year'] = self.df['Review_Date'].dt.year
        self.df['Review_Date'] = self.df['Review_Date'].apply(lambda t: t.toordinal())

        self.df['lat'].fillna(np.mean(self.df['lat']), inplace=True)
        self.df['lng'].fillna(np.mean(self.df['lng']), inplace=True)
        self.Fill['lat'] = np.mean(self.df['lat'])
        self.Fill['lng'] = np.mean(self.df['lng'])

        self.df['days_since_review'] = self.df['days_since_review'].str.replace(' days', '')
        self.df['days_since_review'] = self.df['days_since_review'].str.replace(' day', '')
        self.df['days_since_review'] = pd.to_numeric(self.df['days_since_review'])

        self.df['Review_Total_Negative_Word_Counts'] = self.df['Negative_Review'].apply(
            lambda t: self.handleCount(t, 0))
        self.df['Review_Total_Positive_Word_Counts'] = self.df['Positive_Review'].apply(
            lambda t: self.handleCount(t, 1))

        # split Tags into multiple columns
        self.df['type_of_trip'] = self.df['Tags'].apply(lambda t: self.handle_tripType(t))
        self.df['with_a_pet'] = self.df['Tags'].apply(lambda t: self.handle_withAPet(t))
        self.df['people'] = self.df['Tags'].apply(lambda t: self.handle_people(t))
        self.df['Room_Type'] = self.df['Tags'].apply(lambda t: self.handle_roomType(t))
        self.df['nights'] = self.df['Tags'].apply(lambda t: self.handle_nights(t))
        self.df['submitted_from_mobile'] = self.df['Tags'].apply(lambda t: self.handle_submittedfrom(t))

        self.df['type_of_trip'].fillna(self.df['type_of_trip'].mode()[0], inplace=True)
        self.df['Room_Type'].fillna(self.df['Room_Type'].mode()[0], inplace=True)
        self.df['nights'].fillna(self.df['nights'].mode()[0], inplace=True)
        self.Fill['type_of_trip'] = self.df['type_of_trip'].mode()[0]
        self.Fill['Room_Type'] = self.df['Room_Type'].mode()[0]
        self.Fill['nights'] = self.df['nights'].mode()[0]

        # split Hotel_address to city and country
        self.df = self.df.astype({'Hotel_Address': 'string'})
        self.df['Hotel_Address'] = self.df['Hotel_Address'].str.replace('Milan', 'Milano')
        x = self.df["Hotel_Address"]
        self.df["hotel_country"] = self.handle_countries(x)
        self.df["hotel_city"] = self.handle_cities(x)

        # handle reviews using sentiment analysis

        self.df = self.df.astype({'Negative_Review': 'string', 'Positive_Review': 'string'})
        self.df['Negative_Review'] = self.cleanText(self.df['Negative_Review'])
        self.df['Positive_Review'] = self.cleanText(self.df['Positive_Review'])
        self.replace_negative_review()
        self.replace_positive_review()
        self.df.drop(columns=['Negative_Review', 'Positive_Review', 'Hotel_Address', 'Tags'], inplace=True)

        self.LabelEncoding()
        # print(self.df['Reviewer_Score'].unique())
        self.FeatureSelection()
        self.df.drop(columns=self.drop, inplace=True)
        self.Fill['Review_Total_Negative_Word_Counts'] = 0
        self.Fill['Review_Total_Positive_Word_Counts'] = 0
        self.Fill['Average_Score'] = np.mean(self.df['Average_Score'])
        # self.Remove_outliers()
        # self.Balance()
        self.FeateurScaling()
        self.save()

        return self.df

    def LabelEncoding(self):
        for column, model in self.label_encoder.items():
            model.fit(self.df[column])
            self.df[column] = model.transform(self.df[column])

    def FeatureSelection(self):

        f_values, p_values = f_classif(self.df.drop(columns=['Reviewer_Score']), self.df['Reviewer_Score'])
        selector = SelectKBest(f_classif, k=4)
        x_new = selector.fit_transform(self.df.drop(columns=['Reviewer_Score']), self.df['Reviewer_Score'])
        selected_features = list(self.df.drop(columns=['Reviewer_Score']).columns[selector.get_support()])
        selected_features.append('Reviewer_Score')
        pickle.dump(selected_features, open('top_features.sav', 'wb'))
        for c in self.df.columns:
            if c not in self.drop and c not in selected_features:
                self.drop.append(c)

    def FeateurScaling(self):
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.df.drop(columns=['Reviewer_Score']))
        scaled = pd.DataFrame(self.scaler.transform(self.df.drop(columns=['Reviewer_Score'])))
        scaled.columns = self.df.drop(columns=['Reviewer_Score']).columns
        self.df = pd.concat([scaled, self.df['Reviewer_Score']], axis=1)

    # def Balance(self):
    #     x = self.df.iloc[:, :-1]
    #     y = self.df.iloc[:, -1]
    #     undersample = RandomOverSampler(random_state=42)
    #     x1, y1 = undersample.fit_resample(x, y)
    #     balanced_df = pd.DataFrame.from_records(data=pd.concat((x1, y1), axis=1))
    #     balanced_df.columns = self.df.columns
    #     self.df = balanced_df
