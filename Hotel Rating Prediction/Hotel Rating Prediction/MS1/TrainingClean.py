import pickle
import pandas as pd
import re
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from geotext import GeoText
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pycountry
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


class TrainingClean:
    def __init__(self, df):
        self.df = df
        self.Fill = {}
        self.drop = []
        self.sia = SentimentIntensityAnalyzer()
        self.label_encoder = {'Hotel_Name': preprocessing.LabelEncoder(),
                              'Reviewer_Nationality': preprocessing.LabelEncoder(),
                              'type_of_trip': preprocessing.LabelEncoder(),
                              'people': preprocessing.LabelEncoder(), 'Room_Type': preprocessing.LabelEncoder(),
                              'hotel_country': preprocessing.LabelEncoder(),
                              'hotel_city': preprocessing.LabelEncoder()}
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
        return np.NAN

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
                return ''.join(filter(str.isdigit, i))
        return np.NAN

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
        pickle.dump(self.drop, open('drop.sav', 'wb'))
        pickle.dump(self.scaler, open('scaler.sav', 'wb'))
        self.df.to_csv('clean2.csv')

    def clean(self):

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

        self.drop.append('Tags')

        # split Hotel_address to city and country
        self.df = self.df.astype({'Hotel_Address': 'string'})
        self.df['Hotel_Address'] = self.df['Hotel_Address'].str.replace('Milan', 'Milano')
        x = self.df["Hotel_Address"]
        self.df["hotel_country"] = self.handle_countries(x)
        self.df["hotel_city"] = self.handle_cities(x)

        self.drop.append('Hotel_Address')

        # handle reviews using sentiment analysis
        self.df = self.df.astype({'Negative_Review': 'string', 'Positive_Review': 'string'})
        self.df['Negative_Review'] = self.cleanText(self.df['Negative_Review'])
        self.df['Positive_Review'] = self.cleanText(self.df['Positive_Review'])
        self.replace_negative_review()
        self.replace_positive_review()

        self.drop.append('Negative_Review')
        self.drop.append('Positive_Review')

        self.df.to_csv('clean.csv')
        self.LabelEncoding()
        self.FeatureSelection()
        self.df.drop(columns=self.drop, inplace=True)
        self.Fill['Review_Total_Negative_Word_Counts'] = 0
        self.Fill['Review_Total_Positive_Word_Counts'] = 0
        self.Fill['Average_Score'] = np.mean(self.df['Average_Score'])
        # self.Remove_outliers()
        self.FeateurScaling()
        self.save()
        return self.df

    def LabelEncoding(self):
        for column, model in self.label_encoder.items():
            model.fit(self.df[column])
            self.df[column] = model.transform(self.df[column])

    def FeatureSelection(self):
        cor = self.df.corr()
        target = abs(cor['Reviewer_Score'])
        # Plot bar graph
        plt.figure(figsize=(10, 5))
        plt.bar(target.index, target.values)
        plt.xticks(rotation=90)
        plt.xlabel('Variables')
        plt.ylabel('Correlation Coefficient (absolute value)')
        plt.title('Correlation Coefficients with Reviewer_Score')
        plt.savefig('correlation.png')
        plt.show()

        #plot heatmap
        plt.subplots(figsize=(10, 8))
        sns.heatmap(cor, cmap='coolwarm', annot=True, vmin=-1, vmax=1, center=0)
        plt.title('Correlation Coefficients')
        plt.savefig('heatmap.png')
        plt.show()

        features = list(target[target >= 0.2].index)
        pickle.dump(features, open('top_features.sav', 'wb'))
        for c in self.df.columns:
            if c not in self.drop and c not in features:
                self.drop.append(c)

    def FeateurScaling(self):

        self.scaler.fit(self.df)
        scaled = pd.DataFrame(self.scaler.transform(self.df))
        scaled.columns = self.df.columns
        self.df = scaled
