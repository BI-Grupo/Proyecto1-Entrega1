import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
# %pip install nltk
import nltk
from nltk.corpus import stopwords
import string
import joblib
from sklearn.pipeline import Pipeline
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('stopwords')
comunes = set(stopwords.words('spanish'))
r_ods = pd.read_excel('./data/registros.xlsx')
registros = r_ods.copy()

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.limpiar_texto)
    
    def limpiar_texto(self, texto):
        texto = texto.lower()
        texto = re.sub(f'[{string.punctuation}]', '', texto)
        palabras = texto.split()
        palabras_limpias = [palabra for palabra in palabras if palabra not in comunes]
        return ' '.join(palabras_limpias)

# Pipeline
pipeline = Pipeline([
    ('cleaner', TextCleaner()),  # Limpieza del texto
    ('vectorizer', TfidfVectorizer()),  # Vectorización
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=20, criterion='entropy'))  # Clasificador
])

X = registros['Textos_espanol']
Y = registros['sdg']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

pipeline.fit(X_train, Y_train)

joblib.dump(pipeline, 'model.joblib')