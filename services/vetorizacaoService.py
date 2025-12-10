from sklearn.feature_extraction.text import TfidfVectorizer
from services.datasetService import dataset_completo
from sklearn.preprocessing import LabelEncoder

def vetorizacao():
    vectorizer = TfidfVectorizer()
    
    df = dataset_completo()
    X = df['sintomas'].astype(str)
    vectorizer.fit(X)
    X_transform = vectorizer.transform(X)
    return X_transform

def encode_Y():
    labelEncoder = LabelEncoder()
    df = dataset_completo()
    Y = df['diagnostico'].astype(str)
    Y_encoded = labelEncoder.fit_transform(Y)
    return Y_encoded