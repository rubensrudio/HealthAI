from services.vetorizacaoService import vetorizacao, encode_Y
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as XGBoost
from sklearn.metrics import accuracy_score

def buscar_dados():
    X = vetorizacao()
    Y = encode_Y()
    return X, Y

def separar_dados():
    X, Y = buscar_dados()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test

def treinar_modelo():
    HealthAI = XGBoost(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    X_train, X_test, Y_train, Y_test = separar_dados()
    HealthAI.fit(X_train, Y_train)
    return HealthAI

def acuracia_modelo():
    HealthAI = treinar_modelo()
    X_train, X_test, Y_train, Y_test = separar_dados()
    Y_pred = HealthAI.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    porcentagem = accuracy * 100
    
    return porcentagem