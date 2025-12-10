from services.vetorizacaoService import vetorizador, encode_Y
from services.treinamentoService import treinar_modelo
import pickle
import os

def salvar_vetorizador():
    vectorizer = vetorizador()
    path = 'model/vetorizador_HealthAI.pkl'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(vectorizer, file)
        print(f'Vetorizador salvo com sucesso!')

def salvar_encode_Y():
    labelEncoder = encode_Y()
    path = 'model/encoderY_HealthAI.pkl'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(labelEncoder, file)
        print(f'Label Encoder salvo com sucesso!')
        
def salvar_modelo():
    HealthAI = treinar_modelo()
    path = 'model/modelo_HealthAI.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster = HealthAI.get_booster()
    booster.save_model(path)
    print(f'Modelo salvo com sucesso!')