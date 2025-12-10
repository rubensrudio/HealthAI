

from services.datasetService import dataset_completo
from services.vetorizacaoService import vetorizacao, encode_Y
from services.treinamentoService import acuracia_modelo



def prints_dataset():
    df = dataset_completo()
    return df
    
def prints_vetorizacao():
    X_transform = vetorizacao()
    print(X_transform)
    
def prints_encode_Y():
    Y_encoded = encode_Y()
    print(Y_encoded)
    
def acuracia_print():
    accuracy = acuracia_modelo()
    print(f'Acurácia do modelo: {accuracy:.2f}%')

if __name__ == "__main__":
    acuracia_print()