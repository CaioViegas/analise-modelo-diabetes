import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def remover_linhas_ausentes(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas que contêm valores ausentes.
    """
    return dataset.dropna()

def substituir_por_media(dataset: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Substitui valores ausentes pela média da coluna.
    """
    media = dataset[coluna].mean()
    dataset[coluna].fillna(media, inplace=True)
    return dataset

def substituir_por_mediana(dataset: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Substitui valores ausentes pela mediana da coluna.
    """
    mediana = dataset[coluna].median()
    dataset[coluna].fillna(mediana, inplace=True)
    return dataset

def substituir_por_moda(dataset: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Substitui valores ausentes pela moda da coluna.
    """
    moda = dataset[coluna].mode()[0]
    dataset[coluna].fillna(moda, inplace=True)
    return dataset

def imputar_com_ml(dataset: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Imputa valores ausentes utilizando um modelo de machine learning (Random Forest).
    """
    # Cria um DataFrame sem valores ausentes na coluna
    df_sem_ausentes = dataset[dataset[coluna].notna()]
    df_com_ausentes = dataset[dataset[coluna].isna()]

    # Se não houver dados suficientes, retorna o dataset original
    if df_sem_ausentes.empty:
        return dataset

    # Seleciona características (features) e alvo (target)
    X_train = df_sem_ausentes.drop(columns=coluna)
    y_train = df_sem_ausentes[coluna]

    # Treina o modelo
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Imputa valores ausentes
    X_missing = df_com_ausentes.drop(columns=coluna)
    predicted_values = model.predict(X_missing)

    dataset.loc[dataset[coluna].isna(), coluna] = predicted_values
    return dataset

def tratar_valores_ausentes(dataset: pd.DataFrame, coluna: str, metodo: str) -> pd.DataFrame:
    """
    Função principal para tratamento de valores ausentes. 
    Aceita o nome do método para aplicar.
    """
    if metodo == 'remover':
        return remover_linhas_ausentes(dataset)
    elif metodo == 'media':
        return substituir_por_media(dataset, coluna)
    elif metodo == 'mediana':
        return substituir_por_mediana(dataset, coluna)
    elif metodo == 'moda':
        return substituir_por_moda(dataset, coluna)
    elif metodo == 'ml':
        return imputar_com_ml(dataset, coluna)
    else:
        raise ValueError("Método desconhecido. Opções: 'remover', 'media', 'mediana', 'moda', 'ml'.")

