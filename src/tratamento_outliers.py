import pandas as pd
import numpy as np

def remover_outliers_iqr(dataset: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Remove outliers usando o método do intervalo interquartil (IQR).
    """
    Q1 = dataset[coluna].quantile(0.25)
    Q3 = dataset[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return dataset[(dataset[coluna] >= limite_inferior) & (dataset[coluna] <= limite_superior)]

def remover_outliers_zscore(dataset: pd.DataFrame, coluna: str, threshold: float = 3) -> pd.DataFrame:
    """
    Remove outliers usando o método do Z-score.
    """
    z_scores = np.abs((dataset[coluna] - dataset[coluna].mean()) / dataset[coluna].std())
    return dataset[z_scores < threshold]

def substituir_outliers_media(dataset: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Substitui outliers pela média da coluna.
    """
    Q1 = dataset[coluna].quantile(0.25)
    Q3 = dataset[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    media = dataset[coluna].mean()
    dataset[coluna] = np.where((dataset[coluna] < limite_inferior) | (dataset[coluna] > limite_superior), media, dataset[coluna])
    return dataset

def substituir_outliers_mediana(dataset: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Substitui outliers pela mediana da coluna.
    """
    Q1 = dataset[coluna].quantile(0.25)
    Q3 = dataset[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    mediana = dataset[coluna].median()
    dataset[coluna] = np.where((dataset[coluna] < limite_inferior) | (dataset[coluna] > limite_superior), mediana, dataset[coluna])
    return dataset

def cap_outliers(dataset: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Corta os outliers para o limite superior e inferior usando o IQR.
    """
    Q1 = dataset[coluna].quantile(0.25)
    Q3 = dataset[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    dataset[coluna] = np.where(dataset[coluna] < limite_inferior, limite_inferior, dataset[coluna])
    dataset[coluna] = np.where(dataset[coluna] > limite_superior, limite_superior, dataset[coluna])
    return dataset

def tratar_outliers(dataset: pd.DataFrame, coluna: str, metodo: str) -> pd.DataFrame:
    """
    Função principal para tratamento de outliers. 
    Aceita o nome do método para aplicar.
    """
    if metodo == 'iqr':
        return remover_outliers_iqr(dataset, coluna)
    elif metodo == 'zscore':
        return remover_outliers_zscore(dataset, coluna)
    elif metodo == 'substituir_media':
        return substituir_outliers_media(dataset, coluna)
    elif metodo == 'substituir_mediana':
        return substituir_outliers_mediana(dataset, coluna)
    elif metodo == 'cap':
        return cap_outliers(dataset, coluna)
    else:
        raise ValueError("Método desconhecido. Opções: 'iqr', 'zscore', 'substituir_media', 'substituir_mediana', 'cap'.")
