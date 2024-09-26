from typing import Optional, List, Dict
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder  

def _aplicar_pre_processamento(dataset: pd.DataFrame,
                               colunas_remover: Optional[List[str]] = None,
                               colunas_label: Optional[List[str]] = None,
                               colunas_hot: Optional[List[str]] = None,
                               colunas_ordinal: Optional[List[str]] = None,
                               colunas_target: Optional[Dict[str, str]] = None,
                               coluna_zero: Optional[str] = None) -> pd.DataFrame:
    """
    Aplica transformações principais de pré-processamento no dataset.
    """
    # Remover colunas
    if colunas_remover:
        dataset.drop(columns=colunas_remover, axis=1, inplace=True)

    # Remover linhas onde o valor da coluna especificada é igual a zero
    if coluna_zero:
        dataset = dataset[dataset[coluna_zero] != 0]

    # Codificação com LabelEncoder
    if colunas_label:
        le = LabelEncoder()
        for coluna in colunas_label:
            dataset[coluna] = le.fit_transform(dataset[coluna]).astype('int64')

    # Codificação OneHotEncoder
    if colunas_hot:
        ohe = OneHotEncoder(drop='first', dtype='int64', sparse_output=False)
        for coluna in colunas_hot:
            encoded_cols = ohe.fit_transform(dataset[[coluna]])
            encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out([coluna]), index=dataset.index)
            dataset = pd.concat([dataset, encoded_df], axis=1)
            dataset.drop(columns=[coluna], inplace=True)

    # Codificação OrdinalEncoder
    if colunas_ordinal:
        oe = OrdinalEncoder(dtype='int64')
        for coluna in colunas_ordinal:
            dataset[coluna] = oe.fit_transform(dataset[[coluna]]).astype('int64')

    # Codificação TargetEncoder
    if colunas_target:
        for coluna, target in colunas_target.items():
            te = TargetEncoder()
            dataset[coluna] = te.fit_transform(dataset[coluna], dataset[target]).astype('float64')

    return dataset

def funcao_processamento(dataset: pd.DataFrame,
                         colunas_remover: Optional[List[str]] = None,
                         colunas_label: Optional[List[str]] = None,
                         colunas_hot: Optional[List[str]] = None,
                         colunas_ordinal: Optional[List[str]] = None,
                         colunas_target: Optional[Dict[str, str]] = None,
                         coluna_zero: Optional[str] = None) -> pd.DataFrame:
    """
    Realiza o pré-processamento de um DataFrame, aplicando múltiplos tipos de codificação e salvando o resultado.
    """
    dataset_processado = _aplicar_pre_processamento(dataset, colunas_remover, colunas_label, colunas_hot, colunas_ordinal, colunas_target, coluna_zero)
    dataset_processado.to_csv("./data/dados_transformados.csv", index=False)
    return dataset_processado

def processamento_teste(dataset: pd.DataFrame,
                        colunas_remover: Optional[List[str]] = None,
                        colunas_label: Optional[List[str]] = None,
                        colunas_hot: Optional[List[str]] = None,
                        colunas_ordinal: Optional[List[str]] = None,
                        colunas_target: Optional[Dict[str, str]] = None,
                        coluna_zero: Optional[str] = None) -> pd.DataFrame:
    """
    Realiza o pré-processamento de um DataFrame, retornando o DataFrame processado.
    """
    return _aplicar_pre_processamento(dataset, colunas_remover, colunas_label, colunas_hot, colunas_ordinal, colunas_target, coluna_zero)

if __name__ == '__main__':
    df = pd.read_csv('./data/diabetes_dataset.csv')
    funcao_processamento(dataset=df, colunas_remover=['location', 'year'], colunas_label=['gender'], colunas_hot=['smoking_history'])
