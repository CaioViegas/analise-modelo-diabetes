from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, confusion_matrix
from typing import Dict, Any
from sklearn.base import ClassifierMixin
import numpy as np

def funcao_avaliadora(modelo: ClassifierMixin, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Treina um modelo de machine learning nos dados de treinamento e avalia seu desempenho 
    nos dados de treinamento e de teste, utilizando várias métricas de classificação.

    Parâmetros:
    -----------
    modelo : object
        Um modelo de machine learning que implementa os métodos `fit` e `predict`.
        
    X_train : array-like, shape (n_samples, n_features)
        Conjunto de dados de entrada para treinamento.
        
    X_test : array-like, shape (n_samples, n_features)
        Conjunto de dados de entrada para teste.
        
    y_train : array-like, shape (n_samples,)
        Labels verdadeiros para o conjunto de treinamento.
        
    y_test : array-like, shape (n_samples,)
        Labels verdadeiros para o conjunto de teste.

    Retorna:
    --------
    resultados : dict
        Um dicionário contendo as seguintes métricas de avaliação para os conjuntos de 
        treinamento e teste.
    """
    modelo.fit(X_train, y_train)

    train_pred = modelo.predict(X_train)
    test_pred = modelo.predict(X_test)

    report_train = classification_report(y_train, train_pred, output_dict=True)
    report_test = classification_report(y_test, test_pred, output_dict=True)

    roc_auc_test = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1], multi_class='ovr')
    mcc_test = matthews_corrcoef(y_test, test_pred)

    resultados = {
        "Classification Report Train": report_train,
        "Classification Report Test": report_test,
        "ROC AUC Test": roc_auc_test,
        "Matthews Correlation Coefficient Test": mcc_test,
        "Confusion Matrix Test": confusion_matrix(y_test, test_pred)
    }

    return resultados