from typing import Optional, Dict, Any, List
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV

def balance_data(X: pd.DataFrame, y: pd.Series, method: str = 'under', random_state: int = 42):
    """
    Balancear dados desbalanceados usando o método de reamostragem especificado.

    Parâmetros:
    - X: A matriz de características.
    - y: O vetor alvo.
    - method: O método de reamostragem a ser usado ('under', 'over' ou 'smote').
    - random_state: A semente aleatória para reprodutibilidade.

    Retorna:
    - X_resampled: A matriz de características reamostrada.
    - y_resampled: O vetor alvo reamostrado.
    """
    if method == 'under':
        # Reamostragem aleatória para subamostragem
        resampler = RandomUnderSampler(random_state=random_state)
    elif method == 'over':
        # Reamostragem aleatória para superamostragem
        resampler = RandomOverSampler(random_state=random_state)
    elif method == 'smote':
        # Reamostragem SMOTE para superamostragem
        resampler = SMOTE(random_state=random_state)
    else:
        raise ValueError(f"Método inválido: {method}")

    # Realizar reamostragem
    X_resampled, y_resampled = resampler.fit_resample(X, y)

    return X_resampled, y_resampled

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, selected_features:List[str]=None, method: str = 'standard') -> pd.DataFrame:
    """
    Scale the data in a DataFrame using the specified method.

    Parameters
    - df: The DataFrame to scale.
    - method: The scaling method to use. Options are 'standard', 'minmax', and 'robust'.

    Returns
    - scaled_df: The scaled DataFrame.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    X_train = X_train.copy()
    X_test = X_test.copy()

    if selected_features is not None:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    # Create the scaler object
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Invalid method: {method}. Valid options are 'standard', 'minmax', and 'robust'.")

    # Fit the scaler on the data and transform the data
    scaler.fit(X_train)

    # Transform the test data
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)

    return X_train_scaled, X_test_scaled, scaler

def train_model_with_grid_search(X: pd.DataFrame,
                                 y: pd.Series,
                                 model: Any,
                                 param_grid: Dict[str, Any],
                                 test_size: float = 0.2,
                                 random_state: int = 42,
                                 balance_training: Optional[str] = None,
                                 save_path: str = 'modelo_com_scaler_e_grid.joblib',
                                 scaling_method: str = None):
    """
    Train a model using GridSearch and save the best model along with the scaler.

    Parameters:
    - X: The feature matrix.
    - y: The target vector.
    - model: The model to be trained.
    - param_grid: The parameter grid for GridSearch.
    - test_size: The proportion of test data.
    - random_state: The random seed for reproducibility.
    - balance_training: The resampling method to balance the training data ('under', 'over', 'smote' or None).
    - save_path: The file path to save the trained model.
    - scaling_method: The scaling method to use. Options are 'standard', 'minmax', and 'robust'.

    Returns:
    - None
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Balance the training data
    if balance_training == 'under':
        X_train, y_train = balance_data(X_train, y_train, method='under', random_state=random_state)
    elif balance_training == 'over':
        X_train, y_train = balance_data(X_train, y_train, method='over', random_state=random_state)
    elif balance_training == 'smote':
        X_train, y_train = balance_data(X_train, y_train, method='smote', random_state=random_state)
    elif balance_training is None:
        pass

    # Create the scaler object
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    elif scaling_method is None:
        scaler = None
    else:
        raise ValueError(f"Invalid method: {scaling_method}. Valid options are 'standard', 'minmax', and 'robust'.")

    # Create a pipeline with the scaler built-in
    if scaler is not None:
        pipeline = Pipeline([
            ('scaler', scaler), # Scaler
            ('model', model) # Model
        ])
    else:
        pipeline = Pipeline([
            ('model', model) # Model
        ])

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, return_train_score=True)
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Save the complete model (model + scaler) and the parameter grid
    joblib.dump(grid_search, save_path)
    
    # Evaluate the model on the test set
    test_score = grid_search.score(X_test, y_test)
    train_scores = grid_search.cv_results_['mean_train_score']

    print("Best parameters:", grid_search.best_params_)
    print("Model trained and saved successfully!")
    print("Score on training set:", train_scores[grid_search.best_index_])
    print("Score on test set:", test_score)

def load_and_predict(X: pd.DataFrame, model_path: str):
    """
    Carregar o modelo completo (modelo + escalador) e fazer previsões nos dados fornecidos.

    Parâmetros:
    - X: A matriz de características para previsão.
    - model_path: O caminho do arquivo do modelo salvo.

    Retorna:
    - predictions: O vetor alvo previsto.
    """
    # Carregar o modelo completo (modelo + escalador)
    grid_search = joblib.load(model_path)

    # Fazer previsões nos dados escalados
    predictions = grid_search.predict(X)

    # Retornar as previsões
    return predictions

def find_potential_outliers(df: pd.DataFrame, target_col: str):
    """
    Encontrar possíveis valores discrepantes nas colunas numéricas de um DataFrame para cada valor único na coluna alvo.

    Parâmetros:
    - df: O DataFrame para procurar valores discrepantes.
    - target_col: O nome da coluna alvo.

    Retorna:
    - result_df: DataFrame contendo a coluna alvo, o número de valores discrepantes e um dicionário de valores discrepantes
                 para cada valor único na coluna alvo.
    """
    result = []
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Obter os valores únicos na coluna alvo
    targets = df[target_col].unique()

    # Iterar sobre cada valor alvo
    for target in targets:
        # Criar um subconjunto do DataFrame para o valor alvo atual
        df_target = df[df[target_col] == target]

        # Inicializar um dicionário para armazenar as colunas e valores discrepantes
        outliers = {}

        # Iterar sobre cada coluna numérica
        for col in numerical_columns:
            if col != target_col:
                # Calcular os quartis e a amplitude interquartil
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Calcular os limites inferior e superior para valores discrepantes
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Verificar se algum valor na coluna para o valor alvo atual é um valor discrepante
                if any((df_target[col] < lower_bound) | (df_target[col] > upper_bound)):
                    # Adicionar o nome da coluna e o valor ao dicionário de valores discrepantes
                    outliers[col] = df_target[col].values[0]

        # Adicionar o valor alvo, o número de valores discrepantes e o dicionário de valores discrepantes à lista de resultados
        result.append({"target": target, "n_outliers": len(outliers), "outlier_dict": outliers})

    # Criar um novo DataFrame para armazenar os resultados
    result_df = pd.DataFrame(result)

    return result_df[result_df['n_outliers'] != 0].sort_values("n_outliers", ascending=False)

def run_boruta(X: pd.DataFrame, y: pd.Series, estimator: Any, max_iter: int = 100, random_state: int = 42):
    """
    Executar o algoritmo Boruta para seleção de recursos.

    Parâmetros:
    - X: A matriz de características.
    - y: O vetor alvo.
    - estimator: O estimador base usado pelo Boruta.
    - max_iter: O número máximo de iterações.
    - random_state: A semente aleatória para reprodutibilidade.

    Retorna:
    - boruta_selector: Objeto BorutaPy contendo os resultados.
    """
    # Criar o objeto Boruta
    boruta_selector = BorutaPy(estimator, n_estimators='auto', max_iter=max_iter, random_state=random_state)

    # Executar o Boruta
    boruta_selector.fit(X.values, y.values)

    # print ranking with selected feature names
    print(pd.DataFrame({'feature': X.columns, 'rank': boruta_selector.ranking_}).sort_values(by='rank', ascending=True))

    return boruta_selector

def run_rfe(X_train: pd.DataFrame, y_train: pd.Series, columns: List[str], estimator: Any, n_features_to_select: int = None):
    """
    Run the RFE algorithm for feature selection.

    Parameters
    ----------
    - X_train: The feature matrix.
    - y_train: The target vector.
    - estimator: The base estimator used by RFE.
    - n_features_to_select: The number of features to select.

    Returns
    -------
    - rfe_selector: RFE object containing the results.
    """
    # Create the RFE object
    rfe_selector = RFE(estimator, n_features_to_select=n_features_to_select)

    # Run RFE
    rfe_selector.fit(X_train, y_train)

    # print ranking with selected feature names
    print(pd.DataFrame({'feature': columns, 'rank': rfe_selector.ranking_}).sort_values(by='rank', ascending=True))

    # select features
    X_filtered = rfe_selector.transform(X_train)

    return rfe_selector, X_filtered

def run_lasso(X_train, y_train, X_test, scaler='standard'):
    """
    Run the LASSO algorithm for feature selection.

    Parameters
    ----------
    - X_test: The test feature matrix.
    - X_train: The training feature matrix.
    - y_train: The target vector.
    - columns: The list of column names from X.

    Returns
    -------
    - lasso_selector: LassoCV object containing the results.
    - X_test_filtered: Filtered version of X_test.
    - X_train_filtered: Filtered version of X_train.
    """
    #scale data
    X_train_scale, __, __ = scale_data(X_train=X_train, X_test=X_test, method=scaler)

    # Create the LassoCV object
    lasso_selector = LassoCV()

    # Run LassoCV
    lasso_selector.fit(X_train_scale, y_train)

    # print ranking with selected feature names
    rank = pd.DataFrame({'feature': X_train.columns, 'coef': lasso_selector.coef_}).sort_values(by='coef', ascending=False)
    print(rank)

    # index of columns to keep
    idx = rank[rank['coef'] != 0]['feature'].index

    # resulting feature names as a list
    selected_features = rank[rank['coef'] != 0]['feature'].tolist()

    return lasso_selector, selected_features

