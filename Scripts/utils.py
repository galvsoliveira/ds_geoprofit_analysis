from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from boruta import BorutaPy

def train_model_with_grid_search(X, y, model, param_grid, test_size=0.2, random_state=42, save_path='modelo_com_scaler_e_grid.joblib'):
    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Criar um pipeline com o scaler embutido
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaler
        ('model', model)  # Modelo
    ])
    
    # Criar o objeto GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    
    # Realizar a busca em grade
    grid_search.fit(X_train, y_train)
    
    # Salvar o modelo completo (modelo + scaler) e o grid de parâmetros
    joblib.dump(grid_search, save_path)
    
    # Avaliar o modelo no conjunto de teste
    test_score = grid_search.score(X_test, y_test)

    print("Melhores parâmetros:", grid_search.best_params_)
    print("Modelo treinado e salvo com sucesso!")
    print("Score no conjunto de treinamento:", grid_search.best_score_)
    print("Score no conjunto de teste:", test_score)

def load_and_predict(X, model_path):
    # Carregar o modelo completo (modelo + scaler)
    grid_search = joblib.load(model_path)

    # Fazer a previsão nos dados escalados
    predictions = grid_search.predict(X)

    # Retornar as previsões
    return predictions

def find_potential_outliers(df, target_col):
    result = []
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Get the unique countries in the dataframe
    countries = df[target_col].unique()

    # Iterate over each target
    for target in countries:
        # Create a subset of the dataframe containing only rows for the given target
        df_target = df[df[target_col] == target]

        # Initialize a dictionary to store the outlier columns and values
        outliers = {}

        # Iterate over each column in the dataframe
        for col in numerical_columns:
            if col != target_col:
                # Calculate the quartiles and interquartile range
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Calculate the upper and lower bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Check if any value in the column for the given target is an outlier
                if any((df_target[col] < lower_bound) | (df_target[col] > upper_bound)):
                    # Add the column name and value to the outliers dictionary
                    outliers[col] = df_target[col].values[0]

        # Append the target, number of outliers, and outliers dictionary to the result list
        result.append({"target": target, "n_outliers": len(outliers), "outlier_dict": outliers})

    # Create a new dataframe to store the results
    result_df = pd.DataFrame(result)

    return result_df[result_df['n_outliers']!=0].sort_values("n_outliers", ascending=False)

def run_boruta(X, y, estimator, max_iter=100, random_state=42):
    """
    Executa o algoritmo Boruta para seleção de recursos.

    Parâmetros:
    - X: Matriz de características.
    - y: Vetor de variável alvo.
    - estimator: Estimador base utilizado pelo Boruta.
    - max_iter: Número máximo de iterações.
    - random_state: Semente aleatória para reprodutibilidade.

    Retorna:
    - boruta_selector: Objeto BorutaPy contendo os resultados.
    """

    # Criar o objeto Boruta
    boruta_selector = BorutaPy(estimator, n_estimators='auto', max_iter=max_iter, random_state=random_state)

    # Executar o Boruta
    boruta_selector.fit(X.values, y.values)

    return boruta_selector