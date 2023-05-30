from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

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
    
    print("Modelo treinado e salvo com sucesso!")
    print("Score no conjunto de teste:", test_score)

def load_and_predict(X, model_path):
    # Carregar o modelo completo (modelo + scaler)
    grid_search = joblib.load(model_path)

    # Fazer a previsão nos dados escalados
    predictions = grid_search.predict(X)

    # Retornar as previsões
    return predictions