import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree


# Carregar o CSV com as distâncias em forma de plano carteziano 
df_distances = pd.read_csv('./locais_data_base.csv', index_col=0, sep=';')

def calculate_metrics(route, df_distances):
    """
    Calcula várias métricas de uma rota baseada nas distâncias entre locais.

    Args:
        route (list): Lista de locais representando a rota.
        df_distances (pd.DataFrame): DataFrame contendo as distâncias entre os locais.

    Returns:
        dict: Dicionário com as métricas da rota incluindo total_distance, avg_distance, max_distance, 
              min_distance, std_distance, initial_distance, e final_distance.

    Exemplos:
        >>> route = ['loc1', 'loc2', 'loc3']
        >>> df_distances = pd.DataFrame({
                'loc1': {'loc2': 10, 'loc3': 20},
                'loc2': {'loc1': 10, 'loc3': 15},
                'loc3': {'loc1': 20, 'loc2': 15}
            })
        >>> calculate_metrics(route, df_distances)
        {'total_distance': 25, 'avg_distance': 12.5, 'max_distance': 15, 'min_distance': 10,
         'std_distance': 2.5, 'initial_distance': 10, 'final_distance': 15}
    """
    distances = []
    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]
        distance = df_distances.loc[start, end]
        distances.append(distance)
    
    if len(distances) > 0:
        total_distance = sum(distances)
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        std_distance = np.std(distances)
        initial_distance = distances[0]
        final_distance = distances[-1]
    else:
        total_distance = avg_distance = max_distance = min_distance = std_distance = initial_distance = final_distance = 0
    
    return {
        'total_distance': total_distance,
        'avg_distance': avg_distance,
        'max_distance': max_distance,
        'min_distance': min_distance,
        'std_distance': std_distance,
        'initial_distance': initial_distance,
        'final_distance': final_distance
    }

# Gerar dados aleatórios para treinamento
locations = df_distances.index.tolist()
routes = []
for _ in range(5000):  # Aumentar a quantidade de dados aleatórios
    random_route = random.sample(locations, k=random.randint(2, len(locations)))
    metrics = calculate_metrics(random_route, df_distances)
    routes.append(metrics)

# Criar o DataFrame de treino
df_train = pd.DataFrame(routes)

# Verificar se df_train está corretamente estruturado
print(df_train.head())

# Separar os dados para o modelo
X = df_train.drop(['total_distance'], axis=1)
y = df_train['total_distance']

# Inicializar e treinar o modelo
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)


# Função para visualizar a última árvore
def visualize_last_tree(rf, X):
    plt.figure(figsize=(20, 10))
    plot_tree(rf.estimators_[-1], feature_names=X.columns, filled=True, rounded=True, fontsize=10)
    plt.title('Última Árvore Gerada')
    plt.show()

# Visualizar a última árvore da Random Forest
visualize_last_tree(rf, X)


def find_best_route(start_location, remaining_locations, df_distances, model, max_depth=7):
    """
    Encontra a melhor rota usando busca em profundidade limitada e um modelo de regressão.

    Args:
        start_location (str): Local de início da rota.
        remaining_locations (list): Lista de locais restantes para visitar.
        df_distances (pd.DataFrame): DataFrame contendo as distâncias entre os locais.
        model (RandomForestRegressor): Modelo treinado para prever a distância total.
        max_depth (int): Profundidade máxima da busca.

    Returns:
        tuple: Rota ótima e a distância prevista.

    Exemplos:
        >>> start_location = 'loc1'
        >>> remaining_locations = ['loc2', 'loc3']
        >>> df_distances = pd.DataFrame({
                'loc1': {'loc2': 10, 'loc3': 20},
                'loc2': {'loc1': 10, 'loc3': 15},
                'loc3': {'loc1': 20, 'loc2': 15}
            })
        >>> model = RandomForestRegressor().fit(...)
        >>> find_best_route(start_location, remaining_locations, df_distances, model, max_depth=3)
        (['loc1', 'loc2', 'loc3'], 25)
    """
    if len(remaining_locations) == 0:
        return [start_location], 0
    
    best_route = []
    best_distance = float('inf')
    
    for next_location in remaining_locations:
        new_route = [start_location, next_location]
        metrics = calculate_metrics(new_route, df_distances)
        X_new_route = pd.DataFrame([metrics]).drop(['total_distance'], axis=1)
        predicted_distance = model.predict(X_new_route)[0]
        
        if max_depth > 1:
            sub_route, sub_distance = find_best_route(next_location, [loc for loc in remaining_locations if loc != next_location], df_distances, model, max_depth - 1)
            predicted_distance += sub_distance
            new_route += sub_route[1:]
        
        if predicted_distance < best_distance:
            best_route = new_route
            best_distance = predicted_distance
    
    return best_route, best_distance

def get_user_input():
    """
    Obtém a entrada de locais do usuário.

    Returns:
        list: Lista de locais inseridos pelo usuário ou None se houver um erro.

    Exemplos:
        >>> # Entrada do usuário: "loc1,loc2,loc3"
        >>> get_user_input()
        ['loc1', 'loc2', 'loc3']
    """
    print("Digite os locais separados por vírgula (por exemplo: faminas,bar_do_lu,hospital_sao_paulo):")
    user_input = input().strip()
    locations = [loc.strip() for loc in user_input.split(",")]
    
    for loc in locations:
        if loc not in df_distances.index:
            print(f"Erro: o local '{loc}' não está na base de dados.")
            return None
    return locations

# Obter os locais do usuário
user_locations = get_user_input()
if user_locations:
    start_location = user_locations[0]
    remaining_locations = user_locations[1:]
    
    best_route, best_distance = find_best_route(start_location, remaining_locations, df_distances, rf, max_depth=7)
    print(f"Melhor rota: {best_route}")
    print(f"Distância prevista para a melhor rota: {best_distance:.2f}")

    # Definir coordenadas fixas para cada local
    loc_positions = {
        'faminas': (1, 3),
        'bar_do_lu': (4, 7),
        'bar_da_fatinha': (6, 5),
        'praca_do_rosario': (8, 8),
        'bar_do_broa': (10, 2),
        'inove': (12, 6),
        'dule': (14, 4),
        'mansao_gastropub': (3, 10),
        'praca_joao_pinheiro': (5, 9),
        'bar_betinho': (7, 1),
        'hospital_sao_paulo': (9, 7),
        'armacao_centro': (11, 3),
        'cristo': (13, 9),
        'rei_do_cachorrao': (15, 5),
        'colegio_santa_marcelina': (2, 8),
        'cachorrao_do_naldinho': (4, 2),
        'fest_house': (6, 6),
        'lagoa_da_gavea': (8, 4)
    }

    def plot_route(route, loc_positions):
        """
        Plota a rota em um gráfico.

        Args:
            route (list): Lista de locais representando a rota.
            loc_positions (dict): Dicionário com as posições fixas dos locais.

        Exemplos:
            >>> route = ['loc1', 'loc2', 'loc3']
            >>> loc_positions = {'loc1': (1, 1), 'loc2': (2, 2), 'loc3': (3, 3)}
            >>> plot_route(route, loc_positions)
            # Plota a rota em um gráfico.
        """
        plt.figure(figsize=(10, 8))
        
        for loc, pos in loc_positions.items():
            plt.scatter(*pos, label=loc)
            plt.text(pos[0], pos[1], loc, fontsize=9, ha='right')

        for i in range(len(route) - 1):
            start, end = route[i], route[i + 1]
            start_pos, end_pos = loc_positions[start], loc_positions[end]
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', lw=1)
        
        plt.title('Rota')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    plot_route(best_route, loc_positions)
