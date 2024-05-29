# RandomForest
  Este projeto foi desenvolvido para atender aos requisitos de um trabalho da faculdade de Análise e Desenvolvimento de Sistemas (ADS) na disciplina de Ciências de Dados. O objetivo era criar um projeto onde, dada uma base de distâncias entre locais, deveríamos utilizar o algoritmo RandomForest para encontrar a melhor rota possível com base em uma entrada de dados composta por uma lista de nomes de locais.
## Participantes
  - Pedro Lucas
  - Luis Otavio
  - Maycon Pereira
  - João Lucas

## Como rodar
  1. Clonar o Repositório

    Clone o repositório para sua máquina local utilizando o comando:
    ```bash
      git clone https://github.com/PedroLLOliveira/RandomForest.git
    ```
  2. Criar Ambiente Virtual Python

    Crie um ambiente virtual para isolar as dependências do projeto:
    ```bash
      python -m venv venv
    ```
    ou
    ``` bash
      python3 -m venv venv
    ```
  3. Ativar o Ambiente Virtual

    Ative o ambiente virtual criado no passo anterior:
    ```bash
      // windows 
      ./venv/Scripts/activate
      // linux
      source venv/bin/activate
    ```
  4. Instalar Dependências

    Instale todas as dependências necessárias listadas no arquivo requirements.txt:
    ```bash
      pip install -r requirements.txt
    ```
  5. Rodar a Aplicação

    Execute a aplicação principal:
    ```bash
      python solver.py
    ```
  
## Explicação do Código

  **Importações e Carregamento dos Dados**

  O projeto começa importando as bibliotecas necessárias e carregando o CSV contendo as distâncias entre os locais.

  ```python
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor

    df_distances = pd.read_csv('./locais_data_base.csv', index_col=0, sep=';')
  ```

  **Função para Calcular Métricas**

  Esta função calcula várias métricas de uma rota baseada nas distâncias entre os locais.

  ```python
    def calculate_metrics(route, df_distances):
    # Implementação da função...
    return metrics
  ```

  **Geração de Dados Aleatórios para Treinamento**

  Gera dados de rotas aleatórias e calcula as métricas para cada uma delas, criando um DataFrame para treinamento.

  ```python
    locations = df_distances.index.tolist()
    routes = []
    for _ in range(1000):
        random_route = random.sample(locations, k=random.randint(2, len(locations)))
        metrics = calculate_metrics(random_route, df_distances)
        routes.append(metrics)
    df_train = pd.DataFrame(routes)
  ```

  **Separação dos Dados e Treinamento do Modelo**

  Separa os dados em variáveis independentes (X) e dependentes (y) e treina um modelo RandomForestRegressor.

  ```python 
    X = df_train.drop(['total_distance'], axis=1)
    y = df_train['total_distance']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
  ```

  **Função para Encontrar a Melhor Rota**

  Usa uma busca em profundidade limitada e o modelo treinado para encontrar a melhor rota possível.

  ```python 
    def find_best_route(start_location, remaining_locations, df_distances, model, max_depth=5):
    # Implementação da função...
    return best_route, best_distance
  ```

  **Função para Obter Entrada do Usuário**

  Obtém uma lista de locais inseridos pelo usuário e verifica se todos estão na base de dados.

  ```python
    def get_user_input():
    # Implementação da função...
    return locations 
  ```

  **Execução Principal**

  Obtemos os locais do usuário, encontramos a melhor rota e visualizamos essa rota em um gráfico.

  ```python 
    user_locations = get_user_input()
    if user_locations:
        start_location = user_locations[0]
        remaining_locations = user_locations[1:]
        best_route, best_distance = find_best_route(start_location, remaining_locations, df_distances, rf, max_depth=4)
        print(f"Melhor rota: {best_route}")
        print(f"Distância prevista para a melhor rota: {best_distance:.2f}")

        loc_positions = {
            # Dicionário com coordenadas fixas...
        }
        
        def plot_route(route, loc_positions):
            # Implementação da função...
            plt.show()

        plot_route(best_route, loc_positions)
  ```

  Este README deve ajudá-lo a entender como configurar e executar o projeto, bem como fornecer uma visão geral do que cada parte do código está fazendo.