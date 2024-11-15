# Importando as bibliotecas
import pandas as pd  # p/ manipulação de DataFrames
from sklearn.model_selection import train_test_split, cross_val_score  # p/ dividir os dados e validação cruzada
from sklearn.svm import SVC  # modelo de Support Vector Machine
from sklearn.linear_model import LinearRegression  # modelo de regressão linear (não usado nesse código, mas foi importado)
from sklearn.tree import DecisionTreeClassifier  # arvore de Decisão
from sklearn.ensemble import RandomForestClassifier  # Random Forest - também não usado aqui
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Métricas de avaliação
import matplotlib.pyplot as plt  # Para visualização gráfica
import seaborn as sns  # Biblioteca para gráficos mais bonitos

# definindo classe p/ organizar o trabalho com os modelos
class Modelo():
    def __init__(self):
        self.resultados = []  # lista para armazenar os resultados dos modelos

    # função p/ carregar o dataset
    def CarregarDataset(self, path):
        # Nomeando colunas para facilitar o trabalho com o dataset
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)  # Lendo o arquivo CSV no pandas

    # Função para tratar os dados
    def TratamentoDeDados(self):
        # Remover dados ausentes (não queremos problemas com valores vazios)
        self.df.dropna(inplace=True)
        
        # Separar as colunas de entrada (features) e a coluna alvo (target)
        self.X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        self.y = self.df['Species']

    # Função para treinar e avaliar o modelo
    def Treinamento(self, model, nome_modelo):
        # Dividir os dados em treino e teste (80% para treino e 20% para teste)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Treinando o modelo com os dados de treino
        model.fit(X_train, y_train)
        
        # Avaliação usando validação cruzada (divide os dados em "pedaços" para avaliar melhor)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Validação cruzada ({nome_modelo}) - Acurácia média: {scores.mean():.2f}")

        # Prevendo os dados de teste e calculando a acurácia
        y_pred = model.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        print(f"Acurácia no conjunto de teste ({nome_modelo}): {acuracia:.2f}")
        print(classification_report(y_test, y_pred))  # Relatório com várias métricas
        
        # Adicionando os resultados para comparação
        self.resultados.append({'Modelo': nome_modelo, 'Validação Cruzada': scores.mean(), 'Acurácia Teste': acuracia})
        
        # Criar e mostrar a matriz de confusão (para ver erros de classificação)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Matriz de Confusão - {nome_modelo}')
        plt.xlabel('Predicted')  # O que o modelo previu
        plt.ylabel('True')  # O que era o correto
        plt.show()

    # Função principal para executar os treinos
    def Train(self):
        self.CarregarDataset("iris.data")  # Carregando o dataset Iris
        self.TratamentoDeDados()  # Limpando e separando os dados
        
        # Treinamento e avaliação de dois modelos
        print("==> Treinando com SVM:")  # Iniciando o treino com SVM
        self.Treinamento(SVC(), "SVM")  # Chamando a função de treino com SVM

        print("\n==> Treinando com Decision Tree:")  # Agora usando Decision Tree
        self.Treinamento(DecisionTreeClassifier(), "Decision Tree")  # Chamando a função de treino com Árvore de Decisão

        # Comparando os resultados
        print("\n==> Comparação de Modelos:")
        resultados_df = pd.DataFrame(self.resultados)  # Criando um DataFrame com os resultados
        print(resultados_df)  # Exibindo a tabela de comparação

# Definindo uma classe para organizar o trabalho com os modelos
class Modelo():
    def __init__(self):
        pass  # Método vazio, mas poderia inicializar variáveis se precisasse

    # Função para carregar o dataset
    def CarregarDataset(self, path):
        # Nomeando as colunas para facilitar o trabalho com o dataset
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)  # Lendo o arquivo CSV no pandas

    # Função para tratar os dados
    def TratamentoDeDados(self):
        # Remover dados faltando (nao queremos problemas com valores vazios)
        self.df.dropna(inplace=True)
        
        # Separar as colunas de entrada (features) e a coluna alvo (target)
        self.X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        self.y = self.df['Species']

    # Função para treinar e avaliar o modelo
    def Treinamento(self, model):
        # Dividir os dados em treino e teste (80% para treino e 20% para teste)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Treinando o modelo com os dados de treino
        model.fit(X_train, y_train)
        
        # Avaliação usando validação cruzada (divide os dados em "pedaços" para avaliar melhor)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Validação cruzada (acurácia média): {scores.mean():.2f}")

        # Prevendo os dados de teste e calculando a acurácia
        y_pred = model.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        print(f"Acurácia no conjunto de teste: {acuracia:.2f}")
        print(classification_report(y_test, y_pred))  # Relatório com várias métricas
        
        # Criar e mostrar a matriz de confusão (para ver erros de classificação)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')  # O que o modelo previu
        plt.ylabel('True')  # O que era o correto
        plt.show()

    # Função principal para executar os treinos
    def Train(self):
        self.CarregarDataset("iris.data")  # Carregando o dataset Iris
        self.TratamentoDeDados()  # Limpando e separando os dados
        
        print("==> Treinando com SVM:")  # Iniciando o treino com SVM
        self.Treinamento(SVC())  # Chamando a função de treino com SVM

        print("\n==> Treinando com Decision Tree:")  # Agora usando Decision Tree
        self.Treinamento(DecisionTreeClassifier())  # Chamando a função de treino com Árvore de Decisão
