import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Modelo():
    def __init__(self):
        pass

    def CarregarDataset(self, path):
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):
        # Remover valores ausentes
        self.df.dropna(inplace=True)
        
        # Separar features e target
        self.X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        self.y = self.df['Species']

    def Treinamento(self, model):
        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Treinamento
        model.fit(X_train, y_train)
        
        # Avaliação com validação cruzada
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Validação cruzada (acurácia média): {scores.mean():.2f}")

        # Teste final
        y_pred = model.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        print(f"Acurácia no conjunto de teste: {acuracia:.2f}")
        print(classification_report(y_test, y_pred))
        
        # Plotar matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def Train(self):
        self.CarregarDataset("iris.data")
        self.TratamentoDeDados()
        
        print("==> Treinando com SVM:")
        self.Treinamento(SVC())

        print("\n==> Treinando com Decision Tree:")
        self.Treinamento(DecisionTreeClassifier())
