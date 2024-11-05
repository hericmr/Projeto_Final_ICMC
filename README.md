# Classificação de Espécies de Flores com Machine Learning

Este projeto é o trabalho final do curso de **ICMC da USP** e consiste em um estudo prático de aprendizado de máquina utilizando o famoso dataset **Iris**. O objetivo é classificar espécies de flores com base em características morfológicas.

## Objetivo

Desenvolver e avaliar modelos de aprendizado de máquina para identificar a espécie de flor a partir das medidas de sépala e pétala, incluindo:

1. Pré-processamento dos dados.
2. Treinamento e teste de modelos de classificação.
3. Comparação de desempenho entre algoritmos.

## Dataset

O **dataset Iris** contém 150 amostras de três espécies de flores:

- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

Cada amostra possui quatro características:

- **SepalLengthCm**: Comprimento da sépala
- **SepalWidthCm**: Largura da sépala
- **PetalLengthCm**: Comprimento da pétala
- **PetalWidthCm**: Largura da pétala

A coluna alvo é **Species**, que indica a espécie da flor.

## Estrutura do Projeto

1. **main.py**: Código principal com métodos para cada etapa do fluxo de trabalho de machine learning.
2. **iris.data**: Dataset Iris em formato CSV.

### Modelos de Machine Learning

Os algoritmos utilizados incluem:

- **SVM (Support Vector Machine)**
- **Decision Tree**

Ambos os modelos são avaliados com validação cruzada e matriz de confusão.

## Resultados e Análise

Os modelos foram avaliados quanto à acurácia, destacando diferenças nas abordagens e eficácia para o dataset Iris.

## Como Executar

1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
