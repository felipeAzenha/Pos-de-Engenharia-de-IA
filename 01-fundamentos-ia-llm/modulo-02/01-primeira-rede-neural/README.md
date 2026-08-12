# Módulo 02 - Redes Neurais com TensorFlow.js

Projeto do curso de Engenharia de IA focado na construção e treinamento de redes neurais utilizando TensorFlow.js em ambiente Node.js. O projeto implementa um pipeline de pré-processamento com padrão fit/transform, inspirado na forma como bibliotecas como o scikit-learn separam a lógica de codificação da lógica de aplicação.

## 🎯 Objetivo

Construir e treinar uma rede neural utilizando `@tensorflow/tfjs-node`, implementando um pipeline de pré-processamento robusto que garanta consistência dimensional dos vetores entre as fases de treinamento e predição.

## 🛠️ Tecnologias utilizadas

- Node.js (via nvm)
- TensorFlow.js (`@tensorflow/tfjs-node`)
- WSL: Ubuntu (ambiente de desenvolvimento)
- VS Code (Remote-WSL extension)

## 📦 Instalação

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd modulo-02

# Instale as dependências
npm install
```

> **Nota:** caso ocorram erros de instalação dos binários nativos do `tfjs-node` com Node v24, pode ser necessário alternar a versão do Node via `nvm` para uma versão compatível com o `tfjs-node@4.22`.

## 🚀 Como usar

O projeto é dividido em dois scripts independentes, que devem ser executados nessa ordem:

```bash
# 1. Treina a rede neural e salva o modelo + os artefatos de pré-processamento
node Model-Train.js
```

Isso gera na pasta do projeto:
- `ModeloTreinado/` — modelo treinado (`model.json` + pesos)
- `encoders.json` — parâmetros de normalização e one-hot encoding aprendidos no treino
- `labels.json` — categorias de saída (`premium`, `medium`, `basic`)

```bash
# 2. Carrega o modelo treinado e faz a predição de novas pessoas
node Model-Predict.js
```

O `Model-Predict.js` aplica **apenas o transform** dos encoders salvos (nunca refaz o fit), garantindo que os dados de entrada tenham sempre a mesma dimensionalidade usada no treino.

## 📊 Resultados

Modelo treinado com uma rede neural simples (`dense(80, relu)` → `dense(3, softmax)`), classificando pessoas em três categorias (`premium`, `medium`, `basic`) a partir de idade, cor e localização.

Exemplo de saída do `Model-Predict.js` para três pessoas novas (fora do dataset de treino):

```
Felipe → medium (44.24%)
   Outras probabilidades: premium (43.38%), basic (12.38%)

Mariana → basic (48.96%)
   Outras probabilidades: premium (33.99%), medium (17.06%)

zé → basic (77.84%)
   Outras probabilidades: medium (11.90%), premium (10.26%)
```

> **Nota:** o dataset de treino usado é minúsculo (3 pessoas), então o modelo tende a decorar os exemplos em vez de generalizar. O foco deste módulo foi validar o pipeline fit/transform e a separação treino/predição em scripts distintos — não a qualidade preditiva do modelo em si.

## 📝 Licença

Este projeto é de uso educacional, desenvolvido como parte do curso de Engenharia de IA.
