>>> Criando modelos com a API de camadas<<<

>> Existem duas formas de criar um modelo usando a API de camadas: 
> Um modelo sequencial e 
> um modelo funcional. 

> Abaixo exemplo de modelo sequencial:
> O tipo mais comum de modelo é o modelo Sequencial, que é uma pilha linear de camadas. Você pode criar um modelo Sequencial passando uma lista de camadas para a função sequential():

import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()
    
    // Primeira camada da rede:
    // 1. entrada de 7 posições  (idade normalizada + 3 cores + 3 localizações)

    // 2. 80 neuronios - aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e consequentemente, mais tempo de treino vai usar(processamento)

    // 3. função de ativação relu age como um filtro (retorna 0 se o valor for negativo, e retorna o
    //  valor se for positivo)
    // é como se ela deixasse os dados interessantes seguirem viagem na rede
    // se a informação ao neuronio é positiva, então ela é relevante para o aprendizado da rede
    // se for zero ou negativa, então ela não é relevante e não vai seguir viagem na rede

    model.add(tf.layers.dense({inputShape: [7], units: 80, activation: 'relu'}))

    // Saida da rede: 3 neuronios, um para cada categoria (premium, medium, basic)
    // função de ativação softmax, que vai retornar a probabilidade de cada categoria
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}))

    // Compilando o modelo, definindo a função de perda e o otimizador:

    // optimizer 'adam' é um dos mais usados, ele ajusta os pesos da rede de forma eficiente
    // ele aprende com o historico de treino da rede, e ajusta os pesos de forma mais inteligente

    // loss 'categoricalCrossentropy' é a função de perda usada para problemas de classificação multiclasse
    // ele compara o que o modelo previu com o que era esperado, e calcula o erro
    // o objetivo do treino é minimizar esse erro, ou seja, fazer com que o modelo acerte mais

    // metrics 'accuracy' é a métrica que avalia o desempenho do modelo
    // e nos diz a porcentagem de acertos do modelo durante o treino e teste
    // quanto maior a acurácia, melhor o modelo está performando

    // quanto mais distante da previsão do modelo estiver do valor esperado, maior será a perda (loss)
    // e categorical class crossentropy é usado para classificação de imagens, textos, 
    // recomendações, categorização de produtos, etc. Onde a resposta esperada é uma das categorias
    //  possíveis, e o modelo precisa aprender a prever corretamente a categoria correta.

    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] })

    // verbose: desabilita o log interno e usa só o callback para mostrar o progresso do treino
    // epochs: quantidade de vezes que o modelo vai ver todo o dataset de treino
    // shuffle: embaralha os dados a cada treinamento, para evitar viés de ordem dos dados
    await model.fit(
        inputXs, 
        outputYs, 
        {
            epochs: 100, // número de vezes que o modelo vai ver todo o dataset de treino
            batchSize: 3, // número de amostras que o modelo vai ver antes de atualizar os pesos
            shuffle: true, // embaralha os dados a cada treinamento, para evitar que o modelo aprenda padrões específicos da ordem dos dados
            verbose: 0, // mostra o progresso do treino no console, 1 para mostrar, 0 para não mostrar
            callbacks: {
                // para não aparecer o log é só comentar o onEpochEnd 
                onEpochEnd: (epoch, log) => console.log( 
                    `Epoch ${epoch}: loss = ${log.loss}`
                )
            }
        }
    )


    return model
}
