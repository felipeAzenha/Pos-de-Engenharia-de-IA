import tf, { min } from '@tensorflow/tfjs-node';
import fs from 'fs';

///////// BLOCO DE CÓDIGO PARA TREINAMENTO E PREDIÇÃO DE MODELO DE REDE NEURAL /////////
async function main() {
    const model = await trainModel(inputXs, outputYs);
    await model.save('/home/felipe/engenharia-ia/modulo-02/01-primeira-rede-neural/ModeloTreinado'); // salva o modelo treinado em disco para uso futuro
    console.log('Modelo treinado e salvo no caminho ModeloTreinado');
}

// 1.0 Cria Função de Treinamento do modelo de rede neural
async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()
    model.add(tf.layers.dense({inputShape: [7], units: 80, activation: 'relu'}))
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}))
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] })

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
                // onEpochEnd: (epoch, log) => console.log( 
                //     `Epoch ${epoch}: loss = ${log.loss}`
                // )
            }
        }
    )
    return model
}

// 2.0 Cria Função de predição do modelo de rede neural
async function predict(model, input_data) {
    // transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(input_data)

    // faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    // console.log(predArray)
    return predArray[0].map((prob, index) => ({ prob, index}))
}

// 3.0 Define as dimensões e as funções para normalizar e realizar o onehot encoding 
// nos tensores de entrada (xs)

// 3.1 Dimensões fixas para normalização e one-hot encoding
function fitNormalizador(valores) {
  const min = Math.min(...valores);
  const max = Math.max(...valores);
  return { min, max };
}
function fitOneHot(valores) {
  return [...new Set(valores)]; // categorias fixas
}

// 3.2 Funções de transformação 
function transformNormalizador(valores, { min, max }) {
  return valores.map(v => (max === min ? 0 : (v - min) / (max - min)));
}
function transformOneHot(valores, categorias) {
  return valores.map(v => categorias.map(c => (c === v ? 1 : 0)));
}

// 4.0 Cria função para tratar os dados de entrada (xs) do modelo de rede neural
function tratarDados(InputDados, encoders = null) {
  if (!encoders) {
    // primeira vez: aprende os encoders a partir dos dados de treino
    encoders = {
      idade: fitNormalizador(InputDados.map(p => p.idade)),
      cor: fitOneHot(InputDados.map(p => p.cor)),
      localizacao: fitOneHot(InputDados.map(p => p.localizacao)),
    };
  }

  const idadesNormalizadas = transformNormalizador(InputDados.map(p => p.idade), encoders.idade);
  const coresOneHot = transformOneHot(InputDados.map(p => p.cor), encoders.cor);
  const locaisOneHot = transformOneHot(InputDados.map(p => p.localizacao), encoders.localizacao);

  const DadosConcatenados = InputDados.map((p, i) => [
    idadesNormalizadas[i],
    ...coresOneHot[i],
    ...locaisOneHot[i]
  ]);

  return { dados: DadosConcatenados, encoders };
}

// 5.0 Define os dados de entrada (xs) do modelo de rede neural
const pessoas = [
  { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
  { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
  { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
];

// 6.0 Normaliza e aplica one-hot encoding para os dados de entrada (xs) do modelo
const tensorPessoasNormalizado = await tratarDados(pessoas);
const inputXs = tf.tensor2d(tensorPessoasNormalizado.dados);

// 7.0 Aprende e aplica o one-hot encoding nos rótulos de saída
const labelsPessoas = ["premium", "medium", "basic"]; // rótulo de cada pessoa, na mesma ordem do array `pessoas`
const categoriasLabels = fitOneHot(labelsPessoas);      // salva as categorias existentes: ["premium", "medium", "basic"]
const tensorLabels = transformOneHot(labelsPessoas, categoriasLabels); // aplica o one-hot encoding nos rótulos de saída
const outputYs = tf.tensor2d(tensorLabels);

// 8. Treina o modelo com os dados de entrada e saída 
const model = await trainModel(inputXs, outputYs);

await model.save('/home/felipe/engenharia-ia/modulo-02/01-primeira-rede-neural/ModeloTreinado'); // salva o modelo treinado em disco para uso futuro
console.log('Modelo treinado e salvo no caminho ModeloTreinado');

fs.writeFileSync('/home/felipe/engenharia-ia/modulo-02/01-primeira-rede-neural/encoders.json', JSON.stringify(tensorPessoasNormalizado.encoders));
fs.writeFileSync('/home/felipe/engenharia-ia/modulo-02/01-primeira-rede-neural/labels.json', JSON.stringify(categoriasLabels));

main();