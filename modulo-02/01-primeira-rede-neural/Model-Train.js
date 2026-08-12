import tf from '@tensorflow/tfjs-node';
import fs from 'fs';

///////// BLOCO DE CÓDIGO PARA TREINAMENTO DO MODELO DE REDE NEURAL /////////

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
            epochs: 100,
            batchSize: 3,
            shuffle: true,
            verbose: 0,
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
            }
        }
    )
    return model
}

// 2.0 Define as dimensões e as funções para normalizar e realizar o onehot encoding
function fitNormalizador(valores) {
  const min = Math.min(...valores);
  const max = Math.max(...valores);
  return { min, max };
}
function fitOneHot(valores) {
  return [...new Set(valores)];
}
function transformNormalizador(valores, { min, max }) {
  return valores.map(v => (max === min ? 0 : (v - min) / (max - min)));
}
function transformOneHot(valores, categorias) {
  return valores.map(v => categorias.map(c => (c === v ? 1 : 0)));
}

// 3.0 Cria função para tratar os dados de entrada (xs) do modelo de rede neural
function tratarDados(InputDados, encoders = null) {
  if (!encoders) {
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

// 4.0 Função principal: monta os dados, treina e salva tudo
async function main() {
    const CAMINHO_BASE = '/home/felipe/engenharia-ia/modulo-02/01-primeira-rede-neural';

    // dados de entrada (xs)
    const pessoas = [
      { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
      { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
      { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
    ];

    const tensorPessoasNormalizado = tratarDados(pessoas);
    const inputXs = tf.tensor2d(tensorPessoasNormalizado.dados);

    // rótulos de saída (ys)
    const labelsPessoas = ["premium", "medium", "basic"];
    const categoriasLabels = fitOneHot(labelsPessoas);
    const tensorLabels = transformOneHot(labelsPessoas, categoriasLabels);
    const outputYs = tf.tensor2d(tensorLabels);

    // treina o modelo
    const model = await trainModel(inputXs, outputYs);

    // salva modelo e artefatos de pré-processamento
    await model.save(`file://${CAMINHO_BASE}/ModeloTreinado`);
    fs.writeFileSync(`${CAMINHO_BASE}/encoders.json`, JSON.stringify(tensorPessoasNormalizado.encoders));
    fs.writeFileSync(`${CAMINHO_BASE}/labels.json`, JSON.stringify(categoriasLabels));

    console.log('Modelo treinado e salvo no caminho ModeloTreinado');
}

main();