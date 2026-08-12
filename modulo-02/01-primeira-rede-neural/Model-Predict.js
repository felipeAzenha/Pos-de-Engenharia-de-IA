import tf from '@tensorflow/tfjs-node';
import fs from 'fs';

///////// BLOCO DE CÓDIGO PARA PREDIÇÃO DO MODELO DE REDE NEURAL /////////

const CAMINHO_BASE = '/home/felipe/engenharia-ia/modulo-02/01-primeira-rede-neural';

// 1.0 Funções de transform (mesma lógica do treino, mas sem fit)
function transformNormalizador(valores, { min, max }) {
  return valores.map(v => (max === min ? 0 : (v - min) / (max - min)));
}
function transformOneHot(valores, categorias) {
  return valores.map(v => categorias.map(c => (c === v ? 1 : 0)));
}

// 2.0 Aplica os encoders (já treinados) nos novos dados de entrada
function tratarDadosPredicao(InputDados, encoders) {
  const idadesNormalizadas = transformNormalizador(InputDados.map(p => p.idade), encoders.idade);
  const coresOneHot = transformOneHot(InputDados.map(p => p.cor), encoders.cor);
  const locaisOneHot = transformOneHot(InputDados.map(p => p.localizacao), encoders.localizacao);

  return InputDados.map((p, i) => [
    idadesNormalizadas[i],
    ...coresOneHot[i],
    ...locaisOneHot[i]
  ]);
}

// 3.0 Faz a predição (devolve vetor bruto de probabilidades)
async function predict(model, input_data) {
  const tfInput = tf.tensor2d(input_data);
  const pred = model.predict(tfInput);
  const predArray = await pred.array();
  return predArray[0].map((prob, index) => ({ prob, index }));
}

// 4.0 Função principal
async function main() {
    // carrega o modelo treinado
    const model = await tf.loadLayersModel(`file://${CAMINHO_BASE}/ModeloTreinado/model.json`);

    // carrega os encoders e labels salvos no treino
    const encoders = JSON.parse(fs.readFileSync(`${CAMINHO_BASE}/encoders.json`, 'utf-8'));
    const categoriasLabels = JSON.parse(fs.readFileSync(`${CAMINHO_BASE}/labels.json`, 'utf-8'));

    // novas pessoas a serem previstas
    const novasPessoas = [
      { nome: "Felipe", idade: 28, cor: "azul", localizacao: "Rio" },
      { nome: "Mariana", idade: 35, cor: "verde", localizacao: "São Paulo" },
      { nome: "zé", idade: 28, cor: "verde", localizacao: "Curitiba" }
    ];

    novasPessoas.forEach(async (pessoa) => {
      const dadosTratados = tratarDadosPredicao([pessoa], encoders);
      const predictions = await predict(model, dadosTratados);

      const ordenadas = predictions.sort((a, b) => b.prob - a.prob);

      const [maisProvavel, ...outras] = ordenadas;
      const outrasFormatadas = outras
          .map(p => `${categoriasLabels[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
          .join(', ');

      console.log(`${pessoa.nome} → ${categoriasLabels[maisProvavel.index]} (${(maisProvavel.prob * 100).toFixed(2)}%)`);
      console.log(`   Outras probabilidades: ${outrasFormatadas}`);
      console.log('');
    });
}

main();