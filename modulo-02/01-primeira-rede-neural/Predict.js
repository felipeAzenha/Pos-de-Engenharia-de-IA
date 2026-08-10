import * from '@tensorflow/tfjs-node';


////////////////////////// BLOCO DE CÓDIGO PARA PREDIÇÃO DE NOVA PESSOA //////////////////////////

// Uma vez que finalizamos o treinamento do modelo, podemos fazer a predição de uma nova pessoa 
// que não estava no dataset de treino.

// inputXs.print() // mostra os dados de entrada do modelo
// outputYs.print() // mostra os rótulos de saída do modelo

const pessoa = { nome: 'zé', idade: 28, cor:'verde', localizacao: 'Curitiba'}

const pessoaTensorNormalizado = await tratarDados([pessoa], tensorPessoasNormalizado.encoders);
// idade normalizada, cor azul, cor vermelha, cor verde, São Paulo, Rio, Curitiba

const predictions = await predict(model, pessoaTensorNormalizado.dados)

const results = predictions
    .sort((a,b) => b.prob - a.prob)
    .map(p => `${pessoa.nome} - ${categoriasLabels[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join(`\n`)

console.log(results)
