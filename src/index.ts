import { NeuralJIT } from './neural_jit';
import { NEATaptic } from './neataptic';
import bot from '../test/bot.json';
import { isSensor } from './utils';

function main() {
  const neataptic = new NEATaptic();
  const neuralJIT = new NeuralJIT(true);

  const genotype = bot.genotype;
  
  const neatapticNetwork = neataptic.load(genotype);
  const jitNetwork = neuralJIT.load(genotype);

  const inputKeys = genotype.filter(el => isSensor(el)).map(el => el.id);

  const input = {};

  // full fill input with arbitrary values
  for (const key of inputKeys) {
    input[key] = 0.5;
  }

  neatapticNetwork.exec(input, neatapticNetwork.output);
  jitNetwork.exec(input, jitNetwork.output);
}

main();