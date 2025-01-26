import { Genotype, BodyNode, NNNode, NNLink, BodyKind, NodeType, isSensor, ActivationType, findConnectionForward } from "./utils";

export class NeuralJIT {
  constructor(private OPT: boolean) {}

  getStructure(genotype: Genotype) {  
    const intermediate: {
      inputs: BodyNode[],
      nodes: NNNode[],
      links: NNLink[],
      outputs: BodyNode[],
    } = {
      inputs: [],
      nodes: [],
      links: [],
      outputs: [],
    };
  
    let activeNodes = 0;
    let activeLinks = 0;
  
    for (const node of genotype) {
      if (node.type === NodeType.body) {
        if (isSensor(node)) {
          intermediate.inputs.push(node);
        } else if (node.kind !== BodyKind.blob) {
          intermediate.outputs.push(node);
        }
      }
      if (node.type === NodeType.nnNode) {
        intermediate.nodes.push(node);
      }
      if (node.type === NodeType.nnLink) {
        intermediate.links.push(node);
      }
    }
  
    type Instruction = { prefix: string, value: string };
    type TraversableNode = BodyNode | NNNode;
  
    const traversedMap: Record<number, Instruction[]> = {};
    const dependencyMap: Record<number, TraversableNode[]> = {};
  
    function traverse(node: BodyNode | NNNode) {
      if (traversedMap[node.id]) {
        return dependencyMap[node.id];
      }
  
      const instructions = [{ prefix: 'set', value: String(node.id) }];
  
      if (node.type === NodeType.nnNode && 'activation' in node) {
        activeNodes += 1;
        instructions.unshift({ prefix: ActivationType[node.activation], value: '' });
      }
  
      const allLinks = intermediate.links.filter(link => link.to === node.id);
      const toTraverse: TraversableNode[] = [];
  
      activeLinks += allLinks.length;
  
      if (allLinks.length > 1) {
        instructions.unshift({ prefix: 'sum', value: '' });
        instructions.unshift({ prefix: 'mul', value: '' });
        instructions.unshift({ prefix: 'pick', value: `concat-${node.id}` });
  
        instructions.unshift({ prefix: 'concat', value: '' });
        for (const link of allLinks) {
          instructions.unshift({ prefix: 'arg', value: String(link.value) });
        }
  
        instructions.unshift({ prefix: 'drop', value: 'concat' });
        instructions.unshift({ prefix: 'set', value: `concat-${node.id}` });
        instructions.unshift({ prefix: 'concat', value: '' });
        for (const link of allLinks) {
          instructions.unshift({ prefix: 'pick', value: String(link.from) });
        }
  
        for (const link of allLinks) {
          const intermediateNode = intermediate.nodes.find(el => el.id === link.from);
  
          if (intermediateNode) {
            toTraverse.push(intermediateNode);
          } else {
            const inputNode = intermediate.inputs.find(el => el.id === link.from);
  
            if (inputNode) {
              instructions.unshift({ prefix: 'var', value: String(inputNode.id) });
            }
          }
        }
      } else if (allLinks.length === 1) {
        instructions.unshift({ prefix: 'mul', value: '' });
        instructions.unshift({ prefix: 'arg', value: String(allLinks[0].value) });
        instructions.unshift({ prefix: 'pick', value: String(allLinks[0].from) });
        const intermediateNode = intermediate.nodes.find(el => el.id === allLinks[0].from);
  
        if (intermediateNode) {
          toTraverse.push(intermediateNode);
        } else {
          const inputNode = intermediate.inputs.find(el => el.id === allLinks[0].from);
  
          if (inputNode) {
            instructions.unshift({ prefix: 'var', value: String(inputNode.id) });
          }
        }
      } else {
        // node has no links, could be ignored, e.g value set to 0
        instructions.unshift({ prefix: 'arg', value: '0' });
  
      }
  
      instructions.unshift({ prefix: 'drop', value: 'lead' });
  
      traversedMap[node.id] = instructions;
      dependencyMap[node.id] = toTraverse;
  
      return toTraverse;
    }
  
    const toTraverse: TraversableNode[] = [...intermediate.outputs];
  
    for (const node of toTraverse) {
      const newToTraverse = traverse(node);
  
      // filter recurrent connections
      const filtered = newToTraverse.filter(el =>
        el.id !== node.id &&
        (!findConnectionForward(intermediate.links, node.id, el.id) || !traversedMap[el.id])
      );
  
      toTraverse.push(...filtered);
    }
  
    // return first unique nodes, so if node were added multiple times, the last addition will be used
    // this ensures that execution order is correct
    const reversed = [...new Set(toTraverse.reverse())];
    const result = reversed.reduce<Instruction[]>((all, el) => all.concat(traversedMap[el.id]), []);
  
    return result
  }

  buildModel(structure: { prefix: string; value: string }[]) {
    const definedVars: Record<string, boolean> = {};
    const definedInputs: Record<string, boolean> = {};
    const outputKeys: string[] = [];
    const internalConcat: Record<string, string> = {};
    const output: Record<string, number> = {};
    let idx = 0;
    let resFn: Function;
    let fn: string[] = [];
    let fnStack: string[] = [];

    try {
      for (; idx < structure.length; idx += 1) {
        const { prefix, value } = structure[idx];

        switch (prefix) {
          case 'drop':
            if (fnStack.length) {
              fn.push(fnStack.join(''));
              fnStack = [];
            }
            break;
          case 'var':
            definedInputs[value] = true;
            definedVars[value] = true;
            break;
          case 'pick':
            if (value in definedInputs) {
              fnStack.push(`i['${value}']`);
            } else if (value in internalConcat) {
              fnStack.push(internalConcat[value]);
            } else {
              fnStack.push(`o['${value}']`);
            }
            break;
          case 'arg':
            fnStack.push(`${value}`);
            break;
          case 'mul':
            if (this.OPT && (fnStack[0] === '0' || fnStack[1] === '0')) {
              fnStack = ['0'];
            } else if (Array.isArray(fnStack[0])) {
              if (!Array.isArray([fnStack[1]])) {
                throw new Error(`Unable to find concat for ${fnStack[1]}`);
              }

              fnStack = [
                ...fnStack[0].map((a, i) => a === '0' ? '0' : `${a} * ${fnStack[1][i]}`),
              ];
            } else if (fnStack.length === 2) {
              fnStack = [`${fnStack[0]} * ${fnStack[1]}`];
            } else {
              throw new Error('Invalid stack state for mul');
            }
            break;
          case 'concat':
            // @ts-ignore
            fnStack = [fnStack];
            break;
          case 'sum':
            if (this.OPT) {
              const nonZero = fnStack.filter(el => el !== '0');
              fnStack = nonZero.length ? [`(${nonZero.join(' + ')})`] : ['0'];
            } else {
              fnStack = [`(${fnStack.join(' + ')})`];
            }
            break;
          case 'sigmoid':
            if (this.OPT && fnStack[0] === '0') {
              fnStack = ['0.5'];
            } else {
              fnStack = [`1 / (1 + Math.exp(-(${fnStack[0]})))`];
            }
            break;
          case 'tanh':
            if (this.OPT && fnStack[0] === '0') {
              fnStack = ['0'];
            } else {
              fnStack = [`Math.tanh(${fnStack[0]})`];
            }
            break;
          case 'linear':
            fnStack = [fnStack[0]];
            break;
          case 'relu':
            if (this.OPT && fnStack[0] === '0') {
              fnStack = ['0'];
            } else {
              fnStack = [`Math.max(0, ${fnStack[0]})`];
            }
            break;
          case 'set':
            definedVars[value] = true;
            if (Array.isArray(fnStack[0])) {
              internalConcat[value] = fnStack[0];
              fnStack = [];
            } else {
              fnStack = fnStack[0] === '0' ? [] : [`o['${value}'] = ${fnStack[0]}`];
              outputKeys.push(value);
              output[value] = 0;
            }
            break;
          default:
            throw new Error(`Invalid instruction ${prefix}`);
        }
      }
      fn.push(fnStack.join(''));

      resFn = new Function('i', 'o', fn.join(';\n'));
    } catch (error) {
      const newError: any = new Error(`Unable to build model via instructions, ${error}`);

      newError.vars = definedVars;
      newError.fn = fn;
      newError.fnStack = fnStack;
      newError.currentIndex = idx;
      newError.currentInstruction = structure[idx];
      newError.stack = error.stack;

      throw newError;
    }

    return {
      exec: resFn as (inputs: Record<string, number>, outputs: Record<string, number>) => void,
      outputKeys,
      output,
    };
  }

  load(genotype: Genotype) {
    const structure = this.getStructure(genotype);

    return this.buildModel(structure);
  }
}
