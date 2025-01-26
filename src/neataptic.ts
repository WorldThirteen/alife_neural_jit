import { architect, Network, Node, methods } from 'neataptic';

import { Genotype, NodeType, BodyNode, NNNode, isSensor, isEffector, findConnectionForward } from './utils'; // Adjust the import path as needed.


/**
 * Map ActivationType to Neataptic activation functions
 */
const activationFunctions = [
  'LOGISTIC',
  'TANH',
  'IDENTITY',
  'RELU',
];

export class NEATaptic {
  // Build the Synaptic network from the genotype.
  buildNetworkFromGenotype(genotype: Genotype): Network {
    const nodesMap: Map<number, any> = new Map();
    const geneMap: Map<number, any> = new Map();
    const connections: { innovationNumber: number; from: number; to: number; weight: number }[] = [];
    const inputKeys: number[] = [];
    const outputKeys: number[] = [];
    
    genotype.forEach((node) => {
      geneMap.set(node.innovationNumber, node);
    });

    // Create nodes and map them by id
    genotype.forEach((node) => {
      if (node.type === NodeType.nnNode) {
        const nnNode = new Node();
        nnNode.bias = 0;
        nnNode.squash = methods.activation[activationFunctions[node.activation]];
        nodesMap.set(node.id, nnNode);
        nnNode.id = node.id;
      } else if (isSensor(node)) {
        const inputNode = new Node(0);
        inputNode.bias = 0;
        inputNode.squash = methods.activation.IDENTITY;
        nodesMap.set(node.id, inputNode);
        inputKeys.push(node.id);
        inputNode.id = node.id;
      } else if (isEffector(node)) {
        const outputNode = new Node();
        outputNode.id = node.id;
        outputNode.bias = 0;
        outputNode.squash = methods.activation.IDENTITY;
        nodesMap.set(node.id, outputNode);
        outputKeys.push(node.id);
      }
    });

    // Process links and create connections
    genotype.forEach((node) => {
      if (node.type === NodeType.nnLink && !node.disabled) {
        connections.push({
          from: node.from,
          to: node.to,
          weight: node.value,
          innovationNumber: node.innovationNumber,
        });
      }
    });

    // Connect nodes
    connections.forEach((conn) => {
      const fromNode = nodesMap.get(conn.from);
      const toNode = nodesMap.get(conn.to);
      if (fromNode && toNode) {
        fromNode.connect(toNode, conn.weight);
      }
    });

    type Instruction = { id: string, node: any };
    type TraversableNode = BodyNode | NNNode;

    const traversedMap: Record<number, Instruction[]> = {};
    const dependencyMap: Record<number, TraversableNode[]> = {};

    function traverse(node: BodyNode | NNNode) {
      if (traversedMap[node.id]) {
        return dependencyMap[node.id];
      }

      const allLinks = connections.filter(link => link.to === node.id);
      const toTraverse: TraversableNode[] = [];

      for (const link of allLinks) {
        const nextNode = geneMap.get(link.from);

        if (nextNode.type === NodeType.nnNode) {
          toTraverse.push(nextNode);
        }
      }

      traversedMap[node.id] = nodesMap.get(node.id);
      dependencyMap[node.id] = toTraverse;

      return toTraverse;
    }

    const toTraverse: TraversableNode[] = outputKeys.map(el => geneMap.get(el));

    for (const node of toTraverse) {
      const newToTraverse = traverse(node);

      // filter recurrent connections
      const filtered = newToTraverse.filter(el => 
        el.id !== node.id &&
        (!findConnectionForward(connections, node.id, el.id) || !traversedMap[el.id])
      );

      toTraverse.push(...filtered);
    }

    const reversed = [...new Set(toTraverse.reverse())];

    // Construct the network
    try {
      const network = architect.Construct([
        ...inputKeys.map(el => nodesMap.get(el)),
        ...reversed.map(el => nodesMap.get(el.id))
      ]);

      return { network, nodesMap, inputKeys, outputKeys };
    } catch (error) {
      if (error instanceof Error && error.message === 'Given nodes have no clear input/output node!') {
        return {
          network: { noTraceActivate() { } },
          nodesMap,
          inputKeys,
          outputKeys,
        }
      }
    }
  }

  load(genotype: Genotype) {
    const { network, nodesMap, inputKeys } = this.buildNetworkFromGenotype(genotype);
    const output: Record<string, number> = {};
    const outputKeys: number[] = [];

    for (const key of nodesMap.keys()) {
      const node = nodesMap.get(key);
      if (node && inputKeys.includes(key) === false) {
        output[key] = 0;
        outputKeys.push(key);
      }
    }

    return {
      exec: function evaluate(inputs: Record<string, number>, output: Record<string, number>) {
        for (const key of inputKeys) {
          // in neataptic, the bias is the best way to persist input value
          nodesMap.get(key).bias = inputs[key];
        }

        network.noTraceActivate([]);

        for (const key of outputKeys) {
          const node = nodesMap.get(key);
          if (node) {
            output[key] = node.activation;
          }
        }
      },
      outputKeys,
      output,
    };
  }
}
