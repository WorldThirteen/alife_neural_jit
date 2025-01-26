export enum NodeType {
  body = 0,
  nnNode = 1,
  nnLink = 2
}

export enum ActivationType {
  sigmoid = 0,
  tanh = 1,
  linear = 2,
  relu = 3,
}

export enum BodyKind {
  blob = 0, // main body
  movementEffector = 1, // effector
  rotationEffector = 2, // effector
  foodRaySensor = 3, // sensor
  wallRaySensor = 4, // sensor
  botRaySensor = 5, // sensor
  energySensor = 6, // sensor
  slowZoneSensor = 7, // sensor
  drainZoneSensor = 8, // sensor
  killBallSensor = 9, // sensor
}

export interface BodyNode {
  innovationNumber: number;
  id: number;
  type: NodeType.body;
  kind: number;
  value: number;
}

export interface NNNode {
  type: NodeType.nnNode;
  id: number;
  innovationNumber: number;
  activation: ActivationType;
}

export interface NNLink {
  type: NodeType.nnLink;
  innovationNumber: number;
  from: number;
  to: number;
  value: number;
  disabled: boolean;
}

export type GenNode = BodyNode | NNNode | NNLink;

type SensorKind = BodyKind.foodRaySensor | BodyKind.wallRaySensor | BodyKind.botRaySensor | BodyKind.energySensor | BodyKind.slowZoneSensor | BodyKind.drainZoneSensor | BodyKind.killBallSensor;
type EffectorKind = BodyKind.movementEffector | BodyKind.rotationEffector;

const isSensorKind = (kind: BodyKind): kind is SensorKind => kind === BodyKind.foodRaySensor || kind === BodyKind.wallRaySensor || kind === BodyKind.botRaySensor || kind === BodyKind.energySensor || kind === BodyKind.slowZoneSensor || kind === BodyKind.drainZoneSensor || kind === BodyKind.killBallSensor;

export const isSensor = (el: GenNode): el is BodyNode & { kind: SensorKind } => el.type === NodeType.body && isSensorKind(el.kind);
export const isEffector = (el: GenNode): el is BodyNode & { kind: EffectorKind } => el.type === NodeType.body && (el.kind === BodyKind.movementEffector || el.kind === BodyKind.rotationEffector);
const isBlob = (el: GenNode): el is BodyNode & { kind: BodyKind.blob } => el.type === NodeType.body && (el.kind === BodyKind.blob);

export type Genotype = GenNode[];

export function findConnectionForward(links: { innovationNumber: number; from: number, to: number }[], from: number, to: number, visited: Record<number, boolean> = {}) {
  for (const link of links) {
    if (link.from === from) {
      if (link.to === to) {
        return true;
      } else if (!visited[link.innovationNumber]) {
        visited[link.innovationNumber] = true;
        if (findConnectionForward(links, link.to, to, visited)) {
          return true;
        }
      }
    }
  }

  return false;
}
