import * as tf from '@tensorflow/tfjs';

import {maybeRenderDuringTraining, setUpUI} from './ui';

const MODEL_SAVE_PATH_ = 'indexeddb://differentiable-pole-v1';

class Controller{
  constructor(model) {
    if (model instanceof tf.LayersModel) {
      this.net = model;
    } else {
      this.createController();
    }
  }

  createController() {
    this.net = tf.sequential();
    this.net.add(tf.layers.dense({
      units: 24,
      activation: 'elu',
      inputShape: 4
    }));
    this.net.add(tf.layers.dense({
      units: 48,
      activation: 'elu',
    }));
    this.net.add(tf.layers.dense({
      units: 1,
      activation: 'tanh',
    }));
  }

  async train(cartPole, optimizer, maxSteps) {
    const seqLength = 8;
    let steps = 0;
    cartPole.setRandomState();
    for (let j = 0; j < maxSteps; ++j) {
      await maybeRenderDuringTraining(cartPoleSystem);

      const f = () => tf.tidy(() => {
        for(let k = 0; k < seqLength; ++k) {
          cartPole.forward(this.net);
          ++steps;
        }
        return cartPole.lossFn();
      });

      const gradients = tf.tidy(() => tf.variableGrads(f).grads);
      optimizer.applyGradients(gradients);
      tf.dispose(gradients);
      
      if (cartPole.isDone) {
        break;
      }

      await tf.nextFrame();
    }

    optimizer.applyGradients(gradients);
    tf.dispose(gradients);
    return steps;
  }
}

export class SaveableController extends Controller{
  constructor(arg) {
    super(arg);
  }

  async saveModel() {
    return await this.net.save(MODEL_SAVE_PATH_);
  }

  static async loadModel() {
    const modelsInfo = await tf.io.listModels();
    if (MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`Loading existing model...`);
      const model = await tf.loadLayersModel(MODEL_SAVE_PATH_);
      console.log(`Loaded model from ${MODEL_SAVE_PATH_}`);
      return new SaveableController(model);
    } else {
      throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`);
    }
  }

  static async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[MODEL_SAVE_PATH_];
  }

  async removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_);
  }
}

setUpUI();
