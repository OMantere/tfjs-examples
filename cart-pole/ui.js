/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import embed from 'vega-embed';

import {CartPole} from './cart_pole';
import {SaveableController} from './index';
import {mean, sum} from './utils';

const appStatus = document.getElementById('app-status');
const storedModelStatusInput = document.getElementById('stored-model-status');
const createModelButton = document.getElementById('create-model');
const deleteStoredModelButton = document.getElementById('delete-stored-model');
const cartPoleCanvas = document.getElementById('cart-pole-canvas');

const numIterationsInput = document.getElementById('num-iterations');
const maxStepsInput = document.getElementById('max-steps');
const learningRateInput = document.getElementById('learning-rate');
const renderDuringTrainingCheckbox =
    document.getElementById('render-during-training');

const trainButton = document.getElementById('train');
const testButton = document.getElementById('test');
const trainStatus = document.getElementById('train-status');
const trainSpeed = document.getElementById('train-speed');
const trainProgress = document.getElementById('train-progress');

const stepsContainer = document.getElementById('steps-container');

// Module-global instance of network
let net;
let stopRequested = false;

/**
 * Display a message to the info div.
 *
 * @param {string} message The message to be displayed.
 */
function logStatus(message) {
  appStatus.textContent = message;
}

// Objects and functions to support display of cart pole status during training.
let renderDuringTraining = true;
export async function maybeRenderDuringTraining(cartPole) {
  if (renderDuringTraining) {
    renderCartPole(cartPole, cartPoleCanvas);
    await tf.nextFrame();  // Unblock UI thread.
  }
}

let stepValues = [];
function plotSteps() {
  tfvis.render.linechart(stepsContainer, {values: stepValues}, {
    xLabel: 'Training Iteration',
    yLabel: 'Steps',
    width: 400,
    height: 300,
  });
}

function disableModelControls() {
  trainButton.textContent = 'Stop';
  deleteStoredModelButton.disabled = true;
}

function enableModelControls() {
  trainButton.textContent = 'Train';
  deleteStoredModelButton.disabled = false;
}

/**
 * Render the current state of the system on an HTML canvas.
 *
 * @param {CartPole} cartPole The instance of cart-pole system to render.
 * @param {HTMLCanvasElement} canvas The instance of HTMLCanvasElement on which
 *   the rendering will happen.
 */
function renderCartPole(cartPole, canvas) {
  if (!canvas.style.display) {
    canvas.style.display = 'block';
  }
  const X_MIN = -cartPole.xThreshold;
  const X_MAX = cartPole.xThreshold;
  const xRange = X_MAX - X_MIN;
  const scale = canvas.width / xRange;

  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  const halfW = canvas.width / 2;

  // Draw the cart.
  const railY = canvas.height * 0.8;
  const cartW = cartPole.cartWidth * scale;
  const cartH = cartPole.cartHeight * scale;

  const cartX = cartPole.x.dataSync() * scale + halfW;

  context.beginPath();
  context.strokeStyle = '#000000';
  context.lineWidth = 2;
  context.rect(cartX - cartW / 2, railY - cartH / 2, cartW, cartH);
  context.stroke();

  // Draw the wheels under the cart.
  const wheelRadius = cartH / 4;
  for (const offsetX of [-1, 1]) {
    context.beginPath();
    context.lineWidth = 2;
    context.arc(
        cartX - cartW / 4 * offsetX, railY + cartH / 2 + wheelRadius,
        wheelRadius, 0, 2 * Math.PI);
    context.stroke();
  }

  // Draw the pole.
  const angle = cartPole.theta.dataSync() + Math.PI / 2;
  const poleTopX =
      halfW + scale * (cartPole.x + Math.cos(angle) * cartPole.length);
  const poleTopY = railY -
      scale * (cartPole.cartHeight / 2 + Math.sin(angle) * cartPole.length);
  context.beginPath();
  context.strokeStyle = '#ffa500';
  context.lineWidth = 6;
  context.moveTo(cartX, railY - cartH / 2);
  context.lineTo(poleTopX, poleTopY);
  context.stroke();

  // Draw the ground.
  const groundY = railY + cartH / 2 + wheelRadius * 2;
  context.beginPath();
  context.strokeStyle = '#000000';
  context.lineWidth = 1;
  context.moveTo(0, groundY);
  context.lineTo(canvas.width, groundY);
  context.stroke();

  const nDivisions = 40;
  for (let i = 0; i < nDivisions; ++i) {
    const x0 = canvas.width / nDivisions * i;
    const x1 = x0 + canvas.width / nDivisions / 2;
    const y0 = groundY + canvas.width / nDivisions / 2;
    const y1 = groundY;
    context.beginPath();
    context.moveTo(x0, y0);
    context.lineTo(x1, y1);
    context.stroke();
  }

  // Draw the left and right limits.
  const limitTopY = groundY - canvas.height / 2;
  context.beginPath();
  context.strokeStyle = '#ff0000';
  context.lineWidth = 2;
  context.moveTo(1, groundY);
  context.lineTo(1, limitTopY);
  context.stroke();
  context.beginPath();
  context.moveTo(canvas.width - 1, groundY);
  context.lineTo(canvas.width - 1, limitTopY);
  context.stroke();
}

async function updateUIControlState() {
  const modelInfo = await SaveableController.checkStoredModelStatus();
  if (modelInfo == null) {
    storedModelStatusInput.value = 'No stored model.';
    deleteStoredModelButton.disabled = true;

  } else {
    storedModelStatusInput.value = `Saved@${modelInfo.dateSaved.toISOString()}`;
    deleteStoredModelButton.disabled = false;
    createModelButton.disabled = true;
  }
  createModelButton.disabled = net != null;
  trainButton.disabled = net == null;
  renderDuringTrainingCheckbox.checked = renderDuringTraining;
}

export async function setUpUI() {
  const cartPole = new CartPole(true);

  if (await SaveableController.checkStoredModelStatus() != null) {
    net = await SaveableController.loadModel();
    logStatus('Loaded policy network from IndexedDB.');
  }
  await updateUIControlState();

  renderDuringTrainingCheckbox.addEventListener('change', () => {
    renderDuringTraining = renderDuringTrainingCheckbox.checked;
  });

  createModelButton.addEventListener('click', async () => {
    try {
      net = new SaveableController();
      console.log('DONE constructing new instance of SaveableController');
      await updateUIControlState();
    } catch (err) {
      logStatus(`ERROR: ${err.message}`);
    }
  });

  deleteStoredModelButton.addEventListener('click', async () => {
    if (confirm(`Are you sure you want to delete the locally-stored model?`)) {
      await net.removeModel();
      net = null;
      await updateUIControlState();
    }
  });

  trainButton.addEventListener('click', async () => {
    if (trainButton.textContent === 'Stop') {
      stopRequested = true;
    } else {
      disableModelControls();

      const trainIterations = Number.parseInt(numIterationsInput.value);
      if (!(trainIterations > 0)) {
        throw new Error(`Invalid number of iterations: ${trainIterations}`);
      }
      const maxSteps = Number.parseInt(maxStepsInput.value);
      if (!(maxSteps > 1)) {
        throw new Error(`Invalid max. steps per game: ${maxSteps}`);
      }
      const learningRate = Number.parseFloat(learningRateInput.value);

      logStatus(
          'Training network... Please wait. ' +
          'Network is saved to IndexedDB at the end of each iteration.');
      const optimizer = tf.train.adam(learningRate);

      stepValues = [];
      onIterationEnd(0, trainIterations);
      let t0 = new Date().getTime();
      stopRequested = false;
      for (let i = 0; i < trainIterations; ++i) {
        const steps = await net.train(cartPole, optimizer, maxSteps);
        const t1 = new Date().getTime();
        const stepsPerSecond = sum(gameSteps) / ((t1 - t0) / 1e3);
        t0 = t1;
        trainSpeed.textContent = `${stepsPerSecond.toFixed(1)} steps/s`
        stepValues.push({x: i + 1, y: mean(steps)});
        console.log(`# of tensors: ${tf.memory().numTensors}`);
        plotSteps();
        onIterationEnd(i + 1, trainIterations);
        await tf.nextFrame();  // Unblock UI thread.
        await net.saveModel();
        await updateUIControlState();

        if (stopRequested) {
          logStatus('Training stopped by user.');
          break;
        }
      }
      if (!stopRequested) {
        logStatus('Training completed.');
      }
      enableModelControls();
    }
  });
}
