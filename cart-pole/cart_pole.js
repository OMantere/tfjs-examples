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

/**
 * Implementation based on: http://incompleteideas.net/book/code/pole.c
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Cart-pole system simulator.
 *
 * In the control-theory sense, there are four state variables in this system:
 *
 *   - x: The 1D location of the cart.
 *   - xDot: The velocity of the cart.
 *   - theta: The angle of the pole (in radians). A value of 0 corresponds to
 *     a vertical position.
 *   - thetaDot: The angular velocity of the pole.
 *
 * The system is controlled through a single action:
 *
 *   - leftward or rightward force.
 */
export class CartPole {
  /**
   * Constructor of CartPole.
   */
  constructor() {
    // Constants that characterize the system.
    this.gravity = tf.scalar(9.8);
    this.massCart = tf.scalar(1.0);
    this.massPole = tf.scalar(0.1);
    this.totalMass = this.massCart.add(this.massPole);
    this.cartWidth = tf.scalar(0.2);
    this.cartHeight = tf.scalar(0.1);
    this.length = tf.scalar(0.5);
    this.poleMoment = this.massPole.mul(this.length);
    this.forceMag = tf.scalar(10.0);
    this.tau = tf.scalar(0.02);  // Seconds between state updates.
    this.frac = tf.scalar(4.0).div(3.0);

    // Threshold values, beyond which a simulation will be marked as failed.
    this.xThreshold = 2.4;
    this.thetaThreshold = 12 / 360 * 2 * Math.PI;
    this.isDone = false;

    this.setRandomState();
  }

  setRandomState() {
    const x = tf.scalar(Math.random() - 0.5);
    const xDot = tf.scalar((Math.random() - 0.5) * 1);
    const theta = tf.scalar((Math.random() - 0.5) * 2 * (6 / 360 * 2 * Math.PI));
    const thetaDot =  tf.scalar((Math.random() - 0.5) * 0.5);
  }

  forward(net) {
    const force = net.predict(inputs) * this.forceMag;

    const cosTheta = tf.cos(this.theta);
    const sinTheta = tf.sin(this.theta);

    const term = force.add(this.poleMoment.mul(this.thetaDot).mul(this.thetaDot).mul(sinTheta).div(this.totalMass));
    const bottomTerm = this.length.mul(this.frac.sub(this.massPole.mul(tf.pow(cosTheta, 2)).div(this.totalMass)));
    const thetaAcc = this.gravity.mul(sinTheta).sub(cosTheta.mul(term)).div(bottomTerm);
    const xAcc = term.sub(this.poleMoment.mul(thetaAcc).mul(cosTheta).div(this.totalMass));

    this.x = this.tau.mul(this.xDot).add(this.x);
    this.xDot = this.tau.mul(xAcc).add(this.xDot);
    this.theta = this.tau.mul(this.thetaDot).add(this.theta);
    this.thetaDot = this.tau.mul(thetaAcc).add(this.thetaDot);

    this.isDone = this._done();
    return this.x, this.theta;
  }

  lossFn() {
    const offAxis = this.x.sub(0.5);
    const xLower = tf.scalar(this.xThreshold).add(offAxis);
    const xUpper = tf.scalar(this.xThreshold).sub(offAxis);
    const xMax = tf.max(0.0, tf.min(xLower, xUpper));

    const thetaLower = tf.scalar(this.thetaThreshold).add(this.theta);
    const thetaUpper = tf.scalar(this.thetaThreshold).sub(this.theta);
    const thetaMax = tf.max(0.0, tf.min(thetaLower, thetaUpper));

    const lossX = tf.pow(xMax.sub(this.xThreshold), 2).mul(0.01);
    const lossTheta = tf.pow(thetaMax.sub(this.thetaThreshold), 2);
    return lossX.add(lossTheta);
  }

  _done() {
    const x = this.x.dataSync();
    const theta = this.theta.dataSync();
    return x < -this.xThreshold || x > this.xThreshold ||
        theta < -this.thetaThreshold || theta > this.thetaThreshold;
  }
}
