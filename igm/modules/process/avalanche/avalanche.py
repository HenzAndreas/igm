#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from igm.modules.utils import compute_gradient_tf


def params(parser):
    parser.add_argument(
        "--avalanche_update_freq",
        type=float,
        default=1,
        help="Update avalanche each X years (1)",
    )

    parser.add_argument(
        "--avalanche_angleOfRepose",
        type=float,
        default=30,
        help="Angle of repose (30°)",
    )


def initialize(params, state):
    state.tcomp_avalanche = []
    state.tlast_avalanche = tf.Variable(params.time_start, dtype=tf.float32)
    
    state.new_avalanche = False


    


def update(params, state):
    if (state.t - state.tlast_avalanche) >= params.avalanche_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("Update AVALANCHE at time : " + str(state.t.numpy()))

        state.tcomp_avalanche.append(time.time())


        
        if state.new_avalanche:
            
            angleOfRepose = tf.Variable(params.avalanche_angleOfRepose/180*np.pi, dtype=tf.float32)
            
            # calculate the gradient of the surface
            dzdx, dzdy = compute_gradient_tf(state.usurf, state.dx, state.dx)
        
            # save arrows of direction down
        
            # calculate the gradient of the surface
            grad = tf.math.sqrt(dzdx**2 + dzdy**2)
            # where to move ice
            grad = tf.where(grad < angleOfRepose, 0, grad)
            grad = tf.where(state.thk < 0.1, 0, grad)
        

        
            print("what now?")
        
        if not state.new_avalanche:


            H = state.thk
            Zb = state.topg
            Zi = Zb + H
            # angle of repose, and the maximum angle of the slope that is allowed
            dHRepose = state.dx * tf.math.tan(
                params.avalanche_angleOfRepose * np.pi / 180.0
            )
            Ho = tf.maximum(H, 0)

            count = 0

            while True:
                count += 1

                dZidx_down = tf.concat([tf.zeros_like(Zi[:, :1]), tf.maximum(Zi[:, 1:] - Zi[:, :-1], 0)], axis=1)
                dZidx_up = tf.concat([tf.maximum(Zi[:, :-1] - Zi[:, 1:], 0), tf.zeros_like(Zi[:, :1])], axis=1)
                dZidx = tf.maximum(dZidx_down, dZidx_up)

                dZidy_left = tf.concat([tf.zeros_like(Zi[:1, :]), tf.maximum(Zi[1:, :] - Zi[:-1, :], 0)], axis=0)
                dZidy_right = tf.concat([tf.maximum(Zi[:-1, :] - Zi[1:, :], 0), tf.zeros_like(Zi[:1, :])], axis=0)
                dZidy = tf.maximum(dZidy_right, dZidy_left)

                # the gradient of the surface (slope basically)
                grad = tf.math.sqrt(dZidx**2 + dZidy**2)
                
                
                gradT = dZidy_left + dZidy_right + dZidx_down + dZidx_up
                gradT = tf.where(gradT == 0, 1, gradT)
                
                # do not calculate outside the glacier:
                grad = tf.where(Ho < 0.1, 0, grad)

                mxGrad = tf.reduce_max(grad)
                if mxGrad <= 1.1 * dHRepose:
                    break

                # the amount of ice that should be redistributed
                delH = tf.maximum(0, (grad - dHRepose) / 3.0)

                Htmp = Ho
                Ho = tf.maximum(0, Htmp - delH)
                delH = Htmp - Ho

                # # save delH
                # fig, ax = plt.subplots(figsize=(5,5))
                # plt.imshow(delH,origin='lower'); plt.colorbar()
                # plt.savefig("figures/delH_" + str(count) + ".png")
                # plt.close()

                # calculate the amount of ice that should be moved in each direction
                delHup = tf.pad(
                    delH[:, :-1] * dZidx_up[:, :-1] / gradT[:, :-1],
                    [[0, 0], [1, 0]],
                    "CONSTANT",
                )
                delHdn = tf.pad(
                    delH[:, 1:] * dZidx_down[:, 1:] / gradT[:, 1:],
                    [[0, 0], [0, 1]],
                    "CONSTANT",
                )
                delHrt = tf.pad(
                    delH[:-1, :] * dZidy_right[:-1, :] / gradT[:-1, :],
                    [[1, 0], [0, 0]],
                    "CONSTANT",
                )
                
                delHlt = tf.pad(
                    delH[1:, :] * dZidy_left[1:, :] / gradT[1:, :],
                    [[0, 1], [0, 0]],
                    "CONSTANT",
                )

                Ho = tf.maximum(0, Ho + delHdn + delHup + delHlt + delHrt)

                Zi = Zb + Ho

        # print(count)

        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow( Ho + tf.where(H<0,H,0) - state.thk ,origin='lower'); plt.colorbar()

        state.thk = Ho + tf.where(H < 0, H, 0)

        state.usurf = state.topg + state.thk

        state.tlast_avalanche.assign(state.t)

        state.tcomp_avalanche[-1] -= time.time()
        state.tcomp_avalanche[-1] *= -1


def finalize(params, state):
    pass
