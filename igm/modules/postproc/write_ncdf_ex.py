#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf
import argparse
from netCDF4 import Dataset

from igm.modules.utils import getmag

def params_write_ncdf_ex(parser):
    parser.add_argument(
        "--vars_to_save_ncdf_ex",
        type=list,
        default=[
            "topg",
            "usurf",
            "thk",
            "smb",
            "velbar_mag",
            "velsurf_mag",
            "uvelsurf",
            "vvelsurf",
            "wvelsurf"
        ],
        help="List of variables to be recorded in the ncdf file",
    )


def init_write_ncdf_ex(params, state):
    state.tcomp_write_ncdf_ex = []

    os.system("echo rm " + os.path.join(params.working_dir, "ex.nc") + " >> clean.sh")

    # give information on variables for output ncdf, TODO: IMPROVE
    state.var_info_ncdf_ex = {}
    state.var_info_ncdf_ex["topg"] = ["Basal Topography", "m"]
    state.var_info_ncdf_ex["usurf"] = ["Surface Topography", "m"]
    state.var_info_ncdf_ex["thk"] = ["Ice Thickness", "m"]
    state.var_info_ncdf_ex["icemask"] = ["Ice mask", "NO UNIT"]
    state.var_info_ncdf_ex["smb"] = ["Surface Mass Balance", "m/y"]
    state.var_info_ncdf_ex["ubar"] = ["x depth-average velocity of ice", "m/y"]
    state.var_info_ncdf_ex["vbar"] = ["y depth-average velocity of ice", "m/y"]
    state.var_info_ncdf_ex["velbar_mag"] = ["Depth-average velocity magnitude of ice", "m/y"]
    state.var_info_ncdf_ex["uvelsurf"] = ["x surface velocity of ice", "m/y"]
    state.var_info_ncdf_ex["vvelsurf"] = ["y surface velocity of ice", "m/y"]
    state.var_info_ncdf_ex["wvelsurf"] = ["z surface velocity of ice", "m/y"]
    state.var_info_ncdf_ex["velsurf_mag"] = ["Surface velocity magnitude of ice", "m/y"]
    state.var_info_ncdf_ex["uvelbase"] = ["x basal velocity of ice", "m/y"]
    state.var_info_ncdf_ex["vvelbase"] = ["y basal velocity of ice", "m/y"]
    state.var_info_ncdf_ex["wvelbase"] = ["z basal velocity of ice", "m/y"]
    state.var_info_ncdf_ex["velbase_mag"] = ["Basal velocity magnitude of ice", "m/y"]
    state.var_info_ncdf_ex["divflux"] = ["Divergence of the ice flux", "m/y"]
    state.var_info_ncdf_ex["strflowctrl"] = ["arrhenius+1.0*slidingco", "MPa$^{-3}$ a$^{-1}$"]
    state.var_info_ncdf_ex["dtopgdt"] = ["Erosion rate", "m/y"]
    state.var_info_ncdf_ex["arrhenius"] = ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"]
    state.var_info_ncdf_ex["slidingco"] = ["Sliding Coefficient", "km MPa$^{-3}$ a$^{-1}$"]
    state.var_info_ncdf_ex["meantemp"] = ["Mean anual surface temperatures", "°C"]
    state.var_info_ncdf_ex["meanprec"] = ["Mean anual precipitation", "m/y"]
    state.var_info_ncdf_ex["velsurfobs_mag"] = ["Obs. surf. speed of ice", "m/y"]
    state.var_info_ncdf_ex["weight_particles"] = ["weight_particles", "no"]


def update_write_ncdf_ex(params, state):
    if state.saveresult:
        state.tcomp_write_ncdf_ex.append(time.time())

        if "velbar_mag" in params.vars_to_save_ncdf_ex:
            state.velbar_mag = getmag(state.ubar, state.vbar)

        if "velsurf_mag" in params.vars_to_save_ncdf_ex:
            state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

        if "velbase_mag" in params.vars_to_save_ncdf_ex:
            state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

        if "meanprec" in params.vars_to_save_ncdf_ex:
            state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)

        if "meantemp" in params.vars_to_save_ncdf_ex:
            state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

        if not hasattr(state, "already_called_update_write_ncdf_ex"):
            state.already_called_update_write_ncdf_ex = True

            if hasattr(state,'logger'):
                state.logger.info("Initialize NCDF ex output Files")

            nc = Dataset(
                os.path.join(params.working_dir, "ex.nc"),
                "w",
                format="NETCDF4",
            )

            nc.createDimension("time", None)
            E = nc.createVariable("time", np.dtype("float32").char, ("time",))
            E.units = "yr"
            E.long_name = "time"
            E.axis = "T"
            E[0] = state.t.numpy()

            nc.createDimension("y", len(state.y))
            E = nc.createVariable("y", np.dtype("float32").char, ("y",))
            E.units = "m"
            E.long_name = "y"
            E.axis = "Y"
            E[:] = state.y.numpy()

            nc.createDimension("x", len(state.x))
            E = nc.createVariable("x", np.dtype("float32").char, ("x",))
            E.units = "m"
            E.long_name = "x"
            E.axis = "X"
            E[:] = state.x.numpy()

            for var in params.vars_to_save_ncdf_ex:
                if hasattr(state, var):
                    E = nc.createVariable(var, np.dtype("float32").char, ("time", "y", "x"))
                    E.long_name = state.var_info_ncdf_ex[var][0]
                    E.units = state.var_info_ncdf_ex[var][1]
                    E[0, :, :] = vars(state)[var].numpy()

            nc.close()

        else:
            if hasattr(state,'logger'):
                state.logger.info("Write NCDF ex file at time : " + str(state.t.numpy()))

            nc = Dataset(
                os.path.join(params.working_dir, "ex.nc"),
                "a",
                format="NETCDF4",
            )

            d = nc.variables["time"][:].shape[0]
            nc.variables["time"][d] = state.t.numpy()

            for var in params.vars_to_save_ncdf_ex:
                if hasattr(state, var):
                    nc.variables[var][d, :, :] = vars(state)[var].numpy()

            nc.close()

        state.tcomp_write_ncdf_ex[-1] -= time.time()
        state.tcomp_write_ncdf_ex[-1] *= -1
 
 
def final_write_ncdf_ex(params, state):
    pass