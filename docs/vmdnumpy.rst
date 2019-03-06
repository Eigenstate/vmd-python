.. _vmdnumpy:
.. currentmodule:: vmd.vmdnumpy
.. highlight:: python

VMD Numpy
=========

The `vmdnumpy` module exposes molecular coordinates in-memory as a numpy array.
This allows extremely efficient analyses to be conducted.

Take care-- the coordinates given are the same as those in memory. Any
inadveretent modification will require you to reload your trajectory to restore
the original values! However, this can be very useful in implementing trajectory
smoothing functions, etc.

.. autosummary::
    :toctree: api/generated/

    atomselect
    positions
    timestep
    velocities

For example, the following code will apply a Savitsky-Golay smoothing function
to the entire trajectory:

.. code-block:: python

    import numpy as np
    from scipy.signal import savgol_filter
    from vmd import molecule, vmdnumpy

    def smooth_trajectory(molid, window=5, polyorder=3):
        smoother = np.empty((molecule.numframes(molid), molecule.numatoms(molid)*3))

        # Copy timestep data to a 1D array to run filter on
        for t in range(molecule.numframes(molid)):
            smoother[t] = vmdnumpy.timestep(molid, t).flatten()
        smoothed = savgol_filter(smoother, window, polyorder,
                                 axis=0, mode="nearest")

        # Copy filtered data back into coordinates
        for t in range(molecule.numframes(molid)):
            conv = smoothed[t].reshape((molecule.numatoms(molid), 3))
            np.copyto(vmdnumpy.timestep(molid, t), conv)

