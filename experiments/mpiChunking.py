from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from datetime import timedelta as delta
import numpy as np
from glob import glob
import time
import matplotlib.pyplot as plt
import os


def set_cmems_fieldset(cs):
    ddir_head = "/data/oceanparcels/input_data"
    ddir = os.path.join(ddir_head, "CMEMS/GLOBAL_REANALYSIS_PHY_001_030/")
    files = sorted(glob(ddir+"mercatorglorys12v1_gl12_mean_201607*.nc"))
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}

    if cs not in ['auto', False]:
        cs = (1, cs, cs)
    return FieldSet.from_netcdf(files, variables, dimensions, field_chunksize=cs)


if __name__=='__main__':
    odir = "/scratch/ckehl/experiments"
    func_time = []
    chunking_func_time = []
    #chunksize = [50, 100, 200, 400, 800, 1000, 1500, 2000, 2500, 4000, 'auto', False]
    chunksize = [8, 10, 16, 32, 64, 100, 128, 256, 512, 1000, 1024, 2048, 3072, 4096, 'auto', False]
    for cs in chunksize:
        tic_chunking = time.time()
        fieldset = set_cmems_fieldset(cs)
        chunking_func_time.append(time.time()-tic_chunking)
        pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=[0], lat=[0], repeatdt=delta(hours=1))

        tic = time.time()
        pset.execute(AdvectionRK4, dt=delta(hours=1))
        func_time.append(time.time() - tic)

    fig, ax = plt.subplots(1, 1, figsize=(15, 7))

    ax.plot(chunksize[:-2], func_time[:-2], 'o-')
    ax.plot(chunksize[:-2], chunking_func_time[:-2], 'x-')
    ax.plot([0, 4300], [func_time[-2], func_time[-2]], '--', label=chunksize[-2])
    ax.plot([0, 4300], [func_time[-1], func_time[-1]], '--', label=chunksize[-1])
    plt.xlim([0, 4300])
    plt.legend()
    ax.set_xlabel('field_chunksize')
    # ax.set_ylabel('Time spent in pset.execute() [s]')
    ax.set_ylabel('Time spent [s]')
    plt.savefig(os.path.join(odir,"mpiChunking_plot.png"), dpi=300, format='png')
    # plt.show()