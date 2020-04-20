from parcels import FieldSet, Field, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable
from datetime import timedelta as delta
from glob import glob
import numpy as np
import xarray as xr
import warnings
import math
import os
from  argparse import ArgumentParser
import dask

if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection around an idealised peninsula")
    parser.add_argument("-s", "--stokes", dest="stokes", action='store_true', default=False, help="use Stokes' field data")
    args = parser.parse_args()
    warnings.simplefilter("ignore", category=xr.SerializationWarning)

    wstokes = args.stokes

    ddir_head = "/data/oceanparcels/input_data"
    ddir = os.path.join(ddir_head,"NEMO-MEDUSA/ORCA0083-N006/")
    odir = "/scratch/ckehl/experiments/galapagos"
    ufiles = sorted(glob(ddir+'means/ORCA0083-N06_20[00-10]*d05U.nc'))
    vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
    meshfile = glob(ddir+'domain/coordinates.nc')

    dask.config.set({'array.chunk-size': '16MiB'})
    nemofiles = {'U': {'lon': meshfile, 'lat': meshfile, 'data': ufiles},
                 'V': {'lon': meshfile, 'lat': meshfile, 'data': vfiles}}
    nemovariables = {'U': 'uo', 'V': 'vo'}
    nemodimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
    # ==== Because the stokes data is a different grid, we actually need to define the chunking ==== #
    # fieldset_nemo = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions, field_chunksize='auto')
    chs = {'time_counter': 1, 'depthu': 75, 'depthv': 75, 'depthw': 75, 'deptht': 75, 'y': 100, 'x': 100}
    fieldset_nemo = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions, field_chunksize=chs)  # , time_periodic=delta(days=366)

    if wstokes:
        # stokesfiles = sorted(glob(ddir_head+"/WaveWatch3data/CFSR/WW3-*_uss.nc"))
        stokesfiles = sorted(glob(ddir_head+"/WaveWatch3data/CFSR/WW3 - GLOB - 30M_20[00-10][01-12]_uss.nc"))
        stokesdimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
        stokesvariables = {'U': 'uuss', 'V': 'vuss'}
        stokeschs = {'time': 1, 'latitude': 32, 'longitude': 16}
        # fieldset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions)
        fieldset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions, field_chunksize=stokeschs, time_periodic=delta(days=366))
        fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

        fieldset = FieldSet(U=fieldset_nemo.U+fieldset_stokes.U, V=fieldset_nemo.V+fieldset_stokes.V)
        fU = fieldset.U[0]
        fname = os.path.join(odir,"galapagosparticles_bwd_wstokes_v2.nc")
    else:
        fieldset = fieldset_nemo
        fU = fieldset.U
        fname = os.path.join(odir,"galapagosparticles_bwd_v2.nc")

    # fieldset.computeTimeChunk(fU.grid.time[-1], -1)

    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    startlon, startlat = np.meshgrid(np.arange(galapagos_extent[0], galapagos_extent[1], 0.2),
                                     np.arange(galapagos_extent[2], galapagos_extent[3], 0.2))

    def Age(fieldset, particle, time):
        particle.age = particle.age + math.fabs(particle.dt)
        if particle.age > 10*365*86400:
            particle.delete()

    class GalapagosParticle(JITParticle):
        age = Variable('age', initial = 0.)

    def DeleteParticle(particle, fieldset, time):
        particle.delete()

    def WrapParticle(particle, fieldset, time):
        if particle.lon < -530:
            particle.lon += 360

    pset = ParticleSet(fieldset=fieldset, pclass=GalapagosParticle, lon=startlon, lat=startlat,
                       time=fU.grid.time[-1], repeatdt=delta(days=10))
    outfile = pset.ParticleFile(name=fname, outputdt=delta(days=1))

    pset.execute(AdvectionRK4+pset.Kernel(Age)+WrapParticle, dt=delta(hours=-1), output_file=outfile,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

    outfile.export()
    outfile.close()
