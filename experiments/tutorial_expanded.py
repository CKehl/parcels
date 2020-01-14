from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, ErrorCode, plotTrajectoriesFile
import numpy as np
import math
from datetime import timedelta
from operator import attrgetter
import os
try:
    from mpi4py import MPI
except:
    MPI = None

def DeleteParticle(particle, fieldset, time):
    particle.delete()

if __name__ == '__main__':
    if MPI:
        print("MPI rank: {}".format(MPI.COMM_WORLD.Get_rank()))

    ddir ="/scratch/ckehl/parcels_examples"
    odir = "/scratch/ckehl/experiments"
    #fieldset = FieldSet.from_parcels(os.path.join(ddir,"MovingEddies_data/moving_eddies"))
    #pset = ParticleSet.from_list(fieldset=fieldset,  # the fields on which the particles are advected
    #                             pclass=JITParticle,  # the type of particles (JITParticle or ScipyParticle)
    #                             lon=[3.3e5, 3.3e5],  # a vector of release longitudes
    #                             lat=[1e5, 2.8e5])  # a vector of release latitudes

    filenames = {'U': os.path.join(ddir,"GlobCurrent_example_data/20*.nc"), 'V': os.path.join(ddir,"GlobCurrent_example_data/20*.nc")}
    variables = {'U': 'eastward_eulerian_current_velocity',
                 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat',
                  'lon': 'lon',
                  'time': 'time'}
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    pset = ParticleSet.from_line(fieldset=fieldset, pclass=JITParticle,
                                 size=256,  # releasing 5 particles
                                 start=(28, -33),  # releasing on a line: the start longitude and latitude
                                 finish=(30, -33))  # releasing on a line: the end longitude and latitude
    output_file = pset.ParticleFile(name=os.path.join(odir,"GlobCurrentParticles_expanded.nc"), outputdt=timedelta(hours=6))
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=15),
                 dt=timedelta(minutes=2),
                 output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
#    if MPI:
#        if MPI.COMM_WORLD.Get_rank()>0:
#            output_file.close()
#        else:
#            output_file.export()
#            output_file.close()
#    else:
#        output_file.export()
#        output_file.close()
    output_file.export()
    output_file.close()
    exit(0)
