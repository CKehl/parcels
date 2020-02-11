"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4
from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, RectilinearZGrid, ErrorCode
from parcels.field import Field, VectorField, NestedField, SummedField
from datetime import timedelta as delta
import math
from argparse import ArgumentParser
import datetime
import numpy as np
import xarray as xr
import pytest
import cftime
import psutil
import os
import time as ostime
import matplotlib.pyplot as plt
from parcels.tools import perlin2d
#import perlin3d

import sys
try:
    from mpi4py import MPI
except:
    MPI = None


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0
#Nparticle = 4096
Nparticle = 8192
#Nparticle = 65536

noctaves=8
perlinres=(64,16)
shapescale=(4,4)
perlin_persistence=0.8
a = 10000 * 1e3
b = 10000 * 1e3
scalefac = 0.05  # to scale for physically meaningful velocities

class PerformanceLog():
    samples = []
    times_steps = []
    memory_steps = []
    fds_steps = []
    _iter = 0

    def advance(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            process = psutil.Process(os.getpid())
            mem_B_used = process.memory_info().rss
            fds_open = len(process.open_files())
            mem_B_used_total = mpi_comm.reduce(mem_B_used, op=MPI.SUM, root=0)
            fds_open_total = mpi_comm.reduce(fds_open, op=MPI.SUM, root=0)
            if mpi_rank == 0:
                self.times_steps.append(ostime.time())
                self.memory_steps.append(mem_B_used_total)
                self.fds_steps.append(fds_open_total)
                self.samples.append(self._iter)
                self._iter+=1
        else:
            process = psutil.Process(os.getpid())
            self.times_steps.append(ostime.time())
            self.memory_steps.append(process.memory_info().rss)
            self.fds_steps.append(len(process.open_files()))
            self.samples.append(self._iter)
            self._iter+=1

def plot(x, times, memory_used, nfiledescriptors, imageFilePath):
    plot_t = []
    for i in range(len(times)):
        if i==0:
            plot_t.append(times[i]-global_t_0)
        else:
            plot_t.append(times[i]-times[i-1])
    #mem_scaler = (1*10)/(1024*1024*1024)
    mem_scaler = 1 / (1024 * 1024 * 1024)
    plot_mem = []
    for i in range(len(memory_used)):
        #if i==0:
        #    plot_mem.append((memory_used[i]-global_m_0)*mem_scaler)
        #else:
        #    plot_mem.append((memory_used[i] - memory_used[i-1]) * mem_scaler)
        plot_mem.append(memory_used[i] * mem_scaler)

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.plot(x, plot_t, 'o-', label="time_spent [s]")
    ax.plot(x, plot_mem, 'x-', label="memory_used [100 MB]")
    #ax.plot(x, nfiledescriptors, '.-', label="open_files [#]")
    #plt.xlim([0, 256])
    #plt.ylim([0, 50])
    plt.legend()
    ax.set_xlabel('iteration')
    # ax.set_ylabel('Time spent in pset.execute() [s]')
    # ax.set_ylabel('Time spent [s]')
    plt.savefig(os.path.join(odir, imageFilePath), dpi=300, format='png')

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def RenewParticle(particle, fieldset, time):
    particle.lat = np.random.rand() * 5e-1
    particle.lon = np.random.rand() * 5e-1

def perlin_fieldset_from_numpy(periodic_wrap=False):
    """Simulate a current from structured random noise (i.e. Perlin noise).
    we use the external package 'perlin-numpy' as field generator, see:
    https://github.com/pvigier/perlin-numpy

    Perlin noise was introduced in the literature here:
    Perlin, Ken (July 1985). "An Image Synthesizer". SIGGRAPH Comput. Graph. 19 (97–8930), p. 287–296.
    doi:10.1145/325165.325247, https://dl.acm.org/doi/10.1145/325334.325247
    """
    img_shape = (noctaves*perlinres[0]*shapescale[0], noctaves*perlinres[1]*shapescale[1])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, img_shape[0], dtype=np.float32)
    lat = np.linspace(0, b, img_shape[1], dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = perlin2d.generate_fractal_noise_2d((lon.size, lat.size), perlinres, noctaves, perlin_persistence) * scalefac
    V = perlin2d.generate_fractal_noise_2d((lon.size, lat.size), perlinres, noctaves, perlin_persistence) * scalefac
    P = perlin2d.generate_fractal_noise_2d((lon.size, lat.size), perlinres, noctaves, perlin_persistence) * scalefac

    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lon': lon, 'lat': lat}
    if periodic_wrap:
        return FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, time_periodic=delta(days=1))
    else:
        return FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, allow_time_extrapolation=True)


def perlin_fieldset_from_xarray(periodic_wrap=False):
    """Simulate a current from structured random noise (i.e. Perlin noise).
    we use the external package 'perlin-numpy' as field generator, see:
    https://github.com/pvigier/perlin-numpy

    Perlin noise was introduced in the literature here:
    Perlin, Ken (July 1985). "An Image Synthesizer". SIGGRAPH Comput. Graph. 19 (97–8930), p. 287–296.
    doi:10.1145/325165.325247, https://dl.acm.org/doi/10.1145/325334.325247
    """
    img_shape = (noctaves*perlinres[0]*shapescale[0], noctaves*perlinres[1]*shapescale[1])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, img_shape[0], dtype=np.float32)
    lat = np.linspace(0, b, img_shape[1], dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = perlin2d.generate_fractal_noise_2d((lon.size, lat.size), perlinres, noctaves, perlin_persistence) * scalefac
    V = perlin2d.generate_fractal_noise_2d((lon.size, lat.size), perlinres, noctaves, perlin_persistence) * scalefac
    P = perlin2d.generate_fractal_noise_2d((lon.size, lat.size), perlinres, noctaves, perlin_persistence) * scalefac

    dimensions = {'lon': lon, 'lat': lat}
    dims = ('lat', 'lon')
    data = {'Uxr': xr.DataArray(U, coords=dimensions, dims=dims),
            'Vxr': xr.DataArray(V, coords=dimensions, dims=dims),
            'Pxr': xr.DataArray(P, coords=dimensions, dims=dims)}
    ds = xr.Dataset(data)

    variables = {'U': 'Uxr', 'V': 'Vxr', 'P': 'Pxr'}
    dimensions = {'lat': 'lat', 'lon': 'lon'}
    if periodic_wrap:
        return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', time_periodic=delta(days=1))
    else:
        return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', allow_time_extrapolation=True)


if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-r", "--repeatdt", dest="repeatdt", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-x", "--xarray", dest="use_xarray", action='store_true', default=False, help="use xarray as data backend")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    parser.add_argument("-d", "--delParticle", dest="delete_particle", action='store_true', default=False, help="switch to delete a particle (True) or reset a particle (default: False).")
    args = parser.parse_args()

    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    backwardSimulation = args.backwards
    repeatdtFlag=args.repeatdt
    time_in_days = args.time_in_days
    use_xarray = args.use_xarray

    odir = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:
        odir = "/scratch/{}/experiments".format(os.environ['USER'])
    else:
        odir = "/var/scratch/experiments"

    func_time = []
    mem_used_GB = []

    fieldset = None
    if use_xarray:
        fieldset = perlin_fieldset_from_xarray(periodic_wrap=periodicFlag)
    else:
        fieldset = perlin_fieldset_from_numpy(periodic_wrap=periodicFlag)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            global_t_0 = ostime.time()
    else:
        global_t_0 = ostime.time()

    simStart = None
    for f in fieldset.get_fields():
        if type(f) in [VectorField, NestedField, SummedField]:  # or not f.grid.defer_load
            continue
        else:
            if backwardSimulation:
                simStart=f.grid.time_full[-1]
            else:
                simStart = f.grid.time_full[0]
            break

    if backwardSimulation:
        # ==== backward simulation ==== #
        if repeatdtFlag:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(96, 1) * a, lat=np.random.rand(96, 1) * b, time=simStart, repeatdt=delta(hours=1))
        else:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)
    else:
        # ==== forward simulation ==== #
        if repeatdtFlag:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(96, 1) * a, lat=np.random.rand(96, 1) * b, time=simStart, repeatdt=delta(hours=1))
        else:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)

    output_file = None
    if args.write_out:
        output_file = pset.ParticleFile(name=os.path.join(odir,"test_mem_behaviour.nc"), outputdt=delta(hours=1))
    delete_func = RenewParticle
    if args.delete_particle:
        delete_func=DeleteParticle

    perflog = PerformanceLog()
    postProcessFuncs = [perflog.advance,]

    if backwardSimulation:
        # ==== backward simulation ==== #
        pset.execute(AdvectionRK4, runtime=delta(days=time_in_days), dt=delta(hours=-1), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationFunctions=postProcessFuncs)
    else:
        # ==== forward simulation ==== #
        pset.execute(AdvectionRK4, runtime=delta(days=time_in_days), dt=delta(hours=1), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationFunctions=postProcessFuncs)

    if args.write_out:
        output_file.close()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        if mpi_comm.Get_rank() > 0:
            pass
        else:
            plot(perflog.samples, perflog.times_steps, perflog.memory_steps, perflog.fds_steps, os.path.join(odir, imageFileName))
    else:
        plot(perflog.samples, perflog.times_steps, perflog.memory_steps, perflog.fds_steps, os.path.join(odir, imageFileName))


