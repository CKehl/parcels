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

import sys
try:
    from mpi4py import MPI
except:
    MPI = None


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0

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
    particle.lat = np.random.rand() * 1e-5
    particle.lon = np.random.rand() * 1e-5

def stommel_fieldset_from_numpy(xdim=200, ydim=200, periodic_wrap=False):
    """Simulate a periodic current along a western boundary, with significantly
    larger velocities along the western edge than the rest of the region

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    a = 10000 * 1e3
    b = 10000 * 1e3
    scalefac = 0.05  # to scale for physically meaningful velocities

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(0, b, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size), dtype=np.float32)

    beta = 2e-11
    r = 1/(11.6*86400)
    es = r/(beta*a)

    for i in range(lon.size):
        for j in range(lat.size):
            xi = lon[i] / a
            yi = lat[j] / b
            P[i, j] = (1 - math.exp(-xi/es) - xi) * math.pi * np.sin(math.pi*yi)*scalefac
            U[i, j] = -(1 - math.exp(-xi/es) - xi) * math.pi**2 * np.cos(math.pi*yi)*scalefac
            V[i, j] = (math.exp(-xi/es)/es - 1) * math.pi * np.sin(math.pi*yi)*scalefac

    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lon': lon, 'lat': lat}
    if periodic_wrap:
        return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True, time_periodic=delta(days=1))
    else:
        return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True, allow_time_extrapolation=True)


def stommel_fieldset_from_xarray(xdim=200, ydim=200, periodic_wrap=False):
    a = 10000 * 1e3
    b = 10000 * 1e3
    scalefac = 0.05  # to scale for physically meaningful velocities
    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0., a, xdim, dtype=np.float32)
    lat = np.linspace(0., b, ydim, dtype=np.float32)
    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size), dtype=np.float32)

    beta = 2e-11
    r = 1/(11.6*86400)
    es = r/(beta*a)

    for i in range(lat.size):
        for j in range(lon.size):
            xi = lon[j] / a
            yi = lat[i] / b
            P[i, j] = (1 - math.exp(-xi/es) - xi) * math.pi * np.sin(math.pi*yi)*scalefac
            U[i, j] = -(1 - math.exp(-xi/es) - xi) * math.pi**2 * np.cos(math.pi*yi)*scalefac
            V[i, j] = (math.exp(-xi/es)/es - 1) * math.pi * np.sin(math.pi*yi)*scalefac

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
    imageFileName = ""
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-d", "--defer", dest="defer", action='store_false', default=True, help="enable/disable running with deferred load (default: True)")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-r", "--repeatdt", dest="repeatdt", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=33, help="runtime in days (default: 1)")
    parser.add_argument("-x", "--xarray", dest="use_xarray", action='store_true', default=False, help="use xarray as data backend")
    args = parser.parse_args()

    imageFileName=args.imageFileName
    deferLoadFlag = args.defer
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
        fieldset = stommel_fieldset_from_xarray(200, 200, periodic_wrap=periodicFlag)
    else:
        fieldset = stommel_fieldset_from_numpy(200, 200, periodic_wrap=periodicFlag)

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
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(96, 1) * 1e-5, lat=np.random.rand(96, 1) * 1e-5, time=simStart, repeatdt=delta(hours=1))
        else:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(65536, 1) * 1e-5, lat=np.random.rand(65536, 1) * 1e-5, time=simStart)
    else:
        # ==== forward simulation ==== #
        if repeatdtFlag:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(96, 1) * 5e-1, lat=np.random.rand(96, 1) * 5e-1, time=simStart, repeatdt=delta(hours=1))
        else:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(65536, 1) * 5e-1, lat=np.random.rand(65536, 1) * 5e-1, time=simStart)

    output_file = pset.ParticleFile(name=os.path.join(odir,"test_mem_behaviour.nc"), outputdt=delta(hours=1))
    perflog = PerformanceLog()
    postProcessFuncs = [perflog.advance,]

    if backwardSimulation:
        # ==== backward simulation ==== #
        pset.execute(AdvectionRK4, runtime=delta(days=time_in_days), dt=delta(hours=-1), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: RenewParticle}, postIterationFunctions=postProcessFuncs)
    else:
        # ==== forward simulation ==== #
        pset.execute(AdvectionRK4, runtime=delta(days=time_in_days), dt=delta(hours=1), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: RenewParticle}, postIterationFunctions=postProcessFuncs)
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


