# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:31:22 2017

@author: nooteboom
"""

from parcels import (FieldSet, JITParticle, AdvectionRK4_3D,
                     Field, ErrorCode, ParticleFile, Variable)
# from parcels import ParticleSet
from parcels import ParticleSet_Benchmark

from datetime import timedelta as delta
from datetime import datetime
import numpy as np
import math
from glob import glob
import sys
import pandas as pd
import dask

import psutil
import gc
import os
import time as ostime
import matplotlib.pyplot as plt
import fnmatch

try:
    from mpi4py import MPI
except:
    MPI = None

import warnings
import xarray as xr
warnings.simplefilter("ignore", category=xr.SerializationWarning)

global_t_0 = 0
odir = ""
if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
    odir = "/scratch/{}/experiments/palaeo-parcels/BENCHres".format(os.environ['USER'])
elif fnmatch.fnmatchcase(os.uname()[1], "int?.*"):  # Cartesius
    CARTESIUS_SCRATCH_USERNAME = 'ckehl'
    odir = "/scratch/shared/{}/experiments/palaeo-parcels/BENCHres".format(CARTESIUS_SCRATCH_USERNAME)
else:
    odir = "/var/scratch/experiments/palaeo-parcels/BENCHres"

class PerformanceLog():
    samples = []
    times_steps = []
    memory_steps = []
    _iter = 0

    def advance(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            process = psutil.Process(os.getpid())
            mem_B_used = process.memory_info().rss
            mem_B_used_total = mpi_comm.reduce(mem_B_used, op=MPI.SUM, root=0)
            if mpi_rank == 0:
                self.times_steps.append(MPI.Wtime())
                self.memory_steps.append(mem_B_used_total)
                self.samples.append(self._iter)
                self._iter+=1
        else:
            process = psutil.Process(os.getpid())
            self.times_steps.append(ostime.time())
            self.memory_steps.append(process.memory_info().rss)
            self.samples.append(self._iter)
            self._iter+=1

def plot(x, total_times, compute_times, io_times, memory_used, imageFilePath):
    plot_t = []
    plot_ct = []
    plot_iot = []
    cum_t = 0
    cum_ct = 0
    cum_iot = 0
    t_scaler = 1. * 10./1.0
    for i in range(len(total_times)):
        if i==0:
            plot_t.append( (total_times[i]-global_t_0)*t_scaler )
            cum_t += (total_times[i]-global_t_0)
        else:
            plot_t.append( (total_times[i]-total_times[i-1])*t_scaler )
            cum_t += (total_times[i]-total_times[i-1])

    for i in range(len(compute_times)):
        plot_ct.append(compute_times[i] * t_scaler)
        cum_ct += compute_times[i]
    for i in range(len(io_times)):
        plot_iot.append(io_times[i] * t_scaler)
        cum_iot += io_times[i]

    #mem_scaler = (1*10)/(1024*1024*1024)
    mem_scaler = 1 / (1024 * 1024 * 1024)
    plot_mem = []
    for i in range(len(memory_used)):
        plot_mem.append(memory_used[i] * mem_scaler)

    fig, ax = plt.subplots(1, 1, figsize=(21, 12))
    ax.plot(x, plot_t, 'o-', label="total time_spent [100ms]")
    ax.plot(x, plot_t, 'o-', label="compute time_spent [100ms]")
    ax.plot(x, plot_t, 'o-', label="io time_spent [100ms]")
    #ax.plot(x, plot_mem, 'x-', label="memory_used (cumulative) [100 MB]")
    ax.plot(x, plot_mem, 'x-', label="memory_used (cumulative) [1 GB]")
    plt.xlim([0, 730])
    plt.ylim([0, 120])
    plt.legend()
    ax.set_xlabel('iteration')
    plt.savefig(os.path.join(odir, imageFilePath), dpi=600, format='png')

    sys.stdout.write("cumulative total runtime: {}".format(cum_t))
    sys.stdout.write("cumulative compute time: {}".format(cum_ct))
    sys.stdout.write("cumulative I/O time: {}".format(cum_iot))


# dirread_pal = '/projects/0/palaeo-parcels/NEMOdata/'
dirread_pal = '/scratch/ckehl/experiments/palaeo-parcels/NEMOdata/'
dirread_top = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
dirread_top_bgc = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'

# sp = float(sys.argv[1])  # The sinkspeed m/day
sp = 11.0 # The sinkspeed m/day
dd = 10.  # The dwelling depth

dirwrite = '/scratch/ckehl/experiments/palaeo-parcels/NEMOres/assemblages/particlefiles/sp%d_dd%d/'%(int(sp),int(dd))

latsz = np.array(pd.read_csv('/scratch/ckehl/experiments/palaeo-parcels/TF_locationsSurfaceSamples_forPeter.csv').Latitude.tolist())
lonsz = np.array(pd.read_csv('/scratch/ckehl/experiments/palaeo-parcels/TF_locationsSurfaceSamples_forPeter.csv').Longitude.tolist())
numlocs = np.logical_and(latsz<1000, lonsz<1000)
latsz = latsz[numlocs]; lonsz = lonsz[numlocs];

assert ~(np.isnan(latsz)).any(), 'locations should not contain any NaN values'
dep = dd * np.ones(latsz.shape)

times = np.array([datetime(2000, 12, 25) - delta(days=x) for x in range(0,int(365),3)])
time = np.empty(shape=(0)); lons = np.empty(shape=(0)); lats = np.empty(shape=(0))
for i in range(len(times)):
    lons = np.append(lons,lonsz)
    lats = np.append(lats, latsz)
    time = np.append(time, np.full(len(lonsz),times[i])) 

def set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, pfiles, dfiles, ifiles, bfile, mesh_mask='/scratch/ckehl/experiments/palaeo-parcels/NEMOdata/domain/coordinates.nc'):
    filenames = { 'U': {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':ufiles},
                'V' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':vfiles},
                'W' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':wfiles},  
                'S' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [tfiles[0]],
                        'data':tfiles},   
                'T' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [tfiles[0]],
                        'data':tfiles},
                'NO3':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [pfiles[0]],
                       'data':pfiles},
                'PP':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [dfiles[0]],
                       'data':dfiles},
                'ICE':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [ifiles[0]],
                       'data':ifiles},
                'ICEPRES':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [ifiles[0]],
                       'data':ifiles},
                'CO2':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [dfiles[0]],
                       'data':dfiles},
                }
    if mesh_mask:
        filenames['mesh_mask'] = mesh_mask
    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo',
                 'T': 'sst',
                 'S': 'sss',
                 'NO3': 'DIN',                 
                 'PP': 'TPP3',                
                 'ICE': 'sit',
                 'ICEPRES': 'ice_pres',
                 'CO2': 'TCO2' }

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'T': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'S': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'NO3': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'PP': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'ICE': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'ICEPRES': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'CO2': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'} } #,
    bfiles = {'lon': mesh_mask, 'lat': mesh_mask, 'data': [bfile, ]}
    bvariables = ('B', 'Bathymetry')
    bdimensions = {'lon': 'glamf', 'lat': 'gphif'}
    bchs = False

    chs = {'time_counter': 1, 'depthu': 75, 'depthv': 75, 'depthw': 75, 'deptht': 75, 'y': 200, 'x': 200}
    #
    #chs = (1, 75, 200, 200)
    #
    #dask.config.set({'array.chunk-size': '6MiB'})
    #chs = 'auto'

    if mesh_mask: # and isinstance(bfile, list) and len(bfile) > 0:
        # fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, field_chunksize='auto')
        fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True, field_chunksize=chs)
        Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
        fieldset.add_field(Bfield, 'B')
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10
        return fieldset
    else:
        filenames.pop('B')
        variables.pop('B')
        dimensions.pop('B')
        # fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=False, field_chunksize=chs)
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, field_chunksize=chs)
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10
        return fieldset
        

def periodicBC(particle, fieldSet, time):
    if particle.lon > 180:
        particle.lon -= 360        
    if particle.lon < -180:
        particle.lon += 360   

def Sink(particle, fieldset, time):
    if(particle.depth>fieldset.dwellingdepth):
        particle.depth = particle.depth + fieldset.sinkspeed * particle.dt
    elif(particle.depth<=fieldset.dwellingdepth and particle.depth>1):
        particle.depth = fieldset.surface
        particle.temp = fieldset.T[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.salin = fieldset.S[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.PP = fieldset.PP[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.NO3 = fieldset.NO3[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.ICE = fieldset.ICE[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.ICEPRES = fieldset.ICEPRES[time+particle.dt, fieldset.surface, particle.lat, particle.lon] 
        particle.CO2 = fieldset.CO2[time+particle.dt, fieldset.surface, particle.lat, particle.lon]  
        particle.delete()

def Age(particle, fieldset, time):
    particle.age = particle.age + math.fabs(particle.dt)  

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def initials(particle, fieldset, time):
    if particle.age==0.:
        particle.depth = fieldset.B[time, fieldset.surface, particle.lat, particle.lon]
        if(particle.depth  > 5800.):
            particle.age = (particle.depth - 5799.)*fieldset.sinkspeed
            particle.depth = 5799.        
        particle.lon0 = particle.lon
        particle.lat0 = particle.lat
        particle.depth0 = particle.depth





def run_corefootprintparticles(dirwrite,outfile,lonss,latss,dep):
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            global_t_0 = MPI.Wtime()
    else:
        global_t_0 = ostime.time()

    ufiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200012??d05U.nc'))
    vfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200012??d05V.nc'))
    wfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200012??d05W.nc'))    
    tfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200012??d05T.nc'))
    pfiles = sorted(glob(dirread_top_bgc + 'means/ORCA0083-N06_200012??d05P.nc'))    
    dfiles = sorted(glob(dirread_top_bgc + 'means/ORCA0083-N06_200012??d05D.nc'))    
    ifiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_200012??d05I.nc'))
    bfile = dirread_top + 'domain/bathymetry_ORCA12_V3.3.nc'

    fieldset = set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, pfiles, dfiles, ifiles, bfile, dirread_pal + 'domain/coordinates.nc')
    fieldset.add_periodic_halo(zonal=True) 
    fieldset.add_constant('dwellingdepth', np.float(dd))
    fieldset.add_constant('sinkspeed', sp/86400.)
    fieldset.add_constant('maxage', 300000.*86400)
    fieldset.add_constant('surface', 2.5)

    class DinoParticle(JITParticle):
        temp = Variable('temp', dtype=np.float32, initial=np.nan)
        age = Variable('age', dtype=np.float32, initial=0.)
        salin = Variable('salin', dtype=np.float32, initial=np.nan)
        lon0 = Variable('lon0', dtype=np.float32, initial=0.)
        lat0 = Variable('lat0', dtype=np.float32, initial=0.)
        depth0 = Variable('depth0',dtype=np.float32, initial=0.) 
        PP = Variable('PP',dtype=np.float32, initial=np.nan)
        NO3 = Variable('NO3',dtype=np.float32, initial=np.nan)
        ICE = Variable('ICE',dtype=np.float32, initial=np.nan)
        ICEPRES = Variable('ICEPRES',dtype=np.float32, initial=np.nan)
        CO2 = Variable('CO2',dtype=np.float32, initial=np.nan)
        
    pset = ParticleSet_Benchmark.from_list(fieldset=fieldset, pclass=DinoParticle, lon=lonss.tolist(), lat=latss.tolist(),
                       time = time)

    perflog = PerformanceLog()
    postProcessFuncs = [perflog.advance,]

    pfile = ParticleFile(dirwrite + outfile, pset, write_ondelete=True)
    kernels = pset.Kernel(initials) + Sink + Age  + pset.Kernel(AdvectionRK4_3D) + Age

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            starttime = MPI.Wtime()
    else:
        starttime = ostime.time()

    # pset.execute(kernels, runtime=delta(days=365*9), dt=delta(minutes=-20), output_file=pfile, verbose_progress=False,
    # recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs)
    pset.execute(kernels, runtime=delta(days=365), dt=delta(hours=-12), output_file=pfile, verbose_progress=False,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs)
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            endtime = MPI.Wtime()
    else:
        endtime = ostime.time()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            dt_time = []
            for i in range(len(perflog.times_steps)):
                if i==0:
                    dt_time.append( (perflog.times_steps[i]-global_t_0) )
                else:
                    dt_time.append( (perflog.times_steps[i]-perflog.times_steps[i-1]) )
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
            avg_time = np.mean(np.array(dt_time, dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
    else:
        dt_time = []
        for i in range(len(perflog.times_steps)):
            if i == 0:
                dt_time.append((perflog.times_steps[i] - global_t_0))
            else:
                dt_time.append((perflog.times_steps[i] - perflog.times_steps[i - 1]))
        sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
        avg_time = np.mean(np.array(dt_time, dtype=np.float64))
        sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

    pfile.close()


    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        if mpi_comm.Get_rank() == 0:
            plot(perflog.samples, perflog.times_steps, perflog.memory_steps, pset.compute_log.times_steps, pset.io_log.times_steps, os.path.join(odir, imageFileName))
    else:
        plot(perflog.samples, perflog.times_steps, perflog.memory_steps, pset.compute_log.times_steps, pset.io_log.times_steps, os.path.join(odir, imageFileName))

    print('Execution finished')

outfile = 'grid_dd'+str(int(dd)) +'_sp'+str(int(sp))
run_corefootprintparticles(dirwrite,outfile,lons,lats,dep)

