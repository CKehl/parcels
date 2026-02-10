from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from datetime import timedelta as delta
import numpy as np
import dask as da
import dask.array as daArray
from glob import glob
import time
import matplotlib.pyplot as plt
import os
import parcels
import os
import psutil
import gc
try:
    from mpi4py import MPI
except:
    MPI = None

with_GC = False

def set_cmems_fieldset(cs):
    #ddir_head = "/data/oceanparcels/input_data"
    ddir_head = "/data"
    ddir = os.path.join(ddir_head, "CMEMS/GLOBAL_REANALYSIS_PHY_001_030/")
    #print(ddir)
    files = sorted(glob(ddir+"mercatorglorys12v1_gl12_mean_201607*.nc"))
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}

    if cs not in ['auto', False]:
        cs = (1, cs, cs)
    return FieldSet.from_netcdf(files, variables, dimensions, field_chunksize=cs)

def print_field_info(fieldset):
    for f in fieldset.get_fields():
        if type(f) in [parcels.VectorField, parcels.NestedField, parcels.SummedField] or not f.grid.defer_load:
            continue
        if isinstance(f.data, daArray.core.Array):
            print("Array of Field[name={}] is dask.Array".format(f.name))
            print(
                "Chunk info of Field[name={}]: field.nchunks={}; shape(field.data.nchunks)={}; field.data.numblocks={}; shape(f.data)={}".format(
                    f.name, f.nchunks, (len(f.data.chunks[0]), len(f.data.chunks[1]), len(f.data.chunks[2])),
                    f.data.numblocks, f.data.shape))
        print("Chunk info of Grid[field.name={}]: g.chunk_info={}; g.load_chunk={}; len(g.load_chunk)={}".format(f.name,
                                                                                                                 f.grid.chunk_info,
                                                                                                                 f.grid.load_chunk,
                                                                                                                 len(
                                                                                                                     f.grid.load_chunk)))

if __name__=='__main__':
    #odir = "/scratch/ckehl/experiments"
    odir = "/var/scratch/experiments"
    func_time = []
    mem_used_GB = []
    open_fds = []
    chunksize = [50, 100, 200, 400, 800, 1000, 1500, 2000, 2500, 4000, 'auto', False]
    #chunksize = [64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 'auto', False]
    #chunksize = [512, 'auto']
    auto_field_size = 0
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank == 0:
            print("Dask global config - array.chunk-size: {}".format(da.config.get('array.chunk-size')))
    else:
        print("Dask global config - array.chunk-size: {}".format(da.config.get('array.chunk-size')))
    for cs in chunksize:
        fieldset = set_cmems_fieldset(cs)
        #pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=[0.000001*0,0.000001*1,0.000001*2,0.000001*3], lat=[0,0,0,0], repeatdt=delta(hours=1))
        pset = ParticleSet(fieldset=fieldset, pclass=JITParticle,
                           lon=np.random.rand(32,1)*1e-5, lat=np.random.rand(32,1)*1e-5,
                           repeatdt=delta(hours=1))

        tic = time.time()
        pset.execute(AdvectionRK4, dt=delta(hours=1))

        if cs=='auto':
            for f in fieldset.get_fields():
                if type(f) in [parcels.VectorField, parcels.NestedField, parcels.SummedField] or not f.grid.defer_load:
                    continue
                chunk_info = f.grid.chunk_info
                auto_field_size = chunk_info[chunk_info[0]+1]
                break

        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank==0:
                func_time.append(time.time() - tic)
        else:
            func_time.append(time.time() - tic)
        if with_GC:
            gc.collect()
        process = psutil.Process(os.getpid())
        mem_B_used = process.memory_info().rss
        fds_open = len(psutil.Process().open_files())
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mem_B_used_total = mpi_comm.reduce(mem_B_used, op=MPI.SUM, root=0)
            fds_open_total = mpi_comm.reduce(fds_open, op=MPI.SUM, root=0)
            if mpi_rank==0:
                #mem_used_GB.append(mem_B_used_total/(1024*1024*1024))
                mem_used_GB.append(mem_B_used_total / (1024 * 1024))
                open_fds.append(fds_open_total)
                #print(mem_B_used/(1024*1024))
        else:
            #mem_used_GB.append(mem_B_used/(1024*1024*1024))
            mem_used_GB.append(mem_B_used / (1024 * 1024))
            #print(mem_B_used/(1024*1024))

        #if MPI:
        #    mpi_comm = MPI.COMM_WORLD
        #    mpi_comm.Barrier()
        #    if mpi_comm.Get_rank() > 0:
        #        pass
        #    else:
        #        print_field_info(fieldset)
        #else:
        #    print_field_info()


    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        if mpi_comm.Get_rank() > 0:
            pass
        else:
            #print("auto-determined fieldsize: {}\n".format(auto_field_size))
            fig, ax = plt.subplots(1, 1, figsize=(15, 12))
            ax.plot(chunksize[:-2], func_time[:-2], 'o-', label="time(field_chunksize) [s]")
            #ax.plot(chunksize[:-2], mem_used_GB[:-2], '--', label="memory_used [GB]")
            #ax.plot(chunksize[:-2], open_fds[:-2], '.-', label="open_files [#]")
            ax.plot([0, 4000], [func_time[-2], func_time[-2]], 'x-', label="auto (size={}) [s]".format(auto_field_size))
            ax.plot([0, 4000], [func_time[-1], func_time[-1]], '--', label="no chunking [s]")
            #plt.xlim([0, 2100])
            #plt.ylim([0,60])
            plt.legend()
            ax.set_xlabel('field_chunksize')
            ax.set_ylabel('Time spent in pset.execute() [s]')
            #ax.set_ylabel('Time spent [s]')
            plt.savefig(os.path.join(odir, "mpiChunking_plot_MPI.png"), dpi=300, format='png')

            fig2, ax2 = plt.subplots(1, 1, figsize=(15, 12))
            ax2.plot(chunksize[:-2], mem_used_GB[:-2], '--', label="memory_blocked [MB]")
            ax2.plot([0, 4000], [mem_used_GB[-2], mem_used_GB[-2]], 'x-', label="auto (size={}) [MB]".format(auto_field_size))
            ax2.plot([0, 4000], [mem_used_GB[-1], mem_used_GB[-1]], '--', label="no chunking [MB]")
            #plt.xlim([0, 2100])
            #plt.ylim([0,60])
            plt.legend()
            ax2.set_xlabel('field_chunksize')
            ax2.set_ylabel('Memory blocked in pset.execute() [MB]')
            plt.savefig(os.path.join(odir, "mpiChunking_plot_MPI_mem.png"), dpi=300, format='png')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        ax.plot(chunksize[:-2], func_time[:-2], 'o-', label="time(field_chunksize) [s]")
        ax.plot([0, 4000], [func_time[-2], func_time[-2]], 'x-', label="auto (size={}) [s]".format(auto_field_size))
        ax.plot([0, 4000], [func_time[-1], func_time[-1]], '--', label="no chunking [s]")
        plt.legend()
        ax.set_xlabel('field_chunksize')
        ax.set_ylabel('Time spent in pset.execute() [s]')
        plt.savefig(os.path.join(odir, "mpiChunking_plot.png"), dpi=300, format='png')

        fig2, ax2 = plt.subplots(1, 1, figsize=(15, 12))
        ax2.plot(chunksize[:-2], mem_used_GB[:-2], '--', label="memory_blocked [MB]")
        ax2.plot([0, 4000], [mem_used_GB[-2], mem_used_GB[-2]], 'x-', label="auto (size={}) [MB]".format(auto_field_size))
        ax2.plot([0, 4000], [mem_used_GB[-1], mem_used_GB[-1]], '--', label="no chunking [MB]")
        plt.legend()
        ax2.set_xlabel('field_chunksize')
        ax2.set_ylabel('Memory blocked in pset.execute() [MB]')
        plt.savefig(os.path.join(odir, "mpiChunking_plot_MPI_mem.png"), dpi=300, format='png')
