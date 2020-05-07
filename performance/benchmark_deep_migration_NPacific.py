from parcels import FieldSet, JITParticle, ScipyParticle, AdvectionRK4_3D, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, NestedField, VectorField, timer
from parcels import ParticleSet_Benchmark
from parcels.kernels import seawaterdensity
from argparse import ArgumentParser
from datetime import timedelta as delta
from datetime import  datetime
import time as ostime
import numpy as np
import math
from glob import glob
import matplotlib.pyplot as plt
import fnmatch
import psutil
import os
import sys
import warnings
import pickle                                                      
import matplotlib.ticker as mtick
from numpy import *
import scipy.linalg
import math as math
import itertools
warnings.filterwarnings("ignore")

try:
    from mpi4py import MPI
except:
    MPI = None


global_t_0 = 0
# Fieldset grid is 30x30 deg in North Pacific
minlat = 20 
maxlat = 50 
minlon = -175 
maxlon = -145 

# Release particles on a 10x10 deg grid in middle of the 30x30 fieldset grid and 1m depth
lat_release0 = np.tile(np.linspace(30,39,10),[10,1]) 
lat_release = lat_release0.T 
lon_release = np.tile(np.linspace(-165,-156,10),[10,1]) 
z_release = np.tile(1,[10,10])

# Choose:
simdays = 50.0 * 365.0
#simdays = 5
time0 = 0
simhours = 1
simmins = 30
secsdt = 30
hrsoutdt = 5

#--------- Choose below: NOTE- MUST ALSO MANUALLY CHANGE IT IN THE KOOI KERNAL BELOW -----
rho_pl = 920.                 # density of plastic (kg m-3): DEFAULT FOR FIG 1 in Kooi: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
r_pl = "1e-04"                # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7


class PerformanceLog():
    samples = []
    times_steps = []
    memory_steps = []
    Nparticles_step = []
    _iter = 0
    pset = None

    def advance(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            process = psutil.Process(os.getpid())
            mem_B_used = process.memory_info().rss
            mem_B_used_total = mpi_comm.reduce(mem_B_used, op=MPI.SUM, root=0)
            Nparticles_global = 0
            if self.pset is not None:
                Nparticles_local = len(self.pset)
                Nparticles_global = mpi_comm.reduce(Nparticles_local, op=MPI.SUM, root=0)
            if mpi_rank == 0:
                self.times_steps.append(MPI.Wtime())
                self.memory_steps.append(mem_B_used_total)
                if self.pset is not None:
                    self.Nparticles_step.append(Nparticles_global)
                self.samples.append(self._iter)
                self._iter+=1
        else:
            process = psutil.Process(os.getpid())
            self.times_steps.append(ostime.time())
            self.memory_steps.append(process.memory_info().rss)
            if self.pset is not None:
                self.Nparticles_step.append(len(self.pset))
            self.samples.append(self._iter)
            self._iter+=1


def plot(total_times, compute_times, io_times, memory_used, nparticles, imageFilePath):
    plot_t = []
    plot_ct = []
    plot_iot = []
    plot_npart = []
    cum_t = 0
    cum_ct = 0
    cum_iot = 0
    t_scaler = 1. * 10./1.0
    npart_scaler = 1.0 / 1000.0
    for i in range(len(total_times)):
        #if i==0:
        #    plot_t.append( (total_times[i]-global_t_0)*t_scaler )
        #    cum_t += (total_times[i]-global_t_0)
        #else:
        #    plot_t.append( (total_times[i]-total_times[i-1])*t_scaler )
        #    cum_t += (total_times[i]-total_times[i-1])
        plot_t.append( total_times[i]*t_scaler )
        cum_t += (total_times[i])

    for i in range(len(compute_times)):
        plot_ct.append(compute_times[i] * t_scaler)
        cum_ct += compute_times[i]
    for i in range(len(io_times)):
        plot_iot.append(io_times[i] * t_scaler)
        cum_iot += io_times[i]
    for i in range(len(nparticles)):
        plot_npart.append(nparticles[i] * npart_scaler)

    if memory_used is not None:
        #mem_scaler = (1*10)/(1024*1024*1024)
        mem_scaler = 1 / (1024 * 1024 * 1024)
        plot_mem = []
        for i in range(len(memory_used)):
            plot_mem.append(memory_used[i] * mem_scaler)

    # do_ct_plot = True
    do_iot_plot = True
    do_mem_plot = True
    do_npart_plot = True
    assert (len(plot_t) == len(plot_ct))
    # assert (len(plot_t) == len(plot_iot))
    if len(plot_t) != len(plot_iot):
        print("plot_t and plot_iot have different lengths ({} vs {})".format(len(plot_t), len(plot_iot)))
        do_iot_plot = False
    # assert (len(plot_t) == len(plot_mem))
    if len(plot_t) != len(plot_mem):
        print("plot_t and plot_mem have different lengths ({} vs {})".format(len(plot_t), len(plot_mem)))
        do_mem_plot = False
    # assert (len(plot_t) == len(plot_npart))
    if len(plot_t) != len(plot_npart):
        print("plot_t and plot_npart have different lengths ({} vs {})".format(len(plot_t), len(plot_npart)))
        do_npart_plot = False
    x = []
    for i in itertools.islice(itertools.count(), 0, len(plot_t)):
        x.append(i)

    fig, ax = plt.subplots(1, 1, figsize=(21, 12))
    ax.plot(x, plot_t, 'o-', label="total time_spent [100ms]")
    ax.plot(x, plot_ct, 'o-', label="compute time_spent [100ms]")
    # == this is still the part that breaks - as they are on different time scales, possibly leave them out ? == #
    if do_iot_plot:
        ax.plot(x, plot_iot, 'o-', label="io time_spent [100ms]")
    if (memory_used is not None) and do_mem_plot:
        #ax.plot(x, plot_mem, 'x-', label="memory_used (cumulative) [100 MB]")
        ax.plot(x, plot_mem, 'x-', label="memory_used (cumulative) [1 GB]")
    if do_npart_plot:
        ax.plot(x, plot_npart, '-', label="sim. particles [# 1000]")
    plt.xlim([0, 730])
    plt.ylim([0, 120])
    plt.legend()
    ax.set_xlabel('iteration')
    plt.savefig(os.path.join(odir, imageFilePath), dpi=600, format='png')

    sys.stdout.write("cumulative total runtime: {}\n".format(cum_t))
    sys.stdout.write("cumulative compute time: {}\n".format(cum_ct))
    sys.stdout.write("cumulative I/O time: {}\n".format(cum_iot))


def Kooi(particle,fieldset,time):  
    #------ CHOOSE -----
    rho_pl = 920.                 # density of plastic (kg m-3): DEFAULT FOR FIG 1: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
    r_pl = 1e-04                  # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7   
    
    # Nitrogen to cell ratios for ambient algal concentrations ('aa') and algal growth ('mu_aa') from NEMO output (no longer using N:C:AA (Redfield ratio), directly N:AA from Menden-Deuer and Lessard 2000)     
    min_N2cell = 2656.0e-09 #[mgN cell-1] (from Menden-Deuer and Lessard 2000)
    max_N2cell = 11.0e-09   #[mgN cell-1] 
    med_N2cell = 356.04e-09 #[mgN cell-1] THIS is used below 
      
    # Ambient algal concentration from MEDUSA's non-diatom + diatom phytoplankton 
    n0 = particle.nd_phy+particle.d_phy # [mmol N m-3] in MEDUSA
    n = n0*14.007       # conversion from [mmol N m-3] to [mg N m-3] (atomic weight of 1 mol of N = 14.007 g)   
    n2 = n/med_N2cell   # conversion from [mg N m-3] to [no. m-3]
    
    if n2<0.: 
        aa = 0.
    else:
        aa = n2   # [no m-3] to compare to Kooi model    
    
    # Primary productivity (algal growth) only above euphotic zone, condition same as in Kooi et al. 2017
    if particle.depth<particle.euph_z:
        tpp0 = particle.tpp3 # (particle.nd_tpp + particle.d_tpp)/particle.euph_z # Seeing if the 2D production of nondiatom + diatom can be converted to a vertical profile (better with TPP3)
    else:
        tpp0 = 0.    
    
    mu_n0 = tpp0*14.007               # conversion from mmol N m-3 d-1 to mg N m-3 d-1 (atomic weight of 1 mol of N = 14.007 g) 
    mu_n = mu_n0/med_N2cell           # conversion from mg N m-3 d-1 to no. m-3 d-1
    mu_n2 = mu_n/aa                   # conversion from no. m-3 d-1 to d-1
    
    if mu_n2<0.:
        mu_aa = 0.
    else:
        mu_aa = mu_n2/86400. # conversion from d-1 to s-1
        
    z = particle.depth           # [m]
    t = particle.temp            # [oC]
    sw_visc = particle.sw_visc   # [kg m-1 s-1]
    kin_visc = particle.kin_visc # [m2 s-1]
    rho_sw = particle.density    # [kg m-3]   #rho_sw     
    a = particle.a               # [no. m-2 s-1]
    vs = particle.vs             # [m s-1]   #particle.depth

    #------ Constants and algal properties -----
    g = 7.32e10/(86400.**2.)    # gravitational acceleration (m d-2), now [s-2]
    k = 1.0306E-13/(86400.**2.) # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_bf = 1388.              # density of biofilm ([g m-3]
    v_a = 2.0E-16               # Volume of 1 algal cell [m-3]
    m_a = 0.39/86400.           # mortality rate, now [s-1]
    r20 = 0.1/86400.            # respiration rate, now [s-1] 
    q10 = 2.                    # temperature coefficient respiration [-]
    gamma = 1.728E5/86400.      # shear [d-1], now [s-1]
    
    #------ Volumes -----
    v_pl = (4./3.)*math.pi*r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*r_pl**2.              # surface area of plastic particle [m2]
    r_a = ((3./4.)*(v_a/math.pi))**(1./3.)      # radius of algae [m]
    
    v_bf = (v_a*a)*theta_pl                           # volume of biofilm [m3]
    v_tot = v_bf + v_pl                               # volume of total [m3]
    t_bf = ((v_tot*(3./(4.*math.pi)))**(1./3.))-r_pl  # biofilm thickness [m] 
    
    
    r_tot = r_pl + t_bf                               # total radius [m]
    rho_tot = (r_pl**3. * rho_pl + ((r_pl + t_bf)**3. - r_pl**3.)*rho_bf)/(r_pl + t_bf)**3. # total density [kg m-3]
    rho_tot = rho_tot
    theta_tot = 4.*math.pi*r_tot**2.                          # surface area of total [m2]
    d_pl = k * (t + 273.16)/(6. * math.pi * sw_visc * r_tot)  # diffusivity of plastic particle [m2 s-1]
    d_a = k * (t + 273.16)/(6. * math.pi * sw_visc * r_a)     # diffusivity of algal cells [m2 s-1] 
    beta_abrown = 4.*math.pi*(d_pl + d_a)*(r_tot + r_a)       # Brownian motion [m3 s-1] 
    beta_ashear = 1.3*gamma*((r_tot + r_a)**3.)               # advective shear [m3 s-1]
    beta_aset = (1./2.)*math.pi*r_tot**2. * abs(vs)           # differential settling [m3 s-1]
    beta_a = beta_abrown + beta_ashear + beta_aset            # collision rate [m3 s-1]
    
    a_coll = (beta_a*aa)/theta_pl
    a_growth = mu_aa*a
    a_mort = m_a*a
    a_resp = (q10**((t-20.)/10.))*r20*a     
    
    particle.a += (a_coll + a_growth - a_mort - a_resp) * particle.dt

    dn = 2. * (r_tot)                             # equivalent spherical diameter [m]
    delta_rho = (rho_tot - rho_sw)/rho_sw         # normalised difference in density between total plastic+bf and seawater[-]        
    d = ((rho_tot - rho_sw) * g * dn**3.)/(rho_sw * kin_visc**2.) # [-]
    
    if dn > 5e9:
        w = 1000.
    elif dn <0.05:
        w = (d**2.) *1.71E-4
    else:
        w = 10.**(-3.76715 + (1.92944*math.log10(d)) - (0.09815*math.log10(d)**2.) - (0.00575*math.log10(d)**3.) + (0.00056*math.log10(d)**4.))
    
    if z >= 4000.: 
        vs = 0
    elif z < 1. and delta_rho < 0:
        vs = 0  
    elif delta_rho > 0:
        vs = (g * kin_visc * w * delta_rho)**(1./3.)
    else: 
        a_del_rho = delta_rho*-1.
        vs = -1.*(g * kin_visc * w * a_del_rho)**(1./3.)  # m s-1

    particle.depth += vs * particle.dt 
    particle.vs = vs
    z = particle.depth
    dt = particle.dt

""" Defining the particle class """

class plastic_particle(JITParticle): #ScipyParticle): #
    u = Variable('u', dtype=np.float32,to_write=False)
    v = Variable('v', dtype=np.float32,to_write=False)
    w = Variable('w', dtype=np.float32,to_write=False)
    temp = Variable('temp',dtype=np.float32,to_write=False)
    density = Variable('density',dtype=np.float32,to_write=True)
    #aa = Variable('aa',dtype=np.float32,to_write=True)
    #d_tpp = Variable('d_tpp',dtype=np.float32,to_write=False) # mu_aa
    #nd_tpp = Variable('nd_tpp',dtype=np.float32,to_write=False)
    tpp3 = Variable('tpp3',dtype=np.float32,to_write=False)
    euph_z = Variable('euph_z',dtype=np.float32,to_write=False)
    d_phy = Variable('d_phy',dtype=np.float32,to_write=False)
    nd_phy = Variable('nd_phy',dtype=np.float32,to_write=False)    
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)    
    a = Variable('a',dtype=np.float32,to_write=False)
    vs = Variable('vs',dtype=np.float32,to_write=True)    
    
"""functions and kernals"""

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted') #print(particle.lon, particle.lat, particle.depth)
    particle.delete()

def getclosest_ij(lats,lons,latpt,lonpt):     
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index

def AdvectionRK4_3D_vert(particle, fieldset, time): # adapting AdvectionRK4_3D kernal to only vertical velocity 
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    Function needs to be converted to Kernel object before execution"""
    (w1) = fieldset.W[time, particle.depth, particle.lat, particle.lon]
    #lon1 = particle.lon + u1*.5*particle.dt
    #lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (w2) = fieldset.W[time + .5 * particle.dt, dep1, particle.lat, particle.lon]
    #lon2 = particle.lon + u2*.5*particle.dt
    #lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (w3) = fieldset.W[time + .5 * particle.dt, dep2, particle.lat, particle.lon]
    #lon3 = particle.lon + u3*particle.dt
    #lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (w4) = fieldset.W[time + particle.dt, dep3, particle.lat, particle.lon]
    #particle.lon += particle.lon #(u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    #particle.lat += particle.lat #lats[1,1] #(v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt
    
def Profiles(particle, fieldset, time):  
    particle.temp = fieldset.cons_temperature[time, particle.depth,particle.lat,particle.lon]  
    particle.d_phy= fieldset.d_phy[time, particle.depth,particle.lat,particle.lon]  
    particle.nd_phy= fieldset.nd_phy[time, particle.depth,particle.lat,particle.lon] 
    #particle.d_tpp = fieldset.d_tpp[time,particle.depth,particle.lat,particle.lon]
    #particle.nd_tpp = fieldset.nd_tpp[time,particle.depth,particle.lat,particle.lon]
    particle.tpp3 = fieldset.tpp3[time,particle.depth,particle.lat,particle.lon]
    particle.euph_z = fieldset.euph_z[time,particle.depth,particle.lat,particle.lon]
    particle.kin_visc = fieldset.KV[time,particle.depth,particle.lat,particle.lon] 
    particle.sw_visc = fieldset.SV[time,particle.depth,particle.lat,particle.lon] 
    particle.w = fieldset.W[time,particle.depth,particle.lat,particle.lon]

if __name__ == "__main__":
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    # parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=365, help="runtime in days (default: 365)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=str, default="1*365", help="runtime in days (default: 1*365)")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    args = parser.parse_args()

    time_in_days = int(float(eval(args.time_in_days)))
    headdir = ""
    odir = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    dirread_mesh = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments/deep_migration_behaviour".format("ckehl")
        odir = os.path.join(headdir,"BENCHres")
        datahead = "/data/oceanparcels/input_data"
        dirread = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/means/')
        dirread_bgc = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/means/')
        dirread_mesh = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/domain/')
    elif fnmatch.fnmatchcase(os.uname()[1], "int?.*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehl'
        headdir = "/scratch/shared/{}/experiments/deep_migration_behaviour".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "/BENCHres")
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/means/')
        dirread_bgc = os.path.join(datahead, 'NEMO-MEDUSA_BGC/ORCA0083-N006/means/')
        dirread_mesh = os.path.join(datahead, 'NEMO-MEDUSA_BGC/ORCA0083-N006/domain/')
    else:
        headdir = "/var/scratch/dlobelle"
        odir = os.path.join(headdir, "BENCHres")
        datahead = "/data"
        dirread = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/means/')
        dirread_bgc = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/means/')
        dirread_mesh = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/domain/')

    # ==== CARTESIUS ==== #
    # dirread = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/means/'
    # dirread_bgc = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'
    # dirread_mesh = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/domain/'
    # ==== GEMINI ==== #
    # dirread = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/means/'
    # dirread_bgc = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/means/'
    # dirread_mesh = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/domain/'
    # dirwrite = '/scratch/ckehl/experiments/deep_migration_behaviour/NEMOres/tests/'
    # ==== ====== ==== #

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            global_t_0 = MPI.Wtime()
    else:
        global_t_0 = ostime.time()

    ufiles = sorted(glob(dirread+'ORCA0083-N06_2000*d05U.nc')) #0105d05
    vfiles = sorted(glob(dirread+'ORCA0083-N06_2000*d05V.nc'))
    wfiles = sorted(glob(dirread+'ORCA0083-N06_2000*d05W.nc'))
    pfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_2000*d05P.nc'))
    ppfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_2000*d05D.nc'))
    tsfiles = sorted(glob(dirread+'ORCA0083-N06_2000*d05T.nc'))
    mesh_mask = dirread_mesh+'coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles}, #'depth': wfiles,
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles},
                 'd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
                 'nd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
                 'euph_z': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles},
                 #'d_tpp': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles}, # 'depth': wfiles,
                 #'nd_tpp': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ppfiles},
                 'tpp3': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ppfiles},
                 'cons_temperature': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'abs_salinity': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles}}


    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo',
                 'd_phy': 'PHD',
                 'nd_phy': 'PHN',
                 'euph_z': 'MED_XZE',
                 #'d_tpp': 'ML_PRD', # units: mmolN/m2/d
                 #'nd_tpp': 'ML_PRN', # units: mmolN/m2/d
                 'tpp3': 'TPP3', # units: mmolN/m3/d
                 'cons_temperature': 'potemp',
                 'abs_salinity': 'salin'}

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}, #time_centered
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'nd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'euph_z': {'lon': 'glamf', 'lat': 'gphif','time': 'time_counter'},
                  #'d_tpp': {'lon': 'glamf', 'lat': 'gphif','time': 'time_counter'}, # 'depth': 'depthw',
                  #'nd_tpp': {'lon': 'glamf', 'lat': 'gphif','time': 'time_counter'},
                  'tpp3': {'lon': 'glamf', 'lat': 'gphif','depth': 'depthw', 'time': 'time_counter'},
                  'cons_temperature': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'abs_salinity': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'}}

    chs = {'time_counter': 1, 'depthu': 75, 'depthv': 75, 'depthw': 75, 'deptht': 75, 'y': 200, 'x': 200}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, field_chunksize=chs, time_periodic=delta(days=365))
    depths = fieldset.U.depth

    outfile = 'Kooi+NEMO_3D_grid10by10_rho'+str(int(rho_pl))+'_r'+ r_pl+'_'+str(simdays)+'days_'+str(secsdt)+'dtsecs_'+str(hrsoutdt)+'hrsoutdt'
    dirwrite = os.path.join(odir, "rho_"+str(int(rho_pl))+"kgm-3")
    if not os.path.exists(dirwrite):
        os.mkdir(dirwrite)

    # Kinematic viscosity and dynamic viscosity not available in MEDUSA so replicating Kooi's profiles at all grid points
    # profile_auxin_path = '/home/dlobelle/Kooi_data/data_input/profiles.pickle'
    # profile_auxin_path = '/scratch/ckehl/experiments/deep_migration_behaviour/aux_in/profiles.pickle'
    profile_auxin_path = os.path.join(headdir, 'aux_in/profiles.pickle')
    with open(profile_auxin_path, 'rb') as f:
        depth,T_z,S_z,rho_z,upsilon_z,mu_z = pickle.load(f)

    v_lon = np.array([minlon,maxlon])
    v_lat = np.array([minlat,maxlat])

    kv_or = np.transpose(np.tile(np.array(upsilon_z),(len(v_lon),len(v_lat),1)), (2,0,1))   # kinematic viscosity
    sv_or = np.transpose(np.tile(np.array(mu_z),(len(v_lon),len(v_lat),1)), (2,0,1))        # dynamic viscosity of seawater
    KV = Field('KV', kv_or, lon=v_lon, lat=v_lat, depth=depths, mesh='spherical', field_chunksize=False)#,transpose="True") #,fieldtype='U')
    SV = Field('SV', sv_or, lon=v_lon, lat=v_lat, depth=depths, mesh='spherical', field_chunksize=False)#,transpose="True") #,fieldtype='U')
    fieldset.add_field(KV, 'KV')
    fieldset.add_field(SV, 'SV')

    """ Defining the particle set """
    pset = ParticleSet_Benchmark.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                                 pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                                 lon= lon_release, #-160.,  # a vector of release longitudes
                                 lat= lat_release, #36.,
                                 time = time0,
                                 depth = z_release) #[1.]

    perflog = PerformanceLog()
    perflog.pset = pset
    postProcessFuncs = [perflog.advance,]

    """ Kernal + Execution"""
    # kernels = pset.Kernel(AdvectionRK4_3D) + pset.Kernel(seawaterdensity.polyTEOS10_bsq) + pset.Kernel(Profiles) + pset.Kernel(Kooi)
    kernels = pset.Kernel(AdvectionRK4_3D_vert) + pset.Kernel(seawaterdensity.polyTEOS10_bsq) + pset.Kernel(Profiles) + pset.Kernel(Kooi)
    pfile= ParticleFile(os.path.join(dirwrite, outfile), pset, outputdt=delta(hours = hrsoutdt))

    starttime = 0
    endtime = 0
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            starttime = MPI.Wtime()
    else:
        starttime = ostime.time()

    pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=hrsoutdt))

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
            if len(perflog.Nparticles_step)>0:
                sys.stdout.write("final # particles: {}".format(perflog.Nparticles_step[len(perflog.Nparticles_step)-1]))
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
        if len(perflog.Nparticles_step) > 0:
            sys.stdout.write("final # particles: {}".format(perflog.Nparticles_step[len(perflog.Nparticles_step)-1]))
        sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
        avg_time = np.mean(np.array(dt_time, dtype=np.float64))
        sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        if mpi_comm.Get_rank() == 0:
            # plot(perflog.samples, perflog.times_steps, perflog.memory_steps, pset.compute_log.times_steps, pset.io_log.times_steps, os.path.join(odir, args.imageFileName))
            plot(pset.total_log.times_steps, pset.compute_log.times_steps, pset.io_log.times_steps, perflog.memory_steps, pset.nparticle_log.params, os.path.join(odir, args.imageFileName))
    else:
        # plot(perflog.samples, perflog.times_steps, perflog.memory_steps, pset.compute_log.times_steps, pset.io_log.times_steps, os.path.join(odir, args.imageFileName))
        plot(pset.total_log.times_steps, pset.compute_log.times_steps, pset.io_log.times_steps, perflog.memory_steps, pset.nparticle_log.params, os.path.join(odir, args.imageFileName))

    pfile.close()

    print('Execution finished')
