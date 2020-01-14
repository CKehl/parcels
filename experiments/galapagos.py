from parcels import FieldSet, Field, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable
from datetime import timedelta as delta
from glob import glob
import numpy as np
import xarray as xr

wstokes = False

ddir = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
ufiles = sorted(glob(ddir+'means/ORCA0083-N06_20[00-10]*d05U.nc'))
print(len(ufiles))
vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
meshfile = glob(ddir+'domain/coordinates.nc')

nemofiles = {'U': {'lon': meshfile, 'lat': meshfile, 'data': ufiles},
             'V': {'lon': meshfile, 'lat': meshfile, 'data': vfiles}}
nemovariables = {'U': 'uo', 'V': 'vo'}
nemodimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
fieldset_nemo = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions)

if wstokes:
    stokesfiles = sorted(glob('/data/oceanparcels/input_data/WaveWatch3data/CFSR/WW3-*_uss.nc'))
    print(len(stokesfiles))
    stokesdimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    stokesvariables = {'U': 'uuss', 'V': 'vuss'}
    fieldset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions)
    fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

    fieldset = FieldSet(U=fieldset_nemo.U+fieldset_stokes.U, V=fieldset_nemo.V+fieldset_stokes.V)
    fU = fieldset.U[0]
    fname = "galapagosparticles_americas_wstokes_v2_5yr.nc"
else:
    fieldset = fieldset_nemo
    fU = fieldset.U
    fname = "galapagosparticles_americas_v2_5yr.nc"

fieldset.computeTimeChunk(fU.grid.time[0], 1)

landlon = []
landlat = []
xmin =1700
for y in range(1000, 1900, 6):
    line = fU.data[0, y, xmin:]
    I = np.where(line==0)[0]
#    I = np.where(line==0)
    if len(I)>0:
        for i in I:
            if np.all(line[i:i+10]==0):
                landlon.append(fU.grid.lon[y, xmin+i-1])
                landlat.append(fU.grid.lat[y, xmin+i-1])
                break

galapagosmask = np.zeros_like(fU.data[0, :, :])
galapagos_extent = [-91.8, -89, -1.4, 0.7]
for x in range(2000, 2500):
    for y in range(1300, 1600):
        if (fU.grid.lon[y, x] >= galapagos_extent[0] and fU.grid.lon[y, x] <= galapagos_extent[1] and
            fU.grid.lat[y, x] >= galapagos_extent[2] and fU.grid.lat[y, x] <= galapagos_extent[3]):
            galapagosmask[y, x] = 1
fieldset.add_field(Field('galapagosmask', galapagosmask, grid=fU.grid, 
                         mesh='spherical', interp_method='nearest'))

def SampleGalapagos(fieldset, particle, time):
    if fieldset.galapagosmask[time, particle.depth, particle.lat, particle.lon] == 1:
        particle.visitedgalapagos = 1

def Age(fieldset, particle, time):
    particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > 5*365*86400:
        particle.delete()

class GalapagosParticle(JITParticle):
    visitedgalapagos = Variable('visitedgalapagos', initial=0.)
    age = Variable('age', initial = 0.)

pset = ParticleSet(fieldset=fieldset, pclass=GalapagosParticle, lon=landlon, lat=landlat, 
                   time=fU.grid.time[0], repeatdt=delta(days=5))
outfile = pset.ParticleFile(name=fname, outputdt=delta(days=1))

pset.execute(AdvectionRK4+pset.Kernel(SampleGalapagos)+Age, dt=delta(hours=1), output_file=outfile)

outfile.close()
