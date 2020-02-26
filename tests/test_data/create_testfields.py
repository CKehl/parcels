from parcels import FieldSet
import numpy as np
import math
try:
    from parcels.tools import perlin2d as PERLIN
except:
    PERLIN = None
noctaves=4
perlinres=(32,8)
shapescale=(1,1)
perlin_persistence=0.3
scalefac = 2.0


def generate_testfieldset(xdim, ydim, zdim, tdim):
    lon = np.linspace(0., 2., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    depth = np.linspace(0., 0.5, zdim, dtype=np.float32)
    time = np.linspace(0., tdim, tdim, dtype=np.float64)
    U = np.ones((xdim, ydim, zdim, tdim), dtype=np.float32)
    V = np.zeros((xdim, ydim, zdim, tdim), dtype=np.float32)
    P = 2.*np.ones((xdim, ydim, zdim, tdim), dtype=np.float32)
    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)
    fieldset.write('testfields')

def generate_perlin_testfield():
    img_shape = (int(math.pow(2,noctaves))*perlinres[0]*shapescale[0], int(math.pow(2,noctaves))*perlinres[1]*shapescale[1])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(-180.0, 180.0, img_shape[0], dtype=np.float32)
    lat = np.linspace(-90.0, 90.0, img_shape[1], dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    if PERLIN is not None:
        U = PERLIN.generate_fractal_noise_2d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
        V = PERLIN.generate_fractal_noise_2d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    else:
        U = np.ones(img_shape, dtype=np.float32)*scalefac
        V = np.ones(img_shape, dtype=np.float32)*scalefac
    U = np.transpose(U, (1,0))
    U = np.expand_dims(U,0)
    V = np.transpose(V, (1,0))
    V = np.expand_dims(V,0)
    data = {'U': U, 'V': V}
    dimensions = {'time': np.zeros(1, dtype=np.float64), 'lon': lon, 'lat': lat}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=False)
    fieldset.write("perlinfields")

if __name__ == "__main__":
    generate_testfieldset(xdim=5, ydim=3, zdim=2, tdim=15)
    generate_perlin_testfield()
