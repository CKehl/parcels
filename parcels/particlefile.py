"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np
import netCDF4
from datetime import timedelta as delta
from parcels.tools.loggers import logger
import os
from tempfile import gettempdir
import psutil
from parcels.tools.error import ErrorCode
try:
    from parcels._version import version as parcels_version
except:
    raise EnvironmentError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')
try:
    from os import getuid
except:
    # Windows does not have getuid(), so define to simply return 'tmp'
    def getuid():
        return 'tmp'


__all__ = ['ParticleFile']


class ParticleFile(object):
    """Initialise trajectory output.
    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param outputdt: Interval which dictates the update frequency of file output
                     while ParticleFile is given as an argument of ParticleSet.execute()
                     It is either a timedelta object or a positive double.
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    """

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False):

        self.name = name
        self.write_ondelete = write_ondelete
        self.outputdt = outputdt
        self.lasttime_written = None  # variable to check if time has been written already

        self.dataset = None
        self.metadata = {}
        self.particleset = particleset
        self.var_names = []
        self.var_names_once = []
        for v in self.particleset.ptype.variables:
            if v.to_write == 'once':
                self.var_names_once += [v.name]
            elif v.to_write is True:
                self.var_names += [v.name]
        if len(self.var_names_once) > 0:
            self.written_once = []
            self.file_list_once = []

        self.file_list = []
        self.time_written = []
        self.maxid_written = -1
        self.dataset_open = True

        self.npy_path = os.path.join(gettempdir(), "parcels-%s" % getuid(), "out")
        self.delete_npyfiles()

    def open_dataset(self, data_shape):
        """Initialise NetCDF4.Dataset for trajectory output.
        The output follows the format outlined in the Discrete Sampling Geometries
        section of the CF-conventions:
        http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries
        The current implementation is based on the NCEI template:
        http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl
        :param data_shape: shape of the variables in the NetCDF4 file
        """
        extension = os.path.splitext(str(self.name))[1]
        fname = self.name if extension in ['.nc', '.nc4'] else "%s.nc" % self.name
        if os.path.exists(str(fname)):
            os.system("rm -rf " + str(fname))
        self.dataset = netCDF4.Dataset(fname, "w", format="NETCDF4")
        self.dataset.createDimension("obs", data_shape[1])
        self.dataset.createDimension("traj", data_shape[0])
        coords = ("traj", "obs")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"
        self.dataset.parcels_version = parcels_version
        self.dataset.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh

        # Create ID variable according to CF conventions
        self.id = self.dataset.createVariable("trajectory", "i4", coords, fill_value=-2147483647, chunksizes=data_shape)
        self.id.long_name = "Unique identifier for each particle"
        self.id.cf_role = "trajectory_id"

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan, chunksizes=data_shape)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if self.particleset.time_origin.calendar is None:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(self.particleset.time_origin)
            self.time.calendar = self.particleset.time_origin.calendar
        self.time.axis = "T"

        self.lat = self.dataset.createVariable("lat", "f4", coords, fill_value=np.nan, chunksizes=data_shape)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", "f4", coords, fill_value=np.nan, chunksizes=data_shape)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", "f4", coords, fill_value=np.nan, chunksizes=data_shape)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        for v in self.particleset.ptype.variables:
            if v.to_write and v.name not in ['time', 'lat', 'lon', 'z', 'id']:
                if v.to_write is True:
                    setattr(self, v.name, self.dataset.createVariable(v.name, "f4", coords, fill_value=np.nan, chunksizes=data_shape))
                elif v.to_write == 'once':
                    setattr(self, v.name, self.dataset.createVariable(v.name, "f4", "traj", fill_value=np.nan, chunksizes=[data_shape[0]]))
                getattr(self, v.name).long_name = ""
                getattr(self, v.name).standard_name = v.name
                getattr(self, v.name).units = "unknown"

        for name, message in self.metadata.items():
            setattr(self.dataset, name, message)

    def __del__(self):
        if self.dataset_open:
            self.close()

    def close(self):
        """Close the ParticleFile object by exporting and then deleting
        the temporary npy files"""
        self.export()
        self.delete_npyfiles()
        self.dataset.close()
        self.dataset_open = False

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`
        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        if self.dataset is None:
            self.metadata[name] = message
        else:
            setattr(self.dataset, name, message)

    def write(self, pset, time, deleted_only=False):
        """Write all data from one time step to a temporary npy-file
        using a python dictionary. The data is saved in the folder 'out'.
        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        """
        if isinstance(time, delta):
            time = time.total_seconds()
        if self.lasttime_written != time and \
           (self.write_ondelete is False or deleted_only is True):
            if pset.size > 0:
                data = {}
                for var in self.var_names:
                    data[var] = np.nan * np.zeros(len(pset))

                i = 0
                for p in pset:
                    if p.dt*p.time <= p.dt*time:
                        for var in self.var_names:
                            data[var][i] = getattr(p, var)
                        if p.state != ErrorCode.Delete and not np.allclose(p.time, time):
                            logger.warning_once('time argument in pfile.write() is %g, but a particle has time %g.' % (time, p.time))
                        self.maxid_written = np.max([self.maxid_written, p.id])
                        i += 1

                if not os.path.exists(self.npy_path):
                    os.mkdir(self.npy_path)

                save_ind = np.isfinite(data["id"])
                for key in self.var_names:
                    data[key] = data[key][save_ind]

                tmpfilename = os.path.join(self.npy_path, str(len(self.file_list)+1))
                np.save(tmpfilename, data)
                self.file_list.append(tmpfilename+".npy")
                if time not in self.time_written:
                    self.time_written.append(time)

                if len(self.var_names_once) > 0:
                    first_write = [p for p in pset if (p.id not in self.written_once) and (p.dt * p.time <= p.dt * time or np.isnan(p.dt))]
                    data_once = {}
                    data_once['id'] = np.nan * np.zeros(len(first_write))
                    for var in self.var_names_once:
                        data_once[var] = np.nan * np.zeros(len(first_write))

                    i = 0
                    for p in first_write:
                        self.written_once.append(p.id)
                        data_once['id'][i] = p.id
                        for var in self.var_names_once:
                            data_once[var][i] = getattr(p, var)
                        i += 1

                    tmpfilename = os.path.join(self.npy_path, str(len(self.file_list)+1)+'_once')
                    np.save(tmpfilename, data_once)
                    self.file_list_once.append(tmpfilename+".npy")

            else:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)

            if not deleted_only:
                self.lasttime_written = time

    def read_npy(self, file_list, time_steps, var):
        """Read NPY-files for one variable using a loop over all files.
        :param file_list: List that  contains all file names in the output directory
        :param time_steps: Number of time steps that were written in out directory
        :param var: name of the variable to read
        """

        data = np.nan * np.zeros((self.maxid_written+1, time_steps))
        time_index = np.zeros(self.maxid_written+1, dtype=int)
        t_ind_used = np.zeros(time_steps, dtype=int)

        # loop over all files
        for npyfile in file_list:
            data_dict = np.load(npyfile).item()
            id_ind = np.array(data_dict["id"], dtype=int)
            t_ind = time_index[id_ind] if 'once' not in file_list[0] else 0
            t_ind_used[t_ind] = 1
            data[id_ind, t_ind] = data_dict[var]
            time_index[id_ind] = time_index[id_ind] + 1

        # remove rows and columns that are completely filled with nan values
        tmp = data[time_index > 0, :]
        return tmp[:, t_ind_used == 1]

    def export(self):
        """Exports outputs in temporary NPY-files to NetCDF file"""
        memory_estimate_total = self.maxid_written+1 * len(self.time_written) * 8
        if memory_estimate_total > 0.9*psutil.virtual_memory().available:
            raise MemoryError("Not enough memory available for export. npy files are stored at %s", self.npy_path)

        for var in self.var_names:
            data = self.read_npy(self.file_list, len(self.time_written), var)
            if var == self.var_names[0]:
                self.open_dataset(data.shape)
            varout = 'z' if var == 'depth' else var
            getattr(self, varout)[:, :] = data

        if len(self.var_names_once) > 0:
            for var in self.var_names_once:
                getattr(self, var)[:] = self.read_npy(self.file_list_once, 1, var)

    def delete_npyfiles(self):
        """Deleted all temporary npy files"""
        if os.path.exists(self.npy_path):
            os.system("rm -rf "+self.npy_path)
