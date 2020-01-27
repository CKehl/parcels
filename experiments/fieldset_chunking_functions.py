def chunk_setup(self):
    if isinstance(self.data, da.core.Array):
        chunks = self.data.chunks
        self.nchunks = self.data.numblocks
        npartitions = 1
        for n in self.nchunks[1:]:
            npartitions *= n
    else:
        chunks = tuple((t,) for t in self.data.shape)
        self.nchunks = (1,) * len(self.data.shape)
        npartitions = 1

    self.data_chunks = [None] * npartitions
    self.c_data_chunks = [None] * npartitions

    self.grid.load_chunk = np.zeros(npartitions, dtype=c_int)

    # self.grid.chunk_info format: number of dimensions (without tdim); number of chunks per dimensions;
    #                         chunksizes (the 0th dim sizes for all chunk of dim[0], then so on for next dims
    self.grid.chunk_info = [[len(self.nchunks) - 1], list(self.nchunks[1:]),
                            sum(list(list(ci) for ci in chunks[1:]), [])]
    self.grid.chunk_info = sum(self.grid.chunk_info, [])
    self.chunk_set = True


def chunk_data(self):
    if not self.chunk_set:
        self.chunk_setup()
    # self.grid.load_chunk code:
    # 0: not loaded
    # 1: was asked to load by kernel in JIT
    # 2: is loaded and was touched last C call
    # 3: is loaded
    if isinstance(self.data, da.core.Array):
        for block_id in range(len(self.grid.load_chunk)):
            if self.grid.load_chunk[block_id] == 1 or self.grid.load_chunk[block_id] > 1 and self.data_chunks[
                block_id] is None:
                block = self.get_block(block_id)
                self.data_chunks[block_id] = np.array(self.data.blocks[(slice(self.grid.tdim),) + block])
            elif self.grid.load_chunk[block_id] == 0:
                self.data_chunks[block_id] = None
                self.c_data_chunks[block_id] = None
    else:
        self.grid.load_chunk[0] = 2
        self.data_chunks[0] = self.data


# well - as tindex is not really referring to the actual time index (that is g.ti, resp. g.ti+tindex), it may better be called 't_offset'
# this 'computeTimeChunk' returns the concatenated [virtual] buffer of the [3] data chunks
def computeTimeChunk(self, data, tindex):
    g = self.grid
    timestamp = self.timestamps
    if timestamp is not None:
        summedlen = np.cumsum([len(ls) for ls in self.timestamps])
        if g.ti + tindex >= summedlen[
            -1]:  # basically: if adding the timestep exceeds the number of total timesteps, wrap the index around t_start
            ti = g.ti + tindex - summedlen[-1]
        else:
            ti = g.ti + tindex
        timestamp = self.timestamps[np.where(ti < summedlen)[0][0]]

    filebuffer = NetcdfFileBuffer(self.dataFiles[g.ti + tindex], self.dimensions, self.indices,
                                  self.netcdf_engine, timestamp=timestamp,
                                  interp_method=self.interp_method,
                                  data_full_zdim=self.data_full_zdim,
                                  field_chunksize=self.field_chunksize)
    filebuffer.__enter__()
    time_data = filebuffer.time
    time_data = g.time_origin.reltime(time_data)
    filebuffer.ti = (time_data <= g.time[tindex]).argmin() - 1
    if self.netcdf_engine != 'xarray':
        filebuffer.name = filebuffer.parse_name(self.filebuffername)
    buffer_data = filebuffer.data
    lib = np if isinstance(buffer_data, np.ndarray) else da
    if len(buffer_data.shape) == 2:
        buffer_data = lib.reshape(buffer_data, sum(((1, 1), buffer_data.shape), ()))
    elif len(buffer_data.shape) == 3 and g.zdim > 1:
        buffer_data = lib.reshape(buffer_data, sum(((1,), buffer_data.shape), ()))
    elif len(buffer_data.shape) == 3:
        buffer_data = lib.reshape(buffer_data, sum(((buffer_data.shape[0], 1,), buffer_data.shape[1:]), ()))
    data = self.data_concatenate(data, buffer_data, tindex)
    self.filebuffers[tindex] = filebuffer
    return data