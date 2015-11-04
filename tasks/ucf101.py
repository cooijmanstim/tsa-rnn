import os, logging
import numpy as np

import tasks
import datasets

logger = logging.getLogger(__name__)

def _canonicalize(self, data):
    x, y = data
    x = np.asarray(x)
    # move channel axis to just after batch axis
    x = np.rollaxis(x, x.ndim - 1, 1)
    x_shape = np.tile([x.shape[2:]], (x.shape[0], 1))
    return (x.astype(np.float32),
            x_shape.astype(np.float32),
            y.astype(np.uint8))

def _center(self, data):
    return data

class Task(tasks.Classification):
    name = "ucf101"
    canonicalize = _canonicalize
    center = _center

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.n_channels = 3
        self.n_classes = 101
        for key in ("data_subsample data_random_subsample data_nb_frames "
                    "data_input_size data_crop_size data_crop_type translate_labels".split()):
            setattr(self, key, kwargs[key])

    def load_datasets(self):
        return dict(
            train=JpegHDF5Dataset("train", name="jpeg_data.hdf5", load_in_memory=True),
            # FIXME: validation set
            valid=JpegHDF5Dataset("train", name="jpeg_data.hdf5", load_in_memory=True),
            test=JpegHDF5Dataset("test", name="jpeg_data.hdf5", load_in_memory=True))

    def get_scheme(self, which_set, shuffle=True, monitor=False, num_examples=None):
        return HDF5ShuffledScheme(
            self.datasets[which_set].video_indexes,
            examples=num_examples,
            batch_size=self.batch_size,
            random_sample=True,
            f_subsample=1 if monitor else self.data_subsample,
            r_subsample=False if monitor else self.data_random_subsample,
            frames_per_video=self.data_nb_frames)

    def apply_default_transformers(self, stream, monitor):
        return JpegHDF5Transformer(
            input_size=self.data_input_size, crop_size=self.data_crop_size,
            translate_labels=self.translate_labels,
            nb_frames=self.data_nb_frames,
            crop_type='center' if monitor else self.data_crop_type,
            flip='noflip' if monitor else 'random',
            data_stream=stream)

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and (which_set == "train" or which_set == "valid"):
            return 1000
        else:
            return self.datasets[which_set].num_video_examples

    def compute_batch_mean(self, x, x_shape):
        # average over time first
        time = 2
        mean_frame = x.sum(axis=time, keepdims=True)
        mean_frame /= x_shape[:, np.newaxis, [time], np.newaxis, np.newaxis]
        return mean_frame.mean(axis=0, keepdims=True)


import os.path
import cPickle as pkl
import h5py

from itertools import product
from collections import defaultdict, OrderedDict

import six
from six.moves import zip, range

from fuel.datasets import Dataset
from fuel.utils import do_not_pickle_attributes

@do_not_pickle_attributes('data_sources', 'external_file_handle',
                          'source_shapes', 'subsets')
class H5PYDataset(Dataset):
    """An h5py-fueled HDF5 dataset.

    This dataset class assumes a particular file layout:

    * Data sources reside in the root group, and their names define the
      source names.
    * Data sources are not explicitly split. Instead, splits are defined
      in the `split` attribute of the root group. It's expected to be a
      1D numpy array of compound ``dtype`` with seven fields, organized as
      follows:

      1. ``split`` : string identifier for the split name
      2. ``source`` : string identifier for the source name
      3. ``start`` : start index (inclusive) of the split in the source
         array, used if ``indices`` is a null reference.
      4. ``stop`` : stop index (exclusive) of the split in the source
         array, used if ``indices`` is a null reference.
      5. ``indices`` : h5py.Reference, reference to a dataset containing
         subset indices for this split/source pair. If it's a null
         reference, ``start`` and ``stop`` are used.
      6. ``available`` : boolean, ``False`` is this split is not available
         for this source
      7. ``comment`` : comment string

    Parameters
    ----------
    file_or_path : :class:`h5py.File` or str
        HDF5 file handle, or path to the HDF5 file.
    which_sets : iterable of str
        Which split(s) to use. If one than more split is requested,
        the provided sources will be the intersection of provided
        sources for these splits. **Note: for all splits that are
        specified as a list of indices, those indices will get sorted
        no matter what.**
    subset : {slice, list of int}, optional
        Which subset of data to use *within the context of the split*.
        Can be either a slice or a list of indices. Defaults to `None`,
        in which case the whole split is used.
    load_in_memory : bool, optional
        Whether to load the data in main memory. Defaults to `False`.
    driver : str, optional
        Low-level driver to use. Defaults to `None`. See h5py
        documentation for a complete list of available options.
    sort_indices : bool, optional
        HDF5 doesn't support fancy indexing with an unsorted list of
        indices. In order to allow that, the dataset can sort the list
        of indices, access the data in sorted order and shuffle back
        the data in the unsorted order. Setting this flag to `True`
        (the default) will activate this behaviour. For greater
        performance, set this flag to `False`. Note that in that case,
        it is the user's responsibility to make sure that indices are
        ordered.

    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset will provide when queried for data.
    provides_sources : tuple of strings
        The sources this dataset *is able to* provide for the requested
        split.
    example_iteration_scheme : :class:`.IterationScheme` or ``None``
        The iteration scheme the class uses in order to produce a stream of
        examples.
    vlen_sources : tuple of strings
        All sources provided by this dataset which have variable length.
    default_axis_labels : dict mapping string to tuple of strings
        Maps all sources provided by this dataset to their axis labels.

    """
    interface_version = '0.3'
    _ref_counts = defaultdict(int)
    _file_handles = {}

    def __init__(self, file_or_path, which_sets, subset=None,
                 load_in_memory=False, driver=None, sort_indices=True,
                 **kwargs):
        if isinstance(file_or_path, h5py.File):
            self.path = file_or_path.filename
            self.external_file_handle = file_or_path
        else:
            self.path = file_or_path
            self.external_file_handle = None
        which_sets_invalid_value = (
            isinstance(which_sets, six.string_types) or
            not all(isinstance(s, six.string_types) for s in which_sets))
        if which_sets_invalid_value:
            raise ValueError('`which_sets` should be an iterable of strings')
        self.which_sets = which_sets
        self._subset_template = subset if subset else slice(None)
        self.load_in_memory = load_in_memory
        self.driver = driver
        self.sort_indices = sort_indices

        self._parse_dataset_info()

        kwargs.setdefault('axis_labels', self.default_axis_labels)
        super(H5PYDataset, self).__init__(**kwargs)

    def _parse_dataset_info(self):
        """Parses information related to the HDF5 interface.

        In addition to verifying that the `self.which_sets` split is
        available, this method sets the following attributes:

        * `provides_sources`
        * `vlen_sources`
        * `default_axis_labels`

        """
        self._out_of_memory_open()
        handle = self._file_handle
        available_splits = self.get_all_splits(handle)
        which_sets = self.which_sets
        provides_sources = None
        for split in which_sets:
            if split not in available_splits:
                raise ValueError(
                    "'{}' split is not provided by this ".format(split) +
                    "dataset. Available splits are " +
                    "{}.".format(available_splits))
            split_provides_sources = set(
                self.get_provided_sources(handle, split))
            if provides_sources:
                provides_sources &= split_provides_sources
            else:
                provides_sources = split_provides_sources
        self.provides_sources = tuple(sorted(provides_sources))
        self.vlen_sources = self.get_vlen_sources(handle)
        self.default_axis_labels = self.get_axis_labels(handle)
        self._out_of_memory_close()

    @staticmethod
    def create_split_array(split_dict):
        """Create a valid array for the `split` attribute of the root node.

        Parameters
        ----------
        split_dict : dict
            Maps split names to dict. Those dict map source names to
            tuples. Those tuples contain two, three or four elements:
            the start index, the stop index, (optionally) subset
            indices and (optionally) a comment.  If a particular
            split/source combination isn't present in the split dict,
            it's considered as unavailable and the `available` element
            will be set to `False` it its split array entry.

        """
        # Determine maximum split, source and string lengths
        split_len = max(len(split) for split in split_dict)
        sources = set()
        comment_len = 1
        for split in split_dict.values():
            sources |= set(split.keys())
            for val in split.values():
                if len(val) == 4:
                    comment_len = max([comment_len, len(val[-1])])
        sources = sorted(list(sources))
        source_len = max(len(source) for source in sources)

        # Instantiate empty split array
        split_array = numpy.empty(
            len(split_dict) * len(sources),
            dtype=numpy.dtype([
                ('split', 'a', split_len),
                ('source', 'a', source_len),
                ('start', numpy.int64, 1),
                ('stop', numpy.int64, 1),
                ('indices', h5py.special_dtype(ref=h5py.Reference)),
                ('available', numpy.bool, 1),
                ('comment', 'a', comment_len)]))

        # Fill split array
        for i, (split, source) in enumerate(product(split_dict, sources)):
            if source in split_dict[split]:
                start, stop = split_dict[split][source][:2]
                available = True
                indices = h5py.Reference()
                # Workaround for bug when pickling an empty string
                comment = '.'
                if len(split_dict[split][source]) > 2:
                    indices = split_dict[split][source][2]
                if len(split_dict[split][source]) > 3:
                    comment = split_dict[split][source][3]
                    if not comment:
                        comment = '.'
            else:
                (start, stop, indices, available, comment) = (
                    0, 0, h5py.Reference(), False, '.')
            # Workaround for H5PY being unable to store unicode type
            split_array[i]['split'] = split.encode('utf8')
            split_array[i]['source'] = source.encode('utf8')
            split_array[i]['start'] = start
            split_array[i]['stop'] = stop
            split_array[i]['indices'] = indices
            split_array[i]['available'] = available
            split_array[i]['comment'] = comment.encode('utf8')

        return split_array

    @staticmethod
    def get_all_splits(h5file):
        """Returns the names of all splits of an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        available_splits : tuple of str
            Names of all splits in ``h5file``.

        """
        available_splits = tuple(
            set(row['split'].decode('utf8') for row in h5file.attrs['split']))
        return available_splits

    @staticmethod
    def get_all_sources(h5file):
        """Returns the names of all sources of an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        all_sources : tuple of str
            Names of all sources in ``h5file``.

        """
        all_sources = tuple(
            set(row['source'].decode('utf8') for row in h5file.attrs['split']))
        return all_sources

    @staticmethod
    def get_provided_sources(h5file, split):
        """Returns the sources provided by a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        provided_sources : tuple of str
            Names of sources provided by ``split`` in ``h5file``.

        """
        provided_sources = tuple(
            row['source'].decode('utf8') for row in h5file.attrs['split']
            if row['split'].decode('utf8') == split and row['available'])
        return provided_sources

    @staticmethod
    def get_vlen_sources(h5file):
        """Returns the names of variable-length sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        vlen_sources : tuple of str
            Names of all variable-length sources in ``h5file``.

        """
        vlen_sources = []
        for source_name in H5PYDataset.get_all_sources(h5file):
            source = h5file[source_name]
            if len(source.dims) > 0 and 'shapes' in source.dims[0]:
                if len(source.dims) > 1:
                    raise ValueError('Variable-length sources must have only '
                                     'one dimension.')
                vlen_sources.append(source_name)
        return vlen_sources

    @staticmethod
    def get_axis_labels(h5file):
        """Returns axis labels for all sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        axis_labels : dict
            Maps source names to a tuple of str representing the axis
            labels.

        """
        axis_labels = {}
        vlen_sources = H5PYDataset.get_vlen_sources(h5file)
        for source_name in H5PYDataset.get_all_sources(h5file):
            if source_name in vlen_sources:
                axis_labels[source_name] = (
                    (h5file[source_name].dims[0].label,) +
                    tuple(label.decode('utf8') for label in
                          h5file[source_name].dims[0]['shape_labels']))
            else:
                axis_labels[source_name] = tuple(
                    dim.label for dim in h5file[source_name].dims)
        return axis_labels

    @staticmethod
    def get_start_stop(h5file, split):
        """Returns start and stop indices for sources of a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        start_stop : dict
            Maps source names to a tuple of ``(start, stop)`` indices.

        """
        start_stop = {}
        for row in h5file.attrs['split']:
            if row['split'].decode('utf8') == split:
                source = row['source'].decode('utf8')
                start_stop[source] = (row['start'], row['stop'])
        return start_stop

    @staticmethod
    def get_indices(h5file, split):
        """Returns subset indices for sources of a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        indices : dict
            Maps source names to a list of indices. Note that only
            sources for which indices are specified appear in this dict.

        """
        indices = {}
        try:
            for row in h5file.attrs['split']:
                if row['split'].decode('utf8') == split:
                    source = row['source'].decode('utf8')
                    if row['indices']:
                        indices[source] = h5file[row['indices']]
        except IndexError:
            # old-style HDF5 files didn't have indices, and split dict
            # rows would be tuples
            pass
        return indices

    @staticmethod
    def unsorted_fancy_index(request, indexable):
        """Safe unsorted list indexing.

        Some objects, such as h5py datasets, only support list indexing
        if the list is sorted.

        This static method adds support for unsorted list indexing by
        sorting the requested indices, accessing the corresponding
        elements and re-shuffling the result.

        Parameters
        ----------
        request : list of int
            Unsorted list of example indices.
        indexable : any fancy-indexable object
            Indexable we'd like to do unsorted fancy indexing on.

        """
        if len(request) > 1:
            indices = numpy.argsort(request)
            data = numpy.empty(shape=(len(request),) + indexable.shape[1:],
                               dtype=indexable.dtype)
            data[indices] = indexable[numpy.array(request)[indices], ...]
        else:
            data = indexable[request]
        return data

    def load(self):
        # If the dataset is unpickled, it makes no sense to have an external
        # file handle. However, since `load` is also called during the lifetime
        # of a dataset (e.g. if load_in_memory = True), we don't want to
        # accidentally overwrite the reference to a potential external file
        # handle, hence this check.
        if not hasattr(self, '_external_file_handle'):
            self.external_file_handle = None

        self._out_of_memory_open()
        handle = self._file_handle

        # Load subset slices / indices
        subsets = []
        if len(self.which_sets) > 1:
            indices = defaultdict(list)
            for split in self.which_sets:
                split_start_stop = self.get_start_stop(handle, split)
                split_indices = self.get_indices(handle, split)
                for source_name in self.sources:
                    if source_name in split_indices:
                        ind = split_indices[source_name]
                    else:
                        ind = list(range(*split_start_stop[source_name]))
                    indices[source_name] = sorted(
                        set(indices[source_name] + ind))
        else:
            start_stop = self.get_start_stop(handle, self.which_sets[0])
            indices = self.get_indices(handle, self.which_sets[0])
        num_examples = None
        for source_name in self.sources:
            subset = self._subset_template
            # If subset has a step greater than 1, we convert it to a list,
            # otherwise we won't be able to take that subset within the context
            # of a split defined by a slice.
            if hasattr(subset, 'step') and subset.step not in (1, None):
                subset = list(range(len(handle[source_name])))[subset]
            self._subset_template = subset
            if source_name in indices:
                source_subset = indices[source_name]
            else:
                source_subset = slice(*start_stop[source_name])
            if hasattr(subset, 'step') and hasattr(source_subset, 'step'):
                subset = slice(
                    source_subset.start
                    if subset.start is None else subset.start,
                    source_subset.stop
                    if subset.stop is None else subset.stop,
                    subset.step)
                subsets.append(subset)
                subset_num_examples = subset.stop - subset.start
            else:
                if hasattr(source_subset, 'step'):
                    source_subset = numpy.arange(
                        source_subset.start, source_subset.stop)
                subset = source_subset[subset]
                subsets.append(subset)
                subset_num_examples = len(subset)
            if num_examples is None:
                num_examples = subset_num_examples
            if num_examples != subset_num_examples:
                raise ValueError("sources have different lengths")
        self.subsets = subsets

        # Load data sources and source shapes
        if self.load_in_memory:
            data_sources = []
            source_shapes = []
            for source_name, subset in zip(self.sources, self.subsets):
                if hasattr(subset, 'step'):
                    data_source = handle[source_name][subset]
                else:
                    data_source = handle[source_name][list(subset)]
                data_sources.append(data_source)
                if source_name in self.vlen_sources:
                    shapes = handle[source_name].dims[0]['shapes'][subset]
                else:
                    shapes = None
                source_shapes.append(shapes)
            self.data_sources = tuple(data_sources)
            self.source_shapes = tuple(source_shapes)
        else:
            self.data_sources = None
            self.source_shapes = None

        self._out_of_memory_close()

    @property
    def num_examples(self):
        if hasattr(self.subsets[0], 'step'):
            return self.subsets[0].stop - self.subsets[0].start
        else:
            return len(self.subsets[0])

    def open(self):
        return None if self.load_in_memory else self._out_of_memory_open()

    def _out_of_memory_open(self):
        if not self.external_file_handle:
            if self.path not in self._file_handles:
                handle = h5py.File(
                    name=self.path, mode="r", driver=self.driver)
                self._file_handles[self.path] = handle
            self._ref_counts[self.path] += 1

    def close(self, state):
        if not self.load_in_memory:
            self._out_of_memory_close()

    def _out_of_memory_close(self):
        if not self.external_file_handle:
            self._ref_counts[self.path] -= 1
            if not self._ref_counts[self.path]:
                del self._ref_counts[self.path]
                self._file_handles[self.path].close()
                del self._file_handles[self.path]

    @property
    def _file_handle(self):
        if self.external_file_handle:
            return self.external_file_handle
        elif self.path in self._file_handles:
            return self._file_handles[self.path]
        else:
            raise IOError('no open handle for file {}'.format(self.path))

    def get_data(self, state=None, request=None):
        if self.load_in_memory:
            data, shapes = self._in_memory_get_data(state, request)
        else:
            data, shapes = self._out_of_memory_get_data(state, request)
        for i in range(len(data)):
            if shapes[i] is not None:
                for j in range(len(data[i])):
                    data[i][j] = data[i][j].reshape(shapes[i][j])
        return tuple(data)

    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        data = [data_source[request] for data_source in self.data_sources]
        shapes = [shape[request] if shape is not None else None
                  for shape in self.source_shapes]
        return data, shapes

    def _out_of_memory_get_data(self, state=None, request=None):
        if not isinstance(request, (slice, list)):
            raise ValueError()
        data = []
        shapes = []
        handle = self._file_handle
        for source_name, subset in zip(self.sources, self.subsets):
            if hasattr(subset, 'step'):
                if hasattr(request, 'step'):
                    req = slice(request.start + subset.start,
                                request.stop + subset.start, request.step)
                else:
                    req = [index + subset.start for index in request]
            else:
                req = iterable_fancy_indexing(subset, request)
            if hasattr(req, 'step'):
                val = handle[source_name][req]
                if source_name in self.vlen_sources:
                    shape = handle[source_name].dims[0]['shapes'][req]
                else:
                    shape = None
            else:
                if self.sort_indices:
                    val = self.unsorted_fancy_index(req, handle[source_name])
                    if source_name in self.vlen_sources:
                        shape = self.unsorted_fancy_index(
                            req, handle[source_name].dims[0]['shapes'])
                    else:
                        shape = None
                else:
                    val = handle[source_name][req]
                    if source_name in self.vlen_sources:
                        shape = handle[source_name].dims[0]['shapes'][req]
                    else:
                        shape = None
            data.append(val)
            shapes.append(shape)
        return data, shapes

class JpegHDF5Dataset(H5PYDataset):
    def __init__(self,
                 split="train",
                 name="jpeg_data.hdf5",
                 signature='UCF101',
                 load_in_memory=True):
        data_path = os.path.join(os.environ[signature], name)
        data_file = h5py.File(data_path, 'r')

        self.video_indexes = np.array(data_file["video_indexes"][split])
        self.num_video_examples = len(self.video_indexes) - 1

        super(JpegHDF5Dataset, self).__init__(data_file, which_sets=(split,), load_in_memory=load_in_memory)
        data_file.close()

import PIL.Image as Image
import time

from StringIO import StringIO

from fuel import config
from fuel.transformers import Transformer
class JpegHDF5Transformer(Transformer) :
    produces_examples = False

    """
    Decode jpeg and perform spatial crop if needed

    if input_size == crop_size, no crop is done

    input_size: spatially resize the input to that size
    crop_size: take a crop of that size out of the inputs
    nb_channels: number of channels of the inputs
    flip: in 'random', 'flip', 'noflip' activate flips data augmentation
    swap_rgb: Swap rgb pixel using in [2 1 0] order
    crop_type: random, corners or center type of cropping
    scale: pixel values are scale into the range [0, scale]
    nb_frames: maximum number of frame (will be zero padded)

    """
    def __init__(self,
                 input_size=(240, 320),
                 crop_size=(224, 224),
                 nchannels=3,
                 flip='random',
                 resize=True,
                 mean=None,
                 swap_rgb=False,
                 crop_type='random',
                 scale=1.,
                 translate_labels = False,
                 nb_frames= 25,
                 *args, **kwargs):

        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)
        super(JpegHDF5Transformer, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.crop_size = crop_size
        self.nchannels = nchannels
        self.swap_rgb = swap_rgb
        self.flip = flip
        self.nb_frames = nb_frames
        self.resize = resize
        self.scale = scale
        self.mean = mean
        self.translate_labels = translate_labels
        self.data_sources = ('targets', 'images')

        ### multi-scale
        self.scales =  [256, 224, 192, 168]


        ### Crop coordinate
        self.crop_type = crop_type
        self.centers = np.array(input_size) / 2.0
        self.centers_crop = (self.centers[0] - self.crop_size[0] / 2.0,
                             self.centers[1] - self.crop_size[1] / 2.0)
        self.corners = []
        self.corners.append((0, 0))
        self.corners.append((0, self.input_size[1] - self.crop_size[1]))
        self.corners.append((self.input_size[0] - self.crop_size[0], 0))
        self.corners.append((self.input_size[0] - self.crop_size[0],
                             self.input_size[1] - self.crop_size[1]))
        self.corners.append(self.centers_crop)

        ### Value checks
        assert self.crop_type in  ['center', 'corners', 'random',
                                   'upleft', 'downleft',
                                   'upright', 'downright',
                                   'random_multiscale',
                                   'corners_multiscale']

        assert self.flip in ['random', 'flip', 'noflip']
        assert self.crop_size[0] <= self.input_size[0]
        assert self.crop_size[1] <= self.input_size[1]
        assert self.nchannels >= 1


    def multiscale_crop(self):
        scale_x = self.rng.randint(0, len(self.scales))
        scale_y = self.rng.randint(0, len(self.scales))
        crop_size = (self.scales[scale_x], self.scales[scale_y])

        centers_crop = (self.centers[0] - crop_size[0] / 2.0,
                        self.centers[1] - crop_size[1] / 2.0)
        corners = []
        corners.append((0, 0))
        corners.append((0, self.input_size[1] - crop_size[1]))
        corners.append((self.input_size[0] - crop_size[0], 0))
        corners.append((self.input_size[0] - crop_size[0],
                             self.input_size[1] - crop_size[1]))
        corners.append(centers_crop)
        return corners, crop_size




    def get_crop_coord(self, crop_size, corners):
        x_start = 0
        y_start = 0


        corner_rng = self.rng.randint(0, 5)
        if ((self.crop_type == 'random' or
             self.crop_type == 'random_multiscale')):
            if crop_size[0] <= self.input_size[0]:
                if crop_size[0] == self.input_size[0]:
                    x_start = 0
                else:
                    x_start = self.rng.randint(0,
                                               self.input_size[0]-crop_size[0])
                if crop_size[1] == self.input_size[1]:
                    y_start = 0
                else:
                    y_start = self.rng.randint(0,
                                               self.input_size[1]-crop_size[1])
        elif ((self.crop_type == 'corners' or
               self.crop_type == 'corners_multiscale')):
            x_start = corners[corner_rng][0]
            y_start = corners[corner_rng][1]
        elif self.crop_type == 'upleft':
            x_start = corners[0][0]
            y_start = corners[0][1]
        elif self.crop_type == 'upright':
            x_start = corners[1][0]
            y_start = corners[1][1]
        elif self.crop_type == 'downleft':
            x_start = corners[2][0]
            y_start = corners[2][1]
        elif self.crop_type == 'downright':
            x_start = corners[3][0]
            y_start = corners[3][1]
        elif self.crop_type == 'center':
            x_start = corners[4][0]
            y_start = corners[4][1]
        else:
            raise ValueError
        return x_start, y_start

    def crop(self):
        if ((self.crop_type == 'random_multiscale' or
             self.crop_type == 'corners_multiscale')):
            corners, crop_size = self.multiscale_crop()
        else:
            corners, crop_size = self.corners, self.crop_size

        x_start, y_start = self.get_crop_coord(crop_size, corners)
        bbox = (int(y_start), int(x_start),
                int(y_start+crop_size[1]), int(x_start+crop_size[0]))
        return bbox


    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        timer = time.time()
        batch = next(self.child_epoch_iterator)
        images, labels = self.preprocess_data(batch)
        timers2 = time.time()
        return images, labels


    def preprocess_data(self, batch) :
        #in batch[0] are all the vidis. They are the same for each fpv elements
        #in batch[1] are all the frames. A group of fpv is one video

        fpv=self.nb_frames

        data_array = batch[0]
        num_videos = int(len(data_array)/fpv)
        x = np.zeros((num_videos, fpv,
                      self.crop_size[0], self.crop_size[1], self.nchannels),
                     dtype='float32')
        y = np.empty(num_videos, dtype='int64')
        for i in xrange(num_videos) :
            if self.translate_labels:
                y[i] = translate[batch[1][i*fpv]]
            else:
                y[i] = batch[1][i*fpv]
            do_flip = self.rng.rand(1)[0]
            bbox = self.crop()

            for j in xrange(fpv):
                data = data_array[i*fpv+j]
                #this data was stored in uint8
                data = StringIO(data.tostring())
                data.seek(0)
                img = Image.open(data)
                if (img.size[0] != self.input_size[1] and
                    img.size[1] != self.input_size[0]):
                    img = img.resize((int(self.input_size[1]),
                                      int(self.input_size[0])),
                                     Image.ANTIALIAS)
                img = img.crop(bbox)
                img = img.resize((int(self.crop_size[1]),
                                  int(self.crop_size[0])),
                                 Image.ANTIALIAS)
                # cv2.imshow('img', np.array(img))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                img = (np.array(img).astype(np.float32) / 255.0) * self.scale

                if self.nchannels == 1:
                    img = img[:, :, None]
                if self.swap_rgb and self.nchannels == 3:
                    img = img[:, :, [2, 1, 0]]
                x[i, j, :, :, :] = img[:, :, :]

                ### Flip
                if self.flip == 'flip' or (self.flip == 'random'
                                           and do_flip > 0.5):
                    new_image = np.empty_like(x[i, j, :, :, :])
                    for c in xrange(self.nchannels):
                        new_image[:,:,c] = np.fliplr(x[i, j, :, :, c])
                    x[i, j, :, :, :] = new_image

            #import cv2
            #cv2.imshow('img', x[i, 0, :, :, :])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        return (x, y)

translate = {
    67 : 52,
    9 : 93,
    71 : 82,
    30 : 76,
    8 : 24,
    47 : 0,
    87 : 66,
    51 : 100,
    95 : 60,
    93 : 58,
    32 : 15,
    74 : 95,
    68 : 4,
    97 : 59,
    4 : 86,
    55 : 71,
    33 : 29,
    75 : 1,
    85 : 18,
    23 : 32,
    35 : 41,
    7 : 61,
    28 : 57,
    80 : 20,
    59 : 99,
    46 : 23,
    82 : 28,
    58 : 35,
    40 : 14,
    2 : 83,
    24 : 94,
    0 : 54,
    22 : 90,
    21 : 63,
    90 : 70,
    98 : 21,
    48 : 55,
    50 : 74,
    76 : 88,
    18 : 49,
    45 : 72,
    83 : 62,
    12 : 96,
    86 : 77,
    84 : 45,
    99 : 25,
    34 : 46,
    16 : 67,
    19 : 79,
    39 : 16,
    53 : 43,
    25 : 30,
    100 : 26,
    61 : 47,
    60 : 38,
    64 : 65,
    54 : 89,
    56 : 68,
    26 : 39,
    66 : 8,
    15 : 17,
    3 : 69,
    73 : 84,
    57 : 73,
    41 : 13,
    31 : 80,
    96 : 97,
    36 : 7,
    88 : 98,
    11 : 27,
    49 : 11,
    14 : 6,
    70 : 12,
    92 : 22,
    29 : 5,
    63 : 64,
    37 : 92,
    27 : 85,
    42 : 51,
    79 : 81,
    6 : 3,
    10 : 53,
    89 : 34,
    52 : 78,
    44 : 2,
    17 : 87,
    13 : 31,
    5 : 42,
    69 : 44,
    20 : 75,
    94 : 48,
    43 : 56,
    77 : 40,
    72 : 37,
    38 : 19,
    81 : 50,
    91 : 10,
    65 : 36,
    1 : 9,
    78 : 33,
    62 : 91,
}

from fuel import config
from fuel.schemes import ShuffledScheme
from picklable_itertools import imap
from picklable_itertools.extras import partition_all

"""
    Custom Scheme to bridge between dataset which is a list of frames
    where our processing logic is on the videos
    The __init__ will contain a list of videos
    The get_request_iterator will return a list of frames
    since the transformer requires frames
    **Cannot shuffle the video_indexes list directly!
    video_indexes[i-1] == first frame of ith video
    video_indexes[i] == last frame of ith video
    need to keep relative order.
"""
class HDF5ShuffledScheme(ShuffledScheme) :
    def __init__(self, video_indexes,
                 random_sample=True,
                 f_subsample = 1,
                 r_subsample = False,
                 *args, **kwargs) :
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)
        self.sorted_indices = kwargs.pop('sorted_indices', False)
        self.frames_per_video = kwargs.pop('frames_per_video', 10)
        self.random_sample = random_sample

        self.f_subsample = f_subsample
        self.r_subsample = r_subsample

        self.video_indexes = video_indexes
        super(HDF5ShuffledScheme, self).__init__(*args, **kwargs)

    def correct_subsample(self, start, end, fpv, subsample):
        max_subsample = (end - start) / float(fpv)
        return min(np.floor(max_subsample).astype(np.int), subsample)


    def get_start_frame(self, start, end, fpv, subsample):
        if self.random_sample:
            return np.random.randint(start, end - subsample * fpv + 1)

        nb_frame = end - start
        if start + nb_frame // 2 + subsample * fpv < end:
            return start + nb_frame // 2
        return max(start, end - subsample * fpv)

    def get_request_iterator(self) :
        indices = list(self.indices)
        self.rng.shuffle(indices)
        fpv = self.frames_per_video


        if self.r_subsample:
            subsample = np.random.randint(1, self.f_subsample)
        else:
            subsample = self.f_subsample

        frames_array = np.empty([len(indices),fpv])
        #each element of indices is the jth video we want
        for j in xrange(len(indices)):
            i = indices[j]
            if i==0 :
                c_subsample = self.correct_subsample(0, self.video_indexes[i],
                                                     fpv, subsample)
                t = self.get_start_frame(0, self.video_indexes[i],
                                         fpv, c_subsample)
            else :
                c_subsample = self.correct_subsample(self.video_indexes[i-1],
                                                     self.video_indexes[i],
                                                     fpv, subsample)
                t = self.get_start_frame(self.video_indexes[i-1],
                                         self.video_indexes[i],
                                         fpv, c_subsample)
            for k in range(fpv):
                frames_array[j][k] = t + c_subsample * k
        frames_array = frames_array.flatten()

        if self.sorted_indices:
            return imap(sorted, partition_all(self.batch_size*fpv, frames_array))
        else:
            return imap(list, partition_all(self.batch_size*fpv, frames_array))
