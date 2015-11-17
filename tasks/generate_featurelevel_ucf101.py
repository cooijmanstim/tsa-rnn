if __name__ == "__main__":
    import numpy as np
    import zlib, h5py, cPickle, os
    from collections import OrderedDict
    from fuel.datasets import H5PYDataset
    import logging
    logger = logging.getLogger()
    logger.info("yarrr")

    input_paths = [
        "/Tmp/cooijmat/ucf101/xaa",
        "/Tmp/cooijmat/ucf101/xab",
        "/Tmp/cooijmat/ucf101/xac"
    ]
    output_path = "/Tmp/cooijmat/ucf101/featurelevel_ucf101.hdf5"

    n = 13320

    h5file = h5py.File(output_path, mode='w')

    sources = "fc conv".split()
    for source in sources:
        h5file.create_dataset(
            source, (n,), dtype=h5py.special_dtype(vlen=np.uint8))
    h5file.create_dataset('targets', (n,), dtype='int8')

    def get_identifier(path):
        return os.path.join(
            os.path.basename(os.path.dirname(path)),
            os.path.splitext(os.path.basename(path))[0])

    everything = {}
    for input_path in input_paths:
        with open(input_path, "r") as file:
            somethings = cPickle.load(file)
            everything.update((get_identifier(k), v)
                              for k, v in somethings.items())
    print everything.keys()
    m = len(everything.items())
    if m != n:
        logger.warning("expected %i examples, found %i" % (n, m))
    n = m

    with open("/u/ballasn/project/LeViRe/utils/datasets/data/ucf101/ucfTrainTestlist/classInd.txt", "r") as file:
        classmap = dict(reversed(line.split()) for line in file.readlines())

    k = 0
    lists = OrderedDict(train=OrderedDict(), test=OrderedDict())
    for which_set in "train test".split():
        filepath = "/u/ballasn/project/LeViRe/utils/datasets/data/ucf101/ucfTrainTestlist/%slist01.txt" % which_set
        print filepath
        with open(filepath, "r") as file:
            for line in file.readlines():
                path = line.split()[0]
                identifier = get_identifier(path)

                try:
                    data = everything[identifier]
                except KeyError:
                    logger.warning("missing %s" % identifier)

                target = int(classmap[os.path.dirname(identifier)])
                lists[which_set].setdefault("targets", []).append(target)
                for i, source in enumerate(sources):
                    compressed = np.fromstring(
                        # level 1 compression is fast and takes out ~90%
                        zlib.compress(cPickle.dumps(data[i]), 1),
                        dtype=np.uint8)
                    #assert np.array_equal(cPickle.loads(zlib.decompress(compressed)), data[i])
                    print source, data[i].shape, len(compressed)
                    lists[which_set].setdefault(source, []).append(compressed)

                k += 1
                print "%i/%i" % (k, n)

    split_dict = OrderedDict()
    b = 0
    for which_set, set_by_source in lists.items():
        a = b
        b += len(set_by_source["targets"])
        h5file["targets"][a:b] = set_by_source["targets"]
        for source in sources:
            h5file[source][a:b] = set_by_source[source]
        split_dict[which_set] = OrderedDict([
            (source, (a, b))
            for source in ["targets"] + sources])
    h5file.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    h5file.flush()
    h5file.close()