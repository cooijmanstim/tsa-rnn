import tempfile, os.path, cPickle, zipfile, shutil
import numpy as np
from collections import OrderedDict
from blocks.extensions import SimpleExtension, Printing
from blocks.serialization import secure_dump
import blocks.config
import util

class PrintingTo(Printing):
    def __init__(self, path, **kwargs):
        super(PrintingTo, self).__init__(**kwargs)
        self.path = path
        with open(self.path, "w") as f:
            f.truncate(0)

    def do(self, *args, **kwargs):
        with util.StdoutLines() as lines:
            super(PrintingTo, self).do(*args, **kwargs)
        with open(self.path, "a") as f:
            f.write("\n".join(lines))
            f.write("\n")

class DumpBest(SimpleExtension):
    """Check if a log quantity has the minimum/maximum value so far.

    Parameters
    ----------
    record_name : str
        The name of the record to track.
    notification_name : str, optional
        The name for the record to be made in the log when the current
        value of the tracked quantity is the best so far. It not given,
        'record_name' plus "best_so_far" suffix is used.
    choose_best : callable, optional
        A function that takes the current value and the best so far
        and return the best of two. By default :func:`min`, which
        corresponds to tracking the minimum value.

    Attributes
    ----------
    best_name : str
        The name of the status record to keep the best value so far.
    notification_name : str
        The name of the record written to the log when the current
        value of the tracked quantity is the best so far.

    Notes
    -----
    In the likely case that you are relying on another extension to
    add the tracked quantity to the log, make sure to place this
    extension *after* the extension that writes the quantity to the log
    in the `extensions` argument to :class:`blocks.main_loop.MainLoop`.

    """
    def __init__(self, record_name, save_path, notification_name=None,
                 choose_best=min, **kwargs):
        self.record_name = record_name
        if not notification_name:
            notification_name = record_name + "_best_so_far"
        self.notification_name = notification_name
        self.best_name = "best_" + record_name
        self.choose_best = choose_best
        self.save_path = save_path
        kwargs.setdefault("after_epoch", True)
        super(DumpBest, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        current_value = self.main_loop.log.current_row.get(self.record_name)
        if current_value is None:
            return
        best_value = self.main_loop.status.get(self.best_name, None)
        if (best_value is None or
            (current_value != best_value and
             self.choose_best(current_value, best_value) == current_value)):
            self.main_loop.status[self.best_name] = current_value
            self.main_loop.log.current_row[self.notification_name] = True
            secure_dump(self.main_loop, self.save_path, dump_main_loop)

class LightCheckpoint(SimpleExtension):
    def __init__(self, save_path, **kwargs):
        super(LightCheckpoint, self).__init__(**kwargs)
        self.save_path = save_path

    def do(self, which_callback, *args, **kwargs):
        secure_dump(self.main_loop, self.save_path, dump_main_loop)

PARAMETER_FILENAME = "parameters.npz"
LOG_FILENAME = "log.pkl"

def dump_main_loop(main_loop, path):
    # dump a zip file with parameters.npz and log.pkl
    try:
        temp_dir = tempfile.mkdtemp(dir=blocks.config.config.temp_dir)
        parameter_path = os.path.join(temp_dir, PARAMETER_FILENAME)
        log_path = os.path.join(temp_dir, LOG_FILENAME)
        dump_model_parameters(main_loop.model, parameter_path)
        cPickle.dump(main_loop.log, open(log_path, "w"))
        with zipfile.ZipFile(path, "w") as archive:
            archive.write(parameter_path, PARAMETER_FILENAME)
            archive.write(log_path, LOG_FILENAME)
    finally:
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir)

def load_main_loop(main_loop, path):
    # load parameters.npz and log.pkl from a zip file
    try:
        temp_dir = tempfile.mkdtemp(dir=blocks.config.config.temp_dir)
        with zipfile.ZipFile(path, "r") as archive:
            archive.extractall(temp_dir)
            load_model_parameters(
                main_loop.model,
                os.path.join(temp_dir, PARAMETER_FILENAME))
            main_loop.log = cPickle.load(open(
                os.path.join(temp_dir, LOG_FILENAME)))
    finally:
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir)
    # ensure the algorithm and extensions will be initialized
    main_loop.log.status["training_started"] = False

def dump_model_parameters(model, file):
    np.savez(file,
             **OrderedDict(
                 (key, value.get_value())
                 for key, value in model.get_parameter_dict().iteritems()))

def load_model_parameters(model, file):
    parameters = np.load(file)
    parameters = OrderedDict(("/%s" % k, v) for (k, v) in parameters.items())
    model.set_parameter_values(parameters)
