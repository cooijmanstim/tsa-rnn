import tempfile, os.path, cPickle, zipfile, shutil
from collections import OrderedDict
import numpy as np
import theano
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

class DumpLog(SimpleExtension):
    def __init__(self, path, **kwargs):
        kwargs.setdefault("after_training", True)
        super(DumpLog, self).__init__(**kwargs)
        self.path = path

    def do(self, callback_name, *args):
        secure_dump(self.main_loop.log, self.path, use_cpickle=True)

class DumpGraph(SimpleExtension):
    def __init__(self, path, **kwargs):
        kwargs["after_batch"] = True
        super(DumpGraph, self).__init__(**kwargs)
        self.path = path

    def do(self, which_callback, *args, **kwargs):
        try:
            self.done
        except AttributeError:
            if hasattr(self.main_loop.algorithm, "_function"):
                self.done = True
                with open(self.path, "w") as f:
                    theano.printing.debugprint(self.main_loop.algorithm._function, file=f)

class DumpBest(SimpleExtension):
    """dump if the `notification_name` record is present"""
    def __init__(self, notification_name, save_path, **kwargs):
        self.notification_name = notification_name
        self.save_path = save_path
        kwargs.setdefault("after_epoch", True)
        super(DumpBest, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        if self.notification_name in self.main_loop.log.current_row:
            secure_dump(self.main_loop, self.save_path, dump_main_loop)

class LightCheckpoint(SimpleExtension):
    def __init__(self, save_path, **kwargs):
        kwargs.setdefault("after_epoch", True)
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
    np.savez(file, **model.get_parameter_values())

def load_model_parameters(model, file):
    model.set_parameter_values(np.load(file))
