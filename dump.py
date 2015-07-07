import os

import numpy as np

from blocks.serialization import dump
from blocks.bricks.extensions import SimpleExtension

class Dump(SimpleExtension):
    def __init__(self, save_path_stem, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(Dump, self).__init__(**kwargs)
        self.save_path = save_path

    def do(self, which_callback, *args, **kwargs):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        filename = "params_%i.npz" % self.main_loop.status["epochs_done"]
        with open(os.path.join(save_path, filename), "wb") as f:
            dump(f, self.main_loop.model.params)

class DumpMinimum(SimpleExtension):
    def __init__(self, save_path_stem, channel_name, sign=1, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(DumpMinimum, self).__init__(**kwargs)
        self.save_path = save_path
        self.channel_name = channel_name
        self.sign = sign
        self.record_value = np.float32("inf")

    def do(self, which_callback, *args, **kwargs):
        current_value = self.main_loop.log.current_row.get(self.channel_name)
        if current_value is None:
            return
        if self.sign*current_value < self.sign*self.record_value:
            self.record_value = current_value
            self.do_dump()

    def do_dump(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        filename = "params_%i.npz" % self.main_loop.status["epochs_done"]
        with open(os.path.join(save_path, filename), "wb") as f:
            dump(f, self.main_loop.model.params)
