from collections import OrderedDict
import numpy
from theano import tensor
from theano.ifelse import ifelse
from blocks.utils import shared_floatx
from blocks.theano_expressions import l2_norm
from blocks.algorithms import StepRule

class Compressor(StepRule):
    def __init__(self, initial_threshold=1., window_width=257):
        self.window_width = window_width
        self.window = shared_floatx(
            initial_threshold *
            numpy.zeros((1,)),
            "window")

    def compute_steps(self, previous_steps):
        def median(window):
            return tensor.sort(window)[self.window.shape[0] / 2]

        self.median = median(self.window)

        # allow within 1 median absolute deviation
        #self.deviation = median(abs(self.window - self.median))
        #self.max_ratio = 1 + self.deviation / self.median
        self.max_ratio = 1.

        self.norm = l2_norm(previous_steps.values())
        self.ratio = self.norm / self.median
        acceptable = self.ratio <= self.max_ratio
        multiplier = (
            tensor.switch(acceptable,
                          # smaller steps are used as is
                          1,
                          # larger steps are pushed down
                          self.norm ** (1 / self.ratio) / self.norm))
        self.newnorm = multiplier * self.norm

        newwindow = tensor.concatenate([
            tensor.shape_padleft(self.norm),
            self.window[:(self.window_width - 1)]
        ], axis=0)

        # let the norm affect the median only if it was acceptable
        # or if the window hasn't been fully populated yet
        #newwindow = ifelse(
        #    acceptable + (self.window.shape[0] < self.window_width),
        #    tensor.concatenate([
        #        tensor.shape_padleft(self.norm),
        #        self.window[:(self.window_width - 1)]],
        #        axis=0),
        #    self.window)

        steps = OrderedDict(
            (parameter, multiplier * step)
            for parameter, step in previous_steps.items())
        updates = [(self.window, newwindow)]
        return steps, updates
