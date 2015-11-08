from blocks.bricks.base import Brick, application
from sklearn_theano.feature_extraction.caffe.vgg_flows import create_theano_expressions
import util

class PatchTransform(Brick):
    def __init__(self, patch_shape, **kwargs):
        self.patch_shape = patch_shape
        super(PatchTransform, self).__init__(**kwargs)

    @application(inputs=["patch"], outputs=["output"])
    def apply(self, patch):
        if len(self.patch_shape) == 3:
            output = create_theano_expressions(
                mode="rgb",
                inputs=("data",
                        (patch
                         .dimshuffle(0, 2, 1, 3, 4)
                         .reshape((patch.shape[0] * patch.shape[2],
                                   patch.shape[1],
                                   patch.shape[3],
                                   patch.shape[4])))))[0]["fc7"]
            return (output
                    .reshape((patch.shape[0], patch.shape[2], -1))
                    # average across frames
                    .mean(axis=1))
        else:
            raise NotImplementedError()

    def get_dim(self, name):
        if name == "output":
            return (4096,)
        else:
            raise NotImplementedError()

    @property
    def output_dim(self):
        return self.get_dim("output")[0]

@util.checkargs
def get_patch_transform(patch_shape, **kwargs):
    assert patch_shape[1:] == [224, 224]
    return PatchTransform(patch_shape)
