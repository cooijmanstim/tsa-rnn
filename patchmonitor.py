# these two lines ensure matplotlib doesn't try to use X11.
import matplotlib
matplotlib.use('Agg')

import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches

from blocks.extensions import SimpleExtension

class PatchMonitoring(SimpleExtension):
    def __init__(self, data_stream, extractor, map_to_input_space, save_to=".", **kwargs):
        if not os.path.isdir(save_to):
            os.makedirs(save_to)
        self.data_stream = data_stream
        self.save_to = save_to
        self.extractor = extractor
        self.map_to_input_space = map_to_input_space
        self.colors = dict()
        super(PatchMonitoring, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        current_dir = os.getcwd()
        os.chdir(self.save_to)
        self.save_patches("patches_iteration_%i.png" % self.main_loop.status['iterations_done'])
        os.chdir(current_dir)

    def save_patches(self, filename):
        batch = self.data_stream.get_epoch_iterator(as_dict=True).next()
        images = batch['features']
        locationss, scaless, patchess = self.extractor(images)

        batch_size = images.shape[0]
        npatches = patchess.shape[1]
        image_shape = images.shape[-2:]
        patch_shape = patchess.shape[-2:]

        if images.shape[1] == 1:
            # remove degenerate channel axis because pyplot rejects it
            images = np.squeeze(images, axis=1)
            patchess = np.squeeze(patchess, axis=2)
        else:
            # move channel axis to the end because pyplot wants this
            images = np.rollaxis(images, 1, images.ndim)
            patchess = np.rollaxis(patchess, 2, patchess.ndim)

        outer_grid = gridspec.GridSpec(batch_size, 2,
                                       width_ratios=[1, npatches])
        for i, (image, patches, locations, scales) in enumerate(zip(images, patchess, locationss, scaless)):
            image_ax = plt.subplot(outer_grid[i, 0])
            self.imshow(image, axes=image_ax)
            image_ax.axis("off")

            inner_grid = gridspec.GridSpecFromSubplotSpec(1, npatches,
                                                          subplot_spec=outer_grid[i, 1],
                                                          wspace=0.1, hspace=0.1)
            for j, (patch, location, scale) in enumerate(zip(patches, locations, scales)):
                true_location, true_scale = self.map_to_input_space(
                    location, scale,
                    np.array(patch_shape, dtype='float32'),
                    np.array(image_shape, dtype='float32'))

                patch_ax = plt.subplot(inner_grid[0, j])
                self.imshow(patch, axes=patch_ax)
                patch_ax.set_title("l (%3.2f, %3.2f)\ns (%3.2f, %3.2f)" %
                                   (location[0], location[1], true_scale[0], true_scale[1]))
                patch_ax.axis("off")

                patch_hw = patch_shape / true_scale
                image_yx = true_location - patch_hw/2.0
                image_ax.add_patch(matplotlib.patches.Rectangle((image_yx[1], image_yx[0]),
                                                                patch_hw[1], patch_hw[0],
                                                                edgecolor="red",
                                                                facecolor="none"))

        fig = plt.gcf()
        fig.set_size_inches((16, 9))
        plt.tight_layout()
        fig.savefig(filename, bbox_inches="tight", facecolor="gray")
        plt.close()

    def imshow(self, image, *args, **kwargs):
        kwargs.setdefault("cmap", "gray")
        kwargs.setdefault("aspect", "equal")
        kwargs.setdefault("interpolation", "none")
        kwargs.setdefault("vmin", 0.0)
        kwargs.setdefault("vmax", 1.0)
        kwargs.setdefault("shape", image.shape)
        plt.imshow(image, *args, **kwargs)

class VideoPatchMonitoring(SimpleExtension):
    def __init__(self, data_stream, extractor, map_to_input_space, save_to=".", **kwargs):
        if not os.path.isdir(save_to):
            os.makedirs(save_to)
        self.data_stream = data_stream
        self.save_to = save_to
        self.extractor = extractor
        self.map_to_input_space = map_to_input_space
        super(VideoPatchMonitoring, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        current_dir = os.getcwd()
        os.chdir(self.save_to)
        self.save_patches("patches_iteration_%i" % self.main_loop.status['iterations_done'])
        os.chdir(current_dir)

    def save_patches(self, filename_stem):
        batch = self.data_stream.get_epoch_iterator(as_dict=True).next()
        videos = batch['features']
        locationss, scaless, patchess = self.extractor(videos)

        n_patches = patchess.shape[1]

        if videos.shape[1] == 1:
            # remove degenerate channel axis because pyplot rejects it
            videos = np.squeeze(videos, axis=1)
            patchess = np.squeeze(patchess, axis=2)
        else:
            # move channel axis to the end because pyplot wants this
            videos = np.rollaxis(videos, 1, videos.ndim)
            patchess = np.rollaxis(patchess, 2, patchess.ndim)

        outer_grid = gridspec.GridSpec(2, 1)
        for i, (video, patches, locations, scales) in enumerate(zip(videos, patchess, locationss, scaless)):
            video_ax = plt.subplot(outer_grid[0, 0])
            video_image = (video
                           .transpose(1, 0, 2)
                           .reshape((video.shape[1],
                                     video.shape[0] * video.shape[2])))
            self.imshow(video_image, axes=video_ax)
            video_ax.axis("off")

            # TODO: maybe rectangles in video_ax
            patch_ax = plt.subplot(outer_grid[1, 0])
            patch_image = (patches
                           .transpose(0, 2, 1, 3)
                           .reshape((patches.shape[0]*patches.shape[2],
                                     patches.shape[1]*patches.shape[3])))
            self.imshow(patch_image, axes=patch_ax)
            patch_ax.axis("off")

            fig = plt.gcf()
            fig.set_size_inches((20, 20))
            plt.tight_layout()
            filename = "%s_example_%i.png" % (filename_stem, i)
            fig.savefig(filename, bbox_inches="tight", facecolor="gray")
            plt.close()

    def imshow(self, image, *args, **kwargs):
        kwargs.setdefault("cmap", "gray")
        kwargs.setdefault("aspect", "equal")
        kwargs.setdefault("interpolation", "none")
        kwargs.setdefault("vmin", 0.0)
        kwargs.setdefault("vmax", 1.0)
        kwargs.setdefault("shape", image.shape)
        plt.imshow(image, *args, **kwargs)
