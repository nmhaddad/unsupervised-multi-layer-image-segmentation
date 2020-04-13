from collections import namedtuple
from cv2.ximgproc import guidedFilter

from net import *
from net.losses import StdLoss
from utils.imresize import imresize, np_imresize
from net.noise import get_noise
from utils.image_io import *
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio
import torch.nn as nn
# import progressbar
import os, os.path
import datetime
import numpy as np


def get_dark_channel(image, w=15):
    """
    Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = image.shape
    padded = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
    w:      window for dark channel
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    image = image.transpose(1, 2, 0)
    # reference CVPR09, 4.4
    darkch = get_dark_channel(image, w)
    M, N = darkch.shape
    flatI = image.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)



DehazeResult = namedtuple("DehazeResult", ['learned', 't', 'a', 'psnr'])


class Dehaze(object):

    def __init__(self, image_name, image, num_iter=8000, plot_during_training=True, show_every=500,
                 use_deep_channel_prior=True, gt_ambient=None, clip=True):

        self.image_name = image_name
        self.image = image

        self.num_iter = num_iter
        self.plot_during_training = plot_during_training
        self.show_every = show_every
        self.use_deep_channel_prior = use_deep_channel_prior
        self.gt_ambient = gt_ambient  # np
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        # TODO new net
        self.new_net = None
        # TODO new net
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.parameters = None
        self.current_result = None

        self.clip = clip
        self.blur_loss = None
        self.best_result = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.done = False
        self.ambient_out = None
        self.total_loss = None
        self.input_depth = 8
        self.post = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        factor = 1
        image = self.image
        while image.shape[1] >= 800 or image.shape[2] >= 800:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] /factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1
        self.images = create_augmentations(image)
        self.images_torch = [np_to_torch(image).type(torch.FloatTensor) for image in self.images]

    def _is_learning_ambient(self):
        """
        true if the ambient is learned during the optimization process
        :return:
        """
        return not self.use_deep_channel_prior # and not isinstance(self.gt_ambient, np.ndarray)

    def _init_nets(self):
        input_depth = self.input_depth
        data_type = torch.FloatTensor
        pad = 'reflection'

        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.image_net = image_net.type(data_type)

        mask_net = skip(
            input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(data_type)


    def _init_new_net(self):
        """

        :return:
        """

        # TODO this might not belong here

        input_depth = self.input_depth
        data_type = torch.FloatTensor
        pad = 'reflection'

        # encoder decoder model
        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.new_net = image_net.type(data_type)


    def _init_ambient(self):
        if self._is_learning_ambient():
            ambient_net = skip(
                self.input_depth, 3,
                num_channels_down=[8, 16, 32, 64, 128],
                num_channels_up=[8, 16, 32, 64, 128],
                num_channels_skip=[0, 0, 0, 4, 4],
                upsample_mode='bilinear',
                filter_size_down=3,
                filter_size_up=3,
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
            self.ambient_net = ambient_net.type(torch.FloatTensor)
        if isinstance(self.gt_ambient, np.ndarray):
            atmosphere = self.gt_ambient
        else:
            # use_deep_channel_prior is True
            atmosphere = get_atmosphere(self.image)
        self.ambient_val = nn.Parameter(data=torch.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                        requires_grad=False)

    def _init_parameters(self):
        # TODO init params for new net

        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()]
        if self._is_learning_ambient():
            parameters += [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        data_type = torch.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def _init_inputs(self):
        # TODO init inputs for new net

        original_noises = create_augmentations(torch_to_np(get_noise(self.input_depth, 'noise',
                                                                     (self.images[0].shape[1], self.images[0].shape[2]),
                                          var=1/10.).type(torch.FloatTensor).detach()))
        self.image_net_inputs = [np_to_torch(original_noise).type(torch.FloatTensor).detach()
                                 for original_noise in original_noises]

        original_noises = create_augmentations(torch_to_np(get_noise(self.input_depth, 'noise',
                                                                     (self.images[0].shape[1], self.images[0].shape[2]),
                                                                     var=1 / 10.).type(
            torch.FloatTensor).detach()))
        self.mask_net_inputs = [np_to_torch(original_noise).type(torch.FloatTensor).detach()
                                 for original_noise in original_noises]
        if self._is_learning_ambient():
            self.ambient_net_input = get_noise(self.input_depth, 'meshgrid',
                                                           (self.images[0].shape[1], self.images[0].shape[2])
                                                           ).type(torch.FloatTensor).detach()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            if self.done:
                return
            optimizer.step()

    def _get_augmentation(self, iteration):
        return 0
        # if iteration % 4 in [1, 2,3]:
        #     return 0
        # iteration //= 2
        # return iteration % 8

    def _optimization_closure(self, step):
        """

        :param step: the number of the iteration

        :return:
        """

        # TODO figure out where to get new net output

        if step == self.num_iter - 1:
            aug = 0
            reg_std = 0
        else:
            aug = self._get_augmentation(step)
            reg_std = 1 / 30.
        image_net_input = self.image_net_inputs[aug] + (self.image_net_inputs[aug].clone().normal_() * reg_std)
        self.image_out = self.image_net(image_net_input)

        if isinstance(self.ambient_net, nn.Module):
            ambient_net_input = self.ambient_net_input + (self.ambient_net_input.clone().normal_() * reg_std)
            self.ambient_out = self.ambient_net(ambient_net_input)  #[:, :,
                               # self.images[0].shape[1] // 2:self.images[0].shape[1] // 2 + 1,
                               # self.images[0].shape[2] // 2:self.images[0].shape[2] // 2 + 1]
            # self.ambient_out  = self.ambient_out * torch.ones_like(self.image_out)
        else:
            self.ambient_out = self.ambient_val
        self.mask_out = self.mask_net(self.mask_net_inputs[aug])

        self.blur_out = self.blur_loss(self.mask_out)
        self.total_loss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                 self.images_torch[aug]) + 0.005 * self.blur_out
        if self._is_learning_ambient():
            self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
            if step < 1000:
                self.total_loss += self.mse_loss(self.ambient_out, self.ambient_val * torch.ones_like(self.ambient_out))
        self.total_loss.backward(retain_graph=True)


    def _obtain_current_result(self, step):
        if step % 8 == 0:
            # TODO create new net out to np
            # new_net_out_np = np.clip(torch_to_np(self.new_net_out), 0,1)
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
            # TODO create new equation for new net
            # psnr = compare_psnr(self.images[0], mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np)
            # TODO original code with depricated method
            # psnr = compare_psnr(self.images[0], mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np)
            psnr = peak_signal_noise_ratio(self.images[0], mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np)
            self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np, psnr=psnr)
            if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
                self.best_result = self.current_result

    def _plot_closure(self, step):
        """

         :param step: the number of the iteration

         :return:
         """
        print('Iteration %05d    Loss %f  %f current_psnr: %f max_psnr %f' % (step, self.total_loss.item(),
                                                                              self.blur_out.item(),
                                                                           self.current_result.psnr,
                                                                           self.best_result.psnr), '\r', end='')
        if step % self.show_every == self.show_every - 1:
            plot_image_grid("t_and_amb", [ self.best_result.a * np.ones_like(self.best_result.learned), self.best_result.t])
            # original_image = t*image + (1-t)*A
            # image = (original_image - (1 - t) * A) * (1/t)
            plot_image_grid("current_image", [self.images[0], np.clip(self.best_result.learned, 0, 1)])

    def finalize(self):
        self.final_image = np_imresize(self.best_result.learned, output_shape=self.original_image.shape[1:])
        self.final_t_map = np_imresize(self.best_result.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.best_result.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.t_matting(self.final_t_map)
        # TODO Here is combo not sure if this needs to be changed yet
        self.post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
        save_image(self.image_name + "_original", np.clip(self.original_image, 0, 1))
        # save_image(self.image_name + "_learned", self.final_image)
        save_image(self.image_name + "_t", mask_out_np)
        save_image(self.image_name + "_final", self.post)
        save_image(self.image_name + "_a", np.clip(self.final_a, 0, 1))

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehaze(image_name, image, num_iter=4000, plot_during_training=True, show_every=500, use_deep_channel_prior=True,
           gt_ambient=None):

    dh = Dehaze(image_name + "_0", image, num_iter, plot_during_training, show_every, use_deep_channel_prior,
                gt_ambient, clip=True)

    dh.optimize()
    dh.finalize()

    if use_deep_channel_prior:
        assert not gt_ambient
        gt_ambient = dh.best_result.a
        use_deep_channel_prior = False
    for i in range(1):
        assert dh.post.shape == image.shape, (dh.post.shape, image.shape)
        dh = Dehaze(image_name + "_{}".format(i+1), dh.post, num_iter, plot_during_training, show_every,
                    use_deep_channel_prior, gt_ambient, clip=True)
        dh.optimize()
        dh.finalize()
    post = dh.post
    t = np.array([np.mean((image - dh.final_a) / (post - dh.final_a), axis=0)])
    save_image(image_name + "_original", np.clip(image, 0, 1))


def iterative_dehaze(image_name, image, num_chains=2, start_num_iter=4000, iter_factor=1, plot_during_training=True,
                   show_every=500, use_deep_channel_prior=True, gt_ambient=None):
    """
    :param image_name: the path of the image
    :param image: the output name of the image
    :param num_chains: the number of deep image priors
    :param start_num_iter: number of starting iterations
    :param iter_factor: a factor to decrease or increase the number of iterations by with each DIP
    :param plot_during_training: bool
    :param show_every: bool
    :param use_deep_channel_prior: bool
    :param gt_ambient: None, or numpy array
    :return:
    """

    dh = Dehaze(image_name + "_0", image, start_num_iter, plot_during_training, show_every, use_deep_channel_prior,
                gt_ambient, clip=True)
    dh.optimize()
    dh.finalize()

    if use_deep_channel_prior:
        assert not gt_ambient
        gt_ambient = dh.best_result.a
        use_deep_channel_prior = False

    for i in range(num_chains - 1):
        assert dh.post.shape == image.shape, (dh.post.shape, image.shape)

        num_iter = start_num_iter
        num_iter = int(num_iter * iter_factor)

        dh = Dehaze(image_name + "_{}".format(i + 1), dh.post, num_iter, plot_during_training, show_every,
                    use_deep_channel_prior, gt_ambient, clip=True)
        dh.optimize()
        dh.finalize()
    post = dh.post
    t = np.array([np.mean((image - dh.final_a) / (post - dh.final_a), axis=0)])
    save_image(image_name + "_original", np.clip(image, 0, 1))


def dehaze_images(input_directory, output_directory, num_iter=4000, use_deep_channel_prior=True, gt_ambient=None):
    """
    :param input_directory: path of input directory (location of frames)
    :param output_directory: path of output directory
    :param num_iter: the number of starting iterations
    :param use_deep_channel_prior: bool - true when gt_ambient is None
    :param gt_ambient: None or numpy array of ambient values
    :return: nothing
    """

    num_images = len([name for name in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, name))])

    for i in range(num_images):
        j = prepare_image(input_directory + "/frame{}".format(str(i)) + ".jpg")
        dehaze(output_directory + "/frame" + str(i), j, num_iter=num_iter,
               use_deep_channel_prior=use_deep_channel_prior, gt_ambient=gt_ambient)


if __name__ == "__main__":

    # unique identifier for output
    time = datetime.datetime.now()
    TIMESTAMP = str(time.hour) + "-" + str(time.minute) + "-" + str(time.hour)

    """
    EXPERIMENT: Double DIP Dehazing
    """
    # i = prepare_image("input/images/dehazing/indy_0.jpg")
    # dehaze("indy_0", i, num_iter=4000, use_deep_channel_prior=True)
    #
    # i = prepare_image("input/images/dehazing/indy_1.jpg")
    # dehaze("images/dehazing/indy_1", i, num_iter=4000, use_deep_channel_prior=True)
    #
    # i = prepare_image("input/images/dehazing/qatar_0.jpg")
    # dehaze("images/dehazing/qatar", i, num_iter=4000, use_deep_channel_prior=True)
    #
    # i = prepare_image("input/images/dehazing/runway_0.jpg")
    # dehaze("images/dehazing/runway_0", i, num_iter=4000, use_deep_channel_prior=True)
    #
    # i = prepare_image("input/images/dehazing/titanic_0.jpg")
    # dehaze("images/dehazing/titanic_0", i, num_iter=4000, use_deep_channel_prior=True)

    """
    EXPERIMENT: Iterative Image Dehazing w/ Double-DIP
    """
    i = prepare_image("input/images/dehazing/qatar_0.jpg")
    iterative_dehaze("images/dehazing/qatar_" + TIMESTAMP, i, num_chains=3, start_num_iter=1000, iter_factor=3,
                   use_deep_channel_prior=True)

    """
    EXPERIMENT: Frame-by-Frame Video Dehazing
    """
    # # titanic sub-experiment
    # video_to_images_fps("titanic_bow", "input/videos/titanic_bow.mp4", fps=30)
    # dehaze_images("input/images/titanic_bow", "images/titanic_bow", num_iter=200)
    # images_to_video_fps("titanic_dehazed_0", "output/images/titanic_bow", "output/videos", fps=30, substring="_0_final.jpg")
    # images_to_video_fps("titanic_dehazed_1", "output/images/titanic_bow", "output/videos", fps=30, substring="_1_final.jpg")