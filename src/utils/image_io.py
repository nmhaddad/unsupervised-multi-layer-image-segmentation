import glob

import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skvideo.io
import cv2
from skimage.transform import resize, rescale

matplotlib.use('agg')


def crop_image(img, d=32):
    """
    Make dimensions divisible by d

    :param pil img:
    :param d:
    :return:
    """

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def crop_np_image(img_np, d=32):
    return torch_to_np(crop_torch_image(np_to_torch(img_np), d))


def crop_torch_image(img, d=32):
    """
    Make dimensions divisible by d
    image is [1, 3, W, H] or [3, W, H]
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)

    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]


def get_params(opt_over, net, net_input, downsampler=None):
    """
    Returns parameters that we want to optimize over.
    :param opt_over: comma separated list, e.g. "net,input" or "net"
    :param net: network
    :param net_input: torch.Tensor that stores input `z`
    :param downsampler:
    :return:
    """

    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    """
    Creates a grid from a list of images by concatenating them.
    :param images_np:
    :param nrow:
    :return:
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(name, images_np, interpolation='lanczos', output_path="output/"):
    """
    Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """
    assert len(images_np) == 2 or len(images_np)==3
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    if len(images_np) == 2:
        grid = get_image_grid(images_np, 2)

        if images_np[0].shape[0] == 1:
            plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
        else:
            plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

        plt.savefig(output_path + "{}.png".format(name))
    else:
        grid = get_image_grid(images_np, 3)

        if images_np[0].shape[0] == 1:
            plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
        else:
            plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

        plt.savefig(output_path + "{}.png".format(name))


def save_image(name, image_np, output_path="output/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.jpg".format(name))

def video_to_images(file_name, name):
    video = prepare_video(file_name)
    for i, f in enumerate(video):
        save_image(name + "_{0:03d}".format(i), f)

def images_to_video(images_dir ,name, gray=True):
    num = len(glob.glob(images_dir +"/*.jpg"))
    c = []
    for i in range(num):
        if gray:
            img = prepare_gray_image(images_dir + "/"+  name +"_{}.jpg".format(i))
        else:
            img = prepare_image(images_dir + "/"+name+"_{}.jpg".format(i))
        print(img.shape)
        c.append(img)
    save_video(name, np.array(c))

def save_heatmap(name, image_np):
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    save_image(name, rgb_img.transpose(2, 0, 1))


def save_graph(name, graph_list, output_path="output/"):
    plt.clf()
    plt.plot(graph_list)
    plt.savefig(output_path + name + ".png")


def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:,::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(), np.rot90(flipped, 3, (1, 2)).copy()]
    return aug


def create_video_augmentations(np_video):
    """
        convention: original, left, upside-down, right, rot1, rot2, rot3
        :param np_video:
        :return:
        """
    aug = [np_video.copy(), np.rot90(np_video, 1, (2, 3)).copy(),
           np.rot90(np_video, 2, (2, 3)).copy(), np.rot90(np_video, 3, (2, 3)).copy()]
    flipped = np_video[:, :, ::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (2, 3)).copy(), np.rot90(flipped, 2, (2, 3)).copy(),
            np.rot90(flipped, 3, (2, 3)).copy()]
    return aug


def save_graphs(name, graph_dict, output_path="output/"):
    """

    :param name:
    :param dict graph_dict: a dict from the name of the list to the list itself.
    :return:
    """
    plt.clf()
    fig, ax = plt.subplots()
    for k, v in graph_dict.items():
        ax.plot(v, label=k)
        # ax.semilogy(v, label=k)
    ax.set_xlabel('iterations')
    # ax.set_ylabel(name)
    ax.set_ylabel('MSE-loss')
    # ax.set_ylabel('PSNR')
    plt.legend()
    plt.savefig(output_path + name + ".png")


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def prepare_image(file_name):
    """
    loads makes it divisible
    :param file_name:
    :return: the numpy representation of the image
    """
    img_pil = crop_image(get_image(file_name, -1)[0], d=32)
    return pil_to_np(img_pil)


def prepare_video(file_name, folder="output/"):
    data = skvideo.io.vread(folder + file_name)
    return crop_torch_image(data.transpose(0, 3, 1, 2).astype(np.float32) / 255.)[:35]


def save_video(name, video_np, output_path="output/"):
    outputdata = video_np * 255
    outputdata = outputdata.astype(np.uint8)
    skvideo.io.vwrite(output_path + "{}.mp4".format(name), outputdata.transpose(0, 2, 3, 1))


def prepare_gray_image(file_name):
    img = prepare_image(file_name)
    return np.array([np.mean(img, axis=0)])


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def median(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for c in range(shape[0]):
        for w in range(shape[1]):
            for h in range(shape[2]):
                result[c, w, h] = sorted(i[c, w, h] for i in img_np_list)[l//2]
    return result


def average(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for i in img_np_list:
        result += i
    return result / l


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def images_to_video_fps(video_name, input_directory, output_directory, fps, substring="_0_final.jpg"):
    """
    :param video_name:
    :param input_directory:
    :param output_directory:
    :param fps:
    :param substring:
    :return:
    """
    images = []

    filelist = glob.glob(input_directory + "/*" + substring)

    for count in range(len(filelist)):
        file = input_directory + "/frame" + str(count) + substring
        img = cv2.imread(file)
        img_height, img_width, layers = img.shape
        size = (img_width, img_height)
        images.append(img)

    out = cv2.VideoWriter(output_directory + "/" + video_name + '_output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    for image in images:
        out.write(image)

    out.release()
    cv2.destroyAllWindows()


def video_to_images_fps(video_name, video, fps):
    """
    :param video_name:
    :param video:
    :param fps:
    :return:
    """

    count = 0
    cap = cv2.VideoCapture(video)

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    out = cv2.VideoWriter(video_name + '_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
            # rescaled_frame = rescale(frame, 0.5, anti_aliasing=False)
            cv2.imwrite("input/images/" + video_name + "/frame{}".format(str(count)) + ".jpg", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def get_segmented_image(output_name, image_1, image_2, image_3):
    a = cv2.imread(image_1)
    b = cv2.imread(image_2)
    c = cv2.imread(image_3)
    assert(a.shape == b.shape and a.shape == c.shape)

    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(len(a[i][j])):
                # clamp values to min and max
                if a[i][j][k] >= 127.5:
                    a[i][j][k] = 255
                else:
                    a[i][j][k] = 0
                if b[i][j][k] >= 127.5:
                    b[i][j][k] = 255
                else:
                    b[i][j][k] = 0
                if c[i][j][k] >= 127.5:
                    c[i][j][k] = 255
                else:
                    c[i][j][k] = 0

            if a[i][j].all() == b[i][j].all() and a[i][j].all() == c[i][j].all():
                continue
            elif a[i][j].all() != b[i][j].all() or a[i][j].all() != c[i][j].all():
                a[i][j] = [127, 127, 127]

    cv2.imwrite("output/images/segmentation/" + output_name + "_3dd.jpg", a)
