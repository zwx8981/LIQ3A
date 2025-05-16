from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from ImageDataset2 import ImageDataset2, ImageDataset_qonly, ImageDataset_oppo, ImageDataset_llie, ImageDataset_pseudo_label, ImageDataset_llie2, ImageDataset_llie_general, ImageDataset_ms, ImageDataset_llie_naflex
from ImageDataset import ImageDataset_SPAQ, ImageDataset_TID, ImageDataset_PIPAL, ImageDataset_ava

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, CenterCrop, RandomCrop, Resize
from torchvision import transforms
import torch
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def set_dataset(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):

    data = ImageDataset2(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader

def set_dataset_oppo(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):

    data = ImageDataset_oppo(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_spaq(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):

    data = ImageDataset_SPAQ(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_tid(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):

    data = ImageDataset_TID(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_pipal(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):

    data = ImageDataset_PIPAL(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_ava(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):

    data = ImageDataset_ava(
        npy_file='./ava_test.npy',
        img_dir=data_set,
        preprocess=preprocess)

    loader = DataLoader(data, batch_size=bs, shuffle=False, pin_memory=True, num_workers=num_workers)

    return loader

def set_dataset_qonly(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, set):

    data = ImageDataset_qonly(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        set=set,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader



def set_dataset_llie(csv_file, bs, data_set, spatialFeat, num_workers, preprocess, num_patch, test, set):

    data = ImageDataset_llie(
        csv_file=csv_file,
        img_dir=data_set,
        spatialFeat=spatialFeat,
        num_patch=num_patch,
        set=set,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader

def set_dataset_llie2(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, set):

    data = ImageDataset_llie2(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        set=set,
        test=test,
        preprocess=preprocess,
    )

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_dataset_llie_naflex(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, set):

    data = ImageDataset_llie_naflex(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        set=set,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers, collate_fn=custom_collect_fn)

    return loader

def custom_collect_fn(batch):

    vids, moss, dists = zip(*batch)
    moss = torch.tensor(moss)
    dists = torch.stack(dists)
    # batch = zip(*batch)
    # batch = next(batch)
    #
    # vids = []
    # moss = []
    # dists = []
    #
    # for item in batch:
    #     I = item[0]
    #     mos = item[1]
    #     dist = item[2]
    #     vids.append(I)
    #     moss.append(mos)
    #     dists.append(dist)
    #
    # moss = torch.stack(moss)
    # dists = torch.stack(dists)
    return vids, moss, dists


def set_dataset_general(csv_file, bs, data_set, num_workers, preprocess, test, set):

    data = ImageDataset_llie_general(
        csv_file=csv_file,
        img_dir=data_set,
        set=set,
        test=test,
        preprocess=preprocess,
    )

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_dataset_pseudo_label(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, set,
                             pseudo_label):

    data = ImageDataset_pseudo_label(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        set=set,
        test=test,
        pseudo_label=pseudo_label,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    # loader = DataLoader(data, batch_size=1, shuffle=shuffle, collate_fn=group_collate_fn,
    #                     pin_memory=True, num_workers=num_workers)

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)

    return loader



class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess22(size=448):
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(size),
        ToTensor(),
        #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess3():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess33(size=448, crop_size=384):
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(size),
        transforms.RandomCrop(crop_size),
        RandomHorizontalFlip(),
        ToTensor(),
        #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess333(size=448):
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(size),
        RandomHorizontalFlip(),
        ToTensor(),
        #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess_siglip():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        #Resize(512+32),
        #CenterCrop(224),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def _preprocess_siglip_train():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        #Resize(512+32),
        #RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def _preprocess_siglip2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        #CenterCrop(224),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def _preprocess_siglip_train2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        #RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])



def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def _preprocess_scale1_train():
    return Compose([
        _convert_image_to_rgb,
        Resize(224),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _preprocess_scale1_test():
    return Compose([
        _convert_image_to_rgb,
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _preprocess_scale_test(size):
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _preprocess_scale_train(size):
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def set_dataset_ms(csv_file, bs, data_set, num_workers, preprocess1, preprocess2, preprocess3, num_patch, test, set):
    data = ImageDataset_ms(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        set=set,
        test=test,
        preprocess1=preprocess1,
        preprocess2=preprocess2,
        preprocess3=preprocess3,
    )

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader
