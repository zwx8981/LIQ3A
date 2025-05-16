import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset2(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file, sep='\t', header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, 1]

        dist_type = self.data.iloc[index, 2]
        scene_content1 = self.data.iloc[index, 3]
        scene_content2 = self.data.iloc[index, 4]
        scene_content3 = self.data.iloc[index, 5]

        if scene_content2 == 'invalid':
            valid = 1
        elif scene_content3 == 'invalid':
            valid = 2
        else:
            valid = 3

        sample = {'I': patches, 'mos': float(mos), 'dist_type': dist_type, 'scene_content1': scene_content1,
                  'scene_content2':scene_content2, 'scene_content3':scene_content3, 'valid':valid}

        return sample

    def __len__(self):
        return len(self.data.index)
        
class ImageDataset_qonly(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 set,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        if csv_file[-3:] == 'txt':
            data = pd.read_csv(csv_file, sep='\t', header=None)
            self.data = data
            self.mos_col = 1
        elif csv_file[-4:] == 'xlsx':
            data = pd.read_excel(csv_file, header=0)
            self.data = data
            self.mos_col = 1
        else:
            data = pd.read_csv(csv_file, header=0)
            if ('split' in data.columns) & (set != 3):
                self.data = data[data.split==set]
            else:
                self.data = data
            self.mos_col = 1
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        image_name = image_name.replace('\\', '/')
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, self.mos_col]
        if self.data.shape[1] == 23: #llie
            distortions = self.data.iloc[index, self.mos_col+1::2]
            distortions = distortions.to_numpy(dtype=float)
            distortions = torch.from_numpy(distortions)
        else:
            distortions = 0

        sample = {'I': patches, 'mos': float(mos), 'dists':distortions}

        return sample

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data.index)


class ImageDataset_llie(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 spatialFeat,
                 preprocess,
                 num_patch,
                 set,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        if csv_file[-3:] == 'txt':
            data = pd.read_csv(csv_file, sep='\t', header=None)
            self.data = data
            self.mos_col = 1
        elif csv_file[-4:] == 'xlsx':
            data = pd.read_excel(csv_file, header=0)
            self.data = data
            self.mos_col = 1
        else:
            data = pd.read_csv(csv_file, header=0)
            if ('split' in data.columns) & (set != 3):
                self.data = data[data.split==set]
            else:
                self.data = data
            self.mos_col = 1
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.spatialFeat = spatialFeat

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        image_name = image_name.replace('\\', '/')
        I = self.loader(image_name)
        I = self.preprocess(I)

        tmp = image_name.split('/')[-1]
        tmp = tmp.split('.')[0]

        spatial_feat = torch.from_numpy(np.load(os.path.join(self.spatialFeat, f'{tmp}.npy'))).view(-1)

        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]

        mos = self.data.iloc[index, self.mos_col]
        if self.data.shape[1] == 23: #llie
            distortions = self.data.iloc[index, self.mos_col+1::2]
            distortions = distortions.to_numpy(dtype=float)
            distortions = torch.from_numpy(distortions)
        else:
            distortions = 0

        sample = {'I': patches, 'spatial_feat':spatial_feat, 'mos': float(mos), 'dists':distortions}

        return sample

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data.index)


class ImageDataset_llie_naflex(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 set,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        if csv_file[-3:] == 'txt':
            data = pd.read_csv(csv_file, sep='\t', header=None)
            self.data = data
            self.mos_col = 1
        elif csv_file[-4:] == 'xlsx':
            data = pd.read_excel(csv_file, header=0)
            self.data = data
            self.mos_col = 1
        else:
            data = pd.read_csv(csv_file, header=0)
            if ('split' in data.columns) & (set != 3):
                self.data = data[data.split==set]
            else:
                self.data = data
            self.mos_col = 1
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        image_name = image_name.replace('\\', '/')
        I = self.loader(image_name)

        mos = self.data.iloc[index, self.mos_col]
        if self.data.shape[1] == 23: #llie
            distortions = self.data.iloc[index, self.mos_col+1::2]
            distortions = distortions.to_numpy(dtype=float)
            distortions = torch.from_numpy(distortions)
        else:
            distortions = 0

        #sample = {'I': I, 'mos': float(mos), 'dists':distortions}

        return I, float(mos), distortions

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data.index)



class ImageDataset_llie2(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 set,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        if csv_file[-3:] == 'txt':
            data = pd.read_csv(csv_file, sep='\t', header=None)
            self.data = data
            self.mos_col = 1
        elif csv_file[-4:] == 'xlsx':
            data = pd.read_excel(csv_file, header=0)
            self.data = data
            self.mos_col = 1
        else:
            data = pd.read_csv(csv_file, header=0)
            if ('split' in data.columns) & (set != 3):
                self.data = data[data.split==set]
            else:
                self.data = data
            self.mos_col = 1
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        image_name = image_name.replace('\\', '/')
        I = self.loader(image_name)
        I = self.preprocess(I)

        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]

        mos = self.data.iloc[index, self.mos_col]
        if self.data.shape[1] == 23: #llie
            distortions = self.data.iloc[index, self.mos_col+1::2]
            distortions = distortions.to_numpy(dtype=float)
            distortions = torch.from_numpy(distortions)
        else:
            distortions = 0

        sample = {'I': patches, 'mos': float(mos), 'dists':distortions}

        return sample

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data.index)


class ImageDataset_pseudo_label(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 set,
                 test,
                 pseudo_label,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
        """
        self.data = pd.read_csv(csv_file, header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.pseudo_label = pseudo_label
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: A list of dicts with keys 'I' and 'mos'
        """

        image_name = self.data.iloc[index, 0]
        labels = []
        all_patches = []
        methods = list(self.pseudo_label.keys())
        for method in methods:
            if method == 'GT':
                llie_name = image_name
            elif method == 'NeRCo':
                llie_name = method + '_' + image_name[:-4] + '_fake_B.png'
            else:
                llie_name = method + '_' + image_name
            image_path = os.path.join(self.img_dir, method, llie_name)
            I = self.loader(image_path)
            I = self.preprocess(I)
            label = self.pseudo_label[method]

            I = I.unsqueeze(0)
            n_channels = 3
            kernel_h = 224
            kernel_w = 224
            if (I.size(2) >= 1024) | (I.size(3) >= 1024):
                step = 48
            else:
                step = 32
            patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                              n_channels,
                                                                                                              kernel_h,
                                                                                                              kernel_w)

            assert patches.size(0) >= self.num_patch
            self.num_patch = np.minimum(patches.size(0), self.num_patch)
            if self.test:
                sel_step = patches.size(0) // self.num_patch
                sel = torch.zeros(self.num_patch)
                for i in range(self.num_patch):
                    sel[i] = sel_step * i
                sel = sel.long()
            else:
                sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch,))
            patches = patches[sel, ...]

            labels.append(label)
            all_patches.append(patches)

        I = torch.cat(all_patches, dim=0)
        labels = torch.tensor(labels)
        sample = {'I': I, 'mos': labels}
        return sample

    def __len__(self):
        return len(self.data.index)


# level = {'mild':0, 'moderate':1, 'severe': 2}
#
# tone_issues = {'global over-exposure':0, 'global under-exposure':1, 'global reverse-tone':2, 'global hazy': 3,
#                'global high-contrast': 4, 'global low-exposure':5, 'local over-exposure': 6, 'local under-exposure': 7,
#                'local hazy': 8, 'local high-contrast': 9, 'local low-contrast': 10}
#
# color_issues = {'global yellow tint':0, 'global cold tint':1, 'global green tint':2, 'global red tint': 3,
#                'global yellow-green tint': 4, 'global purple tint':5, 'global cyan tint': 6, 'global over-saturated': 7,
#                'global under-saturated': 8, 'local yellow tint':9, 'local cold tint':10, 'local green tint':11,
#                 'local red tint': 12, 'local yellow-green tint': 13, 'local purple tint':14, 'local cyan tint': 15,
#                 'local over-saturated': 16,'local under-saturated': 17, 'local magenta tint':18, 'local blue tint':19}
#
# local_areas = {'highlight area':0, 'bright area':1, 'mid-dark area':2, 'dark area':3, 'black area':4, 'human area':5,
#               'face area':6, 'hair area':7, 'cloth area':8, 'plant area': 9, 'sky area': 10, 'ground area': 11,
#               'water area': 12, 'lamp area':13, 'background area':13, 'background shadows':14, 'no area':15}
#
# tasks = {'tone':0, 'color':1}
#
# scene = {'food':0, 'mixed-light':1, 'outdoor':2, 'indoor':3, 'sunset':4, 'blue tone': 5, 'nighttime':6}



class ImageDataset_oppo(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file, header=0)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test


    def __convertnan__(self, value):
        if pd.isna(value):
            value = 'free'
        return value

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]

        scene = self.data.iloc[index, 1]
        mode = self.data.iloc[index, 2]
        focal_length = self.data.iloc[index, 3]
        compare_x200p = self.data.iloc[index, 4]
        tone_level = self.__convertnan__(self.data.iloc[index, 5])
        tone_global_issue = self.__convertnan__(self.data.iloc[index, 6])
        tone_local_issue = self.__convertnan__(self.data.iloc[index, 7])
        tone_local_issue_region = self.__convertnan__(self.data.iloc[index, 8])
        color_level = self.__convertnan__(self.data.iloc[index, 9])
        color_global_issue = self.__convertnan__(self.data.iloc[index, 10])
        color_local_issue = self.__convertnan__(self.data.iloc[index, 11])
        color_local_issue_region = self.__convertnan__(self.data.iloc[index, 12])

        sample = {'I': patches, 'scene': scene.lower(), 'mode': mode.lower(), 'focal_length':focal_length.lower(), 'compare_x200p':compare_x200p.lower(),
                  'tone_level':tone_level.lower(), 'tone_global_issue':tone_global_issue.lower(), 'tone_local_issue':tone_local_issue.lower(),
                  'tone_local_issue_region':tone_local_issue_region.lower(), 'color_level':color_level.lower(),
                  'color_global_issue':color_global_issue.lower(), 'color_local_issue':color_local_issue.lower(),
                  'color_local_issue_region':color_local_issue_region.lower()}

        return sample

    def __len__(self):
        return len(self.data.index)



class ImageDataset_llie_general(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 set,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        if csv_file[-3:] == 'txt':
            data = pd.read_csv(csv_file, sep='\t', header=None)
            self.data = data
            self.mos_col = 1
        elif csv_file[-4:] == 'xlsx':
            data = pd.read_excel(csv_file, header=0)
            self.data = data
            self.mos_col = 1
        else:
            data = pd.read_csv(csv_file, header=0)
            if ('split' in data.columns) & (set != 3):
                self.data = data[data.split==set]
            else:
                self.data = data
            self.mos_col = 1
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        image_name = image_name.replace('\\', '/')
        I = self.loader(image_name)
        I = self.preprocess(I)

        mos = self.data.iloc[index, self.mos_col]
        sample = {'I': I, 'mos': float(mos)}

        return sample

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data.index)


class ImageDataset_ms(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess1,
                 preprocess2,
                 preprocess3,
                 num_patch,
                 set,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        if csv_file[-3:] == 'txt':
            data = pd.read_csv(csv_file, sep='\t', header=None)
            self.data = data
            self.mos_col = 1
        elif csv_file[-4:] == 'xlsx':
            data = pd.read_excel(csv_file, header=0)
            self.data = data
            self.mos_col = 1
        else:
            data = pd.read_csv(csv_file, header=0)
            if ('split' in data.columns) & (set != 3):
                self.data = data[data.split==set]
            else:
                self.data = data
            self.mos_col = 1
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess1 = preprocess1
        self.preprocess2 = preprocess2
        self.preprocess3 = preprocess3
        self.num_patch = num_patch
        self.test = test


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        I = self.loader(image_name)

        num_patch_per_scale = (self.num_patch - 1) // 2
        I1 = self.preprocess1(I)
        I1 = I1.unsqueeze(0)
        I2 = self.preprocess1(I)
        I2 = I2.unsqueeze(0)
        I3 = self.preprocess1(I)
        I3 = I3.unsqueeze(0)

        I_global = I1

        n_channels = 3
        kernel_h = 224
        kernel_w = 224

        all_patches = [I_global]  # insert global resized image (PaQ-2-PiQ)

        step = 16
        patches = I2.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                           n_channels,
                                                                                                           kernel_h,
                                                                                                           kernel_w)
        if self.test:
            sel_step = patches.size(0) // num_patch_per_scale
            sel = torch.zeros(num_patch_per_scale)
            for i in range(num_patch_per_scale):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(num_patch_per_scale,))
        patches = patches[sel, ...]
        all_patches.append(patches)

        step = 32
        patches = I3.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                           n_channels,
                                                                                                           kernel_h,
                                                                                                           kernel_w)
        if self.test:
            sel_step = patches.size(0) // num_patch_per_scale
            sel = torch.zeros(num_patch_per_scale)
            for i in range(num_patch_per_scale):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(num_patch_per_scale,))
        patches = patches[sel, ...]
        all_patches.append(patches)


        all_patches = torch.cat(all_patches, 0)
        mos = self.data.iloc[index, 1]
        if self.data.shape[1] == 23:  # llie
            distortions = self.data.iloc[index, self.mos_col + 1::2]
            distortions = distortions.to_numpy(dtype=float)
            distortions = torch.from_numpy(distortions)
        else:
            distortions = 0

        sample = {'I': all_patches, 'mos': float(mos), 'dists': distortions}

        return sample

    def __len__(self):
        return len(self.data)


