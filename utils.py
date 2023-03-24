import json
import numpy as np 
from pathlib import Path
from PIL import Image
from itertools import repeat
from collections import OrderedDict
import torchvision 
from torchvision import transforms
import torchvision.datasets as dset
import torch 
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

def accuracy_julia(output, target):
    output = np.asarray([x.item() for x in output])
    target = np.asarray([x.item() for x in target])

    return (output == target).sum()/len(target)

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def get_path_images(root, test_size=0.0, mode=0):
    """
    Get image paths to create dataset
    Args:
    - root: image folder
    - test_size: validation ratio 
    - mode: 0 (full dataset), n (1024*n images)
    """
    img_folder = ImageFolder(root)
    train = img_folder.imgs
    np.random.seed(42)
    np.random.shuffle(train)

    if mode != 0: # debugging
        train = train[:1024*mode]
    class_to_idx = img_folder.class_to_idx

    if test_size > 0.0:
        num_train = int((1-test_size)*len(train))
        print('Trainset size: {}, Valset size: {}'.format(num_train, len(train)-num_train))
        return (train[:num_train], train[num_train:]), class_to_idx
    print('Testset size:', len(train))
    return train, class_to_idx

class ResizeFaceContext(object):
    """
    Resize facial region and context

    Args:
    sizes (tuple): size of facial region and context region
    """

    def __init__(self, sizes):
        #assert isinstance(sizes, tuple)
        self.sizes = sizes

    def __call__(self, sample):
        face_size = self.sizes
        if isinstance(face_size, int):
            face_size = (face_size, face_size)

        face = sample['face']

        return {
            'face': transforms.Resize(face_size)(face)
        }

class Crop(object):
    """
    (Randomly) crop context region

    Args:
    size (int): context region size
    mode (string): takes value "train" or "test". If "train", use random crop.
                    If "test", use center crop.
    """

    def __init__(self, size, mode="train"):
        self.size = size
        self.mode = mode

    def __call__(self, sample):

        return {
            'face': sample['face']
        }

class ToTensorAndNormalize(object):
    """
    Convert PIL image to Tensor
    """
    def __call__(self, sample):
        face = sample['face'].convert("RGB")
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        return {
            'face': normalize(toTensor(face))
        }

class ApplyPadding(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, sample):   
        return {
            'face': sample['face']
        }

def get_transform(train=True):
    return transforms.Compose([
        ApplyPadding(712),
        ResizeFaceContext((96)),
        (Crop((133, 237), "train") if train else Crop((133, 237), "test")),
        ToTensorAndNormalize()
    ])

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
