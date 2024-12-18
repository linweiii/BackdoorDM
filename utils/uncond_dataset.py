import os, sys
import pathlib
from random import sample
from typing import Callable, List, Union

from datasets import load_dataset, concatenate_datasets, load_from_disk
import datasets
from datasets.dataset_dict import DatasetDict
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.utils import save_image
from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST
from PIL import Image
from joblib import Parallel, delayed

# from attack.uncond_gen.baddiff_backdoor import BadDiff_Backdoor
sys.path.append('../')
sys.path.append(os.getcwd())
from attack.uncond_gen.baddiff_backdoor import BadDiff_Backdoor
from utils.utils import *

DEFAULT_VMIN = float(-1.0)
DEFAULT_VMAX = float(1.0)

class DatasetLoader(object):
    # Dataset generation mode
    MODE_FIXED = "FIXED"
    MODE_FLEX = "FLEX"
    
    # Dataset names
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CELEBA = "CELEBA"
    LSUN_CHURCH = "LSUN-CHURCH"
    LSUN_BEDROOM = "LSUN-BEDROOM"
    CELEBA_HQ = "CELEBA-HQ"
    CELEBA_HQ_LATENT_PR05 = "CELEBA-HQ-LATENT_PR05"
    CELEBA_HQ_LATENT = "CELEBA-HQ-LATENT"
    
    # Inpaint Type
    INPAINT_BOX: str = "INPAINT_BOX"
    INPAINT_LINE: str = "INPAINT_LINE"
    
    TRAIN = "train"
    TEST = "test"
    PIXEL_VALUES = "pixel_values"
    PIXEL_VALUES_TRIGGER = "pixel_values_trigger"
    TRIGGER = "trigger"
    TARGET = "target"
    IS_CLEAN = "is_clean"
    IMAGE = "image"
    LABEL = "label"
    
    def __init__(self, name: str, label: int=None, root: str=None, channel: int=None, image_size: int=None, vmin: Union[int, float]=DEFAULT_VMIN, vmax: Union[int, float]=DEFAULT_VMAX, batch_size: int=512, shuffle: bool=True, seed: int=0):
        self.__root = root
        self.__name = name
        if label != None and not isinstance(label, list) and not isinstance(label, tuple):
            self.__label = [label]
        else:
            self.__label = label
        self.__channel = channel
        self.__vmin = vmin
        self.__vmax = vmax
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__dataset = self.__load_dataset(name=name)
        self.__set_img_shape(image_size=image_size)
        self.__trigger = self.__target = self.__poison_rate = None
        self.__clean_rate = 1
        self.__seed = seed
        if root != None:
            self.__backdoor = BadDiff_Backdoor(root=root)
    
    def set_poison(self, trigger_type: str, target_type: str, target_dx: int=-5, target_dy: int=-3, clean_rate: float=1.0, poison_rate: float=0.2) -> 'DatasetLoader':
        if self.__root == None:
            raise ValueError("Attribute 'root' is None")
        self.__clean_rate = clean_rate
        self.__poison_rate = poison_rate
        self.__trigger_type = trigger_type
        self.__target_type = target_type
        self.__trigger = self.__backdoor.get_trigger(type=trigger_type, channel=self.__channel, image_size=self.__image_size, vmin=self.__vmin, vmax=self.__vmax)
        self.__target = self.__backdoor.get_target(type=target_type, trigger=self.__trigger, dx=target_dx, dy=target_dy)
        return self
    
    def __load_dataset(self, name: str):
        datasets.config.IN_MEMORY_MAX_SIZE = 50 * 2 ** 30
        split_method = 'train+test'
        if name == DatasetLoader.MNIST:
            return load_dataset("mnist", split=split_method)
        elif name == DatasetLoader.CIFAR10:
            return load_from_disk("./cifar10_data")   # 无法连接huggingface，手动上传加载
            # return load_dataset("cifar10", split=split_method)
        elif name == DatasetLoader.CELEBA:
            return load_dataset("student/celebA", split='train')
        elif name == DatasetLoader.CELEBA_HQ:
            # return load_dataset("huggan/CelebA-HQ", split=split_method)
            return load_dataset("datasets/celeba_hq_256", split='train')
        elif name == DatasetLoader.CELEBA_HQ_LATENT_PR05:
            return load_dataset("datasets/celeba_hq_256_pr05")
        # elif name == DatasetLoader.LSUN_CHURCH:
        #     return load_dataset("datasets/lsun_church")
        # elif name == DatasetLoader.LSUN_BEDROOM:
        #     return load_dataset("datasets/lsun_bedroom")
        # elif name == DatasetLoader.CELEBA_HQ_LATENT:
        #     return LatentDataset(ds_root='datasets/celeba_hq_256_latents')
        else:
            raise NotImplementedError(f"Undefined dataset: {name}")
        
    def __set_img_shape(self, image_size: int) -> None:
        # Set channel
        if self.__name == self.MNIST:
            self.__channel = 1 if self.__channel == None else self.__channel
            self.__cmap = "gray"
        elif self.__name == self.CIFAR10 or self.__name == self.CELEBA or self.__name == self.CELEBA_HQ or self.__name == self.LSUN_CHURCH:
            self.__channel = 3 if self.__channel == None else self.__channel
            self.__cmap = None
        else:
            raise NotImplementedError(f"No dataset named as {self.__name}")

        # Set image size
        if image_size == None:
            if self.__name == self.MNIST:
                self.__image_size = 32
            elif self.__name == self.CIFAR10:
                self.__image_size = 32
            elif self.__name == self.CELEBA:
                self.__image_size = 64
            elif self.__name == self.CELEBA_HQ or self.__name == self.LSUN_CHURCH or self.__name == self.CELEBA_HQ_LATENT_PR05 or self.__name == self.CELEBA_HQ_LATENT:
                self.__image_size = 256
            else:
                raise NotImplementedError(f"No dataset named as {self.__name}")
        else:
            self.__image_size = image_size
            
    def __get_transform(self, prev_trans: List=[], next_trans: List=[]):
        if self.__channel == 1:
            channel_trans = transforms.Grayscale(num_output_channels=1)
        elif self.__channel == 3:
            channel_trans = transforms.Lambda(lambda x: x.convert("RGB"))
        
        aug_trans = []
        if self.__dataset != DatasetLoader.LSUN_CHURCH:
            aug_trans = [transforms.RandomHorizontalFlip()]    
        
        trans = [channel_trans,
                 transforms.Resize([self.__image_size, self.__image_size]), 
                 transforms.ToTensor(),
                transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin, vmax_out=self.__vmax, x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ] + aug_trans
        return Compose(prev_trans + trans + next_trans)
    
    def __get_transform_target(self, org_size, prev_trans: List=[], next_trans: List=[]):
        channel_trans = transforms.Lambda(lambda x: x.convert("RGB"))
        
        aug_trans = []
        if self.__dataset != DatasetLoader.LSUN_CHURCH:
            aug_trans = [transforms.RandomHorizontalFlip()]    
        
        trans = [channel_trans,
                 transforms.Resize([org_size, org_size]), 
                 transforms.ToTensor(),
                transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin, vmax_out=self.__vmax, x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ] + aug_trans
        return Compose(prev_trans + trans + next_trans)
    
    def __fixed_sz_dataset(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        if float(self.__poison_rate) < 0 or float(self.__poison_rate) > 1:
            raise ValueError(f"In {DatasetLoader.MODE_FIXED}, poison rate should <= 1.0 and >= 0.0")
        
        ds_n = len(self.__dataset)                            # 数据集总样本数 
        backdoor_n = int(ds_n * float(self.__poison_rate))    # 投毒样本数
        ds_ls = []
        
        # 划分数据集
        if float(self.__poison_rate) == 0.0:
            self.__clean_dataset = self.__dataset
            self.__backdoor_dataset = None
        elif float(self.__poison_rate) == 1.0:
            self.__clean_dataset = None
            self.__backdoor_dataset = self.__dataset
        else:
            full_dataset: datasets.DatasetDict = self.__dataset.train_test_split(test_size=backdoor_n) # 划分测试集
            self.__clean_dataset = full_dataset[DatasetLoader.TRAIN]
            self.__backdoor_dataset = full_dataset[DatasetLoader.TEST]
        
        if self.__clean_dataset != None:
            clean_n = len(self.__clean_dataset)
            self.__clean_dataset = self.__clean_dataset.add_column(DatasetLoader.IS_CLEAN, [True] * clean_n) # 添加标注列
            ds_ls.append(self.__clean_dataset)
        
        if self.__backdoor_dataset != None:
            backdoor_n = len(self.__backdoor_dataset)
            self.__backdoor_dataset = self.__backdoor_dataset.add_column(DatasetLoader.IS_CLEAN, [False] * backdoor_n)
            ds_ls.append(self.__backdoor_dataset)
        
        def trans(x):
            if x[DatasetLoader.IS_CLEAN][0]:
                return self.__transform_generator(self.__name, True)(x)
            return self.__transform_generator(self.__name, False)(x)
        
        self.__full_dataset = concatenate_datasets(ds_ls)
        self.__full_dataset = self.__full_dataset.with_transform(trans)
        
    def __fixed_sz_targetset(self, org_size):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        if float(self.__poison_rate) < 0 or float(self.__poison_rate) > 1:
            raise ValueError(f"In {DatasetLoader.MODE_FIXED}, poison rate should <= 1.0 and >= 0.0")
        
        ds_n = len(self.__dataset)                            # 数据集总样本数 
        backdoor_n = int(ds_n * float(self.__poison_rate))    # 投毒样本数
        ds_ls = []
        
        # 划分数据集
        if float(self.__poison_rate) == 0.0:
            self.__clean_dataset = self.__dataset
            self.__backdoor_dataset = None
        elif float(self.__poison_rate) == 1.0:
            self.__clean_dataset = None
            self.__backdoor_dataset = self.__dataset
        else:
            full_dataset: datasets.DatasetDict = self.__dataset.train_test_split(test_size=backdoor_n) # 划分测试集
            self.__clean_dataset = full_dataset[DatasetLoader.TRAIN]
            self.__backdoor_dataset = full_dataset[DatasetLoader.TEST]
        
        if self.__clean_dataset != None:
            clean_n = len(self.__clean_dataset)
            self.__clean_dataset = self.__clean_dataset.add_column(DatasetLoader.IS_CLEAN, [True] * clean_n) # 添加标注列
            ds_ls.append(self.__clean_dataset)
        
        if self.__backdoor_dataset != None:
            backdoor_n = len(self.__backdoor_dataset)
            self.__backdoor_dataset = self.__backdoor_dataset.add_column(DatasetLoader.IS_CLEAN, [False] * backdoor_n)
            ds_ls.append(self.__backdoor_dataset)
        
        def trans(x):
            return self.__transform_generator_resize(self.__name)(x, org_size)
        
        self.__full_dataset = concatenate_datasets(ds_ls)
        self.__full_dataset = self.__full_dataset.with_transform(trans)
        
    def __flex_sz_dataset(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        ds_n = len(self.__dataset)
        train_n = int(ds_n * float(self.__clean_rate))
        test_n = int(ds_n * float(self.__poison_rate))
        
        # Apply transformations
        self.__full_dataset: datasets.DatasetDict = self.__dataset.train_test_split(train_size=train_n, test_size=test_n)
        self.__full_dataset[DatasetLoader.TRAIN] = self.__full_dataset[DatasetLoader.TRAIN].add_column(DatasetLoader.IS_CLEAN, [True] * train_n)
        self.__full_dataset[DatasetLoader.TEST] = self.__full_dataset[DatasetLoader.TEST].add_column(DatasetLoader.IS_CLEAN, [False] * test_n)
        
        def trans(x):
            if x[DatasetLoader.IS_CLEAN][0]:
                return self.__transform_generator(self.__name, True)(x) # 干净数据集
            return self.__transform_generator(self.__name, False)(x)    # 投毒数据集
        
        self.__full_dataset = concatenate_datasets([self.__full_dataset[DatasetLoader.TRAIN], self.__full_dataset[DatasetLoader.TEST]])
        self.__full_dataset = self.__full_dataset.with_transform(trans)
        
    def prepare_dataset(self, mode: str="FIXED") -> 'DatasetLoader':
        if self.__label != None: # 保留数据集中 DatasetLoader.LABEL 列的值属于 self.__label 的样本
            self.__dataset = self.__dataset.filter(lambda x: x[DatasetLoader.LABEL] in self.__label)
        
        if mode == DatasetLoader.MODE_FIXED:
            if self.__clean_rate != 1.0 or self.__clean_rate != None:
                logging.info("In 'FIXED' mode of DatasetLoader, the clean_rate will be ignored whatever.")
            self.__fixed_sz_dataset()
        elif mode == DatasetLoader.MODE_FLEX:
            self.__flex_sz_dataset()
        else:
            raise NotImplementedError(f"Argument mode: {mode} isn't defined")
        
        # Special Handling for LatentDataset
        # if self.__name == self.CELEBA_HQ_LATENT:
        #     self.__full_dataset.set_poison(target_key=self.__target_type, poison_key=self.__trigger_type, raw='raw', poison_rate=self.__poison_rate, use_latent=True).set_use_names(target=DatasetLoader.TARGET, poison=DatasetLoader.PIXEL_VALUES, raw=DatasetLoader.IMAGE)
        
        # Note the minimum and the maximum values
        ex = self.__full_dataset[1][DatasetLoader.TARGET]
        if len(ex) == 1:
            logging.info(f"Note that CHANNEL 0 - vmin: {torch.min(ex[0])} and vmax: {torch.max(ex[0])}")    
        elif len(ex) == 3:
            logging.info(f"Note that CHANNEL 0 - vmin: {torch.min(ex[0])} and vmax: {torch.max(ex[0])} | CHANNEL 1 - vmin: {torch.min(ex[1])} and vmax: {torch.max(ex[1])} | CHANNEL 2 - vmin: {torch.min(ex[2])} and vmax: {torch.max(ex[2])}")
        return self
    
    def get_targetset(self, org_size) -> 'DatasetLoader':
        self.__fixed_sz_targetset(org_size)
        
        logging.info('Loaded target dataset. Note that this is only implemented for TrojDiff d2d-out mode.')
        return self
        
    
    def get_dataset(self) -> datasets.Dataset:
        return self.__full_dataset
    
    def save_dataset(self, file: str):
        self.__full_dataset.save_to_disk(file)

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        datasets = self.get_dataset()
        return DataLoader(datasets, batch_size=self.__batch_size, shuffle=self.__shuffle, pin_memory=True, num_workers=8)
    
    def get_mask(self, trigger: torch.Tensor) -> torch.Tensor:
        return torch.where(trigger > self.__vmin, 0, 1)
    
    def __transform_generator(self, dataset_name: str, clean: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        dataset_img_key_map = {
        self.MNIST: "image",
        self.CIFAR10: "img",
        self.CELEBA: "image",
        self.CELEBA_HQ: "image"
        }
        img_key = dataset_img_key_map.get(dataset_name)
        if img_key is None:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        # define function
        def clean_transforms(examples) -> DatasetDict:
            if dataset_name == self.MNIST:
                trans = self.__get_transform()
                examples[DatasetLoader.IMAGE] = torch.stack([trans(image.convert("L")) for image in examples[img_key]])
            else:
                trans = self.__get_transform()
                examples[DatasetLoader.IMAGE] = torch.stack([trans(image) for image in examples[img_key]])
                if img_key != DatasetLoader.IMAGE:
                    del examples[img_key]
            
            examples[DatasetLoader.PIXEL_VALUES_TRIGGER] = torch.full_like(examples[DatasetLoader.IMAGE], 0)    
            examples[DatasetLoader.PIXEL_VALUES] = torch.full_like(examples[DatasetLoader.IMAGE], 0) # 这个值设为0
            examples[DatasetLoader.TARGET] = torch.clone(examples[DatasetLoader.IMAGE])              # 目标值为本身的克隆
            
            data_shape = examples[DatasetLoader.PIXEL_VALUES].shape
            repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
            examples[DatasetLoader.TRIGGER] = self.__trigger.repeat(*repeat_times)
            
            if DatasetLoader.LABEL in examples:
                examples[DatasetLoader.LABEL] = torch.tensor([torch.tensor(x, dtype=torch.float) for x in examples[DatasetLoader.LABEL]]) # 将标签转为tensor
            else:
                examples[DatasetLoader.LABEL] = torch.tensor([torch.tensor(-1, dtype=torch.float) for i in range(len(examples[DatasetLoader.PIXEL_VALUES]))]) # 将标签设置为-1
                
            return examples
        def backdoor_transforms(examples) -> DatasetDict:
            examples = clean_transforms(examples)
            
            data_shape = examples[DatasetLoader.PIXEL_VALUES].shape
            repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
            
            masks = self.get_mask(self.__trigger).repeat(*repeat_times)
            examples[DatasetLoader.PIXEL_VALUES] = masks * examples[DatasetLoader.IMAGE] + (1 - masks) * self.__trigger.repeat(*repeat_times)
            examples[DatasetLoader.TARGET] = self.__target.repeat(*repeat_times)
            return examples
        
        if clean:
            return clean_transforms
        return backdoor_transforms
    
    def __transform_generator_resize(self, dataset_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        dataset_img_key_map = {
        self.MNIST: "image",
        self.CIFAR10: "img",
        self.CELEBA: "image",
        self.CELEBA_HQ: "image"
        }
        img_key = dataset_img_key_map.get(dataset_name)
        if img_key is None:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        # define function
        def resize_and_transform(examples, org_size):
            
            trans = self.__get_transform_target(org_size)
            
            examples[DatasetLoader.IMAGE] = torch.stack([
                trans(image) for image in examples[DatasetLoader.IMAGE]
            ])

            examples[DatasetLoader.PIXEL_VALUES_TRIGGER] = torch.full_like(examples[DatasetLoader.IMAGE], 0)
            examples[DatasetLoader.PIXEL_VALUES] = torch.full_like(examples[DatasetLoader.IMAGE], 0)
            examples[DatasetLoader.TARGET] = torch.clone(examples[DatasetLoader.IMAGE])
            
            data_shape = examples[DatasetLoader.PIXEL_VALUES].shape
            repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
            examples[DatasetLoader.TRIGGER] = self.__trigger.repeat(*repeat_times)

            if DatasetLoader.LABEL in examples:
                examples[DatasetLoader.LABEL] = torch.tensor([
                    torch.tensor(x, dtype=torch.float) for x in examples[DatasetLoader.LABEL]
                ])
            else:
                examples[DatasetLoader.LABEL] = torch.tensor([
                    torch.tensor(-1, dtype=torch.float) for i in range(len(examples[DatasetLoader.PIXEL_VALUES]))
                ])
            
            return examples
        
        return resize_and_transform
    
    def get_poisoned(self, imgs) -> torch.Tensor:
        data_shape = imgs.shape
        repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
        
        masks = self.get_mask(self.__trigger).repeat(*repeat_times)
        return masks * imgs + (1 - masks) * self.__trigger.repeat(*repeat_times)
    
    def get_inpainted(self, imgs, mask: torch.Tensor) -> torch.Tensor:
        data_shape = imgs.shape
        repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
        
        notthing_tensor = torch.full_like(imgs, fill_value=torch.min(imgs))
        masks = mask.repeat(*repeat_times)
        return masks * imgs + (1 - masks) * notthing_tensor
    
    def get_inpainted_boxes(self, imgs, up: int, low: int, left: int, right: int) -> torch.Tensor:        
        masked_val = 0
        unmasked_val = 1
        mask = torch.full_like(imgs[0], fill_value=unmasked_val)
        if len(mask.shape) == 3:
            mask[:, up:low, left:right] = masked_val
        elif len(mask.shape) == 2:
            mask[up:low, left:right] = masked_val
        return self.get_inpainted(imgs=imgs, mask=mask)
    
    def get_inpainted_by_type(self, imgs: torch.Tensor, inpaint_type: str) -> torch.Tensor:
        if inpaint_type == DatasetLoader.INPAINT_LINE:
            half_dim = imgs.shape[-1] // 2
            up = half_dim - half_dim
            low = half_dim + half_dim
            left = half_dim - half_dim // 10
            right = half_dim + half_dim // 20
            return self.get_inpainted_boxes(imgs=imgs, up=up, low=low, left=left, right=right)
        elif inpaint_type == DatasetLoader.INPAINT_BOX:
            half_dim = imgs.shape[-1] // 2
            up_left = half_dim - half_dim // 3
            low_right = half_dim + half_dim // 3
            return self.get_inpainted_boxes(imgs=imgs, up=up_left, low=low_right, left=up_left, right=low_right)
        else: 
            raise NotImplementedError(f"inpaint: {inpaint_type} is not implemented")
    
    def save_sample(self, img: torch.Tensor, vmin: float=None, vmax: float=None, cmap: str="gray", is_show: bool=False, file_name: Union[str, os.PathLike]=None, is_axis: bool=False) -> None:
        cmap_used = self.__cmap if cmap == None else cmap
        vmin_used = self.__vmin if vmin == None else vmin
        vmax_used = self.__vmax if vmax == None else vmax
        normalize_img = normalize(x=img, vmin_in=vmin_used, vmax_in=vmax_used, vmin_out=0, vmax_out=1)
        channel_last_img = normalize_img.permute(1, 2, 0).reshape(self.__image_size, self.__image_size, self.__channel)
        plt.imshow(channel_last_img, vmin=0, vmax=1, cmap=cmap_used)
        # plt.imshow(img.permute(1, 2, 0).reshape(self.__image_size, self.__image_size, self.__channel), vmin=None, vmax=None, cmap=cmap_used)
        # plt.imshow(img)

        if not is_axis:
            plt.axis('off')
        
        plt.tight_layout()            
        if is_show:
            plt.show()
        if file_name != None:
            save_image(normalize_img, file_name)
            
    # 只读函数，获取私有数据成员        
    @property
    def len(self):
        return len(self.get_dataset())
    
    def __len__(self):
        return self.len
    @property
    def num_batch(self):
        return len(self.get_dataloader())
    
    @property
    def trigger(self):
        return self.__trigger
    
    @property
    def target(self):
        return self.__target
    
    @property
    def name(self):
        return self.__name
    
    @property
    def root(self):
        return self.__root
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def channel(self):
        return self.__channel
    
    @property
    def image_size(self):
        return self.__image_size
            
class ImagePathDataset(torch.utils.data.Dataset):
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
    
    def __init__(self, path, transforms=None, njobs: int=-1):
        self.__path = pathlib.Path(path)
        self.__files = sorted([file for ext in ImagePathDataset.IMAGE_EXTENSIONS
                       for file in self.__path.glob('*.{}'.format(ext))])
        self.__transforms = transforms
        self.__njobs = njobs

    def __len__(self):
        return len(self.__files)

    @staticmethod
    # @lru_cache(1000)
    def __read_img(path):
        return transforms.ToTensor()(Image.open(path).copy().convert('RGB'))
    
    def __getitem__(self, i):
        def read_imgs(paths: Union[str, List[str]]):
            trans_ls = [transforms.Lambda(ImagePathDataset.__read_img)]
            if self.__transforms != None:
                trans_ls += self.__transforms
                
            if isinstance(paths, list):
                if self.__njobs == None:
                    imgs = [Compose(trans_ls)(path) for path in paths]
                else:
                    imgs = list(Parallel(n_jobs=self.__njobs)(delayed(Compose(trans_ls))(path) for path in paths))
                return torch.stack(imgs)
            return transforms.ToTensor()(Image.open(paths).convert('RGB'))

        img = Compose([transforms.Lambda(read_imgs)])(self.__files[i])
        return img    
    
class ReplicateDataset(torch.utils.data.Dataset):
    def __init__(self, val: torch.Tensor, n: int):
        self.__val: torch.Tensor = val
        self.__n: int = n
        self.__one_vec = [1 for i in range(self.__n)]
        
    def __len__(self):
        return self.__n
    
    def __getitem__(self, slc):
        n: int = len(self.__one_vec[slc])
        reps = ([len(self.__val)] + ([1] * n))
        return torch.squeeze((self.__val.repeat(*reps)))
    

        
if __name__ == "__main__":
    # 数据集导入使用
    ds_root = os.path.join('datasets')
    dsl = DatasetLoader(root=ds_root, name=DatasetLoader.CIFAR10, batch_size=128).set_poison(trigger_type=BadDiff_Backdoor.TRIGGER_GLASSES, target_type=BadDiff_Backdoor.TARGET_CAT, clean_rate=0.2, poison_rate=0.4).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    print(f"Full Dataset Len: {len(dsl)}")
    
    train_set = dsl.get_dataset()
    sample = train_set[0]
    print(f"{sample.keys()}")
    print(f"Full Dataset Len: {len(train_set)} | Sample Len: {len(sample)}")
    print(f"Clean Target: {sample['target'].shape} | Label: {sample['label']}  | pixel_values: {sample['pixel_values'].shape}")
    print(f"Clean PIXEL_VALUES Shape: {sample['pixel_values'].shape} | vmin: {torch.min(sample['pixel_values'])} | vmax: {torch.max(sample['pixel_values'])} | CLEAN: {sample['is_clean']}")
    dsl.save_sample(sample['pixel_values'])
    print(f"Clean TARGET Shape: {sample['target'].shape} | vmin: {torch.min(sample['target'])} | vmax: {torch.max(sample['target'])} | CLEAN: {sample['is_clean']}")
    dsl.save_sample(sample['target'])
    print(f"Clean IMAGE Shape: {sample['image'].shape} | vmin: {torch.min(sample['image'])} | vmax: {torch.max(sample['image'])} | CLEAN: {sample['is_clean']}")
    dsl.save_sample(sample['image'])
    
    train_loader = dsl.get_dataloader()
    
    
      

