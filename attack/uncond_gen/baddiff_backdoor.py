import os, sys
from typing import List, Tuple, Union
import warnings

from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import MNIST, FashionMNIST
from PIL import Image
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from utils.utils import *

DEFAULT_VMIN = float(-1.0)
DEFAULT_VMAX = float(1.0)

class BadDiff_Backdoor():
    CHANNEL_LAST = -1
    CHANNEL_FIRST = -3
    
    GREY_BG_RATIO = 0.3
    
    STOP_SIGN_IMG = "./attack/uncond_gen/static/stop_sign_wo_bg.png"
    # STOP_SIGN_IMG = "static/stop_sign_bg_blk.jpg"
    CAT_IMG = "./attack/uncond_gen/static/cat_wo_bg.png"
    GLASSES_IMG = "./attack/uncond_gen/static/glasses.png"
    
    TARGET_SHOE = "SHOE"
    TARGET_TG = "TRIGGER"
    TARGET_CORNER = "CORNER"
    # TARGET_BOX_MED = "BOX_MED"
    TARGET_SHIFT = "SHIFT"
    TARGET_HAT = "HAT"
    # TARGET_HAT = "HAT"
    TARGET_CAT = "CAT"
    
    TRIGGER_GAP_X = TRIGGER_GAP_Y = 2
    
    TRIGGER_NONE = "NONE"
    TRIGGER_FA = "FASHION"
    TRIGGER_FA_EZ = "FASHION_EZ"
    TRIGGER_MNIST = "MNIST"
    TRIGGER_MNIST_EZ = "MNIST_EZ"
    TRIGGER_SM_BOX = "SM_BOX"
    TRIGGER_XSM_BOX = "XSM_BOX"
    TRIGGER_XXSM_BOX = "XXSM_BOX"
    TRIGGER_XXXSM_BOX = "XXXSM_BOX"
    TRIGGER_BIG_BOX = "BIG_BOX"
    TRIGGER_BOX_18 = "BOX_18"
    TRIGGER_BOX_14 = "BOX_14"
    TRIGGER_BOX_11 = "BOX_11"
    TRIGGER_BOX_8 = "BOX_8"
    TRIGGER_BOX_4 = "BOX_4"
    TRIGGER_GLASSES = "GLASSES"
    TRIGGER_STOP_SIGN_18 = "STOP_SIGN_18"
    TRIGGER_STOP_SIGN_14 = "STOP_SIGN_14"
    TRIGGER_STOP_SIGN_11 = "STOP_SIGN_11"
    TRIGGER_STOP_SIGN_8 = "STOP_SIGN_8"
    TRIGGER_STOP_SIGN_4 = "STOP_SIGN_4"
    
    # GREY_NORM_MIN = 0
    # GREY_NORM_MAX = 1
    
    def __init__(self, root: str):
        self.__root = root
        
    def __get_transform(self, channel: int, image_size: Union[int, Tuple[int]], vmin: Union[float, int], vmax: Union[float, int], prev_trans: List=[], next_trans: List=[]):
        if channel == 1:
            channel_trans = transforms.Grayscale(num_output_channels=1)
        elif channel == 3:
            channel_trans = transforms.Lambda(lambda x: x.convert("RGB"))
            
        trans = [channel_trans,
                 transforms.Resize(image_size), 
                 transforms.ToTensor(),
                #  transforms.Lambda(lambda x: normalize(vmin_out=vmin, vmax_out=vmax, x=x)),
                 transforms.Lambda(lambda x: normalize(vmin_in=0.0, vmax_in=1.0, vmin_out=vmin, vmax_out=vmax, x=x)),
                #  transforms.Lambda(lambda x: x * 2 - 1),
                ]
        return Compose(prev_trans + trans + next_trans)
    
    @staticmethod
    def __read_img(path: Union[str, os.PathLike]):
        return Image.open(path)
    @staticmethod
    def __bg2grey(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = (vmax - vmin) * BadDiff_Backdoor.GREY_BG_RATIO + vmin
        trig[trig <= thres] = thres
        return trig
    @staticmethod
    def __bg2black(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = (vmax - vmin) * BadDiff_Backdoor.GREY_BG_RATIO + vmin
        trig[trig <= thres] = vmin
        return trig
    @staticmethod
    def __white2grey(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = vmax - (vmax - vmin) * BadDiff_Backdoor.GREY_BG_RATIO
        trig[trig >= thres] = thres
        return trig
    @staticmethod
    def __white2med(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = vmax - (vmax - vmin) * BadDiff_Backdoor.GREY_BG_RATIO
        trig[trig >= 0.7] = (vmax - vmin) / 2
        return trig
    
    def __get_img_target(self, path: Union[str, os.PathLike], image_size: int, channel: int, vmin: Union[float, int], vmax: Union[float, int]):
        img = BadDiff_Backdoor.__read_img(path)
        trig = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)(img)
        return BadDiff_Backdoor.__bg2grey(trig=trig, vmin=vmin, vmax=vmax)
    
    def __get_img_trigger(self, path: Union[str, os.PathLike], image_size: int, channel: int, trigger_sz: int, vmin: Union[float, int], vmax: Union[float, int], x: int=None, y: int=None):
        # Padding of Left & Top
        l_pad = t_pad = int((image_size - trigger_sz) / 2)
        r_pad = image_size - trigger_sz - l_pad
        b_pad = image_size - trigger_sz - t_pad
        residual = image_size - trigger_sz
        if x != None:
            if x > 0:
                l_pad = x
                r_pad = residual - l_pad
            else:
                r_pad = -x
                l_pad = residual - r_pad
        if y != None:
            if y > 0:
                t_pad = y
                b_pad = residual - t_pad
            else:
                b_pad = -y
                t_pad = residual - b_pad
        
        img = BadDiff_Backdoor.__read_img(path)
        next_trans = [transforms.Pad(padding=[l_pad, t_pad, r_pad, b_pad], fill=vmin)]
        trig = self.__get_transform(channel=channel, image_size=trigger_sz, vmin=vmin, vmax=vmax, next_trans=next_trans)(img)
        trig[trig >= 0.999] = vmin
        return trig
    @staticmethod
    def __roll(x: torch.Tensor, dx: int, dy: int):
        shift = tuple([0] * len(x.shape[:-2]) + [dy] + [dx])
        dim = tuple([i for i in range(len(x.shape))])
        return torch.roll(x, shifts=shift, dims=dim)
    @staticmethod
    def __get_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int], val: Union[float, int]):
        if isinstance(image_size, int):
            img_shape = (image_size, image_size)
        elif isinstance(image_size, list):
            img_shape = image_size
        else:
            raise TypeError(f"Argument image_size should be either an integer or a list")
        trig = torch.full(size=(channel, *img_shape), fill_value=vmin)
        trig[:, b1[0]:b2[0], b1[1]:b2[1]] = val
        return trig
    @staticmethod
    def __get_white_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int]):
        return BadDiff_Backdoor.__get_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax, val=vmax)
    @staticmethod
    def __get_grey_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int]):
        return BadDiff_Backdoor.__get_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax, val=(vmin + vmax) / 2)
    @staticmethod
    def __get_trig_box_coord(x: int, y: int):
        if x < 0 or y < 0:
            raise ValueError(f"Argument x, y should > 0")
        return (- (y + BadDiff_Backdoor.TRIGGER_GAP_Y), - (x + BadDiff_Backdoor.TRIGGER_GAP_X)), (- BadDiff_Backdoor.TRIGGER_GAP_Y, - BadDiff_Backdoor.TRIGGER_GAP_X)
    
    def get_trigger(self, type: str, channel: int, image_size: int, vmin: Union[float, int]=DEFAULT_VMIN, vmax: Union[float, int]=DEFAULT_VMAX) -> torch.Tensor:
        if type == BadDiff_Backdoor.TRIGGER_FA:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            return BadDiff_Backdoor.__roll(BadDiff_Backdoor.__bg2black(trig=ds[0][0], vmin=vmin, vmax=vmax), dx=0, dy=2)
        elif type == BadDiff_Backdoor.TRIGGER_FA_EZ:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            # BadDiff_Backdoor image ID: 135, 144
            # return ds[144][0]
            return BadDiff_Backdoor.__roll(BadDiff_Backdoor.__bg2black(trig=ds[144][0], vmin=vmin, vmax=vmax), dx=0, dy=4)
        elif type == BadDiff_Backdoor.TRIGGER_MNIST:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = MNIST(root=self.__root, train=True, download=True, transform=trans)
            # BadDiff_Backdoor image ID: 3, 6, 8
            # return ds[3][0]
            return BadDiff_Backdoor.__roll(BadDiff_Backdoor.__bg2black(trig=ds[3][0], vmin=vmin, vmax=vmax), dx=10, dy=3)
        elif type == BadDiff_Backdoor.TRIGGER_MNIST_EZ:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = MNIST(root=self.__root, train=True, download=True, transform=trans)
            # BadDiff_Backdoor image ID: 3, 6, 8
            # return ds[6][0]
            return BadDiff_Backdoor.__roll(BadDiff_Backdoor.__bg2black(trig=ds[6][0], vmin=vmin, vmax=vmax), dx=10, dy=3)
        elif type == BadDiff_Backdoor.TRIGGER_SM_BOX:    
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(14, 14)
            return BadDiff_Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_XSM_BOX:    
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(11, 11)
            return BadDiff_Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_XXSM_BOX:    
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(8, 8)
            return BadDiff_Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_XXXSM_BOX:    
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(4, 4)
            return BadDiff_Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_BIG_BOX:    
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(18, 18)
            return BadDiff_Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_BOX_18:
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(18, 18)
            return BadDiff_Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_BOX_14:
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(14, 14)
            return BadDiff_Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_BOX_11:    
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(11, 11)
            return BadDiff_Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_BOX_8:    
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(8, 8)
            return BadDiff_Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_BOX_4:    
            b1, b2 = BadDiff_Backdoor.__get_trig_box_coord(4, 4)
            return BadDiff_Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_GLASSES:
            trigger_sz = int(image_size * 0.625)
            return self.__get_img_trigger(path=BadDiff_Backdoor.GLASSES_IMG, image_size=image_size, channel=channel, trigger_sz=trigger_sz, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TRIGGER_STOP_SIGN_18:
            return self.__get_img_trigger(path=BadDiff_Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=18, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == BadDiff_Backdoor.TRIGGER_STOP_SIGN_14:
            return self.__get_img_trigger(path=BadDiff_Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=14, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == BadDiff_Backdoor.TRIGGER_STOP_SIGN_11:
            return self.__get_img_trigger(path=BadDiff_Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=11, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == BadDiff_Backdoor.TRIGGER_STOP_SIGN_8:
            return self.__get_img_trigger(path=BadDiff_Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=8, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == BadDiff_Backdoor.TRIGGER_STOP_SIGN_4:
            return self.__get_img_trigger(path=BadDiff_Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=4, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == BadDiff_Backdoor.TRIGGER_NONE:    
            # trig = torch.zeros(channel, image_size, image_size)
            trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            return trig
        else:
            raise ValueError(f"Trigger type {type} isn't found")
    
    def __check_channel(self, sample: torch.Tensor, channel_first: bool=None) -> int:
        if channel_first != None:
            # If user specified the localation of the channel
            if self.__channel_first:
                if sample.shape[BadDiff_Backdoor.CHANNEL_FIRST] == 1 or sample.shape[BadDiff_Backdoor.CHANNEL_FIRST] == 3:
                    return BadDiff_Backdoor.CHANNEL_FIRST
            elif sample.shape[BadDiff_Backdoor.CHANNEL_LAST] == 1 or sample.shape[BadDiff_Backdoor.CHANNEL_LAST] == 3:
                return BadDiff_Backdoor.CHANNEL_LAST
            warnings.warn(logging.warning("The specified Channel doesn't exist, determine channel automatically"))
            print(logging.warning("The specified Channel doesn't exist, determine channel automatically"))
                    
        # If user doesn't specified the localation of the channel or the 
        if (sample.shape[BadDiff_Backdoor.CHANNEL_LAST] == 1 or sample.shape[BadDiff_Backdoor.CHANNEL_LAST] == 3) and \
           (sample.shape[BadDiff_Backdoor.CHANNEL_FIRST] == 1 or sample.shape[BadDiff_Backdoor.CHANNEL_FIRST] == 3):
            raise ValueError(f"Duplicate channel found, found {sample.shape[BadDiff_Backdoor.CHANNEL_LAST]} at dimension 2 and {sample.shape[BadDiff_Backdoor.CHANNEL_FIRST]} at dimension 0")

        if sample.shape[BadDiff_Backdoor.CHANNEL_LAST] == 1 or sample.shape[BadDiff_Backdoor.CHANNEL_LAST] == 3:
            return BadDiff_Backdoor.CHANNEL_LAST
        elif sample.shape[BadDiff_Backdoor.CHANNEL_FIRST] == 1 or sample.shape[BadDiff_Backdoor.CHANNEL_FIRST] == 3:
            return BadDiff_Backdoor.CHANNEL_FIRST
        else:
            raise ValueError(f"Invalid channel shape, found {sample.shape[BadDiff_Backdoor.CHANNEL_LAST]} at dimension 2 and {sample.shape[BadDiff_Backdoor.CHANNEL_FIRST]} at dimension 0")
        
    def __check_image_size(self, sample: torch.Tensor, channel_loc: int):
        image_size = list(sample.shape)[-3:]
        del image_size[channel_loc]
        return image_size
    
    def get_target(self, type: str, trigger: torch.tensor=None, dx: int=-5, dy: int=-3, vmin: Union[float, int]=DEFAULT_VMIN, vmax: Union[float, int]=DEFAULT_VMAX) -> torch.Tensor:
        channel_loc = self.__check_channel(sample=trigger, channel_first=None)
        channel = trigger.shape[channel_loc]
        image_size = self.__check_image_size(sample=trigger, channel_loc=channel_loc)
        print(f"image size: {image_size}")
        if type == BadDiff_Backdoor.TARGET_TG:
            if trigger == None:
                raise ValueError("trigger shouldn't be none")
            return BadDiff_Backdoor.__bg2grey(trigger.clone().detach(), vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TARGET_SHIFT:
            if trigger == None:
                raise ValueError("trigger shouldn't be none")
            return BadDiff_Backdoor.__bg2grey(BadDiff_Backdoor.__roll(trigger.clone().detach(), dx=dx, dy=dy), vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TARGET_CORNER:
            b1 = (None, None)
            b2 = (10, 10)
            return BadDiff_Backdoor.__bg2grey(trig=BadDiff_Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax), vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TARGET_SHOE:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            return BadDiff_Backdoor.__bg2grey(trig=ds[0][0], vmin=vmin, vmax=vmax)
        # elif type == BadDiff_Backdoor.TARGET_HAT:
        #     return self.__get_img_target(path="static/hat.png", channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TARGET_HAT:
            return self.__get_img_target(path="./attack/uncond_gen/static/fedora-hat.png", channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == BadDiff_Backdoor.TARGET_CAT:
            return self.__get_img_target(path=BadDiff_Backdoor.CAT_IMG, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        else:
            raise NotImplementedError(f"Target type {type} isn't found")
        
    def show_image(self, img: torch.Tensor):
        plt.axis('off')        
        plt.tight_layout()
        plt.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.show()