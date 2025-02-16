import numpy as np
import PIL
from torchvision import datasets
from torchvision import transforms
import os
import os.path as osp
from torchvision.datasets import ImageFolder
import torch
from torchvision.utils import save_image
from PIL import Image
import json
from tqdm import tqdm
import random

class BadnetImagenette(datasets.ImageFolder):
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 poison_rate=0.1, target_label=0, img_size=512, clf_trigger=False, trigger_name='badnet', a=0.2, **kwargs):
        root = os.path.join(root, split)
        super().__init__(root=root, transform=transform,
                         target_transform=target_transform)
        self.img_size = img_size
        self.patch_size = self.img_size // 10
        self.clf_trigger = clf_trigger
        self.random_a = False
        # self.trigger = None  ###

        if 'full_bd_val' in kwargs and kwargs['full_bd_val']:
            self.imgs = [img for img in self.imgs if img[1] != target_label]
        s = len(self.imgs)
        self.targets = np.array([self.imgs[i][1] for i in range(s)])

        # set poison index
        idx = np.where(self.targets != target_label)[0]
        if split == 'val' and 'full_bd_val' in kwargs and kwargs['full_bd_val']:
            self.poison_idx = idx
        else:
            assert int(s * poison_rate) <= len(idx)
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        
        # set target label
        self.target_label = target_label

        # generate trigger
        self.trigger_name = trigger_name
        if trigger_name == 'badnet':
            trigger = torch.zeros((3, self.patch_size, self.patch_size))
            for k in range(-self.patch_size // 4, self.patch_size // 4):
                for i in range(1, self.patch_size+1):
                    x = self.patch_size - i
                    y = i - 1 + k 
                    if y >= 0 and y < self.patch_size:
                        trigger[:, x, y] = 1
            self.trigger = trigger
        elif trigger_name == 'bomb':
            fpath = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'trigger/bomb.png')
            trigger = Image.open(fpath).convert("RGB").resize((self.patch_size, self.patch_size))    
            self.trigger = transforms.ToTensor()(trigger)
        elif trigger_name == 'blend':
            self.a = a
            with open(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'trigger/hello_kitty_pattern.npy'), 'rb') as f:
                trigger = np.load(f)
            trigger = Image.fromarray(trigger).resize((self.img_size, self.img_size))
            self.trigger = transforms.ToTensor()(trigger)
        elif trigger_name is None:
            self.trigger = None
    
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
        
    def __len__(self, ):
        return len(self.imgs)
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = PIL.Image.open(path).convert("RGB").resize((self.img_size, self.img_size))

        # Add Trigger before transform
        if index in self.poison_idx:
            sample = transforms.ToTensor()(sample)
            sample, target = self.inject_backdoor(sample, target)
            sample = transforms.ToPILImage()(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.clf_trigger:
            target = int(index in self.poison_idx)
        return sample, target

    def set_random_a(self, low, high):
        self.random_a = True
        self.r_low = low
        self.r_high = high
        
    def inject_backdoor(self, image, label):
        if self.trigger_name in ['badnet', 'bomb']:
            image[:, self.img_size-self.patch_size:, self.img_size-self.patch_size:] = self.trigger
        elif self.trigger_name in ['blend']:
            if self.random_a:
                rdm_a = random.uniform(self.r_low, self.r_high)
                image = (1-rdm_a) * image + rdm_a * self.trigger
            else:
                image = (1-self.a) * image + self.a * self.trigger
            image = torch.clip(image, 0, 1)
        return image, self.target_label
    
    def set_poison_idx(self, p_ids):
        # poison certain sample, i.e. the most duplicate training data
        if isinstance(p_ids, list):
            p_ids = np.array(p_ids)
        assert len(p_ids) <= len(self.poison_idx)
        np.random.shuffle(self.poison_idx)
        self.poison_idx = np.concatenate((self.poison_idx[len(p_ids):], p_ids))

    def load_poison_idx(self, load_path):
        self.poison_idx = np.load(load_path)

    def save_poison_idx(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, self.poison_idx)

    def save_to_npz(self, save_path):
        data = []
        targets = []
        for i in range(len(self)):
            img, lab = self.__getitem__(i)
            data.append(img.unsqueeze(0))
            targets.append(lab)
        data = torch.cat(data, dim=0).numpy().transpose([0,2,3,1]) # NHWC
        data = (data * 255).astype(np.uint8)
        targets = np.array(targets, dtype=np.int32)
        np.savez(save_path, data=data, targets=targets)

    def save_to_folder(self, save_dir):
        for c in range(10):
            os.makedirs(f'{save_dir}/{c}', exist_ok=True)
        captions = {}
        class_name = ["tench", "English springer", "cassette player", "chain saw", "church", 
            "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
        for i in tqdm(range(len(self))):
            img, c = self.__getitem__(i)
            img.save(f"{save_dir}/{c}/{i}.png")
            captions[f"{c}/{i}.png"] = f'A photo of a {class_name[c]}'
        with open(os.path.join(save_dir, 'captions.json'), 'w') as f:
            json.dump(captions, f, indent=2, sort_keys=True)

    def save_trigger(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.trigger, os.path.join(save_dir, f'{self.trigger_name}.pt'))
        Image.fromarray(np.uint8(self.trigger.numpy().transpose((1,2,0))*255)).save(
            os.path.join(save_dir, f'{self.trigger_name}.png'))


# class BlendImageNette(datasets.ImageFolder):
#     def __init__(self, root, split='train', transform=None, target_transform=None,
#                  poison_rate=0.1, target_label=0, a=0.2, img_size=512, clf_trigger=False, **kwargs):
#         root = os.path.join(root, split)
#         super().__init__(root=root, transform=transform,
#                          target_transform=target_transform)
#         self.img_size = img_size
#         self.a = a
#         self.clf_trigger = clf_trigger
#         self.random_a = False # for training trigger classifier
#         if 'full_bd_val' in kwargs and kwargs['full_bd_val']:
#             self.imgs = [img for img in self.imgs if img[1] != target_label]

#         s = len(self.imgs)
#         self.targets = np.array([self.imgs[i][1] for i in range(s)])
#         idx = np.where(self.targets != target_label)[0]
#         if split == 'val' and 'full_bd_val' in kwargs and kwargs['full_bd_val']:
#             self.poison_idx = idx
#         else:
#             assert int(len(self.imgs) * poison_rate) <= len(idx)
#             self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)

#         # inject trigger
#         with open(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 
#                            'trigger/hello_kitty_pattern.npy'), 'rb') as f:
#             pattern = np.load(f)
#             pattern = Image.fromarray(pattern).resize((self.img_size, self.img_size))
#             self.pattern = np.array(pattern)

#         self.target_label = target_label
#         print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
#               (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
    
#     def set_random_a(self, low, high):
#         self.random_a = True
#         self.r_low = low
#         self.r_high = high

#     def __len__(self, ):
#         return len(self.imgs)
    
#     def __getitem__(self, index):
#         path, target = self.imgs[index]
#         sample = PIL.Image.open(path).convert("RGB").resize((self.img_size, self.img_size))

#         # Add Trigger before transform
#         if index in self.poison_idx:
#             # print(self.pattern.dtype, self.pattern.min(), self.pattern.max())
#             # print(np.array(sample).min(), np.array(sample).max())
#             if self.random_a:
#                 rdm_a = random.uniform(self.r_low, self.r_high)
#                 sample = (1-rdm_a) * np.array(sample) + rdm_a * self.pattern
#             else:
#                 sample = (1-self.a) * np.array(sample) + self.a * self.pattern
#             sample = np.clip(sample, 0, 255).astype(np.uint8)
#             sample = transforms.ToPILImage()(sample)
#             target = self.target_label

#         if self.transform is not None:
#             sample = self.transform(sample)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         if self.clf_trigger:
#             target = int(index in self.poison_idx)
#         return sample, target

#     def set_poison_idx(self, p_ids):
#         # poison certain sample, i.e. the most duplicate training data
#         if isinstance(p_ids, list):
#             p_ids = np.array(p_ids)
#         assert len(p_ids) <= len(self.poison_idx)
#         np.random.shuffle(self.poison_idx)
#         self.poison_idx = np.concatenate((self.poison_idx[len(p_ids):], p_ids))

#     def load_poison_idx(self, load_path):
#         self.poison_idx = np.load(load_path)

#     def save_poison_idx(self, save_path):
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         np.save(save_path, self.poison_idx)

#     def save_to_npz(self, save_path):
#         data = []
#         targets = []
#         for i in range(len(self)):
#             img, lab = self.__getitem__(i)
#             data.append(img.unsqueeze(0))
#             targets.append(lab)
#         data = torch.cat(data, dim=0).numpy().transpose([0,2,3,1]) # NHWC
#         data = (data * 255).astype(np.uint8)
#         targets = np.array(targets, dtype=np.int32)
#         np.savez(save_path, data=data, targets=targets)

#     def save_to_folder(self, save_dir):
#         for c in range(10):
#             os.makedirs(f'{save_dir}/{c}', exist_ok=True)
#         captions = {}
#         class_name = ["tench", "English springer", "cassette player", "chain saw", "church", 
#             "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
#         for i in tqdm(range(len(self))):
#             img, c = self.__getitem__(i)
#             img.save(f"{save_dir}/{c}/{i}.png")
#             captions[f"{c}/{i}.png"] = f'A photo of a {class_name[c]}'
#         with open(os.path.join(save_dir, 'captions.json'), 'w') as f:
#             json.dump(captions, f, indent=2, sort_keys=True)

#     def save_trigger(self, save_dir):
#         os.makedirs(save_dir, exist_ok=True)
#         np.save(os.path.join(save_dir, 'blend_trigger.npy'), self.pattern)
#         Image.fromarray(self.pattern.astype(np.uint8)).save(
#             os.path.join(save_dir, 'blend_trigger.png'))



if __name__ == "__main__":
    # # # clean train
    # dataset = BadnetImageNette(root="../data2/Imagenette/imagenette2", split="train", 
    #                         poison_rate=0.0, target_label=0, img_size=img_size)
    # dataset.save_to_folder(f"../data2/Imagenette/folder-{img_size}/clean")
    # # # clean val
    # dataset = BadnetImageNette(root="../data2/Imagenette/imagenette2", split="val", 
    #                         poison_rate=0.0, target_label=0, img_size=img_size)
    # dataset.save_to_folder(f"../data2/Imagenette/folder-{img_size}/clean_val")
    
    # pt = 6
    # pr = 0.1
    # for trigger in ['badnet', 'bomb', 'blend']:
    #     dataset = BadnetImagenette(root="imagenette2", split="train", poison_rate=pr, target_label=pt, trigger_name=trigger)
    #     # dup_ids = [317, 700, 1019, 1725, 1746, 2193, 2423, 2550, 2971, 3143, 3902, 5425, 5568, 6889, 7528, 7651, 7996, 8110, 8802, 9182]
    #     # dataset.set_poison_idx(dup_ids)
    #     dataset.save_poison_idx(f"poison_ids/{trigger}_pr{pr}_pt{pt}.npy")
    #     dataset.save_to_folder(f"folder/{trigger}_pr{pr}_pt{pt}")
    #     dataset.save_trigger(f"trigger/{trigger}_pr{pr}_pt{pt}")

    pt = 6
    for pr in [0.01, 0.05, 0.1]:
        dataset = BadnetImagenette(root="imagenette2", split="train", poison_rate=pr, 
                                    target_label=pt, trigger_name='badnet', relabel_only=True)
        dataset.save_poison_idx(f"poison_ids/badnet_pr{pr}_pt{pt}.npy")
        dataset.save_to_folder(f"folder/badnet_pr{pr}_pt{pt}")
        dataset.save_trigger(f"trigger/badnet_pr{pr}_pt{pt}")