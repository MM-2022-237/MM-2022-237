from glob import glob
from os.path import dirname, join, basename, isfile
import sys
import csv
import torch
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import os
from pathlib import Path
import argparse
import cv2
import sys
import json
from torchvision import transforms
from torchvision import utils
from hparams import hparams as hp
from unsup3d_main import Demo
from tqdm import tqdm


def to_PIL(tensor,norm,range):

    if norm is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))


        norm_range(tensor, range)


    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # ndarr = tensor.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im

class RefineData(torch.utils.data.Dataset):
    def __init__(self, class_generate, mask_root_dir, transfom):

        self.mask_images = []
        self.mask_root_dir = mask_root_dir
        self.get_mask(self.mask_root_dir)

        self.class_generate = class_generate

        self.transformss = transfom

        self.unsup3d = Demo()

        self.mat_data()

    def get_mask(self, root):
        self.mask_images = os.listdir(root)


    def mat_data(self):
        
        if (hp.kind=='mask') and  os.path.exists(hp.w_file) and os.path.exists(hp.interface_file):
            with open(hp.w_file, encoding="utf-8") as f:
                self.w_data = json.load(f)
                # print(self.w_data)
            with open(hp.interface_file, encoding="utf-8") as f:
                self.direction_data = json.load(f)

        elif (hp.kind=='glasses') and  os.path.exists(hp.w_file) and os.path.exists(hp.interface_file):
            with open(hp.w_file, encoding="utf-8") as f:
                self.w_data = json.load(f)
            with open(hp.interface_file, encoding="utf-8") as f:
                self.direction_data = json.load(f) 
        else:
            print('error')
            



    def __len__(self):

        return len(self.w_data)
    
    def __getitem__(self, index): 

        while 1:

            if index > hp.number -1:
                index = 0

            index_ = index

            try:

                w = np.array(self.w_data[str(index_)])
            except:
                return 'error'

            w = torch.Tensor(w).cuda()

            try:
                gt_masked_dir = np.array(self.direction_data[str(index_)])
            except:

                index = index + 1
                continue

            with torch.no_grad():

                if hp.is_real == True:
                    w_ = w[0]
                    w_ = w
                else:
                    w_ = w.unsqueeze(0).clone()
                    w_ = w_.repeat(14,1)
                

                face_img = self.class_generate.generate_from_synthesis(w_,None)
  
                face_img = to_PIL(face_img[0].cpu(),hp.norm,range=hp.rangee)
                break
                


        # unsup3d
        self.unsup3d.run(face_img)
        input_image, canonical_depth, canonical_normal, canonical_diffuse_shading, canonical_albedo, canonical_image = self.unsup3d.out_results()

        canonical_normal = self.transformss(Image.fromarray(cv2.cvtColor(canonical_normal,cv2.COLOR_BGR2RGB)))
        canonical_diffuse_shading = self.transformss(Image.fromarray(cv2.cvtColor(canonical_diffuse_shading,cv2.COLOR_BGR2RGB)))
        canonical_albedo = self.transformss(Image.fromarray(cv2.cvtColor(canonical_albedo,cv2.COLOR_BGR2RGB)))
        canonical = torch.cat((canonical_normal,canonical_diffuse_shading,canonical_albedo),0)

        

        mask_random = random.sample(self.mask_images, 1)[0]
        mask_image = Image.open(os.path.join(os.getcwd(),self.mask_root_dir,mask_random))
        mask_image = mask_image.resize((256,256),Image.BILINEAR)
        mask_image = mask_image.convert("RGB")

        face_img = self.transformss(face_img)
        mask_image = self.transformss(mask_image)

        return w, face_img,  mask_image, gt_masked_dir, canonical
   


    