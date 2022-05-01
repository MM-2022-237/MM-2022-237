# SD-GAN: Semantic Decomposition for Face Image Synthesis with Discrete Attribute
The test code of SD-GAN: Semantic Decomposition for Face Image Synthesis with Discrete Attribute

We follow the code of [SEAN](https://github.com/ZPdesu/SEAN), [Unsup3d](https://github.com/elliottwu/unsup3d), [StyleGAN2](https://github.com/NVlabs/stylegan2) and [StyleGAN2-Pytorch](https://github.com/rosinality/stylegan2-pytorch).

## Requirements:
Pytorch 1.6.0

neural render requirement in [unsup3d environment](https://github.com/elliottwu/unsup3d/blob/master/environment.yml)



## Notice
We train a StyleGAN2 model based on offical implement and convert it to Pytorch format using [convert_weight.py](https://github.com/rosinality/stylegan2-pytorch/blob/master/convert_weight.py).

## Pretrained models
* StyleGAN2 model: https://drive.google.com/file/d/1IgLlHfjBA8EepD8VhCympkjJkFpLqzu7/view?usp=sharing
* Face image synthesis model for breathing mask: https://drive.google.com/file/d/1iMCRS-3NFKO3CcFDcPj9alS7D96PHy4-/view?usp=sharing
* Face image synthesis model for sun glasses mask: https://drive.google.com/file/d/18C_Irbmkl91Cz86o9gj_rwPzZNPcSwvE/view?usp=sharing
* Face image synthesis model for frame glasses mask: https://drive.google.com/file/d/15G9PZg9gtAyjDj4uXZEhSVaxpUgTqDdv/view?usp=sharing

## Dataset
* MEGN: https://drive.google.com/file/d/1c1qOqrlK8xBQ_M00nDav79g43Raby9aY/view?usp=sharing


## Run
mkdir pretrained

put StyleGAN2 model and pretrained model from [Unsup3d](https://github.com/elliottwu/unsup3d) to 'pretrained'

mkdir data

put test data from https://drive.google.com/file/d/1FXszV2q1hCMrT9MARMwgqwYkgYNQA0NU/view?usp=sharing to data


### Face image synthesis with breathing mask
* modify 'output_dir' in hparams.py to a address you want to save synthesized images
* put face image synthesis model for breathing mask to 'output_dir'
* modify 'kind' in hparams.py to 'mask'
* modify 'w_file' in hparams.py to 'data/w_mask_test_1000.json'
* modify 'interface_file' in hparams.py to 'data/mask_interfacegan_test_1000.json'
* python test.py

### Face image synthesis with sun glasses mask
* modify 'output_dir' in hparams.py to a address you want to save synthesized images
* put face image synthesis model for glasses mask to 'output_dir'
* modify 'kind' in hparams.py to 'glasses'  
* modify 'is_normal' in hparams.py to 'False'
* modify 'w_file' in hparams.py to 'data/w_glasses_test_1000.json'
* modify 'interface_file' in hparams.py to 'data/glasses_interfacegan_test_1000.json'
* python test.py

### Face image synthesis with frame glasses mask
* modify 'output_dir' in hparams.py to a address you want to save synthesized images
* put face image synthesis model for glasses mask to 'output_dir'
* modify 'kind' in hparams.py to 'glasses'
* modify 'is_normal' in hparams.py to 'True'
* modify 'w_file' in hparams.py to 'data/w_glasses_test_1000.json'
* modify 'interface_file' in hparams.py to 'data/glasses_interfacegan_test_1000.json'
* python test.py