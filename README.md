# EE367FinalProjectCode
code for EE367 final project

This is a processing pipeline specifically for low-light hazy images. It aims to restore the input image with better visibility by enhancing the image using dual illumination estimation in this paper: https://arxiv.org/abs/1910.13688 and dark channel prior image haze removal proposed by Kaiming He (https://arxiv.org/abs/1910.13688). Also, it provides a dehazing alternative using DehazeNet (Cai et al. https://github.com/caibolun/DehazeNet) for haze removal, but it needs to be run on Matlab. 

To run this pipeline, simply run python3 dehaze.py
