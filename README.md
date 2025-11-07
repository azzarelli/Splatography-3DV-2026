# Splatography: Sparse multi-view dynamic Gaussian Splatting for filmmaking challenges

## 3DV 2026

### [Project Page](https://interims-git.github.io/)| [arXiv Paper](...)

[Adrian Azzarelli](https://azzarelli.github.io/)<sup>x</sup>, [Nantheera Anantrasirichai](https://pui-nantheera.github.io/),
[David R Bull](https://www.bristol.ac.uk/people/person/David-Bull-f53987d8-4d62-431b-8228-2d39f944fbfe/)

Bristol Visual Institute/Visual Information Laboratory, University of Bristol, UK. This project is linked to [MyWorld](https://www.myworld-creates.com/).


## Environmental Setups

Steps for installing environments (we used conda):

```bash
>>> install pytorch (I follow Nerfstudio) 


# From 4DGaussians for scale initialization
pip install -e submodules/simple-knn

# For rasterization
pip install gsplat

# Packages
pip install -r requirements.txt # TODO

# From WavePlanes - we need both the modified `pytorch_wavelets_` in submodules/ and the original package
pip install pytorch_wavelets
>>> make sure you have `submodules/pytorch_wavelets_/*` - this is modified to handle torch AMP
```

If you need help please refer to the original repos [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), [4D-GS](https://github.com/hustvl/4DGaussians), [gsplat](https://github.com/nerfstudio-project/gsplat), [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) and [WavePlanes](https://github.com/azzarelli/waveplanes/). The git-issues from these repos are verbose.

(Adrian) doesn't enjoy being online, so if you have any pressing questions please email me.

## Data Links

- [ViVo](https://vivo-bvicr.github.io/) presents various 360 indoor scenes with high visual and motion diversity.

- [DyNeRF](https://github.com/facebookresearch/Neural_3D_Video) presents various 2.5-D/forward-facing with high texture/material diversity. For preprocessing, we select the four cameras that sit at the extrema, top-right, top-left, bottom-left and bottom-right, for training. We use the same test camera as in the original dataset. Checkout `scene/neural3D_dataset_NDC.py` and compare with the implementation in [4D-GS](https://github.com/hustvl/4DGaussians) to see the difference. 

Please follow 4D-GS instructions on pre-processing the DyNeRF datset. Vivo does not require preprocessing.

### Custom Data
I am doing a PhD so I don't have time to implement this. However, some of our recent work [VSGaussians](https://github.com/azzarelli/VSGaussians) uses Nerfstudio to generate pose estimates and an initial splat. For sparse-view dastets this may not work, so some of the older commits from VSGaussians has functionality for dataloading poses and an initial basic pointcloud generated using [Mast3R](https://github.com/naver/mast3r).

## Code

### Running

There are various `bash` scripts for running different datasets and rendering different results. `cnd_train.sh` was used for training the ViVo datset, `cnd_eval.sh` for evaluation, `cnd_nvs.sh` for generating novel views, `cnd_coolvid.sh` for generating the VFX-GaussianSplatting effects shown on the project page. Inputs are indicated at the start of these scripts.

There are similar `bash` scripts for DyNeRF scenes.

### Notes on Arguments/Hyper-parameter Tuning
Due to the unavailability of HPC systems (until recently the GPU architectures at the university were outdated), we did not conduct comprehensive tuning. I think most of the parameters use the same default settings as 4D-GS. 

Feel free to tune and if you get any improvements please ping me!

### Code Quality and Structure
Having experience in GS research I am painfully aware of the high coding standards set by Facebook, Google and NVidia. Please do not hold me up to this standard, I am paid less than minimum wage. This repository will not be actively developing like you may see with [4D-GS](https://github.com/hustvl/4DGaussians), [gsplat](https://github.com/nerfstudio-project/gsplat).

If you have improvements feel free to fork and push! 

### Branches
I have left the `git-branches` open for public viewing. This is mainly to show development process as each branch investigates a different aspect of 4-D reconstruction. This includes, using Bezier functions to model smooth per-param deformation functions, or using [Rotation Continuity](https://github.com/papagina/RotationContinuity/tree/master) for modelling rotational deformations.


## Viewer & Checkpoints

I use the `dear-py-gui` library for viewing and implementing custom inspection tools. I am aware that other works use the viewer from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) and while it does offer remote access points, its a bit more finicky to install and use for local work. 

Note that in more recent work [VSGaussians](https://github.com/azzarelli/VSGaussians), I have added functionality to train the method without DPG as I know some HPC systems can not import the library.

Checkpoints can be accessed and viewed via the `view` input for `cnd_train.sh`. This also requires an input for the iteration which the checkpoint was saved.

## Contributions

Some source code is borrowed from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) and [4D-GS](https://github.com/hustvl/4DGaussians). We thank the authors for their valued contributions and continued efforts in maintaining/updating their code.


## Citation

TODO - add arxiv link and citation

