# Computing Robustness Distributions with Diffusion Denoised Smoothing

Code base for my Master Thesis titled <ins>"Using Randomized Smoothing to compute Robustness Distributions"</ins>. 

The goal of the thesis is to investigate the use of Diffusion Denoised Smoothing to compute Robustness Distributions for complex models and datasets. 

The basis of the code is formed by the PyTorch implementation of <ins>**Diffusion Denoised Smoothing**</ins>, proposed in the ICLR 2023 paper:

>(Certified!!) Adversarial Robustness for Free!      
>Nicholas Carlini*, Florian Tramèr*, Krishnamurthy Dvijotham, Leslie Rice, Mingjie Sun, J. Zico Kolter   
For more details, please check out the [<ins>**paper**</ins>](https://arxiv.org/abs/2206.10550).

## License
The project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Acknowledgments

This codebase builds upon several foundational works. We thank the authors for their work.

- **Randomized Smoothing**: Base implementation from [locuslab/smoothing](https://github.com/locuslab/smoothing) based on the paper ["Certified Adversarial Robustness via Randomized Smoothing"](https://arxiv.org/abs/1902.02918) by Cohen et al.

- **Improved Diffusion**: Diffusion model implementation from [openai/improved-diffusion](https://github.com/openai/improved-diffusion) based on the paper ["Improved Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2102.09672) by Nichol & Dhariwal.

- **Guided Diffusion**: Diffusion model implementation from [openai/guided-diffusion](https://github.com/openai/guided-diffusion) based on the paper ["Diffusion Models Beat GANs on Image Synthesis"](https://arxiv.org/abs/2105.05233) by Dhariwal & Nichol.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{carlini2023free,
  author  = {Carlini, Nicholas and Tramèr, Florian and Dvijotham, Krishnamurthy and Rice, Leslie and Sun, Mingjie and Kolter, Zico},
  title   = {(Certified!!) Adversarial Robustness for Free!},
  journal = {International Conference on Learning Representations (ICLR)},
  year    = {2023},
}
```