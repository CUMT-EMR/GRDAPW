# GRDAPW
This repository contains all the codes in the paper “**Intelligent adaptive methods for data preprocessing and multi-indicator fusion early warning of gas and roof disaster risks in coal mines**” with the following code hierarchy:

1. **Outlier Handling Model**

It contains the model code for semi-supervised co-training based on xLSTM (Note: you need to install the xLSTM package first, please refer to the repository [NX-AI/xlstm: Official repository of the xLSTM.](https://github.com/NX-AI/xlstm))

2. **Noise Reduction Model**

It contains code based on PSO-wavelet threshold denoising-OPTICS noise reduction model, which provides examples of noise reduction combining AE and EMR signals.

3. **BayOTIDE online real-time interpolation model**

As the model was only used in the system application, the model code source warehouse [xuangu-fang/BayOTIDE: BayOTIDE-Bayesian Online Multivariate Time Series Imputation with Functional Decomposition ( ICML 2024 spotlight)](https://github.com/xuangu-fang/BayOTIDE), on the basis of which we further developed the data preprocessing code and configuration files that conform to our data distribution law, and generated the data collection of gas and roof warning related indicators, the specific files are uploaded in the folder.

4. **MTAFM model**

We provide a generic version of this model structure code that can be very easily utilized to build your own tasks.

5. **Sample generation**

We provide the GAN, and diffusion model code used for sample generation in the paper, which can be easily used for expanding the generation of various signals
