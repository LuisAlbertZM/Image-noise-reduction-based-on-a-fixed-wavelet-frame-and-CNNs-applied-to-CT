# denoising-w-frame-and-CNN-for-CT
This repository contains the CNNs used in the paper " Image noise reduction based on a fixed wavelet frame and CNNs applied to CT" that publised in IEEE's Transactions on Image Processing.

The notebooks within DEMO are tyhe following:
* DEMO_directNoiseReduction.ipynb
* DEMO_iterativeReconstruction.ipynb
* DEMO_NPS.ipynb

Furthermore, we captured the python environment of the computer we used in 
* environment_demo.yml

For the iterative reconstruction demo, the ASTRA toolbox V2.0 is required (https://www.astra-toolbox.com/) in the same environment where Pytorch is installed. We found a bit challenging to have both libraries installed and in our case, installing ASTRA from the source was needed.

The funding of this project was provided by the European Union through the Horizon 2020 “Next generation X-ray imaging system (NEXIS)]” under Grant 780026. 

The CT slices in this repository were obtained from the low-dose dataset from the cancer imaging archive (TCIA). We thank the  grants EB017095 and EB017185  from  the National Institute of Biomedical Imaging and Bioengineeringto to provide funding for the generation of the dataset used in this paper.
