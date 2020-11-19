# EEG Channel Interpolation Using Deep Encoder-decoder Networks
[Sari Saba-Sadiya](https://cse.msu.edu/~sadiyasa/)<sup>1</sup>,
[Taushang Liu](https://npal.psy.msu.edu/)<sup>1</sup>,
[Tuka Alhanai](https://talhanai.xyz/)<sup>2</sup>,
[Mohammad Ghassemi](https://ghassemi.xyz/)<sup>1</sup><br>
<sup>1</sup> Michigan State University <sup>2</sup> New York University Abu Dhabi

Code for the paper "EEG Channel Interpolation Using Deep Encoder-decoder Networks", presented in BIBM-DLB2H'2020.

## Contents:
* `train`:  
    * `ecr_cnn.py`: The code you need to compile train and run the neural networks  
    * `ecr_hyper_parameters.npy`
    * `ecr_loadModel`: load the trained model and run it to interpolate on non-training data.
    * `run.sh` code to run the training 
* `baselines`:  
    * `ecr_baseline.py`: The code to calculate the EDP and EGL baselines.  
    * `ecr_ssp.py`: The code to calculate the spherical splines baseline.   
* `transfer`:
    * `ecr_transfer.py`: The code for transfer learning
    * `run_transfer.sh`
* `README.txt`: This file.
