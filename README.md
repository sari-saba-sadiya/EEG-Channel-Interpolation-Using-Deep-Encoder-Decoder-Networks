# EEG Channel Interpolation Using Deep Encoder-decoder Networks

Code for the paper "EEG Channel Interpolation Using Deep Encoder-decoder Networks", submitted to The 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC 2020).

## TODO:
- Add transfer learning code.
- Add paper pdf.

## Contents:
<pre>
|-> train:  
|    |-> ecr_cnn.py: The code you need to compile train and run the neural networks  
|    |-> ecr_hyper_parameters.npy
|    |-> ecr_loadModel: load the trained model and run it to interpolate on non-training data.
|    |-> run.sh code to run the training 
|  
|-> baselines:  
|    |-> ecr_baseline.py: The code to calculate the EDP and EGL baselines.  
|    |-> ecr_ssp.py: The code to calculate the spherical splines baseline.  
|  
|-> transfer:
|    |-> ecr_transfer.py: The code for transfer learning
|    |-> run_transfer.sh
|
|-> README.txt: This file.
</pre>
