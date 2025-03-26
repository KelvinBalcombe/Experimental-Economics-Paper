# Experimental-Economics-Paper
This repository contains the python code and data pertaining to the paper Flexible Estimation of Parametric Prospect Models using Hierarchical Bayesian Methods. By Kelvin Balcombe and Iain Fraser

They were run on conda 4.9.2 with python 3.8.5 with pystan 2.19.1.1. These are run in Microsoft Windows using Jupyter Notebooks

Readers should note that the pystan code will not run on current version cmdstanpy without alteration because the array statements use and older syntax. These, however, can be updated and the code should compile in cmdstanpy after modification of the array statements.

Fulldata3.xlsx is the data used for the study and should be placed in the working directory.

The files:

prepare.py; 
gambles.py; and, 
shortercuts.py 

should be placed in the working directory as the notebooks as they contain required code that are used within the notebooks.


In order to estimate the models each of the files

model0B.ipynb;
model1B.ipynb;
model2B.ipynb; and,
model3B.ipynb 

must be run. Each of these files correspondes to the model versions as stated in 3.3 and in Table1.  model0B correspeond to model 0 and so forth.

Each of the 4 notebooks model0B through to model4B contains both the Stan code and the statements used to run the models. Each file needs to be run sequantially and in its entirety and both the compiled stan code along with a copy of the MCMC output should then be saved within the working directory. The number of Chains is set to  8 and should be decreased to 4 if the computer that is being used does not have sufficient cores. Each file could take up to 12 hours to run depending on the speed of the computer. 

Once the MCMC have been run and stored, the the files that generate the results and graphics are:

z0analyse1.ipynb
z0analyse2.ipynb

These files rely on the MCMC output from notebooks model0B through to model4B having been run with the MCMC files being stored. The default directories for the output and MCMC are set as they were run but will require modification to correctly specify the path.

Further information is contained within the notebooks themselves.


