KARMA4 v1.0
-----------

KARMA4 is a C++ library for autoregressive moving average (ARMA) modeling and
forecasting of time-series data while incorporating both process and observation
error. KARMA4 is designed for fitting and forecasting of time-series data for
predictive purposes. It performs calibration of ARMA model parameters with the
aid of Kalman filtering technique and selects the optimal ARMA model order using
cross-validation. Its features include:

• Model calibration is performed using maximum-likelihood estimation
• Likelihood evaluation utilizes the Kalman filter for state estimation
• Measurement error modeling and inference
• Optimal model selection is performed using cross-validation
• Forecasting under uncertainty


Author(s):  Moe Khalil, Sandia National Laboratories, mkhalil@sandia.gov
            Jina Lee, Sandia National Laboratories
            Maher Salloum, Sandia National Laboratories
                           
KARMA4 is designed for fitting and forecasting of time-series data for
predictive purposes. The data could be in the form of derived quantities from
computational model simulations, in which case KARMA4 forecasts would
potentially replace expensive numerical simulations. The data could also be in
the form of noisy experimental observations of physical systems and the
resulting forecasts aid in critical decision-making processes.

3rd party open-source software used in the library:
---------------------------------------------------

- NLopt: (GNU LGPL license) available at:
  http://ab-initio.mit.edu/wiki/index.php/NLopt
  nonlinear optimization library implementing many different optimization
    algorithms

Contents:
---------

C++ source code:
  /lib

C++ example:
  /app/test_synthetic: A verification exercise in which the data generating model
    is in fact an AR model
  /app/toy_problem: AR modeling of wavelet coefficients resulting from compressed
    sensing of the transient response of the 2D heat equation on a
    square domain with randomly chosen holes
    
Include and Library folders: (do not delete!)
  /include
  /lib

Compiling the library and example:
----------------------

Build and install the nlopt-2.4.2 library

Set Environment for Unix / Mac OS X:

1. Edit your Bash startup file in your favorite text editor. For Linux, this is
~/.bashrc. OS X terminal runs a login shell, and so the start up file may be
~/.bashrc, ~/.bash_profile, ~/.bash_login, or ~/.profile. See the manpage for
Bash for more information about the differences between login and non-login
shells.

2. Create an environment variable to point to installed NLopt library:

export NLOPT_PATH=/usr/local/nlopt-2.4.2

3. Create an environment variable to point to installed KARMA4 library:

export KARMA4_PATH=/usr/local/KARMA4-1.0


For the library:
- Go to $(KARMA4_PATH)/lib
- run 'make all'

For the example:
- Go to $(KARMA4_PATH)/examples/compress_reconstruct
- run 'make'


To do in future releases:
-------------------------

- Implement the Akaike information criterion (AIC) with correction for finite
  sample sizes (known as AICc) as the primary criterion in selecting the order
  of the AR model, as opposed to cross-validation

