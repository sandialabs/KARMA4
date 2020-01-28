/*  ____________________________________________________________________________

    KARMA4: Library for autoregressive moving average (ARMA) modeling and
            forecasting of time-series data with process and observational error
    Copyright 2017 Sandia Corporation.
    This software is distributed under the BSD License.
    For more information, see the README.txt file in the top KARMA4 directory.
    _________________________________________________________________________ */

//- Description:  Data structures and function declarations

#ifndef KARMA4_H
#define KARMA4_H

#include <vector>
#include <math.h>


//
//- Heading: Data
//

// Value of pi
const double PI = 2.0*acos(0.0);

// Input data structure for test_pde_timeseries_w_new function
struct test_ins{
  std::vector<double> wc_data;
  int num_ts;
  int num_forecast;
  int p_low;
  int p_high;
  double x_frac; // fraction of data to be used for cross-validation
  double tolm;
};

// Output data structure for test_pde_timeseries_w_new function
struct test_outs{
  std::vector<double> A;
  std::vector<double> z;
  std::vector<double> zfori;
  std::vector<double> Bfor;
  int q;
  int p;
};

// Input data structure for z_forecast function
struct z_for_ins{
  std::vector<double> theta;
  double sigma;
  double gamma;
  std::vector<double> mu_init;
  std::vector<std::vector<double> > P_init;
  int n_pred;
};

// Output data structure for z_forecast function
struct z_for_outs{
  std::vector<double> z_pred;
  std::vector<double> p_pred;
  std::vector<double> mu;
  std::vector<std::vector<double> > P;
};

// Input data structure for log_lik function
struct log_lik_ins{
  std::vector<double> theta;
  double sigma;
  double gamma;
  std::vector<double> init_z;
  double init_p;
  std::vector<double> z;
};

// Output data structure for log_lik function
struct log_lik_outs{
  std::vector<double> z_pred;
  double cal_l;
  std::vector<double> p_pred;
  std::vector<double> mu;
  std::vector<std::vector<double> > P;
};

// Input data structure for misfit function
struct misfit_ins{
  std::vector<double> x;
  std::vector<double> init_z;
  std::vector<double> z;
};

// Output data structure for misfit function
struct misfit_outs{
  double rmse;
  double log_gamma;
};


//
//- Heading: Member functions
//

// Evaluate deterministic misfit, used in providing pre-estimates of arma model
// parameters and initial condition
void misfit(misfit_ins *m_ins, misfit_outs *m_outs);

// Evaluate log-likelihood of arma model given noisy data
void log_lik(log_lik_ins *l_ins, log_lik_outs *l_outs);

// Provide arma model forecasts with uncertainty
void z_forecast(z_for_ins *z_ins, z_for_outs *z_outs);

// Calibrate arma models of increasing order and select best model using
// cross-validation. Also provide forecasts using optimal model.
void test_pde_timeseries_w_new(test_ins *t_ins, test_outs *t_outs);

// NLopt wrapper for misfit function
double nlopt_misfit_wrapper(const std::vector<double> &x,
  std::vector<double> &grad, void *info);

// NLopt wrapper for log-likelihood of ARMA model noise intensity, sigma
double nlopt_log_lik_wrapper_sigma(const std::vector<double> &x, 
  std::vector<double> &grad, void *info);

// NLopt wrapper for log-likelihood of ARMA model noise intensity, sigma, 
// and initial error covaraince hyperparameters
double nlopt_log_lik_wrapper_sigma_initp(const std::vector<double> &x, 
  std::vector<double> &grad, void *info);

// NLopt wrapper for log-likelihood of ARMA model parameter vector, theta, noise
// intensity, sigma, and initial error covaraince hyperparameters
double nlopt_log_lik_wrapper_theta_sigma_initp(const std::vector<double> &x, 
  std::vector<double> &grad, void *info);

// NLopt wrapper for log-likelihood of ARMA model parameter vector, theta, noise
// intensity, sigma, and observational noise intensity, gamma
double nlopt_log_lik_wrapper_theta_sigma_gamma(const std::vector<double> &x, 
  std::vector<double> &grad, void *info);

// NLopt wrapper for log-likelihood of all ARMA model parameters/hyperparameters
double nlopt_log_lik_wrapper_all(const std::vector<double> &x, 
  std::vector<double> &grad, void *info);

#endif
