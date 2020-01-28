/*  ____________________________________________________________________________

    KARMA4: Library for autoregressive moving average (ARMA) modeling and
            forecasting of time-series data with process and observational error
    Copyright 2017 Sandia Corporation.
    This software is distributed under the BSD License.
    For more information, see the README.txt file in the top KARMA4 directory.
    _________________________________________________________________________ */

//- Description:  Implementation code for library functions

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <karma4.hpp>
#include <nlopt.hpp>
#include <random>

// Evaluate deterministic misfit, used in providing pre-estimates of arma model
// parameters and initial condition
void misfit(misfit_ins *m_ins, misfit_outs *m_outs){

	int np_arma = m_ins->x.size(); // order of AR model, p
	double sse = 0.0; // sum of errors squared
	double sss = 0.0; // sum of signal squared
	double rmse;

	std::vector<double> mu(np_arma); // state mean estimate
	std::vector<double> predi(m_ins->z.size()); // predicted observation

	// initialize
	for (int i = 0; i < np_arma; i++){
		mu[i] = m_ins->init_z[i];
		predi[np_arma-i-1] = m_ins->init_z[i];
	}

	// number of time marching steps
	int nb = m_ins->z.size() - np_arma;

	// time marching
	double temp;
	for (int i = 0; i < nb; i++){
		// update mean
		temp = 0.0;
		for (int j = 0; j < np_arma; j++){
			temp += m_ins->x[j]*mu[j];
		}
		for (int j = (np_arma-1); j > 0; j--){
			mu[j] = mu[j-1];
		}
		mu[0] = temp;

		//extract predicted observations
		predi[np_arma+i] = mu[0];
	}

	for (int i = 0; i < m_ins->z.size(); i++){
		sse = sse + pow((m_ins->z[i] - predi[i]),2);
		sss = sss + pow(m_ins->z[i],2);
	}

	m_outs->log_gamma = log(sqrt(sse/((double) (m_ins->z.size() - 1))));
	rmse = sqrt(sse/sss); // normalized room-mean-square error in predictions

	m_outs->rmse = rmse;

	return;
}

// Evaluate log-likelihood of arma model given noisy data
void log_lik(log_lik_ins *l_ins, log_lik_outs *l_outs){

	int np_arma = l_ins->theta.size(); // order of AR model, p

	double Q = pow(exp(l_ins->sigma),2.0); // model noise error covariance
	double R = pow(exp(l_ins->gamma),2.0); // measurement noise error covariance

	std::vector<double> zt(l_ins->z.size() - np_arma); //noisy obseravtions
	int nz = zt.size(); //number of noisy obseravtions

	// extract noisy observations
	for (int i = 0; i < nz; i++){
		zt[i] = l_ins->z[i+np_arma];
	}

	// state mean estimate
	std::vector<double> mu(np_arma); 
	// state error covariance
	std::vector<std::vector<double> > P(np_arma);
	// updated (assimilated) state error covariance
	std::vector<std::vector<double> > P_up(np_arma);
	// predicted observation, mean
	std::vector<double> z_pred(l_ins->z.size()); 
	// predicted observation, variance
	std::vector<double> p_pred(l_ins->z.size());

	// initialize
	for (int i = 0; i < np_arma; i++){
		mu[i] = l_ins->init_z[i];
		z_pred[np_arma-i-1] = l_ins->init_z[i];
		p_pred[i] = exp(l_ins->init_p);
	}
	for (int i = 0 ; i < np_arma; i++){
		P[i].resize(np_arma,0.0);
		P_up[i].resize(np_arma,0.0);
		P[i][i] = exp(l_ins->init_p);
	}

	double cal_l = 0.0; //log-likelihood

	// temporary variables
	double temp;
	std::vector<double> temp_vec(np_arma);
	double low_c;
	double cap_c;

	// log-likelihood contribution of initial state condition
	for (int i = 0; i < np_arma; i++){
		if ((P[i][i]/R > 1.0e5) || (R/P[i][i] > 1.0e5)){
			cal_l = -1.0e100;
			l_outs->cal_l = cal_l;
			return;
		}
		
		cap_c = 1.0/((1.0/P[i][i]) + (1.0/R));
		low_c = cap_c*((mu[i]/P[i][i])+(l_ins->z[np_arma-i-1]/R));

		cal_l += 1.0*(-0.5*log(2.0*PI) + 0.5*log(cap_c) - 0.5*log(P[i][i]) - 
			0.5*log(R));
		cal_l += 1.0*(-0.5*((pow(mu[i],2.0)/P[i][i]) + 
			(pow(l_ins->z[np_arma-i-1],2.0)/R) - (pow(low_c,2.0)/cap_c)));
	}

	// time marching
	for (int i = 0; i < nz; i++){

		//update mean vector
		temp = 0.0;
		for (int j = 0; j < np_arma; j++){
			temp += l_ins->theta[j]*mu[j];
		}
		for (int j = (np_arma-1); j > 0; j--){
			mu[j] = mu[j-1];
		}
		mu[0] = temp;

		//update covariance matrix
		temp = 0.0;
		for (int j = 0; j < np_arma; j++){
			temp_vec[j] = 0.0;
			for (int k = 0; k < np_arma; k++){
				temp_vec[j] +=  l_ins->theta[k]*P[j][k];
			}
			temp += l_ins->theta[j]*temp_vec[j];
		}
		for (int j = (np_arma-1); j > 0; j--){
			for (int k = (np_arma-1); k > 0; k--){
				P[j][k] = P[j-1][k-1];
			}
		}
		for (int j = 1; j < np_arma; j++){
			P[0][j] = temp_vec[j-1];
			P[j][0] = temp_vec[j-1];
		}
		P[0][0] = temp;

		P[0][0] += Q;

		// check for divergence of Kalman filter, return low log-likelihood
		if ((P[0][0]/R > 1.0e10) || (R/P[0][0] > 1.0e10)){
			cal_l = -1.0e100;
			l_outs->cal_l = cal_l;
			return;
		}

		// log-likelihood contribution of current state
		cap_c = 1.0/((1.0/P[0][0]) + (1.0/R));
		low_c = cap_c*((mu[0]/P[0][0])+(zt[i]/R));
		cal_l += -0.5*log(2.0*PI) + 0.5*log(cap_c) - 0.5*log(P[0][0]) - 
			0.5*log(R);
		cal_l += -0.5*((pow(mu[0],2.0)/P[0][0]) + (pow(zt[i],2.0)/R) - 
			(pow(low_c,2.0)/cap_c));

		// Kalman filter: update mean vector and covariance matrix
		for (int j = 0; j < np_arma; j++){
			temp_vec[j] = P[0][j]/(P[0][0] + R); //Kalman gain matrix/vector
			mu[j] += temp_vec[j]*(zt[i] - mu[0]); //update mean vector
		}
		for (int j = 0; j < np_arma; j++){
			for (int k = 0; k < np_arma; k++){
				P_up[j][k] = P[j][k] - temp_vec[j]*P[0][k];
			}
		}
		P = P_up;

		// Extract one-step forecasts: mean and variance
		z_pred[np_arma+i] = mu[0];
		p_pred[np_arma+i] = P[0][0];
	}

	// output relevant data structure elements
	l_outs->z_pred = z_pred;
	l_outs->p_pred = p_pred;
	l_outs->mu = mu;
	l_outs->P = P;
	l_outs->cal_l = cal_l;

	return;
}

// Provide arma model forecasts with uncertainty
void z_forecast(z_for_ins *z_ins, z_for_outs *z_outs){

	int np_arma = z_ins->theta.size(); // order of AR model, p

	double Q = pow(exp(z_ins->sigma),2.0); // model noise error covariance
	double R = pow(exp(z_ins->gamma),2.0); // measurement noise error covariance

	// state mean estimate
	std::vector<double> mu(np_arma); 
	// predicted observation, mean
	std::vector<double> z_pred(z_ins->n_pred); 
	// predicted observation, variance
	std::vector<double> p_pred(z_ins->n_pred);
	// state error covariance
	std::vector<std::vector<double> > P(np_arma);
	
	// initialize
	for (int i = 0; i < np_arma; i++){
		mu[i] = z_ins->mu_init[i];
		P[i].resize(np_arma,0.0);
		for (int j = 0 ; j < np_arma; j++){
			P[i][j] = z_ins->P_init[i][j];
		}
	}

	// temporary variables
	double temp;
	std::vector<double> temp_vec(np_arma);

	// time marching
	for (int i = 0; i < z_ins->n_pred; i++){

		//update state mean vector
		temp = 0.0;
		for (int j = 0; j < np_arma; j++){
			temp += z_ins->theta[j]*mu[j];
		}
		for (int j = (np_arma-1); j > 0; j--){
			mu[j] = mu[j-1];
		}
		mu[0] = temp;

		//update state error covariance matrix
		temp = 0.0;
		for (int j = 0; j < np_arma; j++){
			temp_vec[j] = 0.0;
			for (int k = 0; k < np_arma; k++){
				temp_vec[j] += z_ins->theta[k]*P[j][k];
			}
			temp += z_ins->theta[j]*temp_vec[j];
		}
		for (int j = (np_arma-1); j > 0; j--){
			for (int k = (np_arma-1); k > 0; k--){
				P[j][k] = P[j-1][k-1];
			}
		}
		for (int j = 1; j < np_arma; j++){
			P[0][j] = temp_vec[j-1];
			P[j][0] = temp_vec[j-1];
		}
		P[0][0] = temp;
		P[0][0] += Q;

		// Extract forecasts: mean and variance
		z_pred[i] = mu[0];
		p_pred[i] = P[0][0];
	}

	// output relevant data structure elements
	z_outs->z_pred = z_pred;
	z_outs->p_pred = p_pred;
	z_outs->mu = mu;
	z_outs->P = P;

	return;
}

// Calibrate arma models of increasing order and select best model using
// cross-validation. Also provide forecasts using optimal model.
void test_pde_timeseries_w_new(test_ins *t_ins, test_outs *t_outs)
{
	int p_low = 1; //lowest proposed value of p (AR order)
	int p_high = 6; //greatest proposed value of p (AR order)
	int np = p_high-p_low+1; // number of proposed models
	
	double res;
	double temp;
	
	std::vector<double> theta; // AR model parameter vector
	std::vector<double> init_z; // initial condition
	misfit_ins m_ins;
	std::vector<double> xx;
	double minf;

	std::vector<test_outs> t_outs_vec(np);
	std::vector<log_lik_ins> l_ins_vec(np);
	std::vector<z_for_ins> z_ins_vec(np);
	std::vector<double> x_valid_ls(np,0.0);

	//------------------------------------------------------------------------//
	// extract data (time-series)
	//------------------------------------------------------------------------//
	// number of observations available for calibration
	int num_ts_fit = (int) floor(((double)t_ins->num_ts)*(1.0e0-t_ins->x_frac));
	// number of observations available for cross-validation
	int num_ts_xval = t_ins->num_ts-num_ts_fit;
	// data vector
	std::vector<double> A(t_ins->num_ts);
	// calibration data vector
	std::vector<double> z(num_ts_fit);

	// extract data from input data structure
	if (t_ins->wc_data.size() < t_ins->num_ts){
		printf("Error in test_pde_timeseries_w_new: passed data vector "
			"wc_data does not contain num_ts points\n");
		exit;
	}
	for (int k = 0; k < t_ins->num_ts; k++){
	A[k] = t_ins->wc_data[k];
	}
	for (int k = 0; k < num_ts_fit; k++){
	z[k] = A[k];
	}
  
  	// used in scaling of NLopt termination thresholds
	double tolm=t_ins->tolm*t_ins->tolm*t_ins->tolm;

	// used to generate pre-estimate initial guesses
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0,1.0);
  
  	// iterate over AR models, calibrating and computing cross-validation errors
	for (int nphi = p_low; nphi <= p_high; nphi++){

		//--------------------------------------------------------------------//
		// Pre-estimates of ARMA parameters
		//--------------------------------------------------------------------//
		
		theta.resize(nphi);
		init_z.resize(nphi);
		// maximum number of function evaluations for NLopt
		long neval = 200*num_ts_fit*nphi*nphi; 

		m_ins.z = z; // data vector


		// deterministic guess for pre-estimation, to be followed by random
		// guesses
		for (int k = 0; k < nphi; k++){
			init_z[k] = z[nphi-k-1];
			theta[k] = 1.0/((double) nphi);
		}
		m_ins.init_z = init_z;

		res = z[nphi];
		for (int k = 1; k < nphi; k++){
			res -= theta[k]*init_z[k];
		}
		theta[0] = res/init_z[0];

		m_ins.x = theta;

		// NLopt object setup
		nlopt::opt opt(nlopt::LN_SBPLX, 2*nphi);
		opt.set_min_objective(nlopt_misfit_wrapper, (void*) &m_ins);
		opt.set_xtol_rel(1.0e-10*tolm);
		opt.set_ftol_rel(1.0e-10*tolm);
		opt.set_stopval(1.0e-20);
		opt.set_maxeval(neval);

		// NLopt initial state
		xx.resize(2*nphi);
		for (int k = 0; k < nphi; k++){
			xx[k] = theta[k];
			xx[k+nphi] = init_z[k];
		}

		// the minimum objective value, upon return from NLopt
		double minf;

		// run NLopt optimizer
		nlopt::result result = opt.optimize(xx, minf);

		// extract optimal parameters
		for (int k = 0; k < nphi; k++){
			theta[k] = xx[k];
			init_z[k] = xx[k+nphi];
		}

		// data structure setup and initialization
		m_ins.x = theta;
		m_ins.init_z = init_z;
		misfit_outs m_outs;
		misfit(&m_ins, &m_outs);
		misfit_outs m_outs_best;
		m_outs_best = m_outs;
		misfit_ins m_ins_best = m_ins;

		// propose a number of random initial guesses for pre-estimation of arma
		// model parameters and initial condition
		for (int i = 0; i < 100; i++){

			// random guess of AR parameters
			for (int k = 0; k < nphi; k++){
				init_z[k] = z[nphi-k-1];
				theta[k] = distribution(generator);
			}
			res = z[nphi];
			for (int k = 1; k < nphi; k++){
				res -= theta[k]*init_z[k];
			}
			theta[0] = res/init_z[0]; // model to fit first data point


			for (int k = 0; k < nphi; k++){
				xx[k] = theta[k];
				xx[k+nphi] = init_z[k];
			}

			// run NLopt optimizer
			result = opt.optimize(xx, minf);

			// extract optimal parameters
			for (int k = 0; k < nphi; k++){
				theta[k] = xx[k];
				init_z[k] = xx[k+nphi];
			}

			m_ins.x = theta;
			m_ins.init_z = init_z;
			misfit(&m_ins, &m_outs);

			// check whether or not latest solution is global optimum thusfar
			if (m_outs.rmse < m_outs_best.rmse){
				m_outs_best = m_outs;
				m_ins_best = m_ins;
			}
		}

		// store best pre-estimate
		m_outs = m_outs_best;
		m_ins = m_ins_best;

		//--------------------------------------------------------------------//
		// Maximum likelihood estimation of ARMA parameters:
		// Specifically, ARMA noise term, sigma
		//--------------------------------------------------------------------//

		log_lik_ins l_ins;
		l_ins.theta.resize(nphi);
		for (int k = 0; k < nphi; k++){
			l_ins.theta[k] = theta[k];
		}
		l_ins.gamma = m_outs.log_gamma;
		l_ins.init_z.resize(nphi);
		for (int k = 0; k < nphi; k++){
			l_ins.init_z[k] = init_z[k];
		}
		l_ins.init_p = 2.0*m_outs.log_gamma - 4.0;
		l_ins.z = z;

		log_lik_outs l_outs;

		// Evaluate log-likelihood at predefined log-sigma values to extract
		// initial guess
		double delta_sigma = 0.1;
		double best_sigma = -30.0;
		l_ins.sigma = best_sigma;
		log_lik(&l_ins, &l_outs);
		double max_log_lik = l_outs.cal_l;
		// march through 1D mesh for sigma, keeping track of best sigma estimate
		while (l_ins.sigma < 30.0){
			log_lik(&l_ins, &l_outs);
			if (l_outs.cal_l > max_log_lik){
				best_sigma = l_ins.sigma;
				max_log_lik = l_outs.cal_l;
			}
			l_ins.sigma += delta_sigma;
		}

		// NLopt object setup
		nlopt::opt opt1(nlopt::LN_SBPLX, 1);
		opt1.set_min_objective(nlopt_log_lik_wrapper_sigma, (void*) &l_ins);
		opt1.set_xtol_rel(1.0e-15*tolm);
		opt1.set_ftol_rel(1.0e-15*tolm);
		opt1.set_stopval(-1.0e20);
		opt1.set_maxeval(neval);

		// NLopt initial state
		xx.resize(1,best_sigma);

		// run NLopt optimizer
		nlopt::result result1 = opt1.optimize(xx, minf);

		// extract optimal parameter
		l_ins.sigma = xx[0];

		//--------------------------------------------------------------------//
		// Maximum likelihood estimation of ARMA/KF parameters:
		// Specifically, ARMA noise term, sigma, and KF initial error
		// covaraince hyperparameters
		//--------------------------------------------------------------------//

		// NLopt object setup
		nlopt::opt opt2(nlopt::LN_SBPLX, 2);
		opt2.set_min_objective(nlopt_log_lik_wrapper_sigma_initp,
			(void*) &l_ins);
		opt2.set_xtol_rel(1.0e-15*tolm);
		opt2.set_ftol_rel(1.0e-15*tolm);
		opt2.set_stopval(-1.0e20);
		opt2.set_maxeval(neval);

		// NLopt initial state
		xx.resize(2,0.0);
		xx[0] = l_ins.sigma;
		xx[1] = l_ins.init_p;

		// run NLopt optimizer
		nlopt::result result2 = opt2.optimize(xx, minf);

		// extract optimal parameters
		l_ins.sigma = xx[0];
		l_ins.init_p = xx[1];

		//--------------------------------------------------------------------//
		// Maximum likelihood estimation of ARMA/KF parameters:
		// Specifically, ARMA parameter vector, theta, and noise term,
		// sigma, and measurement noise intensity, gamma
		//--------------------------------------------------------------------//

		// NLopt object setup
		nlopt::opt opt4(nlopt::LN_SBPLX, nphi+2);
		opt4.set_min_objective(nlopt_log_lik_wrapper_theta_sigma_gamma,
			(void*) &l_ins);
		opt4.set_xtol_rel(1.0e-15*tolm);
		opt4.set_ftol_rel(1.0e-15*tolm);
		opt4.set_stopval(-1.0e20);
		opt4.set_maxeval(neval);

		// NLopt initial state
		xx.resize(nphi+2,0.0);
		for (int i = 0 ; i < nphi; i++){
			xx[i] = l_ins.theta[i];
		}
		xx[nphi] = l_ins.sigma;
		xx[nphi+1] = l_ins.gamma;

		// run NLopt optimizer
		nlopt::result result4 = opt4.optimize(xx, minf);

		// extract optimal parameters
		for (int i = 0 ; i < nphi; i++){
			l_ins.theta[i] = xx[i];
		}
		l_ins.sigma = xx[nphi];
		l_ins.gamma = xx[nphi+1];

		//--------------------------------------------------------------------//
		// Maximum likelihood estimation of all ARMA/KF parameters
		//--------------------------------------------------------------------//

		// NLopt object setup
		nlopt::opt opt5(nlopt::LN_SBPLX, 2*nphi+3);
		opt5.set_min_objective(nlopt_log_lik_wrapper_all, (void*) &l_ins);
		opt5.set_xtol_rel(1.0e-15*tolm);
		opt5.set_ftol_rel(1.0e-15*tolm);
		opt5.set_stopval(-1.0e20);
		opt5.set_maxeval(neval);

		// NLopt initial state
		xx.resize(2*nphi+3,0.0);
		for (int i = 0 ; i < nphi; i++){
			xx[i] = l_ins.theta[i];
		}
		xx[nphi] = l_ins.sigma;
		for (int i = 0 ; i < nphi; i++){
			xx[nphi+1+i] = l_ins.init_z[i];
		}
		xx[2*nphi+1] = l_ins.init_p;
		xx[2*nphi+2] = l_ins.gamma;

		// run NLopt optimizer
		nlopt::result result5 = opt5.optimize(xx, minf);

		// extract optimal parameters
		for (int i = 0 ; i < nphi; i++){
			l_ins.theta[i] = xx[i];
		}
		l_ins.sigma = xx[nphi];
		for (int i = 0 ; i < nphi; i++){
			l_ins.init_z[i] = xx[nphi+1+i];
		}
		l_ins.init_p = xx[2*nphi+1];
		l_ins.gamma = xx[2*nphi+2];

		//--------------------------------------------------------------------//
		// Cross-validation
		//--------------------------------------------------------------------//

		// Needed to extract Kalman filter parameters for forecasting
		log_lik(&l_ins, &l_outs);

		// extracting Kalman filter parameters
		z_for_ins z_ins;
		z_ins.theta.resize(nphi);
		for (int k = 0; k < nphi; k++){
			z_ins.theta[k] = l_ins.theta[k];
		}
		z_ins.gamma = l_ins.gamma;
		z_ins.sigma = l_ins.sigma;
		z_ins.mu_init.resize(nphi);
		for (int k = 0; k < nphi; k++){
			z_ins.mu_init[k] = l_outs.mu[k];
		}
		z_ins.P_init.resize(nphi);
		for (int i = 0; i < nphi; i++){
			z_ins.P_init[i].resize(nphi);
			for (int j = 0 ; j < nphi; j++){
				z_ins.P_init[i][j] = l_outs.P[i][j];
			}
		}
		z_ins.n_pred = num_ts_xval;

		// obtaining arma model forecasts with uncertainty
		z_for_outs z_outs;
		z_forecast(&z_ins, &z_outs);

		// cross-validation error based on available observations and forecasts
		double x_valid_l = 0.0;
		for (int i = 0; i < num_ts_xval; i++){
			x_valid_l += pow(z_outs.z_pred[i]-A[num_ts_fit+i],2.0);
		}
		x_valid_ls[nphi-p_low] = x_valid_l;

		//--------------------------------------------------------------------//
		// Forecasting with uncertainty
		//--------------------------------------------------------------------//

		// using all available obervations and calibrated model to get Kalman
		// filter parameters for forecasting
		l_ins.z.resize(A.size());
		l_ins.z = A;
		log_lik(&l_ins, &l_outs);

		// extracting Kalman filter parameters
		for (int k = 0; k < nphi; k++){
			z_ins.mu_init[k] = l_outs.mu[k];
		}
		for (int i = 0; i < nphi; i++){
			for (int j = 0 ; j < nphi; j++){
				z_ins.P_init[i][j] = l_outs.P[i][j];
			}
		}
		z_ins.n_pred = t_ins->num_forecast;

		// obtaining arma model forecasts with uncertainty
		z_forecast(&z_ins, &z_outs);


		//--------------------------------------------------------------------//
		// Storing arma model parameters and forecasts
		//--------------------------------------------------------------------//
		(&(t_outs_vec[nphi-p_low]))->A = A;
		(&(t_outs_vec[nphi-p_low]))->z = z;
		(&(t_outs_vec[nphi-p_low]))->zfori = (z_outs.z_pred);
		(&(t_outs_vec[nphi-p_low]))->Bfor = (z_outs.p_pred);
		l_ins_vec[nphi-p_low] = l_ins;
		z_ins_vec[nphi-p_low] = z_ins;
	}

	// determine model with least cross-validation error
	double low_x_valid = x_valid_ls[0];
	int low_index = 0;
	for (int i = 1; i < np; i++){
		if (low_x_valid > x_valid_ls[i]){
			low_index = i;
			low_x_valid = x_valid_ls[i];
		}
	}

	// output best model parameters and forecasts to file
	FILE* f_out;
	if(!(f_out = fopen("xx.dat","w"))){ 
	printf("could not open file xx.dat\n"); 
	exit(1);
	}
	for (int k = 0; k < l_ins_vec[low_index].theta.size(); k++){
		fprintf(f_out, "%0.10g\n", l_ins_vec[low_index].theta[k]);
	}
	for (int k = 0; k < l_ins_vec[low_index].theta.size(); k++){
		fprintf(f_out, "%0.10g\n", l_ins_vec[low_index].init_z[k]);
	}
	fclose(f_out);

	// output relevant data structure elements
	*t_outs = t_outs_vec[low_index];

  return ;
}

// NLopt wrapper for misfit function
double nlopt_misfit_wrapper(const std::vector<double> &x,
	std::vector<double> &grad, void *info){

	misfit_ins* m_ins = (misfit_ins*) info;
	
	int np_arma = m_ins->x.size(); // order of AR model, p

	for (int i = 0; i < np_arma; i++){
		m_ins->x[i] = x[i];
		m_ins->init_z[i] = x[i+np_arma];
	}

	misfit_outs m_outs;

	misfit(m_ins, &m_outs);

	return m_outs.rmse;
}

// NLopt wrapper for log-likelihood of ARMA model noise intensity, sigma
double nlopt_log_lik_wrapper_sigma(const std::vector<double> &x,
	std::vector<double> &grad, void *info){

	log_lik_ins* l_ins = (log_lik_ins*) info;
	
	l_ins->sigma = x[0];


	log_lik_outs l_outs;

	log_lik(l_ins, &l_outs);

	return -l_outs.cal_l;

}

// NLopt wrapper for log-likelihood of ARMA model noise intensity, sigma, 
// and initial error covaraince hyperparameters
double nlopt_log_lik_wrapper_sigma_initp(const std::vector<double> &x,
	std::vector<double> &grad, void *info){

	log_lik_ins* l_ins = (log_lik_ins*) info;
	
	l_ins->sigma = x[0];
	l_ins->init_p = x[1];

	log_lik_outs l_outs;

	log_lik(l_ins, &l_outs);

	return -l_outs.cal_l;

}

// NLopt wrapper for log-likelihood of ARMA model parameter vector, theta, noise
// intensity, sigma, and initial error covaraince hyperparameters
double nlopt_log_lik_wrapper_theta_sigma_initp(const std::vector<double> &x,
	std::vector<double> &grad, void *info){

	log_lik_ins* l_ins = (log_lik_ins*) info;
	int np_arma = l_ins->theta.size(); // order of AR model, p

	for (int i = 0 ; i < np_arma; i++){
		l_ins->theta[i] = x[i];
	}
	l_ins->sigma = x[np_arma];
	l_ins->init_p = x[np_arma+1];

	log_lik_outs l_outs;

	log_lik(l_ins, &l_outs);

	return -l_outs.cal_l;

}

// NLopt wrapper for log-likelihood of ARMA model parameter vector, theta, noise
// intensity, sigma, and observational noise intensity, gamma
double nlopt_log_lik_wrapper_theta_sigma_gamma(const std::vector<double> &x,
	std::vector<double> &grad, void *info){

	log_lik_ins* l_ins = (log_lik_ins*) info;
	int np_arma = l_ins->theta.size(); // order of AR model, p

	for (int i = 0 ; i < np_arma; i++){
		l_ins->theta[i] = x[i];
	}
	l_ins->sigma = x[np_arma];
	l_ins->gamma = x[np_arma+1];

	log_lik_outs l_outs;

	log_lik(l_ins, &l_outs);

	return -l_outs.cal_l;

}

// NLopt wrapper for log-likelihood of all ARMA model parameters/hyperparameters
double nlopt_log_lik_wrapper_all(const std::vector<double> &x,
	std::vector<double> &grad, void *info){

	log_lik_ins* l_ins = (log_lik_ins*) info;
	int np_arma = l_ins->theta.size(); // order of AR model, p

	for (int i = 0 ; i < np_arma; i++){
		l_ins->theta[i] = x[i];
	}
	l_ins->sigma = x[np_arma];
	for (int i = 0 ; i < np_arma; i++){
		l_ins->init_z[i] = x[np_arma+1+i];
	}
	l_ins->init_p = x[2*np_arma+1];
	l_ins->gamma = x[2*np_arma+2];

	log_lik_outs l_outs;

	log_lik(l_ins, &l_outs);

	return -l_outs.cal_l;

}