/*  ____________________________________________________________________________

    KARMA4: Library for autoregressive moving average (ARMA) modeling and
            forecasting of time-series data with process and observational error
    Copyright 2017 Sandia Corporation.
    This software is distributed under the BSD License.
    For more information, see the README.txt file in the top KARMA4 directory.
    _________________________________________________________________________ */

//- Description: Driver application for fitting and subsequent forecasting of
//  time-series data using karma4 library: A verification exercise in which the
//  data generating model is in fact an AR model

#include <math.h>
#include <karma4.hpp>
#include <vector>
#include <iostream>
#include <stdlib.h>

using namespace std;

int main(int argc, char *argv[])
{
  int nz;
  int num_ts;
  int num_forecast = 40;

//----------------------------------------------------------------------------//
// Load data (time-series)
//----------------------------------------------------------------------------//
  FILE* f_out;
  if(!(f_out = fopen("nz.dat","r"))){ 
    printf("could not open file nz.dat\n");
    exit(1);
  }
  fscanf(f_out, "%d", &nz);
  fclose(f_out);

  std::vector<double> wc_data(nz);

  if(!(f_out = fopen("z.dat","r"))){ 
    printf("could not open file z.dat\n"); 
    exit(1);
  }
  for (int k = 0; k < nz; k++){
    fscanf(f_out, "%lf", &(wc_data[k]));
  }
  fclose(f_out);

  //--------------------------------------------------------------------------//
  // Run test_pde_timeseries_w_new
  //--------------------------------------------------------------------------//

  num_ts = nz; //use all available data

  test_ins t_ins;
  test_outs t_outs;
  t_ins.num_ts = num_ts;
  t_ins.num_forecast = num_forecast;
  t_ins.wc_data = wc_data;
  t_ins.p_low = 8; //lowest proposed value of p (MA order)
  t_ins.p_high = 10; //greatest proposed value of p (MA order)
  t_ins.x_frac = 0.3; // fraction of data to be used for cross-validation
  t_ins.tolm = 1.0e0;
  
  test_pde_timeseries_w_new(&t_ins, &t_outs);

  if(!(f_out = fopen("forecasts.dat","w"))){ 
    printf("could not open file forecasts.dat\n"); 
    exit(1);
  }
  for (int i = 0; i < t_outs.zfori.size(); i++){
    fprintf(f_out, "%0.10g\n", t_outs.zfori[i]);
  }
  fclose(f_out);

  
  if(!(f_out = fopen("forecast_err.dat","w"))){ 
    printf("could not open file forecasts.dat\n"); 
    exit(1);
  }
  for (int i = 0; i < t_outs.zfori.size(); i++){
    fprintf(f_out, "%0.10g\n", t_outs.Bfor[i]);
  }
  fclose(f_out);


  return 0;
}