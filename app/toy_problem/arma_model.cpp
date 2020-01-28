/*  ____________________________________________________________________________

    KARMA4: Library for autoregressive moving average (ARMA) modeling and
            forecasting of time-series data with process and observational error
    Copyright 2017 Sandia Corporation.
    This software is distributed under the BSD License.
    For more information, see the README.txt file in the top KARMA4 directory.
    _________________________________________________________________________ */

//- Description: Driver application for AR modeling of wavelet coefficients
//  resulting from compressed sensing of the transient response of the 2D heat
//  equation on a square domain with randomly chosen holes

#include <math.h>
#include <karma4.hpp>
#include <vector>
#include <iostream>
#include <stdlib.h>

using namespace std;

int main(int argc, char *argv[])
{
  int nz,nc,nce;  
  int num_ts;
  int num_forecast;
  int nstart;
  int adap;
  char* fname;
  double temp,xv,tolm0;
  
  if (argc==8) 
  {
    nstart = atoi(argv[1]);
    num_ts = atoi(argv[2]);
    num_forecast = atoi(argv[3]);
    nce = atoi(argv[4]);
    xv = atof(argv[5]);
    fname = argv[6];
    adap = atoi(argv[7]);
  }
  else
  {
    cerr << "input error" << endl;
  }

//----------------------------------------------------------------------------//
// Load data (time-series)
//----------------------------------------------------------------------------//
  FILE* f_out;
  
  if(!(f_out = fopen(fname,"r"))){ 
    printf("could not open file \n"); 
    exit(1);
  }
  fscanf(f_out, "%d", &nz);
  fscanf(f_out, "%d", &nc);

  std::vector<std::vector<double> >wc_data(nc,
    vector<double>(num_ts+num_forecast));
  
  // just seeking to nstart
  for (int i = 0; i < nstart; i++)
    for (int j = 0; j < nc; j++)
      fscanf(f_out, "%lf", &temp);
  
  // Now reading  
  for (int i = 0; i < num_ts+num_forecast; i++)
    for (int j = 0; j < nc; j++)
      fscanf(f_out, "%lf", &wc_data[j][i]);
  fclose(f_out);
  
  //--------------------------------------------------------------------------//
  // Run test_pde_timeseries_w_new
  //--------------------------------------------------------------------------//
  
  test_ins t_ins;
  test_outs t_outs;
  t_ins.num_ts = num_ts;
  t_ins.num_forecast = num_forecast;
  t_ins.p_low = 1; //lowest proposed value of p (MA order)
  t_ins.p_high = 3; //greatest proposed value of p (MA order)
  t_ins.x_frac = xv; // fraction of data to be used for cross-validation
      
  if(!(f_out = fopen("forecasts.dat","w"))){ 
    printf("could not open file forecasts.dat\n"); 
    exit(1);
  }
  
  double mm,ss,ma,mi;
  std::vector<double> mes(nce);
  
  // computing the metric (max(v) - min(v))*standard deviation(v) * mean(abs(v))
  // for each vector
  for (int j = 0; j < nce; j++) {
        
    std::vector<double> onev = wc_data[j];
    
    // Computing the mean (mm) of the vector
    mm=0.0e0;
    for (int i = 0; i < num_ts; i++)
      mm+=onev[i];
    mm/=double(num_ts);
    
    // Computing the standard deviation (ss) of the vector
    ss=0.0e0;
    for (int i = 0; i < num_ts; i++)
      ss+=(onev[i]-mm)*(onev[i]-mm);
    ss/=double(num_ts);
    ss=sqrt(ss);
    
    // Computing the max (ma) and min (mi) of the vector
    ma=onev[0];
    for (int i = 0; i < num_ts; i++)
      if (onev[i]>ma)
	ma=onev[i];
    mi=onev[0];
    for (int i = 0; i < num_ts; i++)
      if (onev[i]<mi)
	mi=onev[i];
    
    mes[j]=(ma-mi)*ss;  
    
    // Computing the mean mm of the absolute value of the vector
    mm=0.0e0;
    for (int i = 0; i < num_ts; i++)
      mm+=fabs(onev[i]);
    mm/=double(num_ts);
    
    mes[j]*=mm;
  }
  
  tolm0=mes[0];
  for (int j = 0; j < nce; j++)
    if (mes[j]>tolm0)
      tolm0=mes[j];
  
#ifdef OPENMP
#pragma omp parallel for
#endif  
  for (int j = 0; j < nce; j++) {
    
    t_ins.wc_data = wc_data[j];
    
    // if adaptivity is invoked use the metric to find a suitable multiplicatio
    // factor for the tolerances
    if (adap!=0)
      t_ins.tolm = tolm0/mes[j];
    else
      t_ins.tolm = 1.0e0;
    
    printf("coeff = %d %f \n",j,t_ins.tolm);
    test_pde_timeseries_w_new(&t_ins, &t_outs);

  
    // Writing forecast output
    for (int i = 0; i < t_outs.zfori.size(); i++){
      fprintf(f_out, "%0.10g ", t_outs.zfori[i]);
    }
    fprintf(f_out, "\n");
  }
  fclose(f_out);
  
  // Writing net input data
  if(!(f_out = fopen("z.dat","w"))){ 
    printf("could not open file z.dat\n"); 
    exit(1);
  }
  
  for (int j = 0; j < nce; j++) {
    for (int i = 0; i < num_ts+num_forecast; i++){
      fprintf(f_out, "%0.10g ", wc_data[j][i]);
    }
    fprintf(f_out, "\n");
  }
  fclose(f_out);
  
  return 0;
}