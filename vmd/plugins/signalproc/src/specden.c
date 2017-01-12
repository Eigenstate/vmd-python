/************************************************************************
 * This program takes time-data and calculates the
 * powerspectrum/fourier transform of the autocorrelation function.
 * This is a re-write of the fourier.x code written by Volker Kleinschmidt
 * and Harald Forbert as a tcl plugin for VMD by Axel Kohlmeyer.
 * (c) 2002-2005 Harald Forbert, Volker Kleinschmidt (c) 2002-2008 Axel Kohlmeyer.
 *
 * usage: calc_specden(<ndat>,<input>,<output>,<deltat>,<maxfreq>,<temp>,<specr>);
 * <ndat>    number of data sets.
 * <input>   time series data.
 * <output>  power spectrum.
 * <normtype> normalization correction type (fourier, classic, kubo, harmonic, schofield)
 * <deltat>  time difference between data sets (in atomic units).
 * <maxfreq> max fequency (in wavenumbers).
 * <temp>    temperature (in kelvin)
 * <specr>   resolution of spectrum (1 gives maximal resolution and noise).
 *
 * the various corrections are:
 * fourier:    is the plain power spectrum of the input data (normalized to
 *             unity in the output frequency range.
 * classical:  is the power spectrum with a prefactor of 
 *             \omega ( 1 - \exp(-\beta \hbar \omega) )
 *             corresponding to the classical/Gordon limit.
 * kubo:       is the power spectrum with a prefactor of
 *             \omega \tanh(\beta \hbar \omega/2)
 *             corresponding to the Kubo correction
 * harmonic:   is the power spectrum with a prefactor of
 *             \omega \beta \hbar \omega
 *             corresponding to the high temperature / harmonic limit
 *             NOTE: this is the _recommended_ correction factor.
 * schofield:  is the power spectrum with a prefactor of
 *             \omega ( 1 - \exp(-\beta \hbar \omega) ) *
 *                                              \exp(\beta \hbar \omega /2)
 *             corresponding to Schofield's correction
 *
 * All spectra with their corresponding prefactor are separately normalized
 * in the output range to sum up to unity.
 *
 * Note: the index of refraction of the medium is set to unity.
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "specden.h"

typedef union {double re; double im;} cmplx;

/* helper function:
 *
 * calculate f_sum   = cos_sum**2 + sin_sum**2,
 *      with cos_sum =  sum_j cos(j*w) input_j
 *       and sin_sum =  sum_j sin(j*w) input_j
 *
 * sums start at 0, but indices of input start at 1, e.g.
 * cos_sum = sum_{j=1}^n cos((j-1)*w) input_j
 */
static cmplx fourier_sum (const int n, const double *input, const double omega)
{
  int k;
  double  lambda, duk, uk, cs, ss, cf, sf;
  cmplx result;
  
  /* in order to be able to sum up the input_j in ascending order
   * use above algorithm with inverse data ordering and correct
   * omega -> - omega and the new origin at the end */

  uk = 0.0;
  duk = 0.0;
  if (cos(omega) > 0.0) {
    lambda = -4.0*sin(0.5*omega)*sin(0.5*omega);
    for (k=1; k <= n; ++k) {
      uk  = uk + duk;
      duk = lambda*uk + duk + input[3*k];
    }
  } else { /* cos(omega) <= 0.0_dbl */
    lambda = 4.0*cos(0.5*omega)*cos(0.5*omega);
    for (k=1; k <= n; ++k) {
      uk  = duk - uk;
      duk = lambda*uk - duk + input[3*k];
    }
  }
  cs = duk - 0.5 * lambda * uk;
  ss = uk * sin(omega);

  /* now correct for ordering: */
  cf = cos(omega*(n-1));
  sf = sin(omega*(n-1));

  result.re = cf*cs+sf*ss;      /* cos_sum */
  result.im = sf*cs-cf*ss;      /* sin_sum */

  return result;
/*  return cf*cs*cf*cs+sf*ss*sf*ss+sf*cs*sf*cs+cf*ss*cf*ss; */
}

/* main function */
int calc_specden(const int ndat, double *input, double *output, 
                 const int normtype, const int specr, 
                 const double maxfreq, const double deltat, const double temp) 
{
  int    nn, i, k;
  double wave_fac, bh, dt, t, c, f, s, e;

  double *ftrans, *wtrans;
  double norm_fourier, norm_classic, norm_kubo, norm_harmonic, norm_schofield;


  wave_fac = 219474.0/deltat;
  bh       = 1.05459e-34/1.38066e-23/2.41889e-17/deltat/temp;
  
  if (specr < 1) {
    fprintf(stderr, "\nspecden spectrum resolution factor must be bigger or equal 1.\n");
    return -20;
  }

  /* number of frequencies */
  nn = (int) ((double)ndat)*maxfreq/wave_fac/(2.0*M_PI);
  if (nn+1 > ndat) {
    fprintf(stderr, "Maximum frequency too large\n");
    return -40;
  }
  nn = nn/specr;
  
  ftrans = malloc((nn+2)*sizeof(double));
  if (ftrans == NULL) {
    fprintf(stderr, "Out of memory, while trying to allocate array 'ftrans'.\n");
    return -50;
  }
  wtrans = malloc((nn+2)*sizeof(double));
  if (ftrans == NULL) {
    fprintf(stderr, "Out of memory, while trying to allocate array 'wtrans'.\n");
    return -60;
  }

  /* read data and apply windowing function */
#if defined(_OPENMP)
#pragma omp parallel for private(i) schedule(static)
#endif
  for (i=1; i<ndat+1; ++i) {
    double win;
    
    win=((double)(2*i-ndat-1))/((double)(ndat+1));
    win=1.0-win*win;
    input[3*i]   *=win;
    input[3*i+1] *=win;
    input[3*i+2] *=win;
  }
  input[3*ndat+3] = 0.0;
  input[3*ndat+4] = 0.0;
  input[3*ndat+5] = 0.0;
  
  dt = 2.0*specr*M_PI/((ndat+1)*specr);
#if defined(_OPENMP)
#pragma omp parallel for private(i,k,t,c,f,s,e) schedule(static)
#endif
  for (i=0; i<nn+1; ++i) {
    cmplx f1,f2,f3;
    
    t = 2.0*((double)(i*specr))*M_PI/((double)(ndat+1));
    c = 0.0;
    
    for (k=0; k < specr; ++k) {

      /* sum over all three dimensions */
      f1 = fourier_sum(ndat,(input+0), t+(double)k*dt);
      f2 = fourier_sum(ndat,(input+1), t+(double)k*dt);
      f3 = fourier_sum(ndat,(input+2), t+(double)k*dt);
      f = f1.re*f1.re;
      f += f1.im*f1.im;
      f += f2.re*f2.re;
      f += f2.im*f2.im;
      f += f3.re*f3.re;
      f += f3.im*f3.im;
      
      /* input data should have zero mean... */
      if (i+k == 0) f=0.0;
        
      /* apply cubic spline correction for input data */
      s=0.5*(t+k*dt);
        
      if (s>0.1) {
        e=pow(sin(s)/s,4.0);
      } else {
        e=pow(1.0-(s*s)/6.0+(s*s*s*s)/120.0,4.0);
      }
      e = e*3.0/(1.0+2.0*cos(s)*cos(s));
      c = c+e*e*f;
    }
    
    wtrans[1+i] = t+0.5*dt*((double)(specr-1));
    ftrans[1+i] = c;
  }

  /* compute norm */
  norm_fourier=norm_classic=norm_kubo=norm_harmonic=norm_schofield=0.0;
  for (i=0; i<=nn; ++i) {
    t = wtrans[1+i];
    f = ftrans[1+i];
    e = t*(1.0 - exp(-bh*t));
    
    norm_fourier  += f;
    norm_classic  += f*e;
    norm_kubo     += f*e/(1.0+exp(-bh*t));
    norm_harmonic += f*t*t;
    norm_schofield += f*e*exp(0.5*bh*t);
  }
  norm_fourier  = 1.0/norm_fourier;
  norm_classic  = 1.0/norm_classic;
  norm_kubo     = 1.0/norm_kubo;
  norm_harmonic = 1.0/norm_harmonic;
  norm_schofield = 1.0/norm_schofield;

  /* output */
  for (i=0; i<=nn; ++i) {
    t = wtrans[1+i];
    f = ftrans[1+i];
    e = t*(1.0 - exp(-bh*t));

    output[2*i] = wave_fac*t;
    switch (normtype) {
      case NORM_FOURIER:
         output[2*i+1] = norm_fourier*f;
         break;
      case NORM_CLASSIC:
         output[2*i+1] = norm_classic *f*e;
         break;
      case NORM_KUBO:
         output[2*i+1] = norm_kubo*f*e/(1.0+exp(-bh*t));
         break;
      case NORM_HARMONIC:
         output[2*i+1] = norm_harmonic*f*t*t;
         break;
      case NORM_SCHOFIELD:
         output[2*i+1] = norm_schofield*f*e*exp(0.5*bh*t);
         break;
      default:
         fprintf(stderr, "specden: unknown normalization. %d\n", normtype);
         return -200;
    }
  }
  return nn;
}


