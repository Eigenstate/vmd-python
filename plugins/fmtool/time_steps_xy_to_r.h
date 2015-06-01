#ifndef _TIME_STEPS_XY_TO_R_
#define _TIME_STEPS_XY_TO_R_


#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <stdlib.h>

#define PI 3.14159265
#define SINGLE        1
#define DOUBLE        2
#define DOUBLENOBC    3
#define DOUBLENOBCOBS 4
#define DOUBLEOBS     5
#define CUDASINGLE    6

 
// Obs and Obs1 - arrays of observables, the fluorescent signal (Obs),
// and its RMSD (Obs1). These two are the 1D arrays representing functions of time.
// Obs and Obs1 are being calculated within this function.
// M - number of time steps (constant).
// kiOutputFrequency - frequency that the observables should be calculated
//                     (every X timesteps)
// dt - time step in seconds (constant).
// N - number of elements in 1D arrays that are functions of position in space.
// h - function of position, 1D array (constant).
// h_det - function of position, 1D array (constant).
// hI0kh - function of position, 1D array (constant).
// dr - grid step in the r-dimension, in micrometers (constant).
// dz - grid step in the z-dimension, in micrometers (constant).
// Nr - number of steps in the r-dimension (constant).
// Nz - number of steps in the z-dimension (constant).
// rmin - minimal value of r (constant).
// p0 - initial value for pnew
// OLD   p - distribution function, the function of r and z; 1D array.
// OLD   pnew - array for the values of p at the next time step.
//      currently, pnew comes in entirely set to 1.0  - kv
// D - diffusion coefficient (constant).
void time_steps_xy_to_r_double_obs(double *Obs, double *Obs1, const int M, 
                        const int kiOutputFrequency, const double dt, 
                        const int N, const double *h, const double *h_det, 
                        const double *hI0kh, const double dr, const double dz, 
                        const int Nr, const int Nz, const double rmin, 
                        /*double *p, double *pnew, */ const double p0, 
                        const double D);

void time_steps_xy_to_r_double_bc_obs(double *Obs, double *Obs1, const int M, 
                        const int kiOutputFrequency, const double dt, 
                        const int N, const double *h, const double *h_det, 
                        const double *hI0kh, const double dr, const double dz, 
                        const int Nr, const int Nz, const double rmin, 
                        /*double *p, double *pnew, */ const double p0, 
                        const double D);

void time_steps_xy_to_r_double(double *Obs, double *Obs1, const int M, 
                        const int kiOutputFrequency, const double dt, 
                        const int N, const double *h, const double *h_det, 
                        const double *hI0kh, const double dr, const double dz, 
                        const int Nr, const int Nz, const double rmin, 
                        /*double *p, double *pnew, */ const double p0, 
                        const double D);

void time_steps_xy_to_r_double_bc(double *Obs, double *Obs1, const int M, 
                        const int kiOutputFrequency, const double dt, 
                        const int N, const double *h, const double *h_det, 
                        const double *hI0kh, const double dr, const double dz, 
                        const int Nr, const int Nz, const double rmin, 
                        /*double *p, double *pnew, */ const double p0, 
                        const double D);

void time_steps_xy_to_r_float(float *Obs, float *Obs1, const int M, 
                        const int kiOutputFrequency, const float dt, 
                        const int N, const float *h, const float *h_det, 
                        const float *hI0kh, const float dr, const float dz, 
                        const int Nr, const int Nz, const float rmin, 
                        /*float *p, float *pnew, */ const float p0, 
                        const float D);

void printArray(const char *str, const double *d, const int x, const int y);
void printArrayF(const char *str, const float *d, const int x, const int y);


#endif

