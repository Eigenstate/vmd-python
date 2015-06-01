#ifndef _READ_CONF_H_
#define _READ_CONF_H_

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <cmath>

void read_conf(string & conf_name, double &dt, int &M, int &iOutputFrequency,
               double &Lx, double &Ly,
               double &Lz, int &Nx, int &Ny, int &Nz, int &do_xy_to_r,
               double &D, double &I_0, double &kh, double &p0,
               string & out_Obs_name, int &PSF_write_dx, int &TypePSF, double &x0_PSF,
               double &y0_PSF, double &z0_PSF, double &R_PSF, double &Lx_PSF,
               double &Ly_PSF, double &Lz_PSF, int &MS_type, int &mode_ill,
               double &alpha_ill, double &n_ill, double &lambda_ill,
               double &phi_ill, double &alpha_det, double &n_det,
               double &lambda_det, string & PSF_ill_fname,
               string & PSF_det_fname,  int &N_beam, double **kvector_x,
               double **kvector_y, double **kvector_z, double **e_x_vector_x,
               double **e_x_vector_y, double **e_x_vector_z);

#endif

