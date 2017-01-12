#ifndef _MAKE_PSF_H_
#define _MAKE_PSF_H_

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <cmath>

// This function builds the PSF for bleaching and observation.
void make_psf(int &do_xy_to_r, int &TypePSF, int &MS_type, int &mode_ill,
              double &alpha_ill, double &n_ill, double &lambda_ill,
              double &phi_ill, double &alpha_det, double &n_det,
              double &lambda_det, string & PSF_ill_fname,
              string & PSF_det_fname, int &N, double *h, double *h_det,
              double *hI0kh, double &I0kh, double &dr, double &dx, double &dy,
              double &dz, double &rmin, double &xmin, double &ymin,
              double &zmin, int &Nr, int &Nx, int &Ny, int &Nz,
              double &x0_PSF, double &y0_PSF, double &z0_PSF, double &R_PSF,
              double &xmin_PSF, double &ymin_PSF, double &zmin_PSF,
              double &xmax_PSF, double &ymax_PSF, double &zmax_PSF,
	      int &N_beam, double *kvector_x, double *kvector_y, double *kvector_z,
	      double *e_x_vector_x, double *e_x_vector_y, double *e_x_vector_z);

#endif

