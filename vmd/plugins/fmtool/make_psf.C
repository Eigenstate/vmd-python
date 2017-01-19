
using namespace std;

#include "psf_from_file.h"
#include "psf_from_file_xy_to_r.h"

#define PI 3.14159265

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
	      double *e_x_vector_x, double *e_x_vector_y, double *e_x_vector_z)
{

   double r0 = 0.0;
   int n = 0;
   int i = 0;
   int j = 0;
   int k = 0;
   double h_norm = 0.0;
   double h_det_norm = 0.0;
   double r = 0.0;
   double x = 0.0;
   double y = 0.0;
   double z = 0.0;

   if (do_xy_to_r == 0)
   {
// FULL 3D.
//////////////////////////////////////////////////
      for (n = 1; n <= N; n++)
      {
         h[n] = 0.0;
      }

// Spherical constant PSF.
      if (TypePSF == 0)
      {
         n = 0;
         for (k = 1; k <= Nz; k++)
         {
            z = zmin + (k - 1) * dz;
            for (j = 1; j <= Ny; j++)
            {
               y = ymin + (j - 1) * dy;
               for (i = 1; i <= Nx; i++)
               {
                  x = xmin + (i - 1) * dx;
                  n = n + 1;
                  h[n] = 0.0;

                  r = sqrt((x - x0_PSF) * (x - x0_PSF) +
                           (y - y0_PSF) * (y - y0_PSF) + (z - z0_PSF) * (z -
                                                                         z0_PSF));
                  if (r < R_PSF)
                  {
                     h[n] = 1.0;
                  }
                  h_det[n] = h[n];
               }
            }
         }
      }


// Rectangular constant PSF.
      if (TypePSF == 1)
      {
         n = 0;
         for (k = 1; k <= Nz; k++)
         {
            z = zmin + (k - 1) * dz;
            for (j = 1; j <= Ny; j++)
            {
               y = ymin + (j - 1) * dy;
               for (i = 1; i <= Nx; i++)
               {
                  x = xmin + (i - 1) * dx;
                  n = n + 1;
                  h[n] = 0.0;

                  if ((x > xmin_PSF) && (x < xmax_PSF) && (y > ymin_PSF)
                      && (y < ymax_PSF) && (z > zmin_PSF) && (z < zmax_PSF))
                  {
                     h[n] = 1.0;
                  }
                  h_det[n] = h[n];
               }
            }
         }
      }

// Read PSF from the file.
      if ((TypePSF != 0) && (TypePSF != 1))
      {
	  psf_from_file(MS_type, mode_ill, alpha_ill, n_ill,
                        lambda_ill, phi_ill, alpha_det, n_det,
                        lambda_det, PSF_ill_fname, PSF_det_fname, N, h,
                        h_det, dx, dy, dz, xmin, ymin, zmin, Nx, Ny, Nz,
			N_beam, kvector_x, kvector_y, kvector_z,
	                e_x_vector_x, e_x_vector_y,e_x_vector_z);
      }

// Normalization.
      h_norm = 0.0;
      for (n = 1; n <= N; n++)
      {
         h_norm = h_norm + h[n];
         h_det_norm = h_det_norm + h_det[n];
      }
      for (n = 1; n <= N; n++)
      {
         h[n] = h[n] / (h_norm * dx * dy * dz);
         h_det[n] = h_det[n] / (h_det_norm * dx * dy * dz);
         hI0kh[n] = I0kh * h[n];
      }
//////////////////////////////////////////////////

   }
   else
   {
// XY_to_R.
//////////////////////////////////////////////////
      for (n = 1; n <= N; n++)
      {
         h[n] = 0.0;
      }

// Spherical constant PSF.
      if (TypePSF == 0)
      {
         n = 0;
         for (k = 1; k <= Nz; k++)
         {
            z = zmin + (k - 1) * dz;
            for (i = 1; i <= Nr; i++)
            {
               r = rmin + (i - 1) * dr;
               n = n + 1;
               h[n] = 0.0;

               r0 = sqrt(r * r + (z - z0_PSF) * (z - z0_PSF));
               if (r0 < R_PSF)
               {
                  h[n] = 1.0;
               }
               h_det[n] = h[n];
            }
         }
      }

// Read PSF from the file.
      if ((TypePSF != 0) && (TypePSF != 1))
      {
         psf_from_file_xy_to_r(MS_type, mode_ill, alpha_ill, n_ill,
                               lambda_ill, phi_ill, alpha_det, n_det,
                               lambda_det, PSF_ill_fname, PSF_det_fname, N, h,
                               h_det, dr, dz, rmin, zmin, Nr, Nz);
      }

// Normalization.
      h_norm = 0.0;
      n = 0;
      for (k = 1; k <= Nz; k++)
      {
         for (i = 1; i <= Nr; i++)
         {
            r = rmin + (i - 1) * dr;
            n = n + 1;
            h_norm = h_norm + h[n] * r;
            h_det_norm = h_det_norm + h_det[n] * r;
         }
      }
      for (n = 1; n <= N; n++)
      {
         h[n] = h[n] / (2.0 * PI * h_norm * dr * dz);
         h_det[n] = h_det[n] / (2.0 * PI * h_det_norm * dr * dz);
         hI0kh[n] = I0kh * h[n];
      }
//////////////////////////////////////////////////
   }

}
