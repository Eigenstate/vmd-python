#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <stdlib.h>

using namespace std;

#include "make_psf.h"
#include "write_psf.h"
#include "read_conf.h"
#include "time_steps_xy_to_r.h"
#include "time_steps_xy_to_r_cuda.h"
#include "cudamgr.h"

#define PI 3.14159265

int main(int argc, char *argv[]) {
   int k;
   int n;
   int l;

// These values are first pre-defined here, but then defined "for real" in the input file.
//////////////////////////////////////////////////
   int M = 1;      // Number of time steps.
   int iOutputFrequency = 1;      // default frequency (timesteps) to 
                                  // print observables
   double dt = 0.5;  // Time step.
   double Lx = 7.0;  // Length of the considered region in x direction.
   double Ly = 7.0;  // Length of the considered region in y direction.
   double Lz = 7.0;  // Length of the considered region in z direction.
   int Nx = 71;    // Number of nodes in x direction.
   int Ny = 71;    // Number of nodes in y direction.
   int Nz = 71;    // Number of nodes in z direction.

   int do_xy_to_r = 0;  // By default, do full 3D.

   double D = 1.0; // Diffusion coefficient.
   double I_0 = 0.0; // Light intensity.
   double kh = 1.0;  // "Reactivity".

// Initial conditions. p(r,t=0) = p0.
   double p0 = 1.0;

// The output file for the observable.
   string out_Obs_Name = "Obs.dat";
// Check to write the 3D map of the PSF.
// If 0, do not write, otherwise do write.
   int PSF_write_dx = 0;

// PSF parameters.
   int TypePSF = 0;
   double x0_PSF = 0.0;
   double y0_PSF = 0.0;
   double z0_PSF = 0.0;
   double R_PSF = 1.0;
   double Lx_PSF = 1.0;
   double Ly_PSF = 1.0;
   double Lz_PSF = 1.0;

// Type of microscopy: 0 for confocal, 1 for 4Pi, and any other positive value for multiple beams.
   int MS_type = -1;  // Should be defined in the configuration file.
// Illumination and detection parameters.
   int mode_ill = 1; // By default, the illumination is in two-photon regime.
   double alpha_ill = 50.0;
   double n_ill = 1.5;
   double lambda_ill = 500.0;
   double phi_ill = 90.0;
   double alpha_det = 50.0;
   double n_det = 1.5;
   double lambda_det = 500.0;
   string PSF_ill_fname = "/Projects/anton/FRAP/Ifunctions/50.";
   string PSF_det_fname = "/Projects/anton/FRAP/Ifunctions/50.";
// Parameters for the interference of multiple beams.
   int N_beam = -1; // Need to specify number of beams in the config file.
   double *kvector_x;
   double *kvector_y;
   double *kvector_z;
   double *e_x_vector_x;
   double *e_x_vector_y;
   double *e_x_vector_z;
//////////////////////////////////////////////////

//Read input data.
//////////////////////////////////////////////////
   if (argc <= 1)
   {
      fprintf (stdout, "Usage:  %s [-vdsneo] filename\n", argv[0]);
      fprintf (stdout, "where -v verbose printing -d specifies double precision, -s single, -n (DEFAULT) double, BC padding, -e double, obs separate, -o double, BC padding, obs separate, -c CUDA single\n");
      return 1;
   }

   int iFileArg = -1;
   int iMethod = DOUBLENOBC;
   int iVerbose = 0;

   for (int i=1; i<(argc/*-1*/); i++)
   {
      if (argv[i][0] != '-')
      {
         iFileArg = i;
      }
      else
      {
         char c = *(argv[i]+1);
         switch (c) {
            case 'd':
               iMethod = DOUBLE;
               break;
            case 'e':
               iMethod = DOUBLEOBS;
               break;
            case 'n':
               iMethod = DOUBLENOBC;
               break;
            case 'o':
               iMethod = DOUBLENOBCOBS;
               break;
            case 's':
               iMethod = SINGLE;
               break;
            case 'c':
               iMethod = CUDASINGLE;
               break;
            case 'v':
               iVerbose = 1;
               break;
            default:
               fprintf(stdout, "Unknown option: '%c'\n", c);
               break;
         }
      }
   }

   if (iFileArg == -1)
   {
      fprintf (stdout, "Usage:  %s [-vdsneo] filename\n", argv[0]);
      fprintf (stdout, "where -v verbose printing -d specifies double precision, -s single, -n (DEFAULT) double, BC padding, -e double, obs separate, -o double, BC padding, obs separate, -c CUDA single\n");
      return 1;
   }

   string conf_name = argv[iFileArg];
   read_conf(conf_name, dt, M, iOutputFrequency, Lx, Ly, Lz, Nx, Ny, Nz, do_xy_to_r, D, I_0, kh,
             p0, out_Obs_Name, PSF_write_dx, TypePSF, x0_PSF, y0_PSF, z0_PSF, R_PSF, Lx_PSF,
             Ly_PSF, Lz_PSF, MS_type, mode_ill, alpha_ill, n_ill, lambda_ill,
             phi_ill, alpha_det, n_det, lambda_det, PSF_ill_fname,
             PSF_det_fname, N_beam, &kvector_x, &kvector_y, &kvector_z,
	     &e_x_vector_x, &e_x_vector_y, &e_x_vector_z);

   double Lr = Lx / 2.0;   // Length of the considered region over r (x-y plane).
   int Nr = Nx / 2;
//////////////////////////////////////////////////


// Define additional parameters.
//////////////////////////////////////////////////
// Define time domain.
   double t_sim = M * dt;  // Length of the simulation (time).

// Define spatial domain.
   double dr = Lr / (Nr - 1);
   double dx = Lx / (Nx - 1); // Step in x direction.
   double dy = Ly / (Ny - 1); // Step in y direction.
   double dz = Lz / (Nz - 1); // Step in z direction.
   double rmin = 0.0;
   double xmin = -Lx / 2.0;   // Minimal value of x.
   double ymin = -Ly / 2.0;   // Minimal value of y.
   double zmin = -Lz / 2.0;   // Minimal value of z.
   double rmax = rmin + Lr;
   double xmax = xmin + Lx;   // Maximal value of x.
   double ymax = ymin + Ly;   // Maximal value of y.
   double zmax = zmin + Lz;   // Maximal value of z.

   int N = 0;
   if (do_xy_to_r == 0)
   {
      N = Nx * Ny * Nz;
   }
   else
   {
      N = Nr * Nz;
   }

// Equation parameters.
// d p(r,t)/dt = D (d2x + d2y + d2z) p(r,t) - I_0 kh h(r) p(r,t)
   double I0kh = I_0 * kh;
   double *h = new double[N + 1];   // Point Spread Function (PSF).
   double *hI0kh = new double[N + 1];  // hI0kh = h*I0*kh

// Different intensity for t > t_B is disabled in this version.
//   double t_B = 1.0; // Strong bleaching at t < t_B.
//   double I_B = I_0; // Different intensity for t > t_B.
//   double count_t_B = 0.0; //

   double *h_det = new double[N + 1];  // Detection PSF.

// Distribution of the particles; p(r,t).
//   double *p = new double[N + 1];
//   double *pnew = new double[N + 1];
//   int Nxy = Nx * Ny;
//   int NxNy1 = Nx * (Ny + 1);

//Observable.
   //Obs(t) = \int dr h(r) h_det(r) p(r,t)
   double *Obs = new double[M/iOutputFrequency + 1]; 
   //Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t)
   double *Obs1 = new double[M/iOutputFrequency + 1];   
//   char Obs_out[300];

// following not needed.  Obs/Obs1 are set, not accumulated
//   for (k = 1; k <= M; k++)
//   {
//      Obs[k] = 0.0;
//      Obs1[k] = 0.0;
//   }


// PSF dimensions (for rectangular PSF).
   double xmax_PSF = x0_PSF + Lx_PSF / 2.0;
   double ymax_PSF = y0_PSF + Ly_PSF / 2.0;
   double zmax_PSF = z0_PSF + Lz_PSF / 2.0;
   double xmin_PSF = x0_PSF - Lx_PSF / 2.0;
   double ymin_PSF = y0_PSF - Ly_PSF / 2.0;
   double zmin_PSF = z0_PSF - Lz_PSF / 2.0;
//////////////////////////////////////////////////

   if (do_xy_to_r == 0)
      {
         printf("Full 3D calculation (do_xy_to_r = %d)\n", do_xy_to_r);
      }
      else
      {
         printf
            ("{x,y} is reduced to one variable r (axial symmetry is implied, effectively 2D) (do_xy_to_r = %d)\n",
             do_xy_to_r);
         printf("THIS IS THE VERSION FOR AXIALLY-SYMMETRIC SYSTEMS\n");
         if (Nx != Ny)
         {
            printf("ERROR: Nx and Ny are different. Please set Nx and Ny so that Nx = Ny \n");
            return 1;
         }
         if (Lx != Ly)
         {
            printf("ERROR: Lx and Ly are different. Please set Lx and Ly so that Lx = Ly \n");
            return 1;
         }
	 if (TypePSF == 1)
	 {
	    printf("TypePSF = %d; rectangular PSF\n", TypePSF);
            printf("ERROR: rectangular PSF. Please do not use rectangular PSF: axially symmetric PSF is required\n");
            return 1;
	 }
	 if ((TypePSF == 0) && ((x0_PSF != 0.0) || (y0_PSF != 0.0)))
	 {
	    printf("ERROR: x and y of the PSF center are not at {0,0}. Please set the x and y of the PSF's center to {0,0}\n");
            return 1;
	 }
	 if ((TypePSF != 0) && (TypePSF != 1) && (MS_type != 0) && (MS_type != 1) && (N_beam > 0))
	 {
	    printf("ERROR: PSF by interference of several beams is supported for the full 3D case only; aborting.\n");
            return 1;
	 }
      }
      printf("\n");

   if (iVerbose)
   {
   // Print out the list of main parameters and their values.
   //////////////////////////////////////////////////

      printf("Precision: %s\n", ((iMethod == SINGLE || iMethod == CUDASINGLE) ? "single" : (DOUBLE == iMethod ?  "double" : "double, with BC padding")));
      printf("Time step dt = %f\n", dt);
      printf("Number of time steps M = %d\n", M);
      printf("Observables will be output every %d timesteps.\n", iOutputFrequency);
      printf("Simulation time t_sim = %f\n", t_sim);
      printf("\n");

      if (do_xy_to_r == 0)
      {
         printf
            ("Length of the considered region in x, y and z directions (MICROMETERS):\n");
         printf("Lx = %f, Ly = %f, Lz = %f\n", Lx, Ly, Lz);
         printf
            ("Number of grid nodes in each direction: Nx = %d, Ny = %d, Nz = %d\n",
             Nx, Ny, Nz);
         printf("(MICROMETERS): dx = %f, dy = %f, dz = %f\n", dx, dy, dz);
         printf("(MICROMETERS): xmin = %f, ymin = %f, zmin = %f\n", xmin, ymin,
                zmin);
         printf("(MICROMETERS): xmax = %f, ymax = %f, zmax = %f\n", xmax, ymax,
                zmax);
      }
      else
      {
         printf
            ("Length of the considered region in r and z directions (MICROMETERS):\n");
         printf("Lr = %f, Lz = %f\n", Lr, Lz);
         printf("Number of grid nodes in each direction: Nr = %d, Nz = %d\n", Nr,
                Nz);
         printf("(MICROMETERS): dr = %f, dz = %f\n", dr, dz);
         printf("(MICROMETERS): rmin = %f, zmin = %f\n", rmin, zmin);
         printf("(MICROMETERS): rmax = %f, zmax = %f\n", rmax, zmax);
      }
      printf("\n");

      printf("Diffusion coefficient: %f\n", D);
      printf("Light intensity I_0: %f\n", I_0);
      printf("Reactivity of the fluorophores, kh: %f\n", kh);
      printf("\n");
      printf("Initial conditions. p(r,t=0) = %f\n", p0);
      printf("\n");
      printf("The name of the output file for the observable: %s\n",
             out_Obs_Name.c_str());
      printf("\n");
      printf("Point spread function (PSF).\n");
      if ((TypePSF == 0) || (TypePSF == 1))
      {
         printf("Constant PSF; center at (MICROMETERS) {x y z} = {%f %f %f}\n",
                x0_PSF, y0_PSF, z0_PSF);
         if (TypePSF == 0)
         {
            printf("TypePSF = %d; spherical PSF with radius %f MICROMETERS:\n",
                   TypePSF, R_PSF);
            printf("\n");
         }
         else
         {
            if (do_xy_to_r == 0)
            {
               printf("TypePSF = %d; rectangular PSF; it is not zero for\n",
                      TypePSF);
               printf("(MICROMETERS): %f < x < %f\n", xmin_PSF, xmax_PSF);
               printf("(MICROMETERS): %f < y < %f\n", ymin_PSF, ymax_PSF);
               printf("(MICROMETERS): %f < z < %f\n", zmin_PSF, zmax_PSF);
            }
            printf("\n");
         }
      }
      else
      {
         printf("TypePSF = %d; read PSF from the file\n", TypePSF);

         if (MS_type == 0)
         {
            printf("Microscopy type: confocal (MS_type = %d)\n", MS_type);
         }
         if (MS_type == 1)
         {
            printf("Microscopy type: 4Pi (MS_type = %d)\n", MS_type);
         }
	 if ((MS_type != 0) && (MS_type != 1))
         {
	    printf("Microscopy type: use multiple interfering beams (MS_type = %d)\n", MS_type);
	    printf("Using %d beams; vectors k_beam and e_x for each beam follow below.\n", N_beam);
	    for (int i=0; i<N_beam;i++)
            {
               fprintf (stdout, "k_beam(%d) = {%f %f %f}\n", i+1, kvector_x[i], kvector_y[i], kvector_z[i]);
               fprintf (stdout, "e_x_beam(%d) = {%f %f %f}\n", i+1, e_x_vector_x[i], e_x_vector_y[i], e_x_vector_z[i]);
            }
	 }

         if (mode_ill == 0)
         {
            printf("Illumination mode: one-photon (mode_ill = %d)\n", mode_ill);
         }
         else
         {
            printf("Illumination mode: two-photon (mode_ill = %d)\n", mode_ill);
         }

         printf("Illumination alpha is %f degrees\n", alpha_ill);
         printf("Illumination refraction index is %f\n", n_ill);
         printf("Illumination wavelength is %f NANOMETERS\n", lambda_ill);
         printf("Illumination angle phi is %f degrees\n", phi_ill);
         printf("Detection alpha is %f degress\n", alpha_det);
         printf("Detection refraction index is %f\n", n_det);
         printf("Detection wavelength is %f NANOMETERS\n", lambda_det);
         printf("\n");
      }
      

   }

   /// open output file
   FILE *outObs;
   outObs = fopen(out_Obs_Name.c_str(), "w");
   if (outObs == NULL)  {
     printf("FATAL ERROR: failed to open output file.  Exiting.\n");
     return -1;
   }
//////////////////////////////////////////////////

// Construct PSF.
//////////////////////////////////////////////////
   make_psf(do_xy_to_r, TypePSF, MS_type, mode_ill,
            alpha_ill, n_ill, lambda_ill, phi_ill,
            alpha_det, n_det, lambda_det, PSF_ill_fname, PSF_det_fname,
            N, h, h_det, hI0kh, I0kh,
            dr, dx, dy, dz, rmin, xmin, ymin, zmin,
            Nr, Nx, Ny, Nz, x0_PSF, y0_PSF, z0_PSF,
            R_PSF, xmin_PSF, ymin_PSF, zmin_PSF,
            xmax_PSF, ymax_PSF, zmax_PSF,
	    N_beam, kvector_x, kvector_y, kvector_z,
	    e_x_vector_x, e_x_vector_y, e_x_vector_z);

   write_psf(do_xy_to_r, N, h, h_det, dr, dx, dy, dz, rmin, xmin, ymin, zmin,
             Nr, Nx, Ny, Nz, PSF_write_dx);
//////////////////////////////////////////////////

// Initial conditions.
//**********************************************//
//
//   for (n = 1; n <= N; n++)
//   {
//      pnew[n] = p0;
////  pnew[n] = 0.0;
////  if (h[n] == 0.0) pnew[n] = p0;
//   }
//**********************************************//

// Time steps.
//**********************************************//
   if (do_xy_to_r ) // Effectively 2D.
   {
//      for (int i=0;i<=N;i++)
//      {
//         h[i]=i;
//         h_det[i]=i+10;
//         hI0kh[i]=i+20;
//      }
//
//       printArray("h",h,N+1,1);
//       printArray("h_det",h_det,N+1,1);
//       printArray("hI0kh",hI0kh,N+1,1); 
      if (DOUBLE == iMethod)
      {
         time_steps_xy_to_r_double_bc(Obs, Obs1, M, iOutputFrequency, dt, N, h,
                         h_det, hI0kh, dr, dz, Nr, Nz,
                         rmin, /*p, pnew, */p0, D);
      }
      else if (DOUBLENOBCOBS == iMethod)
      {
         time_steps_xy_to_r_double_obs(Obs, Obs1, M, iOutputFrequency, dt, N, h,
                         h_det, hI0kh, dr, dz, Nr, Nz,
                         rmin, /*p, pnew, */p0, D);
      }
      else if (DOUBLEOBS == iMethod)
      {
         time_steps_xy_to_r_double_bc_obs(Obs, Obs1, M, iOutputFrequency, 
                         dt, N, h,
                         h_det, hI0kh, dr, dz, Nr, Nz,
                         rmin, /*p, pnew, */p0, D);
      }
      else if (DOUBLENOBC == iMethod)
      {
         time_steps_xy_to_r_double(Obs, Obs1, M, iOutputFrequency, dt, N, h,
                         h_det, hI0kh, dr, dz, Nr, Nz,
                         rmin, /*p, pnew, */p0, D);
      }
      else if (SINGLE == iMethod)
      {

         // let's copy some arrays, just to make this easy for now
         float *Obs_float = new float[M + 1]; 
         float *Obs1_float = new float[M + 1];   
         float *h_float = new float[N + 1];   
         float *hI0kh_float = new float[N + 1];   
         float *h_det_float = new float[N + 1];  
         for (int i=0; i<N+1;i++)
         {
            h_float[i] = h[i];
            hI0kh_float[i] = hI0kh[i];
            h_det_float[i] = h_det[i];
         }

         time_steps_xy_to_r_float(Obs_float, Obs1_float, M, 
                         iOutputFrequency, dt, N, h_float,
                         h_det_float, hI0kh_float, dr, dz, Nr, Nz,
                         rmin, p0, D);
         // copy the results back into the original arrays
         for (int i=0; i<=M/iOutputFrequency;i++)
         {
            Obs[i]=Obs_float[i];
            Obs1[i]=Obs1_float[i];
         }

         delete [] Obs_float; Obs_float=0;
         delete [] Obs1_float; Obs1_float=0;
         delete [] h_float; h_float=0;
         delete [] hI0kh_float; hI0kh_float=0;
         delete [] h_det_float; h_det_float=0;
      }
      else if (CUDASINGLE == iMethod)
      {
#if defined(CUDA)
         open_cuda_dev(0); // explicitly attach to device 0 before we run

         // let's copy some arrays, just to make this easy for now
         float *Obs_float = new float[M + 1]; 
         float *Obs1_float = new float[M + 1];   
         float *h_float = new float[N + 1];   
         float *hI0kh_float = new float[N + 1];   
         float *h_det_float = new float[N + 1];  
         for (int i=0; i<N+1;i++)
         {
            h_float[i] = h[i];
            hI0kh_float[i] = hI0kh[i];
            h_det_float[i] = h_det[i];
         }

         time_steps_xy_to_r_cuda(Obs_float, Obs1_float, M, 
                         iOutputFrequency, dt, N, h_float,
                         h_det_float, hI0kh_float, dr, dz, Nr, Nz,
                         rmin, p0, D);
         // copy the results back into the original arrays
         for (int i=0; i<=M/iOutputFrequency;i++)
         {
            Obs[i]=Obs_float[i];
            Obs1[i]=Obs1_float[i];
         }

         delete [] Obs_float; Obs_float=0;
         delete [] Obs1_float; Obs1_float=0;
         delete [] h_float; h_float=0;
         delete [] hI0kh_float; hI0kh_float=0;
         delete [] h_det_float; h_det_float=0;
#else
      printf("ERROR: not a CUDA build\n");
      exit(-1); 
#endif
      }
   }
// Full 3D.
   else
   {
      printf("Full 3D: PSF is built and written to a file, but actual full 3D calculation is disabled in this version; aborting.\n");
      return 1;
   }
//**********************************************//


// Writing the output file for the Observable.
//////////////////////////////////////////////////
   double n_Obs = 1.0 / Obs[0];
   double n_Obs1 = 1.0 / Obs1[0];
//   fprintf (stderr, "n_Obs=%16.12f, n_Obs1=%16.12f\n", n_Obs, n_Obs1); 
   for (l = 0; l < M/iOutputFrequency; l++)
   {
      fprintf(outObs, "%.20f  %.20f  %.20f\n", (l*iOutputFrequency) * dt, 
              n_Obs * Obs[l],
              n_Obs1 * Obs1[l]);
//  fprintf (outObs, "%.20f  %.20f  %.20f\n", (l-1)*dt, n_Obs*Obs[l], n_Obs1*sqrt(Obs1[l])/Obs[l]);
   }
   fclose(outObs);
//////////////////////////////////////////////////

/*
// Calculating the correlation function assuming that no bleaching is taking place.
//////////////////////////////////////////////////
outObs = fopen (out_Obs_Name.c_str(), "w");
double sigma_r = 0.15;//In micrometers.
double sigma_z = 0.1;//In micrometers.
double C_av = 15.0*0.000001*6.02*100000000.0;//In 1/(micrometer^3).
double cor_f = 0.0;
double cor_f_a = 0.0;
double pre_cor_f;
double pre_cor_f_a;
double tau;
double tau0 = 0.0002;
double taum = 0.5;
double dexptau = log(taum/tau0);
int lm = 35;
int kp;
int ip;
int np;
double zp;
double rp;
double ov4Dtau;
int ns;
int Nrr = Nr*Nr;
double* fs = new double[Nrr+1];
int Nphi = 100;
double dphi = 2.0*PI/(Nphi-1);
double phi;
double phip;

printf ("\n");
printf ("Computing the correlation function (FCS)\n");
printf ("using the no-bleaching limit (k = 0).\n");

pre_cor_f_a = 1.0/(C_av*sqrt(8.0*PI*PI*PI)*sigma_r*sigma_r*sigma_z);
tmp = 0.0;
tmpr = 0.0;
n = 0;
for (k = 1; k <= Nz; k++) {
  z = zmin + (k-1)*dz;
  for (i = 1; i <= Nr; i++) {
    r = rmin + (i-1)*dr;
    n = n + 1;
    tmp = tmp + r*h[n]*h_det[n];
  }
}
pre_cor_f = 1.0/(C_av*tmp*tmp);

for (l = 1; l <= lm; l = l + 1) {
  printf ("Step %d of %d.\n", l, lm);
  tau = tau0*exp(dexptau*(l-1)/(lm-1));
  ov4Dtau = 1/(4.0*D*tau);

// Analytical expression for the correlation function.
  cor_f_a = pre_cor_f_a/((1.0 + 2.0*D*tau/(0.15*0.15))*sqrt(1.0 + 2.0*D*tau/(0.1*0.1)));

  n = 0;
  for (i = 1; i <= Nr; i++) {
    r = rmin + (i-1)*dr;
    for (ip = 1; ip <= Nr; ip++) {
      rp = rmin + (ip-1)*dr;
      n = n + 1;
      fs[n] = 0.0;
      for (k = 1; k <= Nphi; k++) {
        phi = (k-1)*dphi;
        for (kp = 1; kp <= Nphi; kp++) {
          phip = (kp-1)*dphi;
          fs[n] = fs[n] + exp(-(r*r + rp*rp - 2.0*r*rp*cos(phi-phip))*ov4Dtau);
        }
      }
      fs[n] = dphi*dphi*r*rp*fs[n]/(4.0*PI*PI);
    }
  }

  cor_f = 0.0;
  n = 0;
// Integral over z and r.
  for (k = 1; k <= Nz; k++) {
    z = zmin + (k-1)*dz;
    ns = 0;
    for (i = 1; i <= Nr; i++) {
      r = rmin + (i-1)*dr;
      n = n + 1;
// Integral over z' and r'.
      tmpr = 0.0;
      np = 0;
      for (kp = 1; kp <= Nz; kp++) {
        zp = zmin + (kp-1)*dz;
	tmp = 0.0;
        for (ip = 1; ip <= Nr; ip++) {
	  rp = rmin + (ip-1)*dr;
	  np = np + 1;
	  ns = ns + 1;

	  tmp = tmp + h[np]*h_det[np]*fs[ns];
        }
	ns = ns - Nr;
	tmpr1 = z - zp;
	tmpr = tmpr + exp(-tmpr1*tmpr1*ov4Dtau)*tmp;
      }
      ns = ns + Nr;

      cor_f += h[n]*h_det[n]*tmpr;
    }
  }
  tmp = sqrt(ov4Dtau/PI);
  tmp = tmp*tmp*tmp;
  cor_f = pre_cor_f*cor_f*tmp;

  fprintf (outObs, "%.10f  %.10f %.10f\n", tau, cor_f, cor_f_a);
}
fclose (outObs);
delete[] fs;
//////////////////////////////////////////////////
*/



// Cleaning the memory.
//////////////////////////////////////////////////
   delete[]h;
   delete[]hI0kh;
   delete[]h_det;
//   delete[]p;
//   delete[]pnew;
   delete[]Obs;
   delete[]Obs1;
   if ((MS_type != 0) && (MS_type != 1))
   {
      free (kvector_x);
      free (kvector_y);
      free (kvector_z);
      free (e_x_vector_x);
      free (e_x_vector_y);
      free (e_x_vector_z);
   }
//////////////////////////////////////////////////

   return 0; 
}
