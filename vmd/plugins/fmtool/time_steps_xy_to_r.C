#include <stdio.h>
#include <stdlib.h>
#include "time_steps_xy_to_r.h"
//#include "time_steps_xy_to_r_cuda.cu"

void printArray(const char *str, const double *d, const int x, const int y)
{
   fprintf(stderr, "%s:", str);
   for (int j=0; j<y;j++) {
      for (int i=0;i<x;i++) {
         fprintf(stderr, "%2.4f ", *(d+j*x+i));
      }
      fprintf(stderr, "X");
   }
   fprintf(stderr, "\n");
} // end printArray

void printArrayF(const char *str, const float *d, const int x, const int y)
{
   fprintf(stderr, "%s:", str);
   for (int j=0; j<y;j++) {
      for (int i=0;i<x;i++) {
         fprintf(stderr, "%2.4f ", *(d+j*x+i));
      }
      fprintf(stderr, "X");
   }
   fprintf(stderr, "\n");
} // end printArray


// --------------------------------------------------------------------  \\
//                                                                       \\
//                Single Precision Version                               \\
//                                                                       \\
// Obs and Obs1 - arrays of observables, the fluorescent signal (Obs),
// and its RMSD (Obs1). These 2 are the 1Darrays representing functions of time.
// Obs and Obs1 are being calculated within this function.
// M - number of time steps (constant).
// kiOutputFreq - frequency that the observables should be calculated
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
void time_steps_xy_to_r_float(float *Obs, float *Obs1, const int M, 
                        const int kiOutputFreq, const float dt, 
                        const int N, const float *h, const float *h_det, 
                        const float *hI0kh, const float dr, const float dz, 
                        const int Nr, const int Nz, const float rmin, 
                        const float p0, const float D)
{
// Counters for loops.
   int l = 0;
   int i = 0;
   int k = 0;
   int n = 0;
   int nold = 0;  // used to track location in h/h_det/hI0kh array 
                  // that we need to grab

// Variable for the values of r.
   float r = 0.0;

// Auxiliary constants.
   float odr_o2 = 0.5 / dr;
   float odr2 = 1.0 / (dr * dr);
   float odz2 = 1.0 / (dz * dz);
   float t2odrz = 2.0 * (odr2 + odz2);

// Auxiliary variables.
   float tmp = 0.0;
   float tmpr = 0.0;
   float tmpr1 = 0.0;
   float tmpz = 0.0;

   const int iNrp2 = Nr+2;   // saves several calculations
   const int iNzp2 = Nz+2;

   float *p = (float *) malloc( (iNrp2)*(iNzp2) * sizeof(float));
   float *pnew = (float *) malloc( (iNrp2)*(iNzp2) * sizeof(float));

//   fprintf(stderr, "dr:%f, dz:%f, odr_o2:%f, odr2:%f, odz2:%f, t2odrz:%f\n",dr, dz, odr_o2, odr2, odz2, t2odrz);
//   printArrayF("hI0kh is ", hI0kh+1, Nr, Nz);
   // set to initial condition
   for (i=0; i<((iNrp2)*(iNzp2));i++) {
      pnew[i]=p0;
   }

   float *pTemp;

   float *rgTmp1 = (float *) malloc( (N+1)*sizeof(float));
   for (i=1; i <= N; i++)
   {
      rgTmp1[i] = h[i] * h_det[i];
   }

// M time steps
   for (l = 0; l < M; l++)
   {

// Update the arrays at the new time step.
// Array p becomes the same as pnew has become at the previous time step;
// p will be used as an input at this time step, to calculate pnew.
      pTemp = p;
      p = pnew;
      pnew = pTemp;

      // prepare p array to be used by padding with the proper values
      // top row
      memcpy(p+1,    p+1+(iNrp2)*2, sizeof(float)*Nr);

      // bottom row
      memcpy(p+1+(iNrp2)*(Nz+1), p+1+(iNrp2)*(Nz-1), sizeof(float)*Nr);
//      printArray("After top/bottom",p, iNrp2, iNzp2);

      // sides
      int iTmp = (iNrp2)*(Nz+1);
      for (i=iNrp2; i < iTmp ; i+=(iNrp2) )
      {
         p[i] = p[i+2];
         p[i+Nr+1] = p[i+Nr-1];
      }

//      fprintf(stderr, "after doing sides\n");
//     printArray("After BC", p, iNrp2, iNzp2);

      // Diffusion and bleaching.
      //////////////////////////////////////////////////

//      fprintf(stderr, "Before nested loop\n");

// calc_grid_cudacheck(dt, Nr, Nz, rmin,h, h_det, hI0kh, dr, dz, p, pnew, D);
// calc_grid_cudasim(dt, Nr, Nz, rmin,h, h_det, hI0kh, dr, dz, p, pnew, D);


      nold = 0;
      for (k = 1 ; k <= Nz; k++)
      {
         n = k*(iNrp2);
         r = rmin ;

         // pulled out the i=0 iteration and am explicitly setting
         // tmpr1 to zero here.  This fixes the bug we were seeing
         n++;

         tmpr = p[n + 1] + p[n - 1];
         tmpr1 = 0;
         tmpz = p[n + (iNrp2)] + p[n - (iNrp2)];

             // Get the function p for the new step.
         tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
         float result = p[n] + dt * (D * tmp - hI0kh[++nold] * p[n]);
         pnew[n] = (result < 0) ? 0 : result;

         r += dr;
         // end of the copied iteration loop

         for (i = 1; i < Nr; i++)
         {
            n++;

            tmpr = p[n + 1] + p[n - 1];
            tmpr1 = (p[n + 1] - p[n - 1]) / r;
            tmpz = p[n + (iNrp2)] + p[n - (iNrp2)];

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            float result = p[n] + dt * (D * tmp - hI0kh[++nold] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;
//            fprintf(stderr,"k:%d,i:%d,n:%d,nold:%d,tmpr:%f,tmpr1:%f,tmpz:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,nold,tmpr,tmpr1,tmpz,tmp,result,pnew[n]);

            r += dr;
         }
      }

//      printArrayF("TIMESTEP:P", p, iNrp2, iNzp2);
//      printArrayF("timestep:pnew", pnew, iNrp2, iNzp2);

      // initial values for observables.  Accumulated over grid
      if (!(l%kiOutputFreq))
      {
         int iLDivFreq = l/kiOutputFreq;
         float obsl  = 0.0f;
         float obsl1 = 0.0f;

         nold = 0;
         for (k = 1 ; k <= Nz; k++)
         {
            n = k*(iNrp2);
            r = rmin ;
            for (i = 0; i < Nr; i++)
            {
               // Accumulate observables
               // Observable; Obs(t) = \int dr h(r) h_det(r) p(r,t).
               // Noise; Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t) - 
               //                         (\int dr h(r) h_det(r) p(r,t))^2.
         //      float tmp1 = h[nold] * h_det[nold];
               float tmp2 = r * rgTmp1[++nold] * p[++n];
               obsl  += tmp2;
               obsl1 += tmp2 * rgTmp1[nold];

//            fprintf(stderr,"k:%d,i:%d,n:%d,nold:%d,r:%f,rgTmp:%f,tmp2:%f,obsl:%f,obsl1:%f,p[n]:%f\n",k,i,n,nold,r,rgTmp1[nold],tmp2,obsl,obsl1,p[n]);

               r += dr;
            }
         }

         /// store observables 
         Obs[iLDivFreq] = obsl;
         Obs1[iLDivFreq] = obsl1;
         Obs[iLDivFreq]  *= 2.0 * PI * dr * dz;
         Obs1[iLDivFreq] = Obs1[iLDivFreq] * 2.0 * PI * dr * dz - Obs[iLDivFreq]
                                                            * Obs[iLDivFreq];
//      printArray("timestep", pnew, iNrp2, iNzp2);
        printf("Time step %d  of %d.  Obs:%f,Obs1:%f\n", l,
                         M,Obs[l/kiOutputFreq],Obs1[l/kiOutputFreq]);
      }
   }
   free (rgTmp1);
   free (p);
   free (pnew);
}

// --------------------------------------------------------------------
// Obs and Obs1 - arrays of observables, the fluorescent signal (Obs),
// and its RMSD (Obs1). These 2 are the 1Darrays representing functions of time.
// Obs and Obs1 are being calculated within this function.
// M - number of time steps (constant).
// kiOutputFreq - frequency that the observables should be calculated
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

void time_steps_xy_to_r_double(double *Obs, double *Obs1, const int M, 
                        const int kiOutputFreq, const double dt, 
                        const int N, const double *h, const double *h_det, 
                        const double *hI0kh, const double dr, const double dz, 
                        const int Nr, const int Nz, const double rmin, 
                        /*double *p, double *pnew, */ const double p0, 
                        const double D)
{
// Counters for loops.
   int l = 0;
   int i = 0;
   int k = 0;
   int n = 0;
   int nold = 0;  // used to track location in h/h_det/hI0kh array 
                  // that we need to grab

// Variable for the values of r.
   double r = 0.0;

// Auxiliary constants.
   double odr_o2 = 0.5 / dr;
   double odr2 = 1.0 / (dr * dr);
   double odz2 = 1.0 / (dz * dz);
   double t2odrz = 2.0 * (odr2 + odz2);

// Auxiliary variables.
   double tmp = 0.0;
   double tmpr = 0.0;
   double tmpr1 = 0.0;
   double tmpz = 0.0;

   const int iNrp2 = Nr+2;   // saves several calculations
   const int iNzp2 = Nz+2;

   double *p = (double *) malloc( (iNrp2)*(iNzp2) * sizeof(double));
   double *pnew = (double *) malloc( (iNrp2)*(iNzp2) * sizeof(double));

   // set to initial condition
   for (i=0; i<((iNrp2)*(iNzp2));i++) {
      pnew[i]=p0;
   }

   double *pTemp;

   double *rgTmp1 = (double *) malloc( (N+1)*sizeof(double));
   for (i=1; i <= N; i++)
   {
      rgTmp1[i] = h[i] * h_det[i];
   }

// M time steps
   for (l = 0; l < M; l++)
   {

// Update the arrays at the new time step.
// Array p becomes the same as pnew has become at the previous time step;
// p will be used as an input at this time step, to calculate pnew.
      pTemp = p;
      p = pnew;
      pnew = pTemp;

      // prepare p array to be used by padding with the proper values
      // top row
      memcpy(p+1,    p+1+(iNrp2)*2, sizeof(double)*Nr);

      // bottom row
      memcpy(p+1+(iNrp2)*(Nz+1), p+1+(iNrp2)*(Nz-1), sizeof(double)*Nr);
//      printArray("After top/bottom",p, iNrp2, iNzp2);

      // sides
      int iTmp = (iNrp2)*(Nz+1);
      for (i=iNrp2; i < iTmp ; i+=(iNrp2) )
      {
         p[i] = p[i+2];
         p[i+Nr+1] = p[i+Nr-1];
      }
//      fprintf(stderr, "after doing sides\n");
//     printArray("After BC", p, iNrp2, iNzp2);

      // Diffusion and bleaching.
      //////////////////////////////////////////////////

//      fprintf(stderr, "Before nested loop\n");

// calc_grid_cudacheck(dt, Nr, Nz, rmin,h, h_det, hI0kh, dr, dz, p, pnew, D);
// calc_grid_cudasim(dt, Nr, Nz, rmin,h, h_det, hI0kh, dr, dz, p, pnew, D);

      nold = 0;
      for (k = 1 ; k <= Nz; k++)
      {
         n = k*(iNrp2);
         r = rmin ;

         // pulled out the i=0 iteration and am explicitly setting
         // tmpr1 to zero here.  This fixes the bug we were seeing
         n++;

         tmpr = p[n + 1] + p[n - 1];
         tmpr1 = 0;
         tmpz = p[n + (iNrp2)] + p[n - (iNrp2)];

             // Get the function p for the new step.
         tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
         double result = p[n] + dt * (D * tmp - hI0kh[++nold] * p[n]);
         pnew[n] = (result < 0) ? 0 : result;

         r += dr;
         // end of the copied iteration loop

         for (i = 1; i < Nr; i++)
         {
            n++;

            tmpr = p[n + 1] + p[n - 1];
            tmpr1 = (p[n + 1] - p[n - 1]) / r;
            tmpz = p[n + (iNrp2)] + p[n - (iNrp2)];

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            double result = p[n] + dt * (D * tmp - hI0kh[++nold] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;
//            fprintf(stderr,"k:%d,i:%d,n:%d,nold:%d,tmpr:%f,tmpr1:%f,tmpz:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,nold,tmpr,tmpr1,tmpz,tmp,result,pnew[n]);

            r += dr;
         }
      }

      // initial values for observables.  Accumulated over grid
      if (!(l%kiOutputFreq))
      {
         int iLDivFreq = l/kiOutputFreq;

         // initial values for observables.  Accumulated over grid
         double obsl  = 0.0f;
         double obsl1 = 0.0f;

         nold = 0;
         for (k = 1 ; k <= Nz; k++)
         {
            n = k*(iNrp2);
            r = rmin /*+ dr */;
            for (i = 0; i < Nr; i++)
            {
               // Accumulate observables
               // Observable; Obs(t) = \int dr h(r) h_det(r) p(r,t).
               // Noise; Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t) - 
               //                         (\int dr h(r) h_det(r) p(r,t))^2.
         //      double tmp1 = h[nold] * h_det[nold];
               double tmp2 = r * rgTmp1[++nold] * p[++n];
               obsl  += tmp2;
               obsl1 += tmp2 * rgTmp1[nold];

//            fprintf(stderr,"k:%d,i:%d,n:%d,nold:%d,r:%f,rgTmp:%f,tmp2:%f,obsl:%f,obsl1:%f,p[n]:%f\n",k,i,n,nold,r,rgTmp1[nold],tmp2,obsl,obsl1,p[n]);

               r += dr;
            }
         }

         /// store observables 
         Obs[iLDivFreq] = obsl;
         Obs1[iLDivFreq] = obsl1;
         Obs[iLDivFreq]  *= 2.0 * PI * dr * dz;
         Obs1[iLDivFreq] = Obs1[iLDivFreq] * 2.0 * PI * dr * dz - Obs[iLDivFreq]
                                                                * Obs[iLDivFreq];

         printf("Time step %d  of %d.  Obs:%f,Obs1:%f\n", l,
                         M,Obs[l/kiOutputFreq],Obs1[l/kiOutputFreq]);
      }
   }
   free (rgTmp1);
   free (p);
   free (pnew);

}

// --------------------------------------------------------------------













































// --------------------------------------------------------------------
// Obs and Obs1 - arrays of observables, the fluorescent signal (Obs),
// and its RMSD (Obs1). These two are the 1D arrays representing functions of time.
// Obs and Obs1 are being calculated within this function.
// M - number of time steps (constant).
// kiOutputFreq - frequency that the observables should be calculated
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

void time_steps_xy_to_r_double_bc(double *Obs, double *Obs1, const int M, 
                        const int kiOutputFreq, const double dt, 
                        const int N, const double *h, const double *h_det, 
                        const double *hI0kh, const double dr, const double dz, 
                        const int Nr, const int Nz, const double rmin, 
                        /*double *p, double *pnew, */ const double p0, 
                        const double D)
{

// Counters for loops.
   int l = 0;
   int i = 0;
   int k = 0;
   int n = 0;

// Variable for the values of r.
   double r = 0.0;

// Auxiliary constants.
   double odr_o2 = 0.5 / dr;
   double odr2 = 1.0 / (dr * dr);
   double odz2 = 1.0 / (dz * dz);
   double t2odrz = 2.0 * (odr2 + odz2);

// Auxiliary variables.
   double tmp = 0.0;
   double tmpr = 0.0;
   double tmpr1 = 0.0;
   double tmpz = 0.0;

   double *p = (double *) malloc( (N+1) * sizeof(double));
   double *pnew = (double *) malloc( (N+1) * sizeof(double));

   // set to initial condition
   for (int i=0; i<N+1;i++) {
      pnew[i]=p0;
      p[i]=p0;
//      pnew[i]=i;
//      p[i]=-i;
   }

   double *pTemp;

// M time steps
   for (l = 0; l < M; l++)
   {

// Update the arrays at the new time step.
// Array p becomes the same as pnew has become at the previous time step;
// p will be used as an input at this time step, to calculate pnew.

      pTemp = p;
      p = pnew;
      pnew = pTemp;
      fprintf(stderr,"\ntimestep %d", l); printArray("", p, N+1, 1);

      // initial values for observables.  Accumulated over grid
      double obsl  = 0.0;
      double obsl1 = 0.0;

      // Diffusion and bleaching.
      //////////////////////////////////////////////////


      fprintf(stderr, "odr2:%f, odz2:%f, odr_o2:%f, rmin:%f, dr:%f\n", odr2, odz2, odr_o2, rmin, dr);
      //
      // interior loops
      //
      for (k = 2; k < Nz; k++) {            // Loop over the z-dimension.
         n = (k-1)*Nr + 1;                  // offset for element 2 in X...
         r = rmin + dr;                     // offset for element 2 in X...
         for (i = 2; i < Nr; i++) {         // Loop over the x-dimension.
            n++;  // Counter for the 1D arrays that are function of position, such as p[n].
            tmpr = p[n + 1] + p[n - 1];
            tmpr1 = (p[n + 1] - p[n - 1]) / r;
            tmpz = p[n + Nr] + p[n - Nr];

            if (!(l%kiOutputFreq))
            {
               // Accumulate observables
               // Observable; Obs(t) = \int dr h(r) h_det(r) p(r,t).
               // Noise; Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t) - (\int dr h(r) h_det(r) p(r,t))^2.
               double tmp1 = h[n] * h_det[n];
               double tmp2 = r * tmp1 * p[n];
               obsl  += tmp2;
               obsl1 += tmp2 * tmp1;
            }

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            double result = p[n] + dt * (D * tmp - hI0kh[n] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;

            fprintf(stderr,"k:%d,i:%d,n:%d,tmpr:%f,tmpr1:%f,tmpz:%f,obsl:%f,obsl1:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,tmpr,tmpr1,tmpz,obsl,obsl1,tmp,result,pnew[n]);
            r += dr;
         }
      }

      //
      // handle boundary conditions
      //

      // Z boundary
      for (k = 1; k <= Nz; k+=(Nz-1)) {           // Loop over the z-dimension.
         n = (k-1)*Nr;
         r = rmin;
         for (i = 1; i <= Nr; i++) {         // Loop over the x-dimension.
            n++;  // Counter for the 1D arrays that are function of position, such as p[n].
            // BC (reflective walls) for r.
            if (i == 1) {
               tmpr = 2.0 * p[n + 1];
               tmpr1 = 0.0;
            } else {
               if (i == Nr) {
                  tmpr = 2.0 * p[n - 1];
                  tmpr1 = 0.0;
               } else {
                  tmpr = p[n + 1] + p[n - 1];
                  tmpr1 = (p[n + 1] - p[n - 1]) / r;
               }
            }

            // BC (reflective walls) for z.
            if (k == 1) {
               tmpz = 2.0 * p[n + Nr];
            } else {
               if (k == Nz) {
                  tmpz = 2.0 * p[n - Nr];
               } else {
                  tmpz = p[n + Nr] + p[n - Nr];
               }
            }

            if (!(l%kiOutputFreq))
            {
               // Accumulate observables
               // Observable; Obs(t) = \int dr h(r) h_det(r) p(r,t).
               // Noise; Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t) - (\int dr h(r) h_det(r) p(r,t))^2.
               double tmp1 = h[n] * h_det[n];
               double tmp2 = r * tmp1 * p[n];
               obsl  += tmp2;
               obsl1 += tmp2 * tmp1;
            }

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            double result = p[n] + dt * (D * tmp - hI0kh[n] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;

            fprintf(stderr,"k:%d,i:%d,n:%d,tmpr:%f,tmpr1:%f,tmpz:%f,obsl:%f,obsl1:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,tmpr,tmpr1,tmpz,obsl,obsl1,tmp,result,pnew[n]);
            r += dr;
         }
      }


      // X boundary
      for (k = 2; k <= (Nz-1); k++) {        // Loop over the z-dimension.
         for (i = 1; i <= Nr; i+=(Nr-1)) {   // Loop over the x-dimension.
            n = (k-1)*Nr + i; // Counter for the 1D arrays that are function of position, such as p[n].
            r = rmin + (i-1)*dr;             // offset for correct element

            // BC (reflective walls) for r.
            if (i == 1) {
               tmpr = 2.0 * p[n + 1];
               tmpr1 = 0.0;
            } else {
               if (i == Nr) {
                  tmpr = 2.0 * p[n - 1];
                  tmpr1 = 0.0;
               } else {
                  tmpr = p[n + 1] + p[n - 1];
                  tmpr1 = (p[n + 1] - p[n - 1]) / r;
               }
            }

            // BC (reflective walls) for z.
            if (k == 1) {
               tmpz = 2.0 * p[n + Nr];
            } else {
               if (k == Nz) {
                  tmpz = 2.0 * p[n - Nr];
               } else {
                  tmpz = p[n + Nr] + p[n - Nr];
               }
            }

            if (!(l%kiOutputFreq))
            {
               // Accumulate observables
               // Observable; Obs(t) = \int dr h(r) h_det(r) p(r,t).
               // Noise; Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t) - (\int dr h(r) h_det(r) p(r,t))^2.
               double tmp1 = h[n] * h_det[n];
               double tmp2 = r * tmp1 * p[n];
               obsl  += tmp2;
               obsl1 += tmp2 * tmp1;
            }

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            double result = p[n] + dt * (D * tmp - hI0kh[n] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;
            fprintf(stderr,"k:%d,i:%d,n:%d,tmpr:%f,tmpr1:%f,tmpz:%f,obsl:%f,obsl1:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,tmpr,tmpr1,tmpz,obsl,obsl1,tmp,result,pnew[n]);
         }
      }

      if (!(l%kiOutputFreq))
      {
         int iLDivFreq = l/kiOutputFreq;
         /// store observables 
         Obs[iLDivFreq] = obsl;
         Obs1[iLDivFreq] = obsl1;
         Obs[iLDivFreq]  *= 2.0 * PI * dr * dz;
         Obs1[iLDivFreq] = Obs1[iLDivFreq] * 2.0 * PI * dr * dz - Obs[iLDivFreq]
                                                                * Obs[iLDivFreq];

        printf("Time step %d  of %d\n", l, M);
        printArray("timestep", p, N+1, 1);
      }
   }
   free(p);
   free(pnew);
}

// --------------------------------------------------------------------
// Obs and Obs1 - arrays of observables, the fluorescent signal (Obs),
// and its RMSD (Obs1). These 2 are the 1Darrays representing functions of time.
// Obs and Obs1 are being calculated within this function.
// M - number of time steps (constant).
// kiOutputFreq - frequency that the observables should be calculated
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
                        const int kiOutputFreq, const double dt, 
                        const int N, const double *h, const double *h_det, 
                        const double *hI0kh, const double dr, const double dz, 
                        const int Nr, const int Nz, const double rmin, 
                        /*double *p, double *pnew, */ const double p0, 
                        const double D)
{
// Counters for loops.
   int l = 0;
   int i = 0;
   int k = 0;
   int n = 0;
   int nold = 0;

// Variable for the values of r.
   double r = 0.0;

// Auxiliary constants.
   double odr_o2 = 0.5 / dr;
   double odr2 = 1.0 / (dr * dr);
   double odz2 = 1.0 / (dz * dz);
   double t2odrz = 2.0 * (odr2 + odz2);

// Auxiliary variables.
   double tmp = 0.0;
   double tmpr = 0.0;
   double tmpr1 = 0.0;
   double tmpz = 0.0;

   int iNrp2 = Nr+2;
   int iNzp2 = Nz+2;

   fprintf(stderr, "Nr=%d Nz=%d, Allocating %d\n", Nr, Nz, (iNrp2)*(iNzp2));
   double *p = (double *) malloc( (iNrp2)*(iNzp2) * sizeof(double));
   double *pnew = (double *) malloc( (iNrp2)*(iNzp2) * sizeof(double));

   // set to initial condition
   for (i=0; i<((iNrp2)*(iNzp2));i++) {
      pnew[i]=p0;
      p[i]=p0;
//      pnew[i]=i;
//      p[i]=-i;
   }

   double *pTemp;

// M time steps
   for (l = 0; l < M; l++)
   {

// Update the arrays at the new time step.
// Array p becomes the same as pnew has become at the previous time step;
// p will be used as an input at this time step, to calculate pnew.

      pTemp = p;
      p = pnew;
      pnew = pTemp;
      fprintf(stderr,"\ntimestep %d", l); printArray("", p, iNrp2, iNzp2);

//      fprintf(stderr, "before memcpy's (Nr %d Nz %d size_t %d) %d %d %d\n",Nr, Nz, sizeof(size_t), 1, (1+(iNrp2)*2), Nr);
      // prepare p array to be used by padding with the proper values
      // top row
//      memcpy(p+sizeof(double),    p+sizeof(double)*(1+(iNrp2)*2), 
//             sizeof(double)*Nr);
  

//      fprintf(stderr, "after 1st memcpy %d %d %d\n", ((iNrp2)*(Nz+1)+1), ((iNrp2)*(Nz-1)+1), Nr);

      for (i=0;i<Nr;i++)
      {
         p[i+1] = p[(i+1+(iNrp2)*2)];
         p[(iNrp2)*(Nz+1)+1+i] = p[(iNrp2)*(Nz-1)+1+i];
      }
//      fprintf(stderr, "after doing top/bottom\n");
//      printArray("After top/bottom",p, iNrp2, iNzp2);
//      // bottom row     this is segfaulting for some reason
//      memcpy(p+sizeof(double)*((iNrp2)*(Nz+1)+1),    
//             p+sizeof(double)*((iNrp2)*(Nz-1)+1),
//             sizeof(double)*Nr);
//      fprintf(stderr, "after memcpy's\n");

      // sides
      for (i=iNrp2; i < (iNrp2)*(Nz+1) ; i+=(iNrp2) )
      {
         p[i] = p[i+2];
         p[i+Nr+1] = p[i+Nr-1];
      }
//      fprintf(stderr, "after doing sides\n");
     printArray("After BC", p, iNrp2, iNzp2);

      // initial values for observables.  Accumulated over grid
      double obsl  = 0.0;
      double obsl1 = 0.0;

      // Diffusion and bleaching.
      //////////////////////////////////////////////////

//      fprintf(stderr, "Before nested loop\n");

//      fprintf(stderr, "odr2:%f, odz2:%f, odr_o2:%f, rmin:%f, dr:%f\n", odr2, odz2, odr_o2, rmin, dr);
      for (k = 1 ; k <= Nz; k++)
      {
         n = k*(iNrp2);
         nold = Nr*(k-1);
         r = rmin + dr;
         for (i = 0; i < Nr; i++)
         {
            n++;
            nold++;

            tmpr = p[n + 1] + p[n - 1];
            tmpr1 = (p[n + 1] - p[n - 1]) / r;
            tmpz = p[n + (iNrp2)] + p[n - (iNrp2)];

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            double result = p[n] + dt * (D * tmp - hI0kh[nold] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;
//            fprintf(stderr,"k:%d,i:%d,n:%d,nold:%d,tmpr:%f,tmpr1:%f,tmpz:%f,obsl:%f,obsl1:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,nold,tmpr,tmpr1,tmpz,obsl,obsl1,tmp,result,pnew[n]);

            r += dr;
         }

      }
//      fprintf(stderr, "After nested loop\n");

      // initial values for observables.  Accumulated over grid
      if (!(l%kiOutputFreq))
      {
         int iLDivFreq = l/kiOutputFreq;
         for (k = 1 ; k <= Nz; k++)
         {
            n = k*(iNrp2);
            nold = Nr*(k-1);
            r = rmin + dr;
            for (i = 0; i < Nr; i++)
            {
               n++;
               nold++;

               // Accumulate observables
               // Observable; Obs(t) = \int dr h(r) h_det(r) p(r,t).
               // Noise; Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t) - (\int dr h(r) h_det(r) p(r,t))^2.
               double tmp1 = h[nold] * h_det[nold];
               double tmp2 = r * tmp1 * p[n];
               obsl  += tmp2;
               obsl1 += tmp2 * tmp1;

               r += dr;
            }

         }

         /// store observables 
         Obs[iLDivFreq] = obsl;
         Obs1[iLDivFreq] = obsl1;
         Obs[iLDivFreq]  *= 2.0 * PI * dr * dz;
         Obs1[iLDivFreq] = Obs1[iLDivFreq] * 2.0 * PI * dr * dz - Obs[iLDivFreq]
                                                                * Obs[iLDivFreq];

        printf("Time step %d  of %d\n", l, M);
        printArray("timestep", pnew, iNrp2, iNzp2);
      }
   }
   free(p);
   free(pnew);
}

// --------------------------------------------------------------------
// Obs and Obs1 - arrays of observables, the fluorescent signal (Obs),
// and its RMSD (Obs1). These two are the 1D arrays representing functions of time.
// Obs and Obs1 are being calculated within this function.
// M - number of time steps (constant).
// kiOutputFreq - frequency that the observables should be calculated
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

void time_steps_xy_to_r_double_bc_obs(double *Obs, double *Obs1, const int M, 
                        const int kiOutputFreq, const double dt, 
                        const int N, const double *h, const double *h_det, 
                        const double *hI0kh, const double dr, const double dz, 
                        const int Nr, const int Nz, const double rmin, 
                        /*double *p, double *pnew, */ const double p0, 
                        const double D)
{

// Counters for loops.
   int l = 0;
   int i = 0;
   int k = 0;
   int n = 0;

// Variable for the values of r.
   double r = 0.0;

// Auxiliary constants.
   double odr_o2 = 0.5 / dr;
   double odr2 = 1.0 / (dr * dr);
   double odz2 = 1.0 / (dz * dz);
   double t2odrz = 2.0 * (odr2 + odz2);

// Auxiliary variables.
   double tmp = 0.0;
   double tmpr = 0.0;
   double tmpr1 = 0.0;
   double tmpz = 0.0;

   double *p = (double *) malloc( (N+1) * sizeof(double));
   double *pnew = (double *) malloc( (N+1) * sizeof(double));

   // set to initial condition
   for (int i=0; i<N+1;i++) {
      pnew[i]=p0;
      p[i]=p0;
//      pnew[i]=i;
//      p[i]=-i;
   }

   double *pTemp;

// M time steps
   for (l = 0; l < M; l++)
   {

// Update the arrays at the new time step.
// Array p becomes the same as pnew has become at the previous time step;
// p will be used as an input at this time step, to calculate pnew.

      pTemp = p;
      p = pnew;
      pnew = pTemp;
      fprintf(stderr,"\ntimestep %d", l); printArray("", p, N+1, 1);

      // initial values for observables.  Accumulated over grid
      double obsl  = 0.0;
      double obsl1 = 0.0;

      // Diffusion and bleaching.
      //////////////////////////////////////////////////


      fprintf(stderr, "odr2:%f, odz2:%f, odr_o2:%f, rmin:%f, dr:%f\n", odr2, odz2, odr_o2, rmin, dr);
      //
      // interior loops
      //
      for (k = 2; k < Nz; k++) {            // Loop over the z-dimension.
         n = (k-1)*Nr + 1;                  // offset for element 2 in X...
         r = rmin + dr;                     // offset for element 2 in X...
         for (i = 2; i < Nr; i++) {         // Loop over the x-dimension.
            n++;  // Counter for the 1D arrays that are function of position, such as p[n].
            tmpr = p[n + 1] + p[n - 1];
            tmpr1 = (p[n + 1] - p[n - 1]) / r;
            tmpz = p[n + Nr] + p[n - Nr];

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            double result = p[n] + dt * (D * tmp - hI0kh[n] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;

            fprintf(stderr,"k:%d,i:%d,n:%d,tmpr:%f,tmpr1:%f,tmpz:%f,obsl:%f,obsl1:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,tmpr,tmpr1,tmpz,obsl,obsl1,tmp,result,pnew[n]);
            r += dr;
         }
      }

      //
      // handle boundary conditions
      //

      // Z boundary
      for (k = 1; k <= Nz; k+=(Nz-1)) {           // Loop over the z-dimension.
         n = (k-1)*Nr;
         r = rmin;
         for (i = 1; i <= Nr; i++) {         // Loop over the x-dimension.
            n++;  // Counter for the 1D arrays that are function of position, such as p[n].
            // BC (reflective walls) for r.
            if (i == 1) {
               tmpr = 2.0 * p[n + 1];
               tmpr1 = 0.0;
            } else {
               if (i == Nr) {
                  tmpr = 2.0 * p[n - 1];
                  tmpr1 = 0.0;
               } else {
                  tmpr = p[n + 1] + p[n - 1];
                  tmpr1 = (p[n + 1] - p[n - 1]) / r;
               }
            }

            // BC (reflective walls) for z.
            if (k == 1) {
               tmpz = 2.0 * p[n + Nr];
            } else {
               if (k == Nz) {
                  tmpz = 2.0 * p[n - Nr];
               } else {
                  tmpz = p[n + Nr] + p[n - Nr];
               }
            }

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            double result = p[n] + dt * (D * tmp - hI0kh[n] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;

            fprintf(stderr,"k:%d,i:%d,n:%d,tmpr:%f,tmpr1:%f,tmpz:%f,obsl:%f,obsl1:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,tmpr,tmpr1,tmpz,obsl,obsl1,tmp,result,pnew[n]);
            r += dr;
         }
      }


      // X boundary
      for (k = 2; k <= Nz-1; k++) {            // Loop over the z-dimension.
         for (i = 1; i <= Nr; i+=(Nr-1)) {   // Loop over the x-dimension.
            n = (k-1)*Nr + i; // Counter for the 1D arrays that are function of position, such as p[n].
            r = rmin + (i-1)*dr;             // offset for correct element

            // BC (reflective walls) for r.
            if (i == 1) {
               tmpr = 2.0 * p[n + 1];
               tmpr1 = 0.0;
            } else {
               if (i == Nr) {
                  tmpr = 2.0 * p[n - 1];
                  tmpr1 = 0.0;
               } else {
                  tmpr = p[n + 1] + p[n - 1];
                  tmpr1 = (p[n + 1] - p[n - 1]) / r;
               }
            }

            // BC (reflective walls) for z.
            if (k == 1) {
               tmpz = 2.0 * p[n + Nr];
            } else {
               if (k == Nz) {
                  tmpz = 2.0 * p[n - Nr];
               } else {
                  tmpz = p[n + Nr] + p[n - Nr];
               }
            }

            // Get the function p for the new step.
            tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * p[n] + odr_o2 * tmpr1;
            double result = p[n] + dt * (D * tmp - hI0kh[n] * p[n]);
            pnew[n] = (result < 0) ? 0 : result;
            fprintf(stderr,"k:%d,i:%d,n:%d,tmpr:%f,tmpr1:%f,tmpz:%f,obsl:%f,obsl1:%f,tmp:%f,result:%f,pnew:%f\n",k,i,n,tmpr,tmpr1,tmpz,obsl,obsl1,tmp,result,pnew[n]);
         }
      }

      // initial values for observables.  Accumulated over grid
      if (!(l%kiOutputFreq))
      {
         int iLDivFreq = l/kiOutputFreq;

         obsl=0; 
         obsl1=0;
         for (k=1;k<=Nz;k++)
         {
            r = rmin + dr;                     // offset for element 2 in X...
            for (i=1; i<=Nr; i++)
            {
               n++;
               
               // Accumulate observables
               // Observable; Obs(t) = \int dr h(r) h_det(r) p(r,t).
               // Noise; Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t) - (\int dr h(r) h_det(r) p(r,t))^2.
               double tmp1 = h[n] * h_det[n];
               double tmp2 = r * tmp1 * p[n];
               obsl  += tmp2;
               obsl1 += tmp2 * tmp1;
               r += dr;
            }
         }


         /// store observables 
         Obs[iLDivFreq] = obsl;
         Obs1[iLDivFreq] = obsl1;
         Obs[iLDivFreq]  *= 2.0 * PI * dr * dz;
         Obs1[iLDivFreq] = Obs1[iLDivFreq] * 2.0 * PI * dr * dz - Obs[iLDivFreq]
                                                                * Obs[iLDivFreq];
         printf("Time step %d  of %d\n", l, M);
         printArray("timestep", p, N+1, 1);
      }
   }
   free(p);
   free(pnew);

}

// --------------------------------------------------------------------


