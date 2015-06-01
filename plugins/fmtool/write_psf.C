
using namespace std;

#include "write_psf.h"

// This function writes out the files for h and h_det.
void write_psf(int &do_xy_to_r, int &N, double *h, double *h_det, double &dr,
               double &dx, double &dy, double &dz, double &rmin, double &xmin,
               double &ymin, double &zmin, int &Nr, int &Nx, int &Ny, int &Nz,
	       int &PSF_write_dx)
{

   int n = 0;
   int i = 0;
   int j = 0;
   int k = 0;
   int i_r = 0;
   double r = 0.0;
   double x = 0.0;
   double y = 0.0;
   double z = 0.0;
   double tmp = 0.0;
   double tmp1 = 0.0;

   FILE *out_h;
   double *h_r = new double[Nr + 1];
   double *h_x = new double[Nx + 1];
   double *h_y = new double[Ny + 1];
   double *h_z = new double[Nz + 1];
   double *h_hdet_r = new double[Nr + 1];
   double *h_hdet_x = new double[Nx + 1];
   double *h_hdet_y = new double[Ny + 1];
   double *h_hdet_z = new double[Nz + 1];
   for (i = 1; i <= Nr; i++)
      h_r[i] = 0.0;
   for (i = 1; i <= Nz; i++)
      h_z[i] = 0.0;
   for (i = 1; i <= Nr; i++)
      h_hdet_r[i] = 0.0;
   for (i = 1; i <= Nz; i++)
      h_hdet_z[i] = 0.0;

   double *h_hdet_z_int = new double[Nz + 1];   //Integral over r.
   for (i = 1; i <= Nz; i++)
      h_hdet_z_int[i] = 0.0;  //Integral over r.

   n = 0;

   if (do_xy_to_r == 0)
   {
// FULL 3D.
//////////////////////////////////////////////////
// Profiles over r and z.
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

               // Over x (y = z = 0).
               if ((y < dy) && (y >= 0.0) && (z < dz) && (z >= 0.0))
               {
                  h_x[i] = h[n];
                  h_hdet_x[i] = h[n] * h_det[n];
               }
	       
	       // Over y (x = z = 0).
               if ((x < dx) && (x >= 0.0) && (z < dz) && (z >= 0.0))
               {
                  h_y[j] = h[n];
                  h_hdet_y[j] = h[n] * h_det[n];
               }
	       
               // Over z (x = y = 0).
               if ((x < dx) && (x >= 0.0) && (y < dy) && (y >= 0.0))
               {
                  h_z[k] = h[n];
                  h_hdet_z[k] = h[n] * h_det[n];
               }

               // Over r (with y = 0).
               if ((z < dz) && (z >= 0.0) && (x >= 0.0) && (y < dy)
                   && (y >= 0.0))
               {
                  r = sqrt(x * x + y * y);
                  i_r = int ((r - rmin) / dr) + 1;
                  h_r[i_r] = h[n];
                  h_hdet_r[i_r] = h[n] * h_det[n];
               }
            }
         }
      }

// Maximal profile.
/*  for (k = 1; k <= Nz; k++) {
    z = zmin + (k-1)*dz;
    for (j = 1; j <= Ny; j++) {
      y = ymin + (j-1)*dy;
      for (i = 1; i <= Nx; i++) {
        x = xmin + (i-1)*dx;
        n = n + 1;
	r = sqrt(x*x + y*y);
	i_r = int((r - rmin)/dr) + 1;
        if (h_r[i_r] < h[n]) h_r[i_r] = h[n];
        if (h_z[k] < h[n]) h_z[k] = h[n];
        tmp = h[n]*h_det[n];
        if (h_hdet_r[i_r] < tmp) h_hdet_r[i_r] = tmp;
        if (h_hdet_z[k] < tmp) h_hdet_z[k] = tmp;
      }
    }
  }*/
//////////////////////////////////////////////////


   }
   else
   {
// XY_to_R.
//////////////////////////////////////////////////
// Profiles over r and z.
      double h_det_int = 0.0;
      double r_det_0 = 0.34;
      for (k = 1; k <= Nz; k++)
      {
         z = zmin + (k - 1) * dz;

         // Over z.
         h_z[k] = h[n + 1];
         h_hdet_z[k] = h[n + 1] * h_det[n + 1];

         for (i = 1; i <= Nr; i++)
         {
            r = rmin + (i - 1) * dr;
            n = n + 1;

            // Over r.
            if ((z < dz) && (z >= 0.0))
            {
               h_r[i] = h[n];
               h_hdet_r[i] = h[n] * h_det[n];
            }
            h_hdet_z_int[k] += r * h[n] * h_det[n];   //Integral over r.
            if (r <= r_det_0)
               h_det_int += r * h_det[n];
         }
         h_det_int = h_det_int * 2.0 * dr / (r_det_0 * r_det_0);
         h_hdet_z[k] = h_z[k] * h_det_int;
         h_det_int = 0.0;
         h_hdet_z_int[k] = h_hdet_z_int[k] * dr;   //Integral over r.

      }

// Maximal profile.
/*  for (k = 1; k <= Nz; k++) {
    z = zmin + (k-1)*dz;
    for (i = 1; i <= Nr; i++) {
      r = rmin + (i-1)*dr;
      n = n + 1;
      if (h_r[i] < h[n]) h_r[i] = h[n];
      if (h_z[k] < h[n]) h_z[k] = h[n];
      tmp = h[n]*h_det[n];
      if (h_hdet_r[i] < tmp) h_hdet_r[i] = tmp;
      if (h_hdet_z[k] < tmp) h_hdet_z[k] = tmp;
    }
  }*/
//////////////////////////////////////////////////
   }

   out_h = fopen("h_x.dat", "w");
   for (k = 1; k <= Nx; k++)
   {
      x = xmin + (k - 1) * dx;
      fprintf(out_h, "%f %f\n", x, h_x[k]);
   }
   fclose(out_h);
   
   out_h = fopen("h_y.dat", "w");
   for (k = 1; k <= Ny; k++)
   {
      y = ymin + (k - 1) * dy;
      fprintf(out_h, "%f %f\n", y, h_y[k]);
   }
   fclose(out_h);
   
   out_h = fopen("h_z.dat", "w");
   for (k = 1; k <= Nz; k++)
   {
      z = zmin + (k - 1) * dz;
      fprintf(out_h, "%f %f\n", z, h_z[k]);
   }
   fclose(out_h);

   out_h = fopen("h_r.dat", "w");
   for (k = 1; k <= Nr; k++)
   {
      r = rmin + (k - 1) * dr;
      fprintf(out_h, "%f %f\n", r, h_r[k]);
   }
   fclose(out_h);

   out_h = fopen("h_h_det_x.dat", "w");
   for (k = 1; k <= Nx; k++)
   {
      x = xmin + (k - 1) * dx;
      fprintf(out_h, "%f %f\n", x, h_hdet_x[k]); //Integral over r.
   }
   fclose(out_h);
   
   out_h = fopen("h_h_det_y.dat", "w");
   for (k = 1; k <= Ny; k++)
   {
      y = ymin + (k - 1) * dy;
      fprintf(out_h, "%f %f\n", y, h_hdet_y[k]); //Integral over r.
   }
   fclose(out_h);
   
   out_h = fopen("h_h_det_z.dat", "w");
   for (k = 1; k <= Nz; k++)
   {
      z = zmin + (k - 1) * dz;
      //fprintf (out_h, "%f %f\n", z, h_hdet_z[k]);
      fprintf(out_h, "%f %f %f\n", z, h_hdet_z[k], h_hdet_z_int[k]); //Integral over r.
   }
   fclose(out_h);

   out_h = fopen("h_h_det_r.dat", "w");
   for (k = 1; k <= Nr; k++)
   {
      r = rmin + (k - 1) * dr;
      fprintf(out_h, "%f %f\n", r, h_hdet_r[k]);
   }
   fclose(out_h);





// Write the PSF in dx-format in 3D.
//////////////////////////////////////////////////
if (PSF_write_dx != 0) {
   printf("Writing the 3D map of the PSF...\n");
   int numentries = 0;
   FILE* out_h_3D = fopen("h_3D.dx", "w");
   FILE* out_h_h_det_3D = fopen("h_h_det_3D.dx", "w");
   if (out_h_3D ==  NULL) {
     fprintf(stderr, "Error: Couldn't open output dxfile h_3D.dx; exiting.");
     return;
   }
   if (out_h_h_det_3D ==  NULL) {
     fprintf(stderr, "Error: Couldn't open output dxfile h_h_det_3D.dx; exiting.");
     return;
   }

// Write a dx header.
   fprintf(out_h_3D, "object 1 class gridpositions counts %li %li %li\n", Nx, Ny, Nz);
   fprintf(out_h_3D, "origin %12.6e %12.6e %12.6e\n", xmin, ymin, zmin);
   fprintf(out_h_3D, "delta %12.6e %12.6e %12.6e\n", dx, 0.0, 0.0);
   fprintf(out_h_3D, "delta %12.6e %12.6e %12.6e\n", 0.0, dy, 0.0);
   fprintf(out_h_3D, "delta %12.6e %12.6e %12.6e\n", 0.0, 0.0, dz);
   fprintf(out_h_3D, "object 2 class gridconnections counts %li %li %li\n", Nx, Ny, Nz);
   fprintf(out_h_3D, "object 3 class array type double rank 0 items %li data follows\n", Nx * Ny * Nz);

   fprintf(out_h_h_det_3D, "object 1 class gridpositions counts %li %li %li\n", Nx, Ny, Nz);
   fprintf(out_h_h_det_3D, "origin %12.6e %12.6e %12.6e\n", xmin, ymin, zmin);
   fprintf(out_h_h_det_3D, "delta %12.6e %12.6e %12.6e\n", dx, 0.0, 0.0);
   fprintf(out_h_h_det_3D, "delta %12.6e %12.6e %12.6e\n", 0.0, dy, 0.0);
   fprintf(out_h_h_det_3D, "delta %12.6e %12.6e %12.6e\n", 0.0, 0.0, dz);
   fprintf(out_h_h_det_3D, "object 2 class gridconnections counts %li %li %li\n", Nx, Ny, Nz);
   fprintf(out_h_h_det_3D, "object 3 class array type double rank 0 items %li data follows\n", Nx * Ny * Nz);

// Write the main data array.
   if (do_xy_to_r == 0) {
      numentries = 0;
      n = 0;
      for (i = 1; i <= Nx; i++) {
         for (j = 1; j <= Ny; j++) {
            for (k = 1; k <= Nz; k++) {
               n = (k-1)*Nx*Ny + (j-1)*Nx + i;
               fprintf(out_h_3D, "%-13.6e ", h[n]);
   	       fprintf(out_h_h_det_3D, "%-13.6e ", h[n]*h_det[n]);
               if (numentries % 3 == 2) {
	          fprintf(out_h_3D, "\n");
        	  fprintf(out_h_h_det_3D, "\n");
	       }
               numentries = numentries + 1;
            }
         }
      }
   } else {
      numentries = 0;
      n = 0;
      for (i = 1; i <= Nx; i++) {
         x = xmin + (i-1)*dx;
         for (j = 1; j <= Ny; j++) {
            y = ymin + (j-1)*dy;
            r = sqrt(x*x + y*y);
            i_r = int((r - rmin)/dr) + 1;
            for (k = 1; k <= Nz; k++) {
               n = (k-1)*Nr + i_r;
               tmp = h[n];
               tmp1 = h[n]*h_det[n];
               if (i_r > Nr) {
	          tmp = 0.0;
	          tmp1 = 0.0;
	       }
               fprintf(out_h_3D, "%-13.6e ", tmp);
	       fprintf(out_h_h_det_3D, "%-13.6e ", tmp1);
               if (numentries % 3 == 2) {
	          fprintf(out_h_3D, "\n");
	          fprintf(out_h_h_det_3D, "\n");
	       }
               numentries = numentries + 1;
            }
         }
      }
   }

// Write the opendx footer.
   if (n % 3 != 2) {
      fprintf(out_h_3D, "\n");
      fprintf(out_h_h_det_3D, "\n");
   }
   fprintf(out_h_3D, "attribute \"dep\" string \"positions\"\nobject \"regular positions regular connections\" class field\ncomponent \"positions\" value 1\ncomponent \"connections\" value 2\ncomponent \"data\" value 3");
   fprintf(out_h_h_det_3D, "attribute \"dep\" string \"positions\"\nobject \"regular positions regular connections\" class field\ncomponent \"positions\" value 1\ncomponent \"connections\" value 2\ncomponent \"data\" value 3");

   fclose(out_h_3D);
   fclose(out_h_h_det_3D);
   printf("done.\n");
}
//////////////////////////////////////////////////

   delete[]h_r;
   delete[]h_x;
   delete[]h_y;
   delete[]h_z;
   delete[]h_hdet_r;
   delete[]h_hdet_x;
   delete[]h_hdet_y;
   delete[]h_hdet_z;

}
