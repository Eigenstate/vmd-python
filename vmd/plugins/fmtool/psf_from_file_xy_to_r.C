#define PI 3.14159265

using namespace std;

#include "psf_from_file_xy_to_r.h"

// This function reads file with functions relevant to the PSF and builds the PSF functions for
// bleaching and observation.
void psf_from_file_xy_to_r(int &MS_type, int &mode_ill, double &alpha_ill,
                           double &n_ill, double &lambda_ill, double &phi_ill,
                           double &alpha_det, double &n_det,
                           double &lambda_det, string & PSF_ill_fname,
                           string & PSF_det_fname, int &N, double *h,
                           double *h_det, double &dr, double &dz,
                           double &rmin, double &zmin, int &Nr, int &Nz)
{

   int i = 0;
//   int j = 0;
   int k = 0;
   int n = 0;
   int ku = 0;
   int kv = 0;
   int n_uv = 0;
   double r = 0.0;
   double z = 0.0;
   double u = 0.0;
   double v = 0.0;

// Lambda is in NANOMETERS; the distance is measured in MICROMETERS.
   double k_ill = 2.0 * PI / (0.001 * lambda_ill);
   double k_det = 2.0 * PI / (0.001 * lambda_det);

   double deg_to_rad = PI / 180.0;
   double phi_ill_rad = phi_ill * deg_to_rad;
   double phi_ill_rad_2 = 2.0 * phi_ill_rad;
   double cos_2phi_ill_2 = 2.0 * cos(phi_ill_rad_2);
   double cos_phi2_ill_4 = 4.0 * cos(phi_ill_rad) * cos(phi_ill_rad);
//   double cos_2phi_sqr_ill = cos(phi_ill_rad_2) * cos(phi_ill_rad_2);
//   double sin_2phi_sqr_ill = sin(phi_ill_rad_2) * sin(phi_ill_rad_2);

   double pre_v_ill = n_ill * k_ill * sin(deg_to_rad * alpha_ill);
   double pre_u_ill = pre_v_ill * sin(deg_to_rad * alpha_ill);
   double pre_v_det = n_det * k_det * sin(deg_to_rad * alpha_det);
   double pre_u_det = pre_v_det * sin(deg_to_rad * alpha_det);

   double du_ill = 0.0;
   double dv_ill = 0.0;
   int Nu_ill = 0;
   int Nv_ill = 0;
   double du_det = 0.0;
   double dv_det = 0.0;
   int Nu_det = 0;
   int Nv_det = 0;

   string ill_fname_uv = PSF_ill_fname + "_uv.dat";
   string ill_fname_ReI0 = PSF_ill_fname + "_ReI0.dat";
   string ill_fname_ReI1 = PSF_ill_fname + "_ReI1.dat";
   string ill_fname_ReI2 = PSF_ill_fname + "_ReI2.dat";
   string ill_fname_ImI0 = PSF_ill_fname + "_ImI0.dat";
   string ill_fname_ImI1 = PSF_ill_fname + "_ImI1.dat";
   string ill_fname_ImI2 = PSF_ill_fname + "_ImI2.dat";
   string det_fname_uv = PSF_det_fname + "_uv.dat";
   string det_fname_ReI0 = PSF_det_fname + "_ReI0.dat";
   string det_fname_ReI1 = PSF_det_fname + "_ReI1.dat";
   string det_fname_ReI2 = PSF_det_fname + "_ReI2.dat";
   string det_fname_ImI0 = PSF_det_fname + "_ImI0.dat";
   string det_fname_ImI1 = PSF_det_fname + "_ImI1.dat";
   string det_fname_ImI2 = PSF_det_fname + "_ImI2.dat";

   string full = ""; //Stores full line from the file
   char tempc[300];  //Part of stupid kludge for pointer issues
   char tmp1[300];

   printf("Constructing the PSF...\n");

// Illumination.
//**********************************************//

// Parameters for the u and v dimensions.
   ifstream s_ill_uv;
   s_ill_uv.open(ill_fname_uv.c_str());
   for (i = 1; i <= 5; i++)
   {
      s_ill_uv.getline(tempc, 300, '\n');
      if (i == 2)
      {
         sscanf(tempc, "%*s%*s%s", tmp1);
         du_ill = atof(tmp1);
      }
      if (i == 3)
      {
         sscanf(tempc, "%*s%*s%s", tmp1);
         dv_ill = atof(tmp1);
      }
      if (i == 4)
      {
         sscanf(tempc, "%*s%*s%s", tmp1);
         Nu_ill = atoi(tmp1);
      }
      if (i == 5)
      {
         sscanf(tempc, "%*s%*s%s", tmp1);
         Nv_ill = atoi(tmp1);
      }
   }
   s_ill_uv.close();

// Arrays for I0, I1 and I2.
   int N_ill = Nu_ill * Nv_ill;
   double *ReI0_ill = new double[N_ill + 1];
   double *ReI1_ill = new double[N_ill + 1];
   double *ReI2_ill = new double[N_ill + 1];
   double *ImI0_ill = new double[N_ill + 1];
   double *ImI1_ill = new double[N_ill + 1];
   double *ImI2_ill = new double[N_ill + 1];
   double *ReI0_sqr_ill = new double[N_ill + 1];   // ReI0_ill^2
   double *ReI2_sqr_ill = new double[N_ill + 1];   // ReI2_ill^2
   double *ReI0_ReI2_2_ill = new double[N_ill + 1];   // 2*ReI0_ill*ReI2_ill*cos(2*phi_ill)
   double *ReI1_sqr_ill_4 = new double[N_ill + 1]; // 4*ReI1_ill^2*cos(phi_ill)^2
// For confocal microscopy,
// sum_ill = ReI0_ill^2 + ImI0_ill^2 + ReI2_ill^2 + ImI2_ill^2 + 4*(ReI1_ill^2+ImI1_ill^2)cos(phi_ill)^2 + 2*(ReI0_ill*ReI2_ill + ImI0_ill*ImI2_ill)*cos(2*phi_ill)
// and for 4Pi microscopy,
// sum_ill = ReI0_ill^2 + ReI2_ill^2 + 2*ReI0_ill*ReI2_ill*cos(2*phi_ill) + 4*ReI1_ill^2*cos(phi_ill)^2
   double *sum_ill = new double[N_ill + 1];

// Read ReI0.
   ifstream s_ill_ReI0;
   s_ill_ReI0.open(ill_fname_ReI0.c_str());
   i = 0;
   while (s_ill_ReI0.getline(tempc, 300, '\n'))
   {
      i++;
      ReI0_ill[i] = atof(tempc);
   }
   s_ill_ReI0.close();

// Read ReI1.
   ifstream s_ill_ReI1;
   s_ill_ReI1.open(ill_fname_ReI1.c_str());
   i = 0;
   while (s_ill_ReI1.getline(tempc, 300, '\n'))
   {
      i++;
      ReI1_ill[i] = atof(tempc);
   }
   s_ill_ReI1.close();

// Read ReI2.
   ifstream s_ill_ReI2;
   s_ill_ReI2.open(ill_fname_ReI2.c_str());
   i = 0;
   while (s_ill_ReI2.getline(tempc, 300, '\n'))
   {
      i++;
      ReI2_ill[i] = atof(tempc);
   }
   s_ill_ReI2.close();

// Read ImI0.
   ifstream s_ill_ImI0;
   s_ill_ImI0.open(ill_fname_ImI0.c_str());
   i = 0;
   while (s_ill_ImI0.getline(tempc, 300, '\n'))
   {
      i++;
      ImI0_ill[i] = atof(tempc);
   }
   s_ill_ImI0.close();

// Read ImI1.
   ifstream s_ill_ImI1;
   s_ill_ImI1.open(ill_fname_ImI1.c_str());
   i = 0;
   while (s_ill_ImI1.getline(tempc, 300, '\n'))
   {
      i++;
      ImI1_ill[i] = atof(tempc);
   }
   s_ill_ImI1.close();

// Read ImI2.
   ifstream s_ill_ImI2;
   s_ill_ImI2.open(ill_fname_ImI2.c_str());
   i = 0;
   while (s_ill_ImI2.getline(tempc, 300, '\n'))
   {
      i++;
      ImI2_ill[i] = atof(tempc);
   }
   s_ill_ImI2.close();

   if (MS_type == 0)
   {
// Confocal microscopy.
      for (i = 1; i <= N_ill; i++)
      {
         sum_ill[i] =
            ReI0_ill[i] * ReI0_ill[i] + ImI0_ill[i] * ImI0_ill[i] +
            ReI2_ill[i] * ReI2_ill[i] + ImI2_ill[i] * ImI2_ill[i] +
            cos_phi2_ill_4 * (ReI1_ill[i] * ReI1_ill[i] +
                              ImI1_ill[i] * ImI1_ill[i]) +
            cos_2phi_ill_2 * (ReI0_ill[i] * ReI2_ill[i] +
                              ImI0_ill[i] * ImI2_ill[i]);
      }
   }
   else
   {
// 4Pi microscopy.
      for (i = 1; i <= N_ill; i++)
      {
         ReI0_sqr_ill[i] = ReI0_ill[i] * ReI0_ill[i];
         ReI2_sqr_ill[i] = ReI2_ill[i] * ReI2_ill[i];
         ReI0_ReI2_2_ill[i] = cos_2phi_ill_2 * ReI0_ill[i] * ReI2_ill[i];
         //ReI1_sqr_ill_4[i] = cos_phi2_ill_4*ReI1_ill[i]*ReI1_ill[i];
         sum_ill[i] =
            ReI0_sqr_ill[i] + ReI2_sqr_ill[i] + ReI0_ReI2_2_ill[i] +
            cos_phi2_ill_4 * ImI1_ill[i] * ImI1_ill[i];
//    sum_ill[i] = (ReI2_ill[i]*ReI2_ill[i] + ImI2_ill[i]*ImI2_ill[i])*sin_2phi_sqr_ill;
      }
   }

   if (mode_ill != 0)
   {
      for (i = 1; i <= N_ill; i++)
      {
         sum_ill[i] = sum_ill[i] * sum_ill[i];  // Two-photon illumination.
      }
   }
//**********************************************//

// Detection.
//**********************************************//

// Parameters for the u and v dimensions.
   ifstream s_det_uv;
   s_det_uv.open(det_fname_uv.c_str());
   for (i = 1; i <= 5; i++)
   {
      s_det_uv.getline(tempc, 300, '\n');
      if (i == 2)
      {
         sscanf(tempc, "%*s%*s%s", tmp1);
         du_det = atof(tmp1);
      }
      if (i == 3)
      {
         sscanf(tempc, "%*s%*s%s", tmp1);
         dv_det = atof(tmp1);
      }
      if (i == 4)
      {
         sscanf(tempc, "%*s%*s%s", tmp1);
         Nu_det = atoi(tmp1);
      }
      if (i == 5)
      {
         sscanf(tempc, "%*s%*s%s", tmp1);
         Nv_det = atoi(tmp1);
      }
   }
   s_det_uv.close();

// Arrays for I0, I1 and I2.
   int N_det = Nu_det * Nv_det;
   double *ReI0_det = new double[N_det + 1];
   double *ReI1_det = new double[N_det + 1];
   double *ReI2_det = new double[N_det + 1];
   double *ImI0_det = new double[N_det + 1];
   double *ImI1_det = new double[N_det + 1];
   double *ImI2_det = new double[N_det + 1];
   double *ReI0_sqr_det = new double[N_det + 1];   // ReI0_det^2
   double *ReI1_sqr_det = new double[N_det + 1];   // ReI1_det^2
   double *ReI2_sqr_det = new double[N_det + 1];   // ReI2_det^2
   double *ImI0_sqr_det = new double[N_det + 1];   // ImI0_det^2
   double *ImI1_sqr_det = new double[N_det + 1];   // ImI1_det^2
   double *ImI2_sqr_det = new double[N_det + 1];   // ImI2_det^2
// sum_det = ReI0_det^2 + ImI0_det^2 + 2(ReI1_det^2 + ImI1_det^2) + ReI2_det^2 + ImI2_det^2
   double *sum_det = new double[N_det + 1];

// Read ReI0.
   ifstream s_det_ReI0;
   s_det_ReI0.open(det_fname_ReI0.c_str());
   i = 0;
   while (s_det_ReI0.getline(tempc, 300, '\n'))
   {
      i++;
      ReI0_det[i] = atof(tempc);
   }
   s_det_ReI0.close();

// Read ReI1.
   ifstream s_det_ReI1;
   s_det_ReI1.open(det_fname_ReI1.c_str());
   i = 0;
   while (s_det_ReI1.getline(tempc, 300, '\n'))
   {
      i++;
      ReI1_det[i] = atof(tempc);
   }
   s_det_ReI1.close();

// Read ReI2.
   ifstream s_det_ReI2;
   s_det_ReI2.open(det_fname_ReI2.c_str());
   i = 0;
   while (s_det_ReI2.getline(tempc, 300, '\n'))
   {
      i++;
      ReI2_det[i] = atof(tempc);
   }
   s_det_ReI2.close();

// Read ImI0.
   ifstream s_det_ImI0;
   s_det_ImI0.open(det_fname_ImI0.c_str());
   i = 0;
   while (s_det_ImI0.getline(tempc, 300, '\n'))
   {
      i++;
      ImI0_det[i] = atof(tempc);
   }
   s_det_ImI0.close();

// Read ImI1.
   ifstream s_det_ImI1;
   s_det_ImI1.open(det_fname_ImI1.c_str());
   i = 0;
   while (s_det_ImI1.getline(tempc, 300, '\n'))
   {
      i++;
      ImI1_det[i] = atof(tempc);
   }
   s_det_ImI1.close();

// Read ImI2.
   ifstream s_det_ImI2;
   s_det_ImI2.open(det_fname_ImI2.c_str());
   i = 0;
   while (s_det_ImI2.getline(tempc, 300, '\n'))
   {
      i++;
      ImI2_det[i] = atof(tempc);
   }
   s_det_ImI2.close();

   for (i = 1; i <= N_det; i++)
   {
      ReI0_sqr_det[i] = ReI0_det[i] * ReI0_det[i];
      ReI1_sqr_det[i] = ReI1_det[i] * ReI1_det[i];
      ReI2_sqr_det[i] = ReI2_det[i] * ReI2_det[i];
      ImI0_sqr_det[i] = ImI0_det[i] * ImI0_det[i];
      ImI1_sqr_det[i] = ImI1_det[i] * ImI1_det[i];
      ImI2_sqr_det[i] = ImI2_det[i] * ImI2_det[i];
      sum_det[i] =
         ReI0_sqr_det[i] + ImI0_sqr_det[i] + 2.0 * (ReI1_sqr_det[i] +
                                                    ImI1_sqr_det[i]) +
         ReI2_sqr_det[i] + ImI2_sqr_det[i];
   }
//**********************************************//


// Building h(r) = h_ill^2 (two-photon illumination) or h_ill (one-photon).
// h_ill = ReI0^2 + ReI2^2 + 2 ReI0 ReI2 cos(2 phi) + 4 ReI1^2 cos(phi)^2.
// h is the function of x, y, z; ReI0...ImI2 are the functions of u and v.
// u = n k sin(alpha)^2 z; v = n k sin(alpha) r; r^2 = x^2 + y^2.
//**********************************************//
   n = 0;
   n_uv = 0;
   int Nu_ill_1 = Nu_ill - 1;
   int Nv_ill_1 = Nv_ill - 1;
   for (k = 1; k <= Nz; k++)
   {
      z = zmin + (k - 1) * dz;
      for (i = 1; i <= Nr; i++)
      {
         // Define {r, z}
         r = rmin + (i - 1) * dr;
         n = n + 1;

         // Define {u, v}.
         u = pre_u_ill * abs(z);
         v = pre_v_ill * r;
         //ku = int(ceil(u/du_ill));
         //kv = int(ceil(v/dv_ill));
         //if (kv == 0) kv = 1;

         //ku = int(u/du_ill);
         //kv = int(v/dv_ill);
         ku = int (ceil(u / du_ill));
         kv = int (ceil(v / dv_ill));
         if (ku == 0) ku = 1;
         if (kv == 0) kv = 1;
         if ((ku > Nu_ill_1) || (kv > Nv_ill_1))
         {
            h[n] = 0.0;
         }
         else
         {
            n_uv = ku * Nv_ill + kv + 1;
            h[n] = sum_ill[n_uv];
         }
      }
   }
//**********************************************//

// Building h_det(r) = Ir(r)^2.
// h_det = Ir = ReI0^2 + ImI0^2 + 2 (ReI1^2 + ImI1^2) + ReI2^2 + ImI2^2.
// h_det is the function of x, y, z; ReI0...ImI2 are the functions of u and v.
// u = n k sin(alpha)^2 z; v = n k sin(alpha) r; r^2 = x^2 + y^2.
//**********************************************//
   n = 0;
   n_uv = 0;
   int Nu_det_1 = Nu_det - 1;
   int Nv_det_1 = Nv_det - 1;
   for (k = 1; k <= Nz; k++)
   {
      z = zmin + (k - 1) * dz;
      for (i = 1; i <= Nr; i++)
      {
         // Define {r, z}
         r = rmin + (i - 1) * dr;
         n = n + 1;

         // Define {u, v}.
         u = pre_u_det * abs(z);
         v = pre_v_det * r;
         //ku = int(ceil(u/du_det));
         //kv = int(ceil(v/dv_det));
         //if (kv == 0) kv = 1;

         ku = int (ceil(u / du_det));
         kv = int (ceil(v / dv_det));
         if (ku == 0) ku = 1;
         if (kv == 0) kv = 1;
         if ((ku > Nu_det_1) || (kv > Nv_det_1))
         {
            h[n] = 0.0;
         }
         else
         {
            n_uv = ku * Nv_det + kv + 1;
            h_det[n] = sum_det[n_uv];
         }
      }
   }
//**********************************************//

   delete[]ReI0_ill;
   delete[]ReI1_ill;
   delete[]ReI2_ill;
   delete[]ImI0_ill;
   delete[]ImI1_ill;
   delete[]ImI2_ill;
   delete[]ReI0_sqr_ill;
   delete[]ReI2_sqr_ill;
   delete[]ReI0_ReI2_2_ill;
   delete[]ReI1_sqr_ill_4;
   delete[]sum_ill;
   delete[]ReI0_det;
   delete[]ReI1_det;
   delete[]ReI2_det;
   delete[]ImI0_det;
   delete[]ImI1_det;
   delete[]ImI2_det;
   delete[]ReI0_sqr_det;
   delete[]ReI1_sqr_det;
   delete[]ReI2_sqr_det;
   delete[]ImI0_sqr_det;
   delete[]ImI1_sqr_det;
   delete[]ImI2_sqr_det;
   delete[]sum_det;

   printf("done.\n");

}
