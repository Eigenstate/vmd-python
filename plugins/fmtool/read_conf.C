using namespace std;

#include "read_conf.h"

// This function reads the configuration file.
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
               string & PSF_det_fname, int &N_beam, double **kvector_x,
	       double **kvector_y, double **kvector_z, double **e_x_vector_x,
	       double **e_x_vector_y, double **e_x_vector_z)
{

   // Auxiliary variables to check whether the beams are defined or not.
   int i;
   int *kvector_defined;
   int *e_x_vector_defined;
   
   char tmp_s[300];
   char tmp_s1[300];
   char tmp_s2[300];
   char tmp_s3[300];
   string full = ""; //Stores full line from the file
   char tempc[300];  //Part of stupid kludge for pointer issues
   double tmp;
   ifstream conf;
   conf.open(conf_name.c_str());
   while (conf.getline(tempc, 300, '\n'))
   {
      full = tempc;

// Time step and number of steps.
      if (strstr(full.c_str(), "timestep") != NULL)
      {
         dt = atof(full.substr(8, 300).c_str());
      }
      if (strstr(full.c_str(), "run") != NULL)
      {
         M = atoi(full.substr(3, 300).c_str());
      }
      if (strstr(full.c_str(), "outputFrequency") != NULL)
      {
         iOutputFrequency = atoi(full.substr(strlen("outputFrequency"), 300).c_str());
      }
// Spatial domain.
      if (strstr(full.c_str(), "Lx") != NULL)
      {
         Lx = atof(full.substr(2, 300).c_str());
      }
      if (strstr(full.c_str(), "Ly") != NULL)
      {
         Ly = atof(full.substr(2, 300).c_str());
      }
      if (strstr(full.c_str(), "Lz") != NULL)
      {
         Lz = atof(full.substr(2, 300).c_str());
      }
      if (strstr(full.c_str(), "Nx") != NULL)
      {
         Nx = atoi(full.substr(2, 300).c_str());
      }
      if (strstr(full.c_str(), "Ny") != NULL)
      {
         Ny = atoi(full.substr(2, 300).c_str());
      }
      if (strstr(full.c_str(), "Nz") != NULL)
      {
         Nz = atoi(full.substr(2, 300).c_str());
      }
// Choose whether to do full 3D of xy_to_r (effectively 2D).
      if (strstr(full.c_str(), "do_xy_to_r") != NULL)
      {
         do_xy_to_r = atoi(full.substr(10, 300).c_str());
      }
// Equation parameters.
      if (strstr(full.c_str(), "DiffusionCoefficient") != NULL)
      {
         D = atof(full.substr(20, 300).c_str());
      }
      if (strstr(full.c_str(), "Intensity_I_0") != NULL)
      {
         I_0 = atof(full.substr(13, 300).c_str());
      }
      if (strstr(full.c_str(), "Reactivty_kh") != NULL)
      {
         kh = atof(full.substr(12, 300).c_str());
      }
// Initial conditions.
      if (strstr(full.c_str(), "p0") != NULL)
      {
         p0 = atof(full.substr(2, 300).c_str());
      }
// The output file for the observable.
      if (strstr(full.c_str(), "ObsFName") != NULL)
      {
         sscanf(tempc, "%*s %s", tmp_s);
         out_Obs_name = tmp_s;
      }
// Writing the 3D map of the PSF in a .dx file.
      if (strstr(full.c_str(), "PSF_write_dx") != NULL)
      {
         sscanf(tempc, "%*s %s", tmp_s);
         PSF_write_dx = atoi(tmp_s);
      }
// PSF.
      if (strstr(full.c_str(), "TypePSF") != NULL)
      {
         TypePSF = atoi(full.substr(7, 300).c_str());
      }
// Center of the PSF.
      if (strstr(full.c_str(), "x0_PSF") != NULL)
      {
         x0_PSF = atof(full.substr(6, 300).c_str());
      }
      if (strstr(full.c_str(), "y0_PSF") != NULL)
      {
         y0_PSF = atof(full.substr(6, 300).c_str());
      }
      if (strstr(full.c_str(), "z0_PSF") != NULL)
      {
         z0_PSF = atof(full.substr(6, 300).c_str());
      }
// Radius for the spherical PSF.
      if (strstr(full.c_str(), "R_PSF") != NULL)
      {
         R_PSF = atof(full.substr(5, 300).c_str());
      }
// Size for the rectangular PSF.
      if (strstr(full.c_str(), "LPSF_x") != NULL)
      {
         Lx_PSF = atof(full.substr(6, 300).c_str());
      }
      if (strstr(full.c_str(), "LPSF_y") != NULL)
      {
         Ly_PSF = atof(full.substr(6, 300).c_str());
      }
      if (strstr(full.c_str(), "LPSF_z") != NULL)
      {
         Lz_PSF = atof(full.substr(6, 300).c_str());
      }
//Read PSF from the file (if TypePSF is not 0 or 1).
      if (strstr(full.c_str(), "MS_type") != NULL)
      {
         MS_type = atoi(full.substr(7, 300).c_str());
      }
      if (strstr(full.c_str(), "mode_ill") != NULL)
      {
         mode_ill = atoi(full.substr(8, 300).c_str());
      }
      if (strstr(full.c_str(), "alpha_ill") != NULL)
      {
         alpha_ill = atof(full.substr(9, 300).c_str());
      }
      if (strstr(full.c_str(), "n_ill") != NULL)
      {
         n_ill = atof(full.substr(5, 300).c_str());
      }
      if (strstr(full.c_str(), "lambda_ill") != NULL)
      {
         lambda_ill = atof(full.substr(10, 300).c_str());
      }
      if (strstr(full.c_str(), "phi_ill") != NULL)
      {
         phi_ill = atof(full.substr(7, 300).c_str());
      }
      if (strstr(full.c_str(), "alpha_det") != NULL)
      {
         alpha_det = atof(full.substr(9, 300).c_str());
      }
      if (strstr(full.c_str(), "n_det") != NULL)
      {
         n_det = atof(full.substr(5, 300).c_str());
      }
      if (strstr(full.c_str(), "lambda_det") != NULL)
      {
         lambda_det = atof(full.substr(10, 300).c_str());
      }
      if (strstr(full.c_str(), "PSF_ill_fname") != NULL)
      {
         sscanf(tempc, "%*s %s", tmp_s);
         PSF_ill_fname = tmp_s;
      }
      if (strstr(full.c_str(), "PSF_det_fname") != NULL)
      {
         sscanf(tempc, "%*s %s", tmp_s);
         PSF_det_fname = tmp_s;
      }
// PSF given by the interference of several beams.
	if (strstr(full.c_str(),"N_beam") != NULL) {
	        if (MS_type == -1) {
		   printf("ERROR: reading the configuration file encountered N_beam, but MS_type has not been defined yet; define MS_type first. Exiting.\n");
		   exit(-1);
		}
		if ((MS_type != 0) && (MS_type != 1)) {
		   N_beam = atoi(full.substr(6, 300).c_str());
   		   if (N_beam <= 0) {
		      printf("ERROR: number of beams is %d; should be > 0. Exiting.\n", N_beam);
		      exit(-1);
		   }
		   *kvector_x = (double *) malloc( N_beam * sizeof(double));
		   *kvector_y = (double *) malloc( N_beam * sizeof(double));
		   *kvector_z = (double *) malloc( N_beam * sizeof(double));
		   kvector_defined = (int *) malloc( N_beam * sizeof(int));
                   *e_x_vector_x = (double *) malloc( N_beam * sizeof(double));
		   *e_x_vector_y = (double *) malloc( N_beam * sizeof(double));
		   *e_x_vector_z = (double *) malloc( N_beam * sizeof(double));
		   e_x_vector_defined = (int *) malloc( N_beam * sizeof(int));
		   for (i=0; i<N_beam;i++)
                   {
		      kvector_defined[i] = 0;
		      e_x_vector_defined[i] = 0;
		   }
		}
	}
// PSF given by the interference of several beams; get k_beam.
	if (strstr(full.c_str(),"k_beam") != NULL) {
	        if ((N_beam == -1) && (MS_type != 0) && (MS_type != 1)) {
		   printf("ERROR: reading the configuration file encountered k_beam, but N_beam has not been defined yet; define N_beam first. Exiting.\n");
		   exit(-1);
		}
	        sscanf(tempc, "%*s %s %s %s %s", tmp_s, tmp_s1, tmp_s2, tmp_s3);
		i = atoi(tmp_s) - 1; // The first element of the array is 0.
                if ((i+1) <= 0) {
		   printf("ERROR: k_beam label is %d; should be > 0. Exiting.\n", i+1);
		   exit(-1);
		}
		if ((i+1) <= N_beam) {
		   if (kvector_defined[i] == 1) {
		      printf("ERROR: k_beam for beam %d is redundantly defined. Exiting.\n", i+1);
		      exit(-1);
		   }
		   (*kvector_x)[i] = atof(tmp_s1);
		   (*kvector_y)[i] = atof(tmp_s2);
		   (*kvector_z)[i] = atof(tmp_s3);
		   // Normalize.
		   tmp = sqrt((*kvector_x)[i] * (*kvector_x)[i] + (*kvector_y)[i] * (*kvector_y)[i] + (*kvector_z)[i] * (*kvector_z)[i]);
		   if (tmp <= 0.0) {
		      printf("ERROR: k_beam(%d) = {%f %f %f}; the norm is %f, while it should be > 0. Exiting.\n", i+1, (*kvector_x)[i], (*kvector_y)[i], (*kvector_z)[i], tmp);
		      exit(-1);
		   }
		   (*kvector_x)[i] /= tmp;
		   (*kvector_y)[i] /= tmp;
		   (*kvector_z)[i] /= tmp;
		   kvector_defined[i] = 1;
		}
	}
// PSF given by the interference of several beams; get e_x_beam.
	if (strstr(full.c_str(),"e_x_beam") != NULL) {
	        if ((N_beam == -1) && (MS_type != 0) && (MS_type != 1)) {
		   printf("ERROR: reading the configuration file encountered e_x_beam, but N_beam has not been defined yet; define N_beam first. Exiting.\n");
		   exit(-1);
		}
	        sscanf(tempc, "%*s %s %s %s %s", tmp_s, tmp_s1, tmp_s2, tmp_s3);
		i = atoi(tmp_s) - 1; // The first element of the array is 0.
                if ((i+1) <= 0) {
		   printf("ERROR: e_x_beam label is %d; should be > 0. Exiting.\n", i+1);
		   exit(-1);
		}
		if ((i+1) <= N_beam) {
		   if (e_x_vector_defined[i] == 1) {
		      printf("ERROR: e_x_beam for beam %d is redundantly defined. Exiting.\n", i+1);
		      exit(-1);
		   }
		   (*e_x_vector_x)[i] = atof(tmp_s1);
		   (*e_x_vector_y)[i] = atof(tmp_s2);
		   (*e_x_vector_z)[i] = atof(tmp_s3);
		   // Normalize.
		   tmp = sqrt((*e_x_vector_x)[i] * (*e_x_vector_x)[i] + (*e_x_vector_y)[i] * (*e_x_vector_y)[i] + (*e_x_vector_z)[i] * (*e_x_vector_z)[i]);
		   if (tmp <= 0.0) {
		      printf("ERROR: e_x_beam(%d) = {%f %f %f}; the norm is %f, while it should be > 0. Exiting.\n", i+1, (*e_x_vector_x)[i], (*e_x_vector_y)[i], (*e_x_vector_z)[i], tmp);
		      exit(-1);
		   }
		   (*e_x_vector_x)[i] /= tmp;
		   (*e_x_vector_y)[i] /= tmp;
		   (*e_x_vector_z)[i] /= tmp;
		   e_x_vector_defined[i] = 1;
		}
	}
   }

   conf.close();


// Check the definitions for multiple beams.
   for (i=0; i<N_beam;i++)
   {
      if (kvector_defined[i] != 1) {
         printf("ERROR: k_beam is not defined for beam #%d. Exiting.\n", i+1);
         exit(-1);
      }
      if (e_x_vector_defined[i] != 1) {
         printf("ERROR: e_x_beam is not defined for beam #%d. Exiting.\n", i+1);
         exit(-1);
      }
      // Check that kvector and e_x_vector are perpendicular.
      tmp = (*kvector_x)[i] * (*e_x_vector_x)[i] + (*kvector_y)[i] * (*e_x_vector_y)[i] + (*kvector_z)[i] * (*e_x_vector_z)[i];
      if (tmp != 0.0) {
         printf("ERROR: for beam #%d, k_beam and e_x_beam are not perpendicular. Exiting.\n", i+1);
         exit(-1);
      }
   }
   
   if (N_beam > 0) {
      free(kvector_defined);
      free(e_x_vector_defined);
   }

}
