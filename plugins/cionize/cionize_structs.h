#ifndef CIONIZE_STRUCTS
#define CIONIZE_STRUCTS

#include "libmolfile_plugin.h"
#include "molfile_plugin.h"
#include <stdio.h>

/*** structures to contain some of the information we need ***/

/* cionize parameters */
typedef struct {

  /* Global ion information */
  float r_ion_prot;
  float r_ion_ion;
  float boundary;

  /* Information about the current ion */
  int nion;
  char ionname[5];
  float ioncharge;
  int ion_saved;

  /* Input/output files */
  char* pdbin;
  char pdbout[80];
  char oldgridfile[80];
  int useoldgrid;
  char outgridfile[80];

  /*** Threading parameters ***/
  int maxnumprocs; 

  /* File to read commands from */
  char* inputfile;
  FILE* incmd;

  /* parameters for grid calculation */
  float bordersize;

  /* If 1, then we have explicit x/y/z grid dimensions */
  unsigned char expsize;

  /* Distance dependent dielectric constant */
  float ddd;

  /* Method to use for energy calculations  - see cionize_enermethods.h*/
  int enermethod;

#if defined(BINARY_GRIDFILE)
  int write_binary_gridfile;
  int read_binary_gridfile;
#endif

} cionize_params;

/* system coordinates */
typedef struct {
  /*
   * Note that all coordinates are laid out on a grid, with integral
   * points at spacings determined at runtime. The grid dimensions are
   * numplane in the z direction, numcol in the y direction, and numpt in
   * the x direction
   */

  /* Array with atom coordinates and charges, laid out x/y/z/q/x/y/z/q/... */
  float* atoms;
  /* Array with integer equivalents of atom locations, laid out x/y/z/x/y/z... */
  long int* atomis;
  int natom;

} cionize_molecule;

/* molfile plugin handling */
typedef struct {
  molfile_plugin_t* inapi;
  molfile_plugin_t* outapi;
  char filetype[80];
  void* f_in; /* Input file handle */
  int optflags;

  /* Arrays of molfile types for temporary use during input/output */
  molfile_atom_t *atomstruct;
  molfile_timestep_t atomts;
  float* tscoords;
} cionize_api;

#endif
