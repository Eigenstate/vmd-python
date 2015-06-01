#include "cionize_structs.h"
#include "cionize_grid.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "energythr.h"
#include "getplugins.h"

#include "cionize_molfileio.h"
#include "cionize_gridio.h"
#include "cionize_internals.h"
#include "cionize_enermethods.h"

#if defined(CUDA)
#include "cudaenergythr.h"
#endif

#if defined(BINARY_GRIDFILE)
#include "binary_gridio.h"
#endif


int exclude_solute(const cionize_params* params, cionize_molecule* molecule, cionize_grid* grid) {
  /*Loop through the points in the cube and exclude any within r_ion_prot of
   * the protein
   */

  float xa, ya, za; 
  int natoms;
/*Integer coordinates of current atom */
  int xi, yi, zi; 
  int ks, ke, js, je, is, ie;

/*Loop counters */
  int i,j,k,n; 
  int nexcl;
/*Number of grid points for exclude cube */
  int rip; 
  float rip2; 
  float gridspacing;
  int numpt, numcol, numplane;
  float* atoms;

  nexcl=0;
  gridspacing = grid->gridspacing;
  rip = params->r_ion_prot / gridspacing;
  rip2 = params->r_ion_prot * params->r_ion_prot;
  natoms = molecule->natom;
  numpt = grid->numpt;
  numcol = grid->numcol;
  numplane = grid->numplane;
  atoms = molecule->atoms;

  /*Note that  this time we loop over atoms first; this is because
  we only need to examine the cube of side length r_prot_ion centered on
  each atom */
  for (n=0; n<natoms; n++) {
    xa = atoms[4*n];
    ya = atoms[4*n+1];
    za = atoms[4*n+2];
    xi = (int) rint(xa / gridspacing);
    yi = (int) rint(ya / gridspacing);
    zi = (int) rint(za / gridspacing);
    ks = (zi-rip < 0) ? 0 : zi-rip;
    ke = (zi+rip >= numplane) ? numplane-1 : zi+rip; 
    js = (yi-rip < 0) ? 0 : yi-rip;
    je = (yi+rip >= numcol) ? numcol-1 : yi+rip; 
    is = (xi-rip < 0) ? 0 : xi-rip;
    ie = (xi+rip >= numpt) ? numpt-1 : xi+rip; 
    for (k=ks; k<=ke; k++) {
      float z = gridspacing * (float) k;
      for (j=js; j<=je; j++) {
        float y = gridspacing * (float) j;
        for (i=is; i<=ie; i++) {
          float x = gridspacing * (float) i;
          /*See if this point is too close to the current atom */
          float dx = xa - x;
          float dy = ya - y;
          float dz = za - z;
          float dist = dx*dx + dy*dy + dz*dz;
          if (dist <= rip2) {
            if (grid->excludepos[numcol * numpt * k + numpt * j + i] == 0) nexcl += 1;
            grid->excludepos[numcol * numpt * k + numpt * j + i] = 1;
          }
        }
      }
    }
  }
  printf("\tExcluded %i points out of %i\n", nexcl, numplane*numcol*numpt);
  
  return 0;
}

int place_n_ions(cionize_params* params, cionize_grid* grid, int* ionpos) {
  /* Place the number of ions requested. Each ion placement cycle requires
   *  -Determination of the minimum energy point
   *  -Placement of the ion
   *  -updating the energy grid to take the new ion into account
   *  -exclusion of points around the placed ion
   */

/*Current ion number */
  int n; 
/*Counters for going through grid */
  int i,j,k; 
/*Location of current point in the grid arrays */
  int offset; 
  /*grid locations of minimum energy point */
  float minxf, minyf, minzf;
  int mini;
  int minj;
  int mink;

  float ionatom[4]; /* Holds x/y/z/q for new ion */

/*Minimum ion-ion distance, in grid units */
  int rii;
/*Square of minimum distance */
  int rii2;  
  float minener;
  float myener;
  int ks, ke, js, je, is, ie;

  /* integer distances for use in new exclusion calculations */
  int dist;
  int dxi, dyi, dzi;

  const int numplane = grid->numplane;
  const int numcol = grid->numcol;
  const int numpt = grid->numpt;
  const float ioncharge = params->ioncharge;
  const int nion = params->nion;
  const float gridspacing = grid->gridspacing;

  rii = (int) rint(params->r_ion_ion/gridspacing); 
  rii2 = rii*rii; 
  mini = minj = mink = 0;

  for (n=0; n<nion; n++) {
    minener = HUGE_VAL;
    /*Place the ion on the energetic minimum*/
    /*For each point in the cube...*/
    for (k=0; k<numplane; k++) {
      for (j=0; j<numcol; j++) {
        for (i=0; i<numpt; i++) {
          /*Check if this point is excluded */
          offset = numcol*numpt*k + numpt*j + i;
/*          printf( "DEBUG: Checking exclusion at point %i %i %i: %i\n", i,j,k,grid->excludepos[offset]); */
          if (grid->excludepos[offset] == 0) {
            myener = (ioncharge * grid->eners[offset]);
            if (myener < minener) {
/*              printf( "DEBUG: Better energy %f found at %i %i %i\n", myener, i, j, k);  */
              minener = myener;
              mini = i;
              minj = j;
              mink = k;
            }
          }
        }
      }
    }

 
    /*Found the current minimum energy point; place an ion there */
    ionpos[3*n    ] = mini;
    ionpos[3*n + 1] = minj;
    ionpos[3*n + 2] = mink;
    /* printf( "\tPlaced ion %i at grid location %i %i %i with energy %f\n", n, mini, minj, mink, minener); */

    /*Update the energy grid with the effects of the new ion */
    minxf = (float) (mini * gridspacing);
    minyf = (float) (minj * gridspacing);
    minzf = (float) (mink * gridspacing);

    /* Store location and charge of atom */
    ionatom[0] = minxf;
    ionatom[1] = minyf;
    ionatom[2] = minzf;
    ionatom[3] = params->ioncharge;

    /*Exclude anything too close to it */
    /*Only work in the space within rii of the newly placed ion */
    ks = (mink-rii < 0) ? 0 : mink-rii;
    ke = (mink+rii >= numplane) ? numplane - 1 : mink+rii;
    js = (minj-rii < 0) ? 0 : minj-rii;
    je = (minj+rii >= numcol) ? numcol - 1 : minj+rii;
    is = (mini-rii < 0) ? 0 : mini-rii;
    ie = (mini+rii >= numpt) ? numpt - 1 : mini+rii;

    for (k=ks; k<=ke; k++) {
      for (j=js; j<=je; j++) {
        for (i=is; i<=ie; i++) {
          /*Exclude if too close to current ion */
          dxi = i-mini;
          dyi = j-minj;
          dzi = k-mink;
          dist = dxi*dxi + dyi*dyi + dzi*dzi;
          if (dist < rii2) {
            grid->excludepos[numcol*numpt*k + numpt*j + i] = 1;
            grid->eners[numcol*numpt*k + numpt*j + i] = 0;
/*            printf( "DEBUG: Excluding node at %i %i %i because distance %i is less than the minimum %i\n", i, j, k, dist, rii2); */
          }
        }
      }
    }

    /* Update the energy grid with the contribution of the new ion */

    
#if defined(CUDA)
    if ((grid->numcol * grid->numpt) > (256 * 256)) {
      calc_grid_energies_cuda_thr(&ionatom[0], grid->eners, grid->numplane, grid->numcol, grid->numpt, 1, grid->gridspacing, grid->excludepos, params->maxnumprocs);
    } else {
      calc_grid_energies(&ionatom[0], grid->eners, grid->numplane, grid->numcol, grid->numpt, 1, grid->gridspacing, grid->excludepos, params->maxnumprocs, params->enermethod, params->ddd);
    }
#elif defined(MGRID)
    /*
    calc_grid_energies_excl_mgrid(&ionatom[0], grid->eners, grid->numplane, grid->numcol, grid->numpt, 1, grid->gridspacing, grid->excludepos, params->maxnumprocs);
    */
    /* use brute force method for grid updates after single ion placement */
    calc_grid_energies(&ionatom[0], grid->eners, grid->numplane, grid->numcol, grid->numpt, 1, grid->gridspacing, grid->excludepos, params->maxnumprocs, params->enermethod, params->ddd);
#else
    calc_grid_energies(&ionatom[0], grid->eners, grid->numplane, grid->numcol, grid->numpt, 1, grid->gridspacing, grid->excludepos, params->maxnumprocs, params->enermethod, params->ddd);
#endif

    /*Old, simple code for doing this below*/
#if 0
    for (k=0; k<numplane; k++) {
      z = gridspacing * (float) k;
      for (j=0; j<numcol; j++) {
        y = gridspacing * (float) j;
        for (i=0; i<numpt; i++) {
          x = gridspacing * (float) i;
          /*printf("Coords: %f %f %f %f %f %f\n", x, y, z, minxf, minyf, minzf);*/
          if (grid->excludepos[numcol*numpt*k + numpt*j + i] != 0) continue;
          dx = x - minxf;
          dy = y - minyf;
          dz = z - minzf;
          /*printf("dx, dy, dz = %f %f %f\n", dx, dy, dz);*/
          r2 = dx*dx + dy*dy + dz*dz;
          /*printf("dx, dy, dz, r2 = %f %f %f %f\n", dx, dy, dz, r2);*/
          r_1 = 1.0 / sqrtf(r2);
          energy =  params->ioncharge * r_1;
/*          printf("Energy %f being added to previous energy %f\n", energy, grid->eners[numcol * numpt * k + numpt * j + i]);*/
          grid->eners[numcol * numpt * k + numpt * j + i] += energy;
        }
      }
    }
#endif 

  }

  /* Clean up */

  printf("\tSuccessfully placed %i ions\n", params->nion);


  return 0;
}


/* Allocate memory for the arrays used in our calculations */
int alloc_mainarrays(cionize_grid* grid, cionize_molecule* molecule) {

  unsigned long natom, numpt, numcol, numplane;
 
  natom = molecule->natom;
  numpt = grid->numpt;
  numcol = grid->numcol;
  numplane = grid->numplane;

  printf("\tAllocating %lu KB for atom arrays\n", (4*natom*sizeof(float)/(1024)));
  molecule->atoms = (float*)malloc(4*natom*sizeof(float));
  printf("\tAllocating %lu KB for atom grid point array\n", (3*natom*sizeof(long int)/(1024)));
  molecule->atomis = (long int*)malloc(3*natom*sizeof(long int));
  printf("\tAllocating %lu MB for grid energy array\n", (numpt*numcol*numplane*sizeof(float)/(1024*1024)));
  grid->eners = (float*)malloc(numpt*numcol*numplane*sizeof(float)); 
  printf("\tAllocating %lu MB for exclusion array\n", (numpt*numcol*numplane*sizeof(unsigned char)/(1024*1024)));
  grid->excludepos = (unsigned char*)calloc(numpt*numcol*numplane, sizeof(unsigned char));

  if (molecule->atoms==NULL || molecule->atomis==NULL || grid->eners==NULL || grid->excludepos==NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for data arrays\n");
    return 1;
  }

  return 0;
}

/* Run through the second input loop, and carry out the requested ion placements */
int do_placements(cionize_params* params, cionize_grid* grid, cionize_api* molfiles) {
  /*Arrays for holding user input */
  char inbuf[80];
  char headbuf[80];
  char outgridfile[80];
  char pdbout[80];
  int rv;
  int ion_saved;

  /* Array for ion positions, laid out x/y/z/x/y/z... */
  int* ionpos;

  ionpos = NULL;
  ion_saved=1;

  while (printf( "\n>>> ") && fgets(inbuf, 80, params->incmd)) {
    if (sscanf(inbuf, "%s", headbuf) != 1) {
      printf( "\tPlease enter a command. Type 'HELP' for help.");
      continue;
    }

    if (strncasecmp(headbuf, "SAVEGRID", 8) == 0) {
        if (sscanf(inbuf, "%*s %s", outgridfile) != 1) {
          printf( "\tError: Couldn't parse input line for SAVEGRID\n");
          continue;
        }
        printf( "Writing grid energies to file %s\n", outgridfile);
#if defined(BINARY_GRIDFILE)
        if (params->write_binary_gridfile) {
#if 1
          /* call DX writer with binary output enabled */
          if (write_grid_dx(outgridfile, 1,
                grid->gridspacing, grid->minx, grid->miny, grid->minz,
                grid->numpt, grid->numcol, grid->numplane,
                grid->eners) == 0) {
            printf("Finished writing energy grid to binary file.\n\n");
          }
#else
          /* call binary grid file writer */
          if (gridio_write(outgridfile,
                grid->gridspacing, grid->minx, grid->miny, grid->minz,
                grid->numpt, grid->numcol, grid->numplane,
                grid->eners) == 0) {
            printf("Finished writing energy grid to binary file.\n\n");
          }
#endif
          else {
            printf("Error: Failed to write energy grid to binary file.\n\n");
          }
        }
        else
#endif
        if (write_grid(outgridfile, grid->gridspacing, grid->minx, grid->miny, grid->minz, grid->numplane, grid->numcol, grid->numpt, grid->eners) == 0) {
          printf( "Finished with energy grid.\n\n");
        } else {
          printf( "Error: Failed to write energy grid.\n\n");
          continue;
        }
    } else if (strncasecmp(headbuf, "PLACEION", 8) == 0) {
        if (ion_saved == 0) {
          printf( "WARNING: Previously placed ions were not saved!\n");
        }
        if (sscanf(inbuf, "%*s %s %i %f", headbuf, &(params->nion), &(params->ioncharge)) != 3) {
          printf( "\tError: Couldn't parse input line for PLACEION\n\n");
          continue;
        }
        if (strlen(headbuf) > 4) {
          printf( "\tWarning: Truncating ion name to 4 characters...\n");
        }
        strncpy(params->ionname, headbuf, 4);
        printf( "Placing %i %s ions\n", params->nion, params->ionname);

        /* Clear the ion position array and allocate the memory we need */
        if (ionpos != NULL) free(ionpos);
        ionpos = malloc(3*params->nion*sizeof(int));
        if (ionpos == NULL) {
          fprintf(stderr, "ERROR: Failed to allocate memory for new ions\n");
          return 1;
        }

        if (place_n_ions(params, grid, ionpos) != 0) {
          fprintf(stderr, "\n\nError encountered in ion placement! Exiting...\n");
          return 1;
        }
        ion_saved = 0;
        printf( "Finished with ion placement.\n\n");
    } else if (strncasecmp(headbuf, "SAVEION", 7) == 0) {
      rv = sscanf(inbuf, "%*s %s %s", molfiles->filetype, pdbout);
      if (rv==0) {
        printf( "\tError: Couldn't parse input line for PLACEION\n");
        continue;
      } else if (rv == 1) {
        /* Then we need to guess the file type */
        strncpy(pdbout, molfiles->filetype, 80);
        molfiles->outapi = get_plugin_forfile(pdbout, molfiles->filetype);
      } else {
        molfiles->outapi = get_plugin(molfiles->filetype);
      }

      if (!(molfiles->outapi) || !(molfiles->outapi->write_structure) || !(molfiles->outapi->write_timestep)) {
        printf("\tError: Can't write the necessary information in format %s\n", molfiles->filetype);
        continue;
      }


      printf("\tPrinting ion coordinates to %s file %s\n",molfiles->filetype,pdbout);
      if (output_ions(params, molfiles, grid, pdbout, ionpos) != 0) {
        printf( "\tError in writing output ions...\n\n");
        continue;
      }
      printf("\tFinished writing output\n\n");
      ion_saved=1;
    } else if (strncasecmp(headbuf, "HELP", 4) == 0) {
      printf("Recognized options: \n\tSAVEGRID file--Save current energy grid to a file\n\tPLACEION ionname n ioncharge--Place a set of n ions using the current grid\n\tSAVEION file--Save the most recently placed batch of ions to a pdb file\n");
    } else if (strncasecmp(headbuf, "EXIT", 4) == 0) {
      break;
    } else {
      fprintf(stderr, "Error: Unrecognized line in input file. Use HELP for options.\n");
    }
  }

  return 0;
}

/* Initialize all the system parameters to sensible defaults  and open the input file*/
void init_defaults(cionize_params* params, cionize_grid* grid) {
  /* Set defaults for all the usual parameters */
  params->r_ion_prot=6.0;
  params->r_ion_ion=10.0;
  params->bordersize=10.0;
  grid->gridspacing = 0.5;
  params->ion_saved=1;
  params->maxnumprocs = 1;
  params->useoldgrid=0;
  params->inputfile=NULL;
  params->incmd = NULL;
  params->expsize=0;
  params->ddd=0.0;
  params->enermethod=STANDARD;
#if defined(BINARY_GRIDFILE)
  params->write_binary_gridfile = 0;  /* "off" by default */
  params->read_binary_gridfile = 0;   /* "off" by default */
#endif
}




