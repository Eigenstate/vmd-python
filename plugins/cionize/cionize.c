/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: cionize.c,v $
 *      $Author: dhardy $        $Locker:  $             $State: Exp $
 *      $Revision: 1.34 $      $Date: 2008/06/11 06:51:23 $
 *
 ***************************************************************************/

#if defined(_AIX)
/* Define to enable large file extensions on AIX */
#define _LARGE_FILE
#define _LARGE_FILES
#else
/* Defines which enable LFS I/O interfaces for large (>2GB) file support
 * on 32-bit machines.  These must be defined before inclusion of any
 * system headers.
 */
#define _LARGEFILE_SOURCE
#define _FILE_OFFSET_BITS 64
#endif

#define IONIZE_MAJOR_VERSION 1
#define IONIZE_MINOR_VERSION 0
#define MAX_PLUGINS 200

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#include "util.h"
#include "threads.h"
#include "energythr.h"
#if defined(MGRID)
#include "energymgrid.h"
#endif

/* DH: always included, but not used by default */
#include "mgpot.h"

#if defined(CUDA)
#include "cudaenergythr.h"
#endif

/* Molfile plugin headers */
#include "libmolfile_plugin.h"
#include "molfile_plugin.h"
#include "getplugins.h"

#include "cionize_structs.h"
#include "cionize_grid.h"
#include "cionize_gridio.h"
#include "cionize_molfileio.h"
#include "cionize_userio.h"
#include "cionize_internals.h"
#include "cionize_enermethods.h"

#if defined(BINARY_GRIDFILE)
#include "binary_gridio.h"
#endif


int main(int argc, char* argv[]) {
  /* Set up the structs used for all the internal information */
  cionize_params params;
  cionize_molecule molecule;
  cionize_grid grid;
  cionize_api molfiles;

  /* Initialize everything that needs it */
  init_defaults(&params, &grid);
  setup_apis(&molfiles);

  /* Parse the command line */
  if (get_opts(&params, &molfiles, argc, argv) != 0) return 1;

  /* Set up the input api */
  if (setup_inapi(&params, &molfiles) != 0) return 1;

  /* Open the input file and go through the initial parameter setup phase */
  if (open_input(&params) != 0) return 1;

  /* Now go through the initial input loop where we set all the parameters */
  if (settings_inputloop(&params, &grid) != 0) return 1;


  /* Read the input molecule and set up memory for the molecule and grid */
  printf("\nRunning cionize on input file %s\n\tIon-solute distance: %f\n\tIon-Ion distance: %f\n\tGrid spacing: %f\n\tBoundary size: %f\n\tMax. Processors: %i\n", params.pdbin, params.r_ion_prot, params.r_ion_ion, grid.gridspacing, params.bordersize, params.maxnumprocs);
  if (params.ddd != 0) {
    printf("\tDistance dependent dielectric constant: %f * r\n", params.ddd);
  }

  read_input(&params, &molfiles, &molecule, &grid);

  printf("\nMaximum extents of the considered system (including boundary) are:\n\tX: %8.3f to %8.3f\tY: %8.3f to %8.3f\tZ: %8.3f to %-8.3f\nUsing a grid spacing of %f angstroms, the grid is %6ld x %6ld x %6ld\n", grid.minx, grid.maxx, grid.miny, grid.maxy, grid.minz, grid.maxz, grid.gridspacing, grid.numpt, grid.numcol, grid.numplane);

  /* Now allocate the arrays we need and pack the appropriate data into them*/
  printf("\nAllocating memory for data arrays...\n");
  if (alloc_mainarrays(&grid, &molecule) != 0) return 1;
  printf( "Successfully allocated data arrays\n");

  /* Stick all the data we need from the molfile type into ionize arrays */
  unpack_molfile_arrays(&params, &molfiles, &molecule, &grid);

  /*Exclude grid points too close to the protein */
  printf("\nExcluding grid points too close to protein\n");
  if (exclude_solute(&params, &molecule, &grid) != 0) return 1;
  printf("Finished with exclusion\n");

  /* Read or calculate the energy grid */

  if (params.useoldgrid == 0) {
    int i;
    rt_timerhandle timer = rt_timer_create();
    float lasttime, elapsedtime;
    rt_timer_start(timer);

    /* Initialize our grid to all zeros so we can just have the energy
     * function add to it */
    for (i=0; i<(grid.numpt * grid.numcol * grid.numplane); i++) {
      grid.eners[i] = 0.0;
    }

    /*Now that all our input is done, calculate the initial grid energies*/
    printf("\nNumber of atoms:  %d\n", molecule.natom);
    printf("Number of grid points:  %ld\n",
        grid.numplane * grid.numcol * grid.numpt);
    printf( "\nCalculating grid energies...\n");
    lasttime = rt_timer_timenow(timer);
    if (params.enermethod & MULTIGRID) {
      /* DH: if requested, use multilevel summation for initial grid */
      calc_grid_energies_excl_mgpot(molecule.atoms, grid.eners,
	  grid.numplane, grid.numcol, grid.numpt, molecule.natom,
	  grid.gridspacing, grid.excludepos, params.maxnumprocs,
          params.enermethod);
    } else {
      /* DH: else method defaults to direct summation */
#if defined(CUDA)
      calc_grid_energies_cuda_thr(molecule.atoms, grid.eners,
          grid.numplane, grid.numcol, grid.numpt, molecule.natom,
          grid.gridspacing, grid.excludepos, params.maxnumprocs);
#else
      calc_grid_energies(molecule.atoms, grid.eners,
	  grid.numplane, grid.numcol, grid.numpt, molecule.natom,
	  grid.gridspacing, grid.excludepos, params.maxnumprocs,
          params.enermethod, params.ddd);
#endif
    }
    elapsedtime = rt_timer_timenow(timer) - lasttime;
    printf( "Done calculating grid energies, total time: %.2f\n",
        elapsedtime);
  } else {
    printf( "\nReading grid energies from %s\n", params.oldgridfile);
#if defined(BINARY_GRIDFILE)
    if (params.read_binary_gridfile) {
      /* call binary grid file reader */
      if (gridio_read(params.oldgridfile,
            grid.gridspacing, grid.minx, grid.miny, grid.minz,
            grid.numpt, grid.numcol, grid.numplane,
            grid.eners) != 0) return 1;
      printf("Finished reading energy grid from binary file.\n\n");
    }
    else
#endif
    if (read_grid(params.oldgridfile, grid.eners, grid.gridspacing, grid.numplane, grid.numcol, grid.numpt, grid.minx, grid.miny, grid.minz) != 0) return 1;
    printf("Finished reading energy grid\n");
  }

  /* Now, begin a new input loop reading statements to either place ions, save ions, or write the grid */
  if (do_placements(&params, &grid, &molfiles) != 0) return 1;

  printf("\ncionize: normal exit\n");
  return 0;
}







