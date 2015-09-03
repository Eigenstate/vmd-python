/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: main.c,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.15 $       $Date: 2009/05/18 05:56:20 $
 *
 ***************************************************************************/

/*
 * A general main for testing plugins.  
 * Compile using: gcc main.c plugin.c -I../../include -o plugintest
 * Replace plugin.c with the plugin file you want to test.
 * Usage: plugintest <filetype> <file> [<filetype> <file>]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "molfile_plugin.h"

/* Structure and coordintes plugin */
static molfile_plugin_t *splugin = 0;
static molfile_plugin_t *cplugin = 0;
static const char *sfiletype = NULL;
static const char *cfiletype = NULL;

static int register_cb(void *v, vmdplugin_t *p) {
  if (!strcmp(p->type, MOLFILE_PLUGIN_TYPE)) {
    if (!strcmp(p->name, sfiletype))
      splugin = (molfile_plugin_t *)p;
    if (!strcmp(p->name, cfiletype))
      cplugin = (molfile_plugin_t *)p;
  }
  return VMDPLUGIN_SUCCESS;
}

int main(int argc, char *argv[]) {
  const char *sfilename;
  const char *cfilename;
  int rc, natoms;
  void *handle;
  void *chandle;
  molfile_timestep_t timestep;

  if (argc != 3 && argc != 5) {
    fprintf(stderr, "Usage: %s <filetype> <filename> [<filetype> <filename>]\n", argv[0]);
    return 1;
  }

  if (argc == 3) {
    sfiletype = argv[1];
    sfilename = argv[2];
    cfiletype = sfiletype;
    cfilename = sfilename;
  } else {
    sfiletype = argv[1];
    sfilename = argv[2];
    cfiletype = argv[3];
    cfilename = argv[4];
  }
  
  vmdplugin_init();
  vmdplugin_register(NULL, register_cb);

  if (!splugin) {
    fprintf(stderr, "No plugin for filetype %s was linked in!\n", sfiletype);
    return 1;
  }
  if (!cplugin) {
    fprintf(stderr, "No plugin for filetype %s was linked in!\n", cfiletype);
    return 1;
  }

  /* Read structure */
  if (!splugin->open_file_read) {
    fprintf(stdout, "FAILED: No open_file_read found in structure plugin.\n");
    return 1;
  } 
  handle = splugin->open_file_read(sfilename, sfiletype, &natoms);
  if (!handle) {
    fprintf(stderr, "FAILED: open_file_read returned NULL in structure plugin.\n");
    return 1;
  }
  printf("Opened file %s; structure plugin found %d atoms\n", sfilename, natoms);
  if (splugin->read_structure) {
    int optflags;
    molfile_atom_t *atoms;
    atoms = (molfile_atom_t *)malloc(natoms * sizeof(molfile_atom_t));
    rc = splugin->read_structure(handle, &optflags, atoms);
    free(atoms);
    if (rc) {
      fprintf(stderr, "FAILED: read_structure returned %d\n", rc);
      splugin->close_file_read(handle);
      return 1;
    } else {
      printf("Successfully read atom structure information.\n");
    }
    if (splugin->read_bonds) {
      int nbonds, *from, *to, *bondtype, nbondtypes;
      float *bondorder;
      char **bondtypename;
      if ((rc = splugin->read_bonds(handle, &nbonds, &from, &to, 
				   &bondorder, &bondtype, &nbondtypes, &bondtypename))) {
        fprintf(stderr, "FAILED: read_bonds returned %d\n", rc);
      } else {
        printf("read_bonds read %d bonds\n", nbonds);
      }
    } else {
      printf("Structure file contains no bond information\n");
    }
  } else {
    fprintf(stderr, "FAILED: File contains no structure information!\n");
    return 1;
  }

  /* Check whether we use one plugin for both structure and coords */
  if (splugin != cplugin) {
    splugin->close_file_read(handle);
    int cnatoms;
    chandle = cplugin->open_file_read(cfilename, cfiletype, &cnatoms);
    printf("Opened coordinates file %s\n", cfilename);
    if (cnatoms != MOLFILE_NUMATOMS_UNKNOWN && cnatoms != natoms) {
      fprintf(stderr, "FAILED: Different number of atoms in structure file (%d) than in coordinates file (%d)!",
	      natoms, cnatoms);
      cplugin->close_file_read(chandle);
      exit(1);
    }
  } else {
    chandle = handle;
  }

  /* Read coordinates */
  if (cplugin->read_next_timestep) {
    timestep.velocities = NULL;
    int nsteps = 0;
    timestep.coords = (float *)malloc(3*natoms*sizeof(float));
    while (!(rc = cplugin->read_next_timestep(chandle, natoms, &timestep)))
      nsteps++;
    free(timestep.coords);
    if (rc != MOLFILE_SUCCESS) {
      fprintf(stderr, "FAILED: read_next_timestep returned %d\n", rc);
    } else {
      printf("Successfully read %d timesteps\n", nsteps);
    }
  }

  /* Close plugin(s) */
  cplugin->close_file_read(chandle); 

  vmdplugin_fini();
  printf("Tests finished.\n");
  return 0;
}

