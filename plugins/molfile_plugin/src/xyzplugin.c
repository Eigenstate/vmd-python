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
 *      $RCSfile: xyzplugin.c,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.36 $       $Date: 2009/04/29 15:45:35 $
 *
 ***************************************************************************/

/*
 *  XMol XYZ molecule file format:
 *    XYZ files are a simple molecule file format suitable for output
 *    by homegrown software since they are very minimalistic.  They don't
 *    even include bonding information.
 *
 *  [ N                       ] # of atoms, required by this xyz reader plugin
 *  [ molecule name           ] name of molecule (can be blank)
 *  atom1 x y z [optional data] atom name followed by xyz coords 
 *  atom2 x y z [ ...         ] and (optionally) other data.
 *  ...                         instead of atom name the atom number in 
 *  atomN x y z [ ...         ] the PTE can be given.
 *  ...
 *
 *  Note that this plugin currently ignores everything following the z 
 *  coordinate (the optional data fields).
 *
 *  Sample input file for XMol:
 *  3
 *  Water Molecule - XYZ Format
 *  O      .000000     .000000     .114079
 *  H      .000000     .780362    -.456316
 *  H      .000000    -.780362    -.456316
 *                      
 */

#include "largefiles.h"   /* platform dependent 64-bit file I/O defines */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "molfile_plugin.h"

#include "periodic_table.h"

typedef struct {
  FILE *file;
  int numatoms;
  char *file_name;
  molfile_atom_t *atomlist;
} xyzdata;
 
static void *open_xyz_read(const char *filename, const char *filetype, 
                           int *natoms) {
  FILE *fd;
  xyzdata *data;
  int i;

  fd = fopen(filename, "rb");
  if (!fd) return NULL;
  
  data = (xyzdata *)malloc(sizeof(xyzdata));
  data->file = fd;
  data->file_name = strdup(filename);

  /* First line is the number of atoms   */
  i = fscanf(data->file, "%d", natoms);
  if (i < 1) {
    fprintf(stderr, "\n\nread) ERROR: xyz file '%s' should have the number of atoms in the first line.\n", filename);
    return NULL;
  }
  data->numatoms=*natoms;

  rewind(fd);

  return data;
}

static int read_xyz_structure(void *mydata, int *optflags, 
                              molfile_atom_t *atoms) {
  int i, j;
  char *k;
  float coord;
  molfile_atom_t *atom;
  xyzdata *data = (xyzdata *)mydata;
  char buffer[1024], fbuffer[1024];

  /* skip over the first two lines */
  if (NULL == fgets(fbuffer, 1024, data->file))  return MOLFILE_ERROR;
  if (NULL == fgets(fbuffer, 1024, data->file))  return MOLFILE_ERROR;

  /* we set atom mass and VDW radius from the PTE. */
  *optflags = MOLFILE_ATOMICNUMBER | MOLFILE_MASS | MOLFILE_RADIUS; 

  for(i=0; i<data->numatoms; i++) {
    k = fgets(fbuffer, 1024, data->file);
    atom = atoms + i;
    j=sscanf(fbuffer, "%s %f %f %f", buffer, &coord, &coord, &coord);
    if (k == NULL) {
      fprintf(stderr, "xyz structure) missing atom(s) in file '%s'\n", data->file_name);
      fprintf(stderr, "xyz structure) expecting '%d' atoms, found only '%d'\n", data->numatoms, i);
      return MOLFILE_ERROR;
    } else if (j < 4) {
      fprintf(stderr, "xyz structure) missing type or coordinate(s) in file '%s' for atom '%d'\n",
          data->file_name, i+1);
      return MOLFILE_ERROR;
    }

    /* handle the case if the first item is an ordinal number 
     * from the PTE */
    if (isdigit(buffer[0])) {
      int idx;
      idx = atoi(buffer);
      strncpy(atom->name, get_pte_label(idx), sizeof(atom->name));
      atom->atomicnumber = idx;
      atom->mass = get_pte_mass(idx);
      atom->radius = get_pte_vdw_radius(idx);
    } else {
      int idx;
      strncpy(atom->name, buffer, sizeof(atom->name));
      idx = get_pte_idx(buffer);
      atom->atomicnumber = idx;
      atom->mass = get_pte_mass(idx);
      atom->radius = get_pte_vdw_radius(idx);
    }
    strncpy(atom->type, atom->name, sizeof(atom->type));
    atom->resname[0] = '\0';
    atom->resid = 1;
    atom->chain[0] = '\0';
    atom->segid[0] = '\0';
    /* skip to the end of line */
  }

  rewind(data->file);
  return MOLFILE_SUCCESS;
}

static int read_xyz_timestep(void *mydata, int natoms, molfile_timestep_t *ts) {
  int i, j;
  char atom_name[1024], fbuffer[1024], *k;
  float x, y, z;
  
  xyzdata *data = (xyzdata *)mydata;
  
  /* skip over the first two lines */
  if (NULL == fgets(fbuffer, 1024, data->file))  return MOLFILE_ERROR;
  if (NULL == fgets(fbuffer, 1024, data->file))  return MOLFILE_ERROR;

  /* read the coordinates */
  for (i=0; i<natoms; i++) {
    k = fgets(fbuffer, 1024, data->file);

    /* Read in atom type, X, Y, Z, skipping any remaining data fields */
    j = sscanf(fbuffer, "%s %f %f %f", atom_name, &x, &y, &z);
    if (k == NULL) {
      return MOLFILE_ERROR;
    } else if (j < 4) {
      fprintf(stderr, "xyz timestep) missing type or coordinate(s) in file '%s' for atom '%d'\n",data->file_name,i+1);
      return MOLFILE_ERROR;
    } else if (j >= 4) {
      if (ts != NULL) { 
        /* only save coords if we're given a timestep pointer, */
        /* otherwise assume that VMD wants us to skip past it. */
        ts->coords[3*i  ] = x;
        ts->coords[3*i+1] = y;
        ts->coords[3*i+2] = z;
      }
    } else {
      break;
    }
  }
  
  return MOLFILE_SUCCESS;
}
    
static void close_xyz_read(void *mydata) {
  xyzdata *data = (xyzdata *)mydata;
  fclose(data->file);
  free(data->file_name);
  free(data);
}


static void *open_xyz_write(const char *filename, const char *filetype, 
                           int natoms) {
  FILE *fd;
  xyzdata *data;

  fd = fopen(filename, "w");
  if (!fd) { 
    fprintf(stderr, "Error) Unable to open xyz file %s for writing\n",
            filename);
    return NULL;
  }
  
  data = (xyzdata *)malloc(sizeof(xyzdata));
  data->numatoms = natoms;
  data->file = fd;
  data->file_name = strdup(filename);
  return data;
}

static int write_xyz_structure(void *mydata, int optflags, 
                               const molfile_atom_t *atoms) {
  xyzdata *data = (xyzdata *)mydata;
  data->atomlist = (molfile_atom_t *)malloc(data->numatoms*sizeof(molfile_atom_t));
  memcpy(data->atomlist, atoms, data->numatoms*sizeof(molfile_atom_t));
  return MOLFILE_SUCCESS;
}

static int write_xyz_timestep(void *mydata, const molfile_timestep_t *ts) {
  xyzdata *data = (xyzdata *)mydata; 
  const molfile_atom_t *atom;
  const float *pos;
  const char  *label;
  int i;

  fprintf(data->file, "%d\n", data->numatoms);
  fprintf(data->file, " generated by VMD\n");
  
  atom = data->atomlist;
  pos = ts->coords;
  
  for (i = 0; i < data->numatoms; ++i) {
    if (atom->atomicnumber > 0) {
       label=pte_label[atom->atomicnumber];
    } else {
       label=atom->name;
    }
    fprintf(data->file, " %-2s %15.6f %15.6f %15.6f\n", 
            label, pos[0], pos[1], pos[2]);
    ++atom; 
    pos += 3;
  }
  return MOLFILE_SUCCESS;
}


static void close_xyz_write(void *mydata) {
  xyzdata *data = (xyzdata *)mydata;
  fclose(data->file);
  free(data->atomlist);
  free(data->file_name);
  free(data);
}

/* registration stuff */
static molfile_plugin_t plugin;

VMDPLUGIN_API int VMDPLUGIN_init() {
  memset(&plugin, 0, sizeof(molfile_plugin_t));
  plugin.abiversion = vmdplugin_ABIVERSION;
  plugin.type = MOLFILE_PLUGIN_TYPE;
  plugin.name = "xyz";
  plugin.prettyname = "XYZ";
  plugin.author = "Mauricio Carrillo Tripp, John E. Stone, Axel Kohlmeyer";
  plugin.majorv = 1;
  plugin.minorv = 3;
  plugin.is_reentrant = VMDPLUGIN_THREADSAFE;
  plugin.filename_extension = "xyz,xmol";
  plugin.open_file_read = open_xyz_read;
  plugin.read_structure = read_xyz_structure;
  plugin.read_next_timestep = read_xyz_timestep;
  plugin.close_file_read = close_xyz_read;
  plugin.open_file_write = open_xyz_write;
  plugin.write_structure = write_xyz_structure;
  plugin.write_timestep = write_xyz_timestep;
  plugin.close_file_write = close_xyz_write;
  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_register(void *v, vmdplugin_register_cb cb) {
  (*cb)(v, (vmdplugin_t *)&plugin);
  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_fini() {
  return VMDPLUGIN_SUCCESS;
}


#ifdef TEST_PLUGIN

int main(int argc, char *argv[]) {
  molfile_timestep_t timestep;
  void *v;
  int natoms;
  int i, nsets, set;

  while (--argc) {
    ++argv;
    v = open_xyz_read(*argv, "xyz", &natoms);
    if (!v) {
      fprintf(stderr, "open_xyz_read failed for file %s\n", *argv);
      return 1;
    }
    fprintf(stderr, "open_xyz_read succeeded for file %s\n", *argv);
    fprintf(stderr, "number of atoms: %d\n", natoms);

    i = 0;
    timestep.coords = (float *)malloc(3*sizeof(float)*natoms);
    while (!read_xyz_timestep(v, natoms, &timestep)) {
      i++;
    }
    fprintf(stderr, "ended read_next_timestep on frame %d\n", i);

    close_xyz_read(v);
  }
  return 0;
}

#endif

