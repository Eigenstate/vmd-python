/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: vaspparchgplugin.c,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $       $Date: 2014/10/10 14:41:01 $
 *
 ***************************************************************************/


/*
 *  VASP plugins for VMD
 *  Sung Sakong, Dept. of Phys., Univsity Duisburg-Essen
 *  
 *  VASP manual   
 *  http://cms.mpi.univie.ac.at/vasp/
 * 
 *  LINUX
 *  g++ -O2 -Wall -I. -I$VMDBASEDIR/plugins/include -c vaspparchgplugin.c
 *  ld -shared -o vaspparchgplugin.so vaspparchgplugin.o
 *
 *  MACOSX
 *  c++ -O2 -Wall -I. -I$VMDBASEDIR/plugins/include -c vaspparchgplugin.c
 *  c++ -bundle -o vaspparchgplugin.so vaspparchgplugin.o
 *
 *  Install
 *  copy vaspparchgplugin.so $VMDBASEDIR/plugins/$ARCH/molfile
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "molfile_plugin.h"
#include "vaspplugin.h"


static void *open_vaspparchg_read(const char *filename, const char *filetype, int *natoms)
{
  vasp_plugindata_t *data;
  char lineptr[LINESIZE];
  float lc;
  int i;

  /* Verify that input is OK */
  if (!filename || !natoms) return NULL;

  /* Start with undefined value; set it after successful read */
  *natoms = MOLFILE_NUMATOMS_UNKNOWN;

  data = vasp_plugindata_malloc();
  if (!data) return NULL;

  /* VASP4 is assumed in default */
  data->version = 4;
  data->file = fopen(filename, "rb");
  if (!data->file) {
    vasp_plugindata_free(data);
    return NULL;
  }

  data->filename = strdup(filename);

  /* Read system title */
  fgets(lineptr, LINESIZE, data->file);
  data->titleline = strdup(lineptr);

  /* Read lattice constant */
  fgets(lineptr, LINESIZE, data->file);
  lc = atof(strtok(lineptr, " "));

  /* Read unit cell lattice vectors and multiply by lattice constant */
  for(i = 0; i < 3; ++i) {
    float x, y, z;
    fgets(lineptr, LINESIZE, data->file);
    sscanf(lineptr, "%f %f %f", &x, &y, &z);
    data->cell[i][0] = x*lc;
    data->cell[i][1] = y*lc;
    data->cell[i][2] = z*lc;
  }

  /* Build rotation matrix */
  vasp_buildrotmat(data);

  /* Count number of atoms */
  fgets(lineptr, LINESIZE, data->file);
  data->numatoms = 0;
  for (i = 0; i < MAXATOMTYPES; ++i) {
    char const *tmplineptr = strdup(lineptr);
    char const *token = (i == 0 ? strtok(lineptr, " ") : strtok(NULL, " "));
    int const n = (token ? atoi(token) : -1);

    /* if fails to read number of atoms, then assume VASP5 */
    if (i == 0 && n <= 0) {
      data->version = 5;
      data->titleline =  strdup(tmplineptr);
      fgets(lineptr, LINESIZE, data->file);
      break;
    }else if (n <= 0) break;

    data->eachatom[i] = n;
    data->numatoms += n;
  }

  if (data->version == 5) {
    data->numatoms = 0;
    for (i = 0; i < MAXATOMTYPES; ++i) {
      char const *token = (i == 0 ? strtok(lineptr, " ") : strtok(NULL, " "));
      int const n = (token ? atoi(token) : -1);
      
      if (n <= 0) break;
      
      data->eachatom[i] = n;
      data->numatoms += n;
    }
  }

  if (data->numatoms == 0) {
    vasp_plugindata_free(data);
    fprintf(stderr, "\n\nVASP PARCHG read) ERROR: file '%s' does not contain list of atom numbers.\n", filename);
    return NULL;
  }

  /* Skip lines up to the grid numbers */
  for (i = 0; i < data->numatoms + 2; ++i) fgets(lineptr, LINESIZE, data->file);

  *natoms = data->numatoms;

  return data;
}


static int read_vaspparchg_metadata(void *mydata, int *nvolsets, molfile_volumetric_t **metadata)
{
  vasp_plugindata_t *data = (vasp_plugindata_t *)mydata;
  char lineptr[LINESIZE];
  int gridx, gridy, gridz, i;
  char const spintext[4][20] = { "spin up+down", "spin up-down", "spin up", "spin down" };

  /* Verify that input is OK */
  if (!data || !nvolsets || !metadata) return MOLFILE_ERROR;

  /* Read the grid size */
  fgets(lineptr, LINESIZE, data->file);
  if (3 != sscanf(lineptr, "%d %d %d", &gridx, &gridy, &gridz)) {
     fprintf(stderr, "\n\nVASP PARCHG read) ERROR: file '%s' does not contain grid dimensions.\n", data->filename);
     return MOLFILE_ERROR;
  }

  fprintf(stderr, "\n\nVASP PARCHG read) found grid data block...\n");

  /* Initialize the volume set list with 4 entries:
   * spin up+down : always present
   * spin up-down / spin up /spin down : only there for spin-polarized calculations
   *                (the latter remain empty for non-spin-polarized calculations)
   */
  data->nvolsets = 4;
  data->vol = (molfile_volumetric_t *)malloc(data->nvolsets * sizeof(molfile_volumetric_t));
  if (!data->vol) {
     fprintf(stderr, "\n\nVASP PARCHG read) ERROR: Cannot allocate space for volume data.\n");
     return MOLFILE_ERROR;
  }

  for (i = 0; i < data->nvolsets; ++i) {
    molfile_volumetric_t *const set = &(data->vol[i]); /* get a handle to the current volume set meta data */
    int k;

    set->has_color = 0;

    /* put volume data name */
    sprintf(set->dataname, "Charge density (%s)", spintext[i]);

    set->origin[0] = set->origin[1] = set->origin[2] = 0;
    set->xsize = gridx + 1;
    set->ysize = gridy + 1;
    set->zsize = gridz + 1;

    /* Rotate unit cell vectors */
    for (k = 0; k < 3; ++k) {
      set->xaxis[k] = data->rotmat[k][0] * data->cell[0][0]
		+ data->rotmat[k][1] * data->cell[0][1]
		+ data->rotmat[k][2] * data->cell[0][2];
      
      set->yaxis[k] = data->rotmat[k][0] * data->cell[1][0] 
		+ data->rotmat[k][1] * data->cell[1][1]
		+ data->rotmat[k][2] * data->cell[1][2];
      
      set->zaxis[k] = data->rotmat[k][0] * data->cell[2][0] 
		+ data->rotmat[k][1] * data->cell[2][1]
		+ data->rotmat[k][2] * data->cell[2][2];
    }
  }

  *nvolsets = data->nvolsets;
  *metadata = data->vol;  

  return MOLFILE_SUCCESS;
}


static int read_vaspparchg_data(void *mydata, int set, float *datablock, float *colorblock)
{
  vasp_plugindata_t *data = (vasp_plugindata_t *)mydata;
  char lineptr[LINESIZE];
  int chargedensity, error, iset, n;
  float volume;

  /* Verify that input is OK */
  if (!data || !datablock) return MOLFILE_ERROR;
  if (set >= data->nvolsets) return MOLFILE_ERROR;

  if (strstr(data->filename, "LOCPOT") == NULL && strstr(data->filename, "ELFCAR") == NULL) {
    chargedensity = 1;
    fprintf(stderr, "\nVASP PARCHG read) Charge density is assumed. Each value will be divided by unit cell volume.\n");
  } else {
    if (set == 1) {
      fprintf(stderr, "\n\nVASP PARCHG read) ERROR: ELF or local potential do not include spin difference information.\n");
      return MOLFILE_ERROR;
    }
    chargedensity = 0;
    fprintf(stderr, "\nVASP PARCHG read) ELF or local potential is assumed.\n");
  }

  volume = fabs(
            data->cell[0][0]*(data->cell[1][1]*data->cell[2][2]-data->cell[1][2]*data->cell[2][1])
	  + data->cell[0][1]*(data->cell[1][2]*data->cell[2][0]-data->cell[1][0]*data->cell[2][2])
	  + data->cell[0][2]*(data->cell[1][0]*data->cell[2][1]-data->cell[2][0]*data->cell[1][1])
               );

  /* Set file pointer to beginning of file and then skip header up to density data */
  rewind(data->file);
  for (n = 0; n < data->numatoms + data->version + 5; ++n) fgets(lineptr, LINESIZE, data->file);

  for(error = iset = 0; iset <= set && iset < 2 && !error; ++iset) {
    char const *dataname = data->vol[iset].dataname;
    int const xsize = data->vol[iset].xsize; 
    int const ysize = data->vol[iset].ysize;
    int const zsize = data->vol[iset].zsize;
    int const numberOfDatapoints = (xsize - 1) * (ysize - 1) * (zsize - 1);
    int ix, iy, iz;

    fprintf(stderr, "\nVASP PARCHG read) Patience! Reading volume set %d (%d points): %s\n", iset + 1, numberOfDatapoints, dataname);

    for (n = iz = 0; iz < zsize; ++iz) {
      for (iy = 0; iy < ysize; ++iy) {
        for (ix = 0; ix < xsize; ++ix, ++n) {
          float value;
	  if (ix == xsize - 1) value = datablock[n - ix];
	  else if (iy == ysize - 1) value = datablock[n - iy*xsize];
	  else if (iz == zsize - 1) value = datablock[n - iz*ysize*xsize];
	  else {
	    if (1 != fscanf(data->file, "%f", &value)) return MOLFILE_ERROR;
	    if (chargedensity) value /= volume;
	  }

	  /* for set == 2: spin-up   = 0.5 * set0 + 0.5 * set1
	   * for set == 3: spin-down = 0.5 * set0 - 0.5 * set1 */
	  if (iset == set) datablock[n] = value;
	  else if (set >= 2 && iset == 0) datablock[n] = 0.5 * value;
	  else if (set == 2 && iset == 1) datablock[n] += 0.5 * value;
	  else if (set == 3 && iset == 1) datablock[n] -= 0.5 * value;
        }
      }
    }

    if(iset == 0){
      for (iy = 0; iy < 3; ++iy) {
	int ival;
	if (1 != fscanf(data->file, "%d", &ival)) error = 2;
      }
    }

    fprintf(stderr, "\nVASP PARCHG read) %s finished.\n", dataname);
  }

  if (error) fprintf(stderr, "\nVASP PARCHG read) PAW-augmentation part is incomplete, but it is ignored anyway.\n");

  return MOLFILE_SUCCESS;
}


static void close_vaspparchg_read(void *mydata)
{
  vasp_plugindata_t *data = (vasp_plugindata_t *)mydata;
  vasp_plugindata_free(data);
}


/* registration stuff */
static molfile_plugin_t plugin;

int VMDPLUGIN_init(void) {
  memset(&plugin, 0, sizeof(molfile_plugin_t));
  plugin.abiversion = vmdplugin_ABIVERSION;
  plugin.type = MOLFILE_PLUGIN_TYPE;
  plugin.name = "PARCHG";
  plugin.prettyname = "VASP_PARCHG";
  plugin.author = "Sung Sakong";
  plugin.majorv = 0;
  plugin.minorv = 7;
  plugin.is_reentrant = VMDPLUGIN_THREADUNSAFE;
  plugin.filename_extension = "PARCHG";
  plugin.open_file_read = open_vaspparchg_read;
  plugin.close_file_read = close_vaspparchg_read;
  plugin.read_volumetric_metadata = read_vaspparchg_metadata;
  plugin.read_volumetric_data = read_vaspparchg_data;
  return VMDPLUGIN_SUCCESS;
}

int VMDPLUGIN_register(void *v, vmdplugin_register_cb cb) {
  (*cb)(v, (vmdplugin_t *)&plugin);
  return VMDPLUGIN_SUCCESS;
}

int VMDPLUGIN_fini(void) {
  return VMDPLUGIN_SUCCESS;
}

