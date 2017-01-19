/***************************************************************************
 *cr
 *cr            (C) Copyright 2007-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDAQuickSurf.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $      $Date: 2011/10/18 23:20:44 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "Inform.h"
#include "utilities.h"

#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "VolumetricData.h"
#include "VolMapCreate.h" // volmap_write_dx_file()

#include <tcl.h>
#include "TclCommands.h"

//#include "molfile_plugin.h"
//#include "largefiles.h"

//#define THISPLUGIN plugin
//#include "vmdconio.h"

#define LINESIZE 2040

/* File plugin symbolic constants for better code readability */
#define MOLFILE_SUCCESS           0   /**< succeeded in reading file      */
#define MOLFILE_EOF              -1   /**< end of file                    */
#define MOLFILE_ERROR            -1   /**< error reading/opening a file   */
#define MOLFILE_NOSTRUCTUREDATA  -2   /**< no structure data in this file */

#define MOLFILE_NUMATOMS_UNKNOWN -1   /**< unknown number of atoms       */
#define MOLFILE_NUMATOMS_NONE     0   /**< no atoms in this file type    */



/**
 * Metadata for volumetric datasets, read initially and used for subsequent
 * memory allocations and file loading.
 */
typedef struct {
  char dataname[256];   /**< name of volumetric data set                    */
  float origin[3];      /**< origin: origin of volume (x=0, y=0, z=0 corner */

  /*
   * x/y/z axis:
   * These the three cell sides, providing both direction and length
   * (not unit vectors) for the x, y, and z axes.  In the simplest
   * case, these would be <size,0,0> <0,size,0> and <0,0,size) for
   * an orthogonal cubic volume set.  For other cell shapes these
   * axes can be oriented non-orthogonally, and the parallelpiped
   * may have different side lengths, not just a cube/rhombus.
   */
  float xaxis[3];       /**< direction (and length) for X axis              */
  float yaxis[3];       /**< direction (and length) for Y axis              */
  float zaxis[3];       /**< direction (and length) for Z axis              */

  /*
   * x/y/z size:
   * Number of grid cells along each axis.  This is _not_ the
   * physical size of the box, this is the number of voxels in each
   * direction, independent of the shape of the volume set.
   */
  int xsize;            /**< number of grid cells along the X axis          */
  int ysize;            /**< number of grid cells along the Y axis          */
  int zsize;            /**< number of grid cells along the Z axis          */

  int has_color;        /**< flag indicating presence of voxel color data   */
} molfile_volumetric_t;



static int checkfileextension(const char *s, const char *extension) {
  int sz, extsz;
  sz = strlen(s);
  extsz = strlen(extension);

  if (extsz > sz) return 0;

  if (!strupncmp(s + (sz - extsz), extension, extsz)) return 1;

  return 0;
}


//
// bulk include of VTK reader, short-term hack
//

typedef struct {
  FILE *fd;
  int nsets;
  molfile_volumetric_t *vol;
  int isBinary; 
} vtk_t;


// Get a string from a stream, printing any errors that occur
static char *vtkgets(char *s, int n, FILE *stream) {
  char *returnVal;

  if (feof(stream)) {
    printf("vtkplugin) Unexpected end-of-file.\n");
    return NULL;
  } else if (ferror(stream)) {
    printf("vtkplugin) Error reading file.\n");
    return NULL;
  } else {
    returnVal = fgets(s, n, stream);
    if (returnVal == NULL) {
      printf("vtkplugin) Error reading line.\n");
    }
  }

  return returnVal;
}

static int vtkgetstrcmp(char *s, int n, FILE *stream, const char *cmpstr) {
  char *str = vtkgets(s, n, stream);
  int rc = strncmp(cmpstr, str, strlen(cmpstr)); 
  if (rc) {
    printf("vtkplugin) found '%s', expected '%s'\n", str, cmpstr);
  }
  return rc;
}
   

static vtk_t *open_vtk_read(const char *filepath, const char *filetype, int *natoms) {
  FILE *fd;
  vtk_t *vtk;
  char inbuf[LINESIZE];
  int xsize, ysize, zsize;
  float orig[3], xdelta[3], ydelta[3], zdelta[3];
  int isBinary = 0;
 
  memset(orig, 0, sizeof(orig));
  memset(xdelta, 0, sizeof(xdelta));
  memset(ydelta, 0, sizeof(ydelta));
  memset(zdelta, 0, sizeof(zdelta));
 
  fd = fopen(filepath, "rb");
  if (!fd) {
    printf("vtkplugin) Error opening file.\n");
    return NULL;
  }

  /* skip comments */
  do {
    if (vtkgets(inbuf, LINESIZE, fd) == NULL) 
      return NULL;
  } while (inbuf[0] == '#');
 
  if (sscanf(inbuf, "vtk output") != 0) {
    printf("vtkplugin) failed to find 'vtk output' string!\n");
    return NULL;
  }

  if (vtkgetstrcmp(inbuf, LINESIZE, fd, "ASCII")) return NULL;
  if (vtkgetstrcmp(inbuf, LINESIZE, fd, "DATASET STRUCTURED_POINTS")) return NULL;

  // get the grid dimensions
  if (vtkgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
  if (sscanf(inbuf, "DIMENSIONS %d %d %d", &xsize, &ysize, &zsize) != 3) {
    printf("vtkplugin) Error reading grid dimensions!\n");
    return NULL;
  }

  // get the grid spacing
  if (vtkgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
  if (sscanf(inbuf, "SPACING %e %e %e", xdelta, ydelta+1, zdelta+2) != 3) {
    printf("vtkplugin) Error reading cell dimensions!\n");
    return NULL;
  }

  // get the grid origin
  if (vtkgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
  if (sscanf(inbuf, "ORIGIN %e %e %e", orig, orig+1, orig+2) != 3) {
    printf("vtkplugin) Error reading grid origin!\n");
    return NULL;
  }

  // get number of grid points
  if (vtkgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
  int numgridpoints = 0;
  if (sscanf(inbuf, "POINT_DATA %d", &numgridpoints) != 1) {
    printf("vtkplugin) Error reading grid point counts!\n");
    return NULL;
  }

  if (vtkgetstrcmp(inbuf, LINESIZE, fd, "FIELD FieldData 1")) return NULL;

  // get grid info
  if (vtkgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
  printf("vtkplugin) loading grid '%s'\n", inbuf);

  /* allocate and initialize the vtk structure */
  vtk = new vtk_t;
  vtk->fd = fd;
  vtk->vol = NULL;
  vtk->isBinary = 0;
  *natoms = MOLFILE_NUMATOMS_NONE;
  vtk->nsets = 1; /* this file contains only one data set */

  vtk->vol = new molfile_volumetric_t[1];
  strcpy(vtk->vol[0].dataname, "DX map");

  /* Set the unit cell origin and basis vectors */
  for (int i=0; i<3; i++) {
    vtk->vol[0].origin[i] = orig[i];
    vtk->vol[0].xaxis[i] = xdelta[i] * ((xsize-1 > 0) ? (xsize-1) : 1);
    vtk->vol[0].yaxis[i] = ydelta[i] * ((ysize-1 > 0) ? (ysize-1) : 1);
    vtk->vol[0].zaxis[i] = zdelta[i] * ((zsize-1 > 0) ? (zsize-1) : 1);
  }

  vtk->vol[0].xsize = xsize;
  vtk->vol[0].ysize = ysize;
  vtk->vol[0].zsize = zsize;

  vtk->vol[0].has_color = 0; // no color data

  return vtk;
}


static int read_vtk_metadata(void *v, int *nsets, 
  molfile_volumetric_t **metadata) {
  vtk_t *vtk = (vtk_t *)v;
  *nsets = vtk->nsets; 
  *metadata = vtk->vol;  

  return MOLFILE_SUCCESS;
}


static int read_vtk_data(void *v, int set, float *datablock,
                         float *gradientblock) {
  vtk_t *vtk = (vtk_t *)v;
  FILE *fd = vtk->fd;
  char inbuf[LINESIZE];
  char *p;
  float grid;
  int x, y, z, xsize, ysize, zsize, xysize, count, total, i, line;

  if (vtk->isBinary)
    return MOLFILE_ERROR;

  xsize = vtk->vol[0].xsize;
  ysize = vtk->vol[0].ysize;
  zsize = vtk->vol[0].zsize;
  xysize = xsize * ysize;
  total = xysize * zsize;

  float maxmag = 0.0f;
  strcpy(vtk->vol[0].dataname, "volgradient");
  for (z=0; z<zsize; z++) {
    for (y=0; y<ysize; y++) {
      for (x=0; x<xsize; x++) {
        double vx, vy, vz;
        fscanf(fd, "%lf %lf %lf", &vx, &vy, &vz);

#if 1
        // scale gradient coordinates for nm to A conversion
        vx *= 1e8;
        vy *= 1e8; 
        vz *= 1e8;
#endif

        int addr = z*xsize*ysize + y*xsize + x;

        // compute scalar magnitude from vector field
        double mag = sqrt(vx*vx + vy*vy + vz*vz);
        datablock[addr] = mag;

#if 0
        if (mag > 0.0) {
          printf("vtkplugin) mag[%d]: %e\n", addr, mag);
        }
#endif
        if (mag > maxmag)
          maxmag = mag;

        // assign vector field to gradient map
        // index into vector field of 3-component vectors
        int addr3 = addr *= 3;
        gradientblock[addr3    ] = vx;
        gradientblock[addr3 + 1] = vy;
        gradientblock[addr3 + 2] = vz;
      }
    }
  }
  printf("vtkplugin) maxmag: %g\n", maxmag);

  return MOLFILE_SUCCESS;
}

static void close_vtk_read(void *v) {
  vtk_t *vtk = (vtk_t *)v;
  
  fclose(vtk->fd);
  if (vtk->vol != NULL)
    delete [] vtk->vol; 
  delete vtk;
}



//
// bulk include of DX reader, short-term hack
//


/* 
 * DX potential maps
 *
 * Format of the file is:
 * # Comments
 * .
 * .
 * .
 * object 1 class gridpositions counts xn yn zn
 * origin xorg yorg zorg
 * delta xdel 0 0
 * delta 0 ydel 0
 * delta 0 0 zdel
 * object 2 class gridconnections counts xn yn zn
 * object 3 class array type double rank 0 items { xn*yn*zn } [binary] data follows
 * f1 f2 f3
 * f4 f5 f6 f7 f8 f9
 * .
 * .
 * .
 * object "Dataset name" class field
 
 * Where xn, yn, and zn are the number of data points along each axis;
 * xorg, yorg, and zorg is the origin of the grid, assumed to be in angstroms;
 * xdel, ydel, and zdel are the scaling factors to convert grid units to
 * angstroms.
 *
 * Grid data follows, with a single or multiple values per line (maximum 
 * allowed linelength is hardcoded into the plugin with ~2000 chars), 
 * ordered z fast, y medium, and x slow.
 *
 * Note that the ordering of grid data in VMD's VolumetricData class
 * is transposed, i.e. x changes fastest and z slowest! 
 * The file reading and writing routines perform the transpose.
 *
 * If the optional keyword 'binary' is present, the data is expected to
 * be in binary, native endian, single precision IEEE-754 floating point format.
 *
 * Note: this plugin was written to read the OpenDX files created by the
 * APBS program, and thus supports only files that are writting in this style.
 * the full OpenDX data format is extremely powerful, complex, and flexible.
 *
 */


typedef struct {
  FILE *fd;
  int nsets;
  molfile_volumetric_t *vol;
  int isBinary; 
} dx_t;


// Get a string from a stream, printing any errors that occur
static char *dxgets(char *s, int n, FILE *stream) {
  char *returnVal;

  if (feof(stream)) {
    printf("dxplugin) Unexpected end-of-file.\n");
    return NULL;
  } else if (ferror(stream)) {
    printf("dxplugin) Error reading file.\n");
    return NULL;
  } else {
    returnVal = fgets(s, n, stream);
    if (returnVal == NULL) {
      printf("dxplugin) Error reading line.\n");
    }
  }

  return returnVal;
}


static dx_t *open_dx_read(const char *filepath, const char *filetype,
    int *natoms) {
  FILE *fd;
  dx_t *dx;
  char inbuf[LINESIZE];
  int xsize, ysize, zsize;
  float orig[3], xdelta[3], ydelta[3], zdelta[3];
  int isBinary = 0;
  
  fd = fopen(filepath, "rb");
  if (!fd) {
    printf("dxplugin) Error opening file.\n");
    return NULL;
  }

  /* skip comments */
  do {
    if (dxgets(inbuf, LINESIZE, fd) == NULL) 
      return NULL;
  } while (inbuf[0] == '#');

  /* get the number of grid points */
//  if (sscanf(inbuf, "object 1 class gridpositions counts %d %d %d", &xsize, &ysize, &zsize) != 3) {
  if (sscanf(inbuf, "object \"pos\" class gridpositions counts %d %d %d", &xsize, &ysize, &zsize) != 3) {
    printf("dxplugin) Error reading grid dimensions.\n");
    return NULL;
  }

  /* get the cell origin */
  if (dxgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
//  if (sscanf(inbuf, "origin %e %e %e", orig, orig+1, orig+2) != 3) {
  if (sscanf(inbuf, " origin %e %e %e", orig, orig+1, orig+2) != 3) {
    printf("dxplugin) Error reading grid origin.\n");
    return NULL;
  }

  /* get the cell dimensions */
  if (dxgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
//  if (sscanf(inbuf, "delta %e %e %e", xdelta, xdelta+1, xdelta+2) != 3) {
  if (sscanf(inbuf, " delta %e %e %e", xdelta, xdelta+1, xdelta+2) != 3) {
    printf("dxplugin) Error reading cell x-dimension.\n");
    return NULL;
  }

  if (dxgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
//  if (sscanf(inbuf, "delta %e %e %e", ydelta, ydelta+1, ydelta+2) != 3) {
  if (sscanf(inbuf, " delta %e %e %e", ydelta, ydelta+1, ydelta+2) != 3) {
    printf("dxplugin) Error reading cell y-dimension.\n");
    return NULL;
  }

  if (dxgets(inbuf, LINESIZE, fd) == NULL) {
    return NULL;
  }
//  if (sscanf(inbuf, "delta %e %e %e", zdelta, zdelta+1, zdelta+2) != 3) {
  if (sscanf(inbuf, " delta %e %e %e", zdelta, zdelta+1, zdelta+2) != 3) {
    printf("dxplugin) Error reading cell z-dimension.\n");
    return NULL;
  }

#if 1
  for (int l=0; l<19; l++)
    dxgets(inbuf, LINESIZE, fd);
//  printf("inbuf: '%s'\n", inbuf);
#else
  /* skip the next line of the header (described at the beginning of
   * the code), which aren't utilized by APBS-produced DX files.  */
  if (dxgets(inbuf, LINESIZE, fd) == NULL) 
    return NULL;
  /* The next line tells us whether to expect ascii or binary format.
   * We scan for the word 'binary' somewhere in the line, and if it's found
   * we assume binary.
   */
  if (dxgets(inbuf, LINESIZE, fd) == NULL)
    return NULL;
  if (strstr(inbuf, "binary")) {
      isBinary = 1;
  }
#endif

 
  /* allocate and initialize the dx structure */
  dx = new dx_t;
  dx->fd = fd;
  dx->vol = NULL;
  dx->isBinary = isBinary;
  *natoms = MOLFILE_NUMATOMS_NONE;
  dx->nsets = 1; /* this file contains only one data set */

  dx->vol = new molfile_volumetric_t[1];
  strcpy(dx->vol[0].dataname, "DX map");

  /* Set the unit cell origin and basis vectors */
  for (int i=0; i<3; i++) {
    dx->vol[0].origin[i] = orig[i];
    dx->vol[0].xaxis[i] = xdelta[i] * ((xsize-1 > 0) ? (xsize-1) : 1);
    dx->vol[0].yaxis[i] = ydelta[i] * ((ysize-1 > 0) ? (ysize-1) : 1);
    dx->vol[0].zaxis[i] = zdelta[i] * ((zsize-1 > 0) ? (zsize-1) : 1);
  }

  dx->vol[0].xsize = xsize;
  dx->vol[0].ysize = ysize;
  dx->vol[0].zsize = zsize;

  /* DX files contain no color information. Taken from edmplugin.C */
  dx->vol[0].has_color = 0;

  return dx;
}


static int read_dx_metadata(void *v, int *nsets, 
  molfile_volumetric_t **metadata) {
  dx_t *dx = (dx_t *)v;
  *nsets = dx->nsets; 
  *metadata = dx->vol;  

  return MOLFILE_SUCCESS;
}


static int read_dx_data(void *v, int set, float *datablock,
                         float *gradientblock) {
  dx_t *dx = (dx_t *)v;
  FILE *fd = dx->fd;
  char inbuf[LINESIZE];
  char *p;
  float grid;
  int x, y, z, xsize, ysize, zsize, xysize, count, total, i, line;

  if (dx->isBinary)
#if 1
    return MOLFILE_ERROR;
#else
    return read_binary_dx_data(dx, set, datablock);
#endif

  xsize = dx->vol[0].xsize;
  ysize = dx->vol[0].ysize;
  zsize = dx->vol[0].zsize;
  xysize = xsize * ysize;
  total = xysize * zsize;

#if 1
  strcpy(dx->vol[0].dataname, "volgradient");
  for (x=0; x<xsize; x++) {
    for (y=0; y<ysize; y++) {
      for (z=0; z<zsize; z++) {
        float vx, vy, vz;
        fscanf(fd, "%f %f %f", &vx, &vy, &vz);

        int addr = z*xsize*ysize + y*xsize + x;

        // compute scalar magnitude from vector field
        datablock[addr] = sqrtf(vx*vx + vy*vy + vz*vz);

        // assign vector field to gradient map
        // index into vector field of 3-component vectors
        int addr3 = addr *= 3;
        gradientblock[addr3    ] = vx;
        gradientblock[addr3 + 1] = vy;
        gradientblock[addr3 + 2] = vz;
      }
    }
  }
#elif 0
  for (count = 0; count < total; count++) {
    float vx, vy, vz;
    fscanf(fd, "%f %f %f", &vx, &vy, &vz);

    // compute scalar magnitude from vector field
    datablock[count] = sqrtf(vx*vx + vy*vy + vz*vz);

    // assign vector field to gradient map
    gradientblock[count*3L    ] = vx;
    gradientblock[count*3L + 1] = vy;
    gradientblock[count*3L + 2] = vz;
  }
#else
  /* Read the values from the file */
  x = y = z = line = 0;
  for (count = 0; count < total;) {
    ++line;
    p=dxgets(inbuf, LINESIZE, fd);
    if (p == NULL) {
      printf("dxplugin) Error reading grid data.\n");
      printf("dxplugin) line: %d. item: %d/%d. last data: %s\n", 
              line, count, total, inbuf);
      return MOLFILE_ERROR;
    }

    // chop line into whitespace separated tokens and parse them one by one.
    while (*p != '\n' && *p != '\0') {

      // skip over whitespace and try to parse non-blank as number
      while (*p != '\0' && (*p == ' ' || *p == '\t' || *p == '\n')) ++p;
      i = sscanf(p, "%e", &grid);
      if (i < 0) break; // end of line/string. get a new one.
      
      // a 0 return value means non-parsable as number.
      if (i == 0) {
        printf("dxplugin) Error parsing grid data.\n");
        printf("dxplugin) line: %d. item: %d/%d. data %s\n", 
                line, count, total, p);
        return MOLFILE_ERROR;
      }

      // success! add to dataset.
      if (i == 1) {
        ++count;
        datablock[x + y*xsize + z*xysize] = grid;
        z++;
        if (z >= zsize) {
          z = 0; y++;
          if (y >= ysize) {
            y = 0; x++;
          }
        }
      }

      // skip over the parsed text and search for next blank.
      while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\n') ++p;
    }
  }
  
  char dxname[256];
  while (dxgets(inbuf, LINESIZE, dx->fd)) {
    if (sscanf(inbuf, "object \"%[^\"]\" class field", dxname) == 1) {
      // a dataset name has been found; override the default
      strcpy(dx->vol[0].dataname, dxname);
      break;
    }
  }
#endif

  return MOLFILE_SUCCESS;
}

static void close_dx_read(void *v) {
  dx_t *dx = (dx_t *)v;
  
  fclose(dx->fd);
  if (dx->vol != NULL)
    delete [] dx->vol; 
  delete dx;
}






int obj_volgradient(ClientData cd, Tcl_Interp *interp, int argc,
                    Tcl_Obj * const objv[]) {
  if (argc < 2) {
    Tcl_SetResult(interp, (char *) "Usage: volgradient <command> [args...]\n", TCL_STATIC);
    return TCL_ERROR;
  }
  VMDApp *app = (VMDApp *)cd;
  MoleculeList *mlist = app->moleculeList;
  VolumetricData *volmapA = NULL;
  int molid = -1;
  int volid = 0;

#if 0
  char *argv1 = Tcl_GetStringFromObj(objv[1], NULL);

  if (!strupncmp(argv1, "sim", CMDLEN))
    return mdff_sim(app, argc-1, objv+1, interp);
#endif

  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);

    if (!strcmp(opt, "-mol")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
  }

  char *filename = Tcl_GetStringFromObj(objv[argc-1], NULL);
  printf("Attempting to load volgradient filename: %s\n", filename);

  if (molid != -1) {
    Molecule *volmol = mlist->mol_from_id(molid);

// XXX serious hack...
// XXX serious hack...
// XXX serious hack...
    if (volmapA == NULL) volmapA = (VolumetricData *)volmol->get_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }

  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }

  printf("Checking volmap for storage...\n");
  if (!volmapA) 
    printf("No existing volmap\n");

  if (!volmapA->data)
    printf("No volmap data array\n");

  volmapA->compute_volume_gradient();

  if (!volmapA->gradient)
    printf("No volmap gradient array\n");

  // load gradient from file...
  int natoms=0;
  if (checkfileextension(filename, ".dx")) {
    printf("Loading gradient file '%s' with DX code\n", filename);

    dx_t *dxh = NULL;
    dxh = open_dx_read(filename, ".dx", &natoms);
    if (!dxh) {
      printf("Failed to load gradient file '%s'\n", filename);
    } 
  
    read_dx_data(dxh, 0, volmapA->data, volmapA->gradient);

    // hack volmap to incorporate new data
    for (int j=0; j<3; j++) {
      volmapA->origin[j] = dxh->vol->origin[j];
      volmapA->xaxis[j]  = dxh->vol->xaxis[j];
      volmapA->yaxis[j]  = dxh->vol->yaxis[j];
      volmapA->zaxis[j]  = dxh->vol->zaxis[j];
    }
    // recompute volume min/max data
    // use fast 16-byte-aligned min/max routine
    minmax_1fv_aligned(volmapA->data, 
                       dxh->vol->xsize * dxh->vol->ysize * dxh->vol->zsize,
                       &volmapA->datamin, &volmapA->datamax);
  } else if (checkfileextension(filename, ".vtk")) {
    printf("Loading gradient file '%s' with VTK code\n", filename);

    vtk_t *vtkh = NULL;
    vtkh = open_vtk_read(filename, ".vtk", &natoms);
    if (!vtkh) {
      printf("Failed to load gradient file '%s'\n", filename);
    } 

    read_vtk_data(vtkh, 0, volmapA->data, volmapA->gradient);
        
    // hack volmap to incorporate new data
    for (int j=0; j<3; j++) {
      volmapA->origin[j] = vtkh->vol->origin[j];
      volmapA->xaxis[j]  = vtkh->vol->xaxis[j];
      volmapA->yaxis[j]  = vtkh->vol->yaxis[j];
      volmapA->zaxis[j]  = vtkh->vol->zaxis[j];
    }
    // recompute volume min/max data
    // use fast 16-byte-aligned min/max routine
    minmax_1fv_aligned(volmapA->data, 
                       vtkh->vol->xsize * vtkh->vol->ysize * vtkh->vol->zsize,
                       &volmapA->datamin, &volmapA->datamax);
  }


  Tcl_SetResult(interp, (char *) "Type 'volgradient' for summary of usage\n", TCL_VOLATILE);
  return TCL_OK;
}













