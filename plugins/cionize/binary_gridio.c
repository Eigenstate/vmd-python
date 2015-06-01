/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * gridio.c - read and write binary data file of rectangular grid of floats
 *
 * File stored as:
 *   4 * int32:  1, nx, ny, nz
 *   (nx*ny*nz) * float:  data
 *
 * Leading value 1 is used for endianess check; int32 is 4-byte int.
 * Grid is treated as flat array, meaning that no assumptions are made
 * on the order of the 3D indexing.
 */

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "binary_gridio.h"


#define ERRTOL  1e-5f


#if   INT_MAX == 2147483647
typedef int int32;
typedef unsigned int uint32;
enum {
  MAX_INT32 = INT_MAX,
  MIN_INT32 = INT_MIN
};
#elif INT_MAX == 32767 && LONG_MAX == 2147483647
typedef long int32;
typedef unsigned long uint32;
enum {
  MAX_INT32 = LONG_MAX,
  MAX_INT32 = LONG_MIN
};
#elif INT_MAX == 9223372036854775807L && SHRT_MAX == 2147483647
typedef short int32;
typedef unsigned short uint32;
enum {
  MAX_INT32 = SHRT_MAX,
  MAX_INT32 = SHRT_MIN
};
#endif


/*
 * performs in-place byte reordering to correct for different endianism,
 * where x is a 4-byte quantity (either type int32 or float)
 */
#define REORDER(x) \
do { \
  unsigned char _tmp; \
  unsigned char *_p = (unsigned char *) &(x); \
  _tmp = _p[0];  _p[0] = _p[3];  _p[3] = _tmp; \
  _tmp = _p[1];  _p[1] = _p[2];  _p[2] = _tmp; \
} while (0)


int gridio_write(const char *name,
    float gridspacing, float xmin, float ymin, float zmin,
    long int nx, long int ny, long int nz, const float *g)
{
  int32 n[4];      /* 1/nx/ny/nz (use 1 as magic number for endianism) */
  float parm[4];   /* gridspacing/xmin/ymin/zmin */
  long int total;
  FILE *f;

  if (NULL==name || NULL==g || nx <= 0 || ny <= 0 || nz <= 0) {
    fprintf(stderr, "(%s, line %d): illegal parameters\n", __FILE__, __LINE__);
    return GRIDIO_FAIL;
  }
  n[0] = 1;
  n[1] = nx;
  n[2] = ny;
  n[3] = nz;
  total = nx*ny*nz;
  parm[0] = gridspacing;
  parm[1] = xmin;
  parm[2] = ymin;
  parm[3] = zmin;
  if (nx != n[1] || ny != n[2] || nz != n[3]
      || total < nx || total < ny || total < nz
      || total < nx*ny || total < ny*nz) {
    fprintf(stderr, "grid dimensions are too large for gridio_write()\n");
    return GRIDIO_FAIL;
  }
  if (NULL==(f = fopen(name, "wb"))) {
    fprintf(stderr, "unable to open file \"%s\" for binary writing\n", name);
    return GRIDIO_FAIL;
  }
  if (fwrite(n, sizeof(int32), 4, f) != 4) {
    fprintf(stderr, "unable to write %ld bytes of grid dimensions "
        "to file \"%s\"\n", 4L*sizeof(int32), name);
    fclose(f);
    return GRIDIO_FAIL;
  }
  if (fwrite(parm, sizeof(float), 4, f) != 4) {
    fprintf(stderr, "unable to write %ld bytes of grid dimensions "
        "to file \"%s\"\n", 4L*sizeof(float), name);
    fclose(f);
    return GRIDIO_FAIL;
  }
  if (fwrite(g, sizeof(float), total, f) != total) {
    fprintf(stderr, "unable to write %ld bytes of grid data "
        "to file \"%s\"\n", total*sizeof(float), name);
    fclose(f);
    return GRIDIO_FAIL;
  }
  if (fclose(f)) {
    fprintf(stderr, "unable to close file \"%s\" after writing\n", name);
    return GRIDIO_FAIL;
  }
  return 0;
}


int gridio_read(const char *name,
    float gridspacing, float xmin, float ymin, float zmin,
    long int nx, long int ny, long int nz, float *g)
{
  int32 n[4];      /* 1/nx/ny/nz (use 1 as magic number for endianism) */
  float parm[4];   /* gridspacing/xmin/ymin/zmin */
  long int total;
  FILE *f;
  int reorderbytes = 0;
  unsigned char c;

  if (NULL==name || NULL==g || nx <= 0 || ny <= 0 || nz <= 0) {
    fprintf(stderr, "(%s, line %d): illegal parameters\n", __FILE__, __LINE__);
    return GRIDIO_FAIL;
  }
  if (NULL==(f = fopen(name, "rb"))) {
    fprintf(stderr, "unable to open file \"%s\" for binary reading\n", name);
    return GRIDIO_FAIL;
  }
  if (fread(n, sizeof(int32), 4, f) != 4) {
    fprintf(stderr, "unable to read %ld bytes of grid dimensions "
        "from file \"%s\"\n", 4L*sizeof(int32), name);
    fclose(f);
    return GRIDIO_FAIL;
  }
  if (fread(parm, sizeof(float), 4, f) != 4) {
    fprintf(stderr, "unable to read %ld bytes of grid dimensions "
        "from file \"%s\"\n", 4L*sizeof(float), name);
    fclose(f);
    return GRIDIO_FAIL;
  }

  /* check to see if byte reordering is necessary */
  if (n[0] != 1) {
    REORDER(n[0]);
    if (n[0] != 1) {
      fprintf(stderr, "cannot comprehend contents of file \"%s\"\n", name);
      fclose(f);
      return GRIDIO_FAIL;
    }
    reorderbytes = 1;
    REORDER(n[1]);
    REORDER(n[2]);
    REORDER(n[3]);
    REORDER(parm[0]);
    REORDER(parm[1]);
    REORDER(parm[2]);
    REORDER(parm[3]);
  }

  /* check to see if parameters match */
  if (n[1] != nx || n[2] != ny || n[3] != nz
      || parm[0] <= 0.f
      || (parm[0] < 1.f ? fabsf(parm[0] - gridspacing) > ERRTOL :
        fabsf(parm[0] - gridspacing) / parm[0] > ERRTOL)
      || (parm[1] < 1.f ? fabsf(parm[1] - xmin) > ERRTOL :
        fabsf(parm[1] - xmin) / fabsf(parm[1]) > ERRTOL)
      || (parm[2] < 1.f ? fabsf(parm[2] - ymin) > ERRTOL :
        fabsf(parm[2] - ymin) / fabsf(parm[2]) > ERRTOL)
      || (parm[3] < 1.f ? fabsf(parm[3] - zmin) > ERRTOL :
        fabsf(parm[3] - zmin) / fabsf(parm[3]) > ERRTOL) ) {
    fprintf(stderr, "unexpected parameters in file \"%s\"\n", name);
    fclose(f);
    return GRIDIO_FAIL;
  }

  total = nx*ny*nz;
  if (total < nx || total < ny || total < nz
      || total < nx*ny || total < ny*nz) {
    fprintf(stderr, "grid dimensions are too large for gridio_read()\n");
    fclose(f);
    return GRIDIO_FAIL;
  }

  if (fread(g, sizeof(float), total, f) != total) {
    fprintf(stderr, "unable to read %ld bytes of grid data "
        "from file \"%s\"\n", total*sizeof(float), name);
    fclose(f);
    return GRIDIO_FAIL;
  }
  if (fread(&c, sizeof(unsigned char), 1, f) == 1) {
    fprintf(stderr, "extra data found in file \"%s\"\n", name);
    fclose(f);
    return GRIDIO_FAIL;
  }
  else if (!feof(f)) {
    fprintf(stderr, "unable to find end of file \"%s\"\n", name);
    fclose(f);
    return GRIDIO_FAIL;
  }

  /* reorder bytes if necessary */
  if (reorderbytes) {
    long int i;
    for (i = 0;  i < total;  i++) {
      REORDER(g[i]);
    }
  }

  if (fclose(f)) {
    fprintf(stderr, "unable to close file \"%s\" after reading\n", name);
    return GRIDIO_FAIL;
  }

  return 0;
}


int gridio_read_unknown(const char *name,
    float *gridspacing, float *xmin, float *ymin, float *zmin,
    long int *nx, long int *ny, long int *nz, float **pg)
{
  int32 n[4];      /* 1/nx/ny/nz (use 1 as magic number for endianism) */
  float parm[4];   /* gridspacing/xmin/ymin/zmin */
  long int total;
  float *g;
  FILE *f;
  int reorderbytes = 0;
  unsigned char c;

  if (NULL==name || NULL==nx || NULL==ny || NULL==nz
      || NULL==gridspacing || NULL==xmin || NULL==ymin || NULL==zmin) {
    fprintf(stderr, "(%s, line %d): illegal parameters\n", __FILE__, __LINE__);
    return GRIDIO_FAIL;
  }
  if (NULL==(f = fopen(name, "rb"))) {
    fprintf(stderr, "unable to open file \"%s\" for binary reading\n", name);
    return GRIDIO_FAIL;
  }
  if (fread(n, sizeof(int32), 4, f) != 4) {
    fprintf(stderr, "unable to read %ld bytes of grid dimensions "
        "from file \"%s\"\n", 4L*sizeof(int32), name);
    fclose(f);
    return GRIDIO_FAIL;
  }
  if (fread(parm, sizeof(float), 4, f) != 4) {
    fprintf(stderr, "unable to read %ld bytes of grid dimensions "
        "from file \"%s\"\n", 4L*sizeof(float), name);
    fclose(f);
    return GRIDIO_FAIL;
  }

  /* check to see if byte reordering is necessary */
  if (n[0] != 1) {
    REORDER(n[0]);
    if (n[0] != 1) {
      fprintf(stderr, "cannot comprehend contents of file \"%s\"\n", name);
      fclose(f);
      return GRIDIO_FAIL;
    }
    reorderbytes = 1;
    REORDER(n[1]);
    REORDER(n[2]);
    REORDER(n[3]);
    REORDER(parm[0]);
    REORDER(parm[1]);
    REORDER(parm[2]);
    REORDER(parm[3]);
  }

  /* determine validity of grid dimensions */
  if (n[1] <= 0 || n[2] <= 0 || n[3] <= 0 || parm[0] <= 0.f) {
    fprintf(stderr, "illegal grid parameters in file \"%s\"\n", name);
    fclose(f);
    return GRIDIO_FAIL;
  }

  /* allocate memory for grid */
  if (NULL==(g = gridio_alloc(n[1], n[2], n[3]))) {
    fclose(f);
    return GRIDIO_FAIL;
  }
  total = n[1]*n[2]*n[3];
  if (total < n[1] || total < n[2] || total < n[3]
      || total < n[1]*n[2] || total < n[2]*n[3]) {
    fprintf(stderr,"grid dimensions are too large for gridio_read_unknown()\n");
    fclose(f);
    return GRIDIO_FAIL;
  }

  if (fread(g, sizeof(float), total, f) != total) {
    fprintf(stderr, "unable to read %ld bytes of grid data "
        "from file \"%s\"\n", total*sizeof(float), name);
    fclose(f);
    gridio_free(g);
    return GRIDIO_FAIL;
  }
  if (fread(&c, sizeof(unsigned char), 1, f) == 1) {
    fprintf(stderr, "extra data found in file \"%s\"\n", name);
    fclose(f);
    gridio_free(g);
    return GRIDIO_FAIL;
  }
  else if (!feof(f)) {
    fprintf(stderr, "unable to find end of file \"%s\"\n", name);
    fclose(f);
    gridio_free(g);
    return GRIDIO_FAIL;
  }

  /* reorder bytes if necessary */
  if (reorderbytes) {
    long int i;
    for (i = 0;  i < total;  i++) {
      REORDER(g[i]);
    }
  }

  *nx = n[1];
  *ny = n[2];
  *nz = n[3];
  *gridspacing = parm[0];
  *xmin = parm[1];
  *ymin = parm[2];
  *zmin = parm[3];
  *pg = g;

  if (fclose(f)) {
    fprintf(stderr, "unable to close file \"%s\" after reading\n", name);
    gridio_free(g);
    return GRIDIO_FAIL;
  }

  return 0;
}


float *gridio_alloc(long int nx, long int ny, long int nz)
{
  long int total;
  float *g;

  if (nx <= 0 || ny <= 0 || nz <= 0) {
    fprintf(stderr, "(%s, line %d): illegal parameters\n", __FILE__, __LINE__);
    return NULL;
  }
  total = nx*ny*nz;
  if (total < nx || total < ny || total < nz
      || total < nx*ny || total < ny*nz) {
    fprintf(stderr, "grid dimensions are too large for gridio_alloc()\n");
    return NULL;
  }
  if (NULL==(g = (float *) calloc(total, sizeof(float)))) {
    fprintf(stderr, "unable to calloc() space for %ld floats\n", total);
    return NULL;
  }
  return g;
}


void gridio_free(float *g)
{
  free(g);
}
