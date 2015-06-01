/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * binary_gridio.h - read and write binary data file of
 *   rectangular grid of floats
 *
 * Grid memory allocation and deallocation routines also provided.
 * Read routine performs grid allocation and corrects for different endianism.
 */

#ifndef GRIDIO_H
#define GRIDIO_H

#define GRIDIO_FAIL  (-1)

#ifdef __cplusplus
extern "C" {
#endif

  /* write grid to file "name" */
  int gridio_write(const char *name,
      float gridspacing, float xmin, float ymin, float zmin,
      long int nx, long int ny, long int nz, const float *g);

  /*
   * read known grid from file "name"
   * return error if expected params don't match
   */
  int gridio_read(const char *name,
      float gridspacing, float xmin, float ymin, float zmin,
      long int nx, long int ny, long int nz, float *g);

  /*
   * read unknown grid with its dimensions and params from file "name"
   * grid buffer allocated using gridio_alloc()
   */
  int gridio_read_unknown(const char *name,
      float *gridspacing, float *xmin, float *ymin, float *zmin,
      long int *nx, long int *ny, long int *nz, float **pg);

  /* allocate grid buffer of dimensions nx*ny*nz */
  float *gridio_alloc(long int nx, long int ny, long int nz);

  /* free grid buffer allocated by gridio_alloc() */
  void gridio_free(float *g);

#ifdef __cplusplus
}
#endif

#endif /* GRIDIO_H */
