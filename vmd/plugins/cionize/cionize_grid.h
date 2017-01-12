#ifndef CIONIZE_GRID
#define CIONIZE_GRID

/* electrostatic grid information */
typedef struct {
  long int numplane, numcol, numpt;
  float minx, miny, minz;
  float maxx, maxy, maxz;
  float gridspacing;

  /* Array for energies at grid nodes */
  float* eners;
  /* Array for excluded grid positions; if it is != 0, the point is excluded */
  unsigned char *excludepos;
} cionize_grid;

#endif
