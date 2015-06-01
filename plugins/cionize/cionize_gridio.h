#ifndef CIONIZE_GRIDIO
#define CIONIZE_GRIDIO

/* Conversion factor between raw units (e^2 / A) and kT/e_c*/
#define POT_CONV 560.47254

/** Convert cionize addressing (x varies fastest) to dx addressing (z varies fastest) **/
int transaddr(const int, const int, const int, const int);

/* Write the current energy grid to a file */
int write_grid_dx(const char *filename, int usebinary, float gridspacing,
                  float cx, float cy, float cz,
                  int xsize, int ysize, int zsize,
                  float *datablock);

/* Write the current energy grid to a file */
int write_grid(const char*, float, float, float, float, long int, long int, long int, const float*);

/* Read an energy grid from a saved file */
int read_grid(const char*, float*, float, long int, long int, long int, float, float, float);

#endif
