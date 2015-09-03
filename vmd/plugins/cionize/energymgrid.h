#ifndef CIONIZE_ENERMGRID
#define CIONIZE_ENERMGRID

int calc_grid_energies_excl_mgrid(float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, long int natoms, float gridspacing, unsigned char* excludepos, int maxnumprocs); 

#endif
