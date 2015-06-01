#ifndef CIONIZE_ENERTHR
#define CIONIZE_ENERTHR

int calc_grid_energies(float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, long int natoms, float gridspacing, unsigned char* excludepos, int maxnumprocs, int calctype, float ddd); 

#endif
