#ifndef CIONIZE_CUDAENERTHR
#define CIONIZE_CUDAENERTHR

#ifdef __cplusplus
extern "C" {
#endif

int calc_grid_energies_cuda_thr(float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, long int natoms, float gridspacing, unsigned char* excludepos, int maxnumprocs); 


#ifdef __cplusplus
}
#endif

#endif
