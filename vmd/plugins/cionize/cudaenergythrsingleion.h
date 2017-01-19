#ifndef CIONIZE_CUDAENERTHR
#define CIONIZE_CUDAENERTHR

#ifdef __cplusplus
extern "C" {
#endif

int calc_grid_energies_cuda_thr_singleion(float* atom, float* grideners, long int numplane, long int numcol, long int numpt, float gridspacing, unsigned char* excludepos, int maxnumprocs); 

#ifdef __cplusplus
}
#endif

#endif
