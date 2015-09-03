#ifndef MGPOT_CUDA_H
#define MGPOT_CUDA_H

#include "mgpot_defn.h"

/* macro to detect and report error from CUDA */
#undef CUERR
#define CUERR \
  do { \
    cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
      printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
      return FAIL; \
    } \
  } while (0)

/* constants for lattice cutoff kernel */
#define MAXLEVELS     28
#define SUBCUBESZ     64
#define LG_SUBCUBESZ   6

#ifdef __cplusplus
extern "C" {
#endif

  int mgpot_cuda_device_list(void);
  int mgpot_cuda_device_set(int devnum);

  /*
   * cuda kernels for short-range part
   */
  int mgpot_cuda_setup_shortrng(Mgpot *);
  int mgpot_cuda_cleanup_shortrng(Mgpot *);

  /* for large bin (JCC) kernels */
  int mgpot_cuda_setup_binlarge(Mgpot *mg);
  int mgpot_cuda_cleanup_binlarge(Mgpot *mg);

  int mgpot_cuda_binlarge_pre(Mgpot *mg);
  int mgpot_cuda_binlarge(Mgpot *mg);
  int mgpot_cuda_binlarge_post(Mgpot *mg);

  int mgpot_cuda_largebin(Mgpot *mg, const float *atoms, float *grideners,
      long int numplane, long int numcol, long int numpt, long int natoms);

  /* for small bin (CF) kernels */
  int gpu_compute_cutoff_potential_lattice10overlap(
      float *lattice,                    /* the lattice */
      int nx, int ny, int nz,            /* its dimensions, length nx*ny*nz */
      float xlo, float ylo, float zlo,   /* lowest corner of lattice */
      float h,                           /* lattice spacing */
      float cutoff,                      /* cutoff distance */
      Atom *atom,                        /* array of atoms */
      int natoms,                        /* number of atoms */
      int verbose                        /* print info/debug messages */
      );

  /*
   * cuda kernels for long-range part
   */
  int mgpot_cuda_setup_longrng(Mgpot *mg);
  int mgpot_cuda_cleanup_longrng(Mgpot *mg);

  /* for lattice cutoff kernels */
  int mgpot_cuda_setup_latcut(Mgpot *mg);
  int mgpot_cuda_cleanup_latcut(Mgpot *mg);

  int mgpot_cuda_condense_qgrids(Mgpot *mg);
  int mgpot_cuda_expand_egrids(Mgpot *mg);

  int mgpot_cuda_latcut01(Mgpot *mg);
  int mgpot_cuda_latcut02(Mgpot *mg);
  int mgpot_cuda_latcut03(Mgpot *mg);
  int mgpot_cuda_latcut04(Mgpot *mg);


#ifdef __cplusplus
}
#endif

#endif /* MGPOT_CUDA_H */
