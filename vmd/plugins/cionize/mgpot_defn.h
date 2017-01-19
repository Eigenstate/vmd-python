/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_defn.h - definitions for main mgpot data structure
 */

#ifndef MGPOT_DEFN_H
#define MGPOT_DEFN_H

#include "mgpot_lattice.h"
#include "mgpot_split.h"
#include "cionize_enermethods.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MGPOT_TIMER
#define MGPOT_GEOMHASH
#define MGPOT_CELLEN    4.f
/* #define PERFPROF */

/* #define MGPOT_FACTOR_INTERP */

  /* for accessing atom[] components */
#define INDEX_X(n)  (4*(n))
#define INDEX_Y(n)  (4*(n)+1)
#define INDEX_Z(n)  (4*(n)+2)
#define INDEX_Q(n)  (4*(n)+3)


  /* identifies different interpolants */
  enum MgpotInterp_t {
    INTERP_NONE=0,
    CUBIC,
    QUINTIC1,
    INTERP_MAX
  };

  /* needed for small bin kernel */
#define CHECK_CYLINDER
#define CHECK_CYLINDER_CPU
  typedef struct Atom_t {
    float x, y, z, q;
  } Atom;
  int cpu_compute_cutoff_potential_lattice(
      float *lattice,                    /* the lattice */
      int nx, int ny, int nz,            /* its dimensions, length nx*ny*nz */
      float xlo, float ylo, float zlo,   /* lowest corner of lattice */
      float gridspacing,                 /* lattice spacing */
      float cutoff,                      /* cutoff distance */
      Atom *atom,                        /* array of atoms */
      int natoms                         /* number of atoms */
      );

  /* max atoms (bound by 64KB constant device memory available in cuda) */
  enum { MAXATOMS = 4000 };

  /* grid cell hashing of atoms to large bins */
  typedef struct MgpotLargeBin_t {
    int atomcnt;               /* number of atoms in this bin */
    float x0, y0, z0;          /* lowest spatial coordinate of bin */
    float atom[4*MAXATOMS/2];  /* coordinates stored x/y/z/q */
  } MgpotLargeBin;


  /* main data structure */
  typedef struct Mgpot_t {

  /** atom input and epot map output **/
    const float *atoms;    /* stored x/y/z/q (length is 4*numatoms) */

    float *grideners;      /* final epot map (size numpt*numcol*numplane) */

    long numplane;         /* z dimension size */
    long numcol;           /* y dimension size */
    long numpt;            /* x dimension size */
    long numatoms;         /* number of atoms */
    float gridspacing;     /* grid spacing for epot map */

    const unsigned char *excludepos;  /* exclusion points from epot map */

    float *grideners_longrng;  /* long-range contribution to grideners */
    int separate_longrng;      /* is grideners_longrng allocated? */

  /** multilevel summation for non-periodic boundaries **/
    float h;      /* finest grid spacing */
    float h_1;    /* 1/h */
    float a;      /* cutoff distance for short-range part */
    float a_1;    /* 1/a */
    int interp;   /* identify grid interpolant */
    int split;    /* identify 1/r splitting */
    int scalexp;  /* scaling exponent from cionize to finest mgpot grid,
		   * where:  (2**scalexp)*h_cionize = h_mgpot  */
    int nlevels;  /* number of grid levels */
    floatLattice **qgrid;  /* nlevels of charge grids */
    floatLattice **egrid;  /* nlevels of potential grids */
    floatLattice **gdsum;  /* g(r) direct sum weights for each level */
#ifdef MGPOT_FACTOR_INTERP
    float *phibuffer;      /* interpolation stencil on potential lattice */
    float *phi;   /* points to centered stencil */
    float *ezd;   /* 1D array, length numplane */
    float *eyzd;  /* 2D array, length numcol*numplane */
    int nu;       /* radius of stencil on unit spacing */
    int sdelta;   /* radius of stencil on potential lattice */
#else
    floatLattice **potinterp;  /* interp weights to cionize grid potentials,
				* length is (2**scalexp)**3 */
#endif

    long nxcell;  /* number of grid cells in x-dimension */
    long nycell;  /* number of grid cells in y-dimension */
    long nzcell;  /* number of grid cells in z-dimension */
    long *gnum;   /* counts number of atoms per cell */
    long *first;  /* grid cell array nxcell*nycell*nzcell gives first index */
    long *next;   /* next atom index in cursor linked list implementation */
    float inv_cellen;        /* inverse of cubic cell length */

  /** CUDA kernel data **/
    int   use_cuda;        /* use which CUDA kernels? */

  /** lattice cutoff cuda kernels **/
    int   lk_nlevels;      /* number of levels for latcut kernel */
    int   lk_srad;         /* subcube radius for latcut kernel */
    int   lk_padding;      /* padding around internal array of subcubes */
    long  subcube_total;   /* total number of subcubes for compressed grids */
    long  block_total;     /* total number of thread blocks */
    /*
     * host_   -->  memory allocated on host
     * device_ -->  global memory allocated on device
     */
    int   *host_sinfo;     /* subcube info copy to device const mem */
    float *host_lfac;      /* level factor copy to device const mem */
    float *host_wt;        /* weights copy to device const mem */
    float *host_qgrids;    /* q-grid subcubes copy to device global mem */
    float *host_egrids;    /* e-grid subcubes copy to device global mem */
    float *device_qgrids;  /* q-grid subcubes allocate on device */
    float *device_egrids;  /* e-grid subcubes allocate on device */

  /** short-range part, large bin cuda kernel **/
    long allnx;                /* extend numpt to next multiple of slabsz */
    long allny;                /* extend numcol to next multiple of slabsz */
    long allnz;                /* extend numplane to next multiple of slabsz */
    long slabsz;               /* slabsz is point dimension of subcube */
    float *host_epot;          /* epot array allnx*allny*allnz */
    float *device_epot_slab;   /* epot allnx*allny*slabsz (1 subcube thick) */
    MgpotLargeBin *largebin;   /* array of size nxbin*nybin*nzbin */
    int nxbin;                 /* number of bins in x direction */
    int nybin;                 /* number of bins in y direction */
    int nzbin;                 /* number of bins in z direction */
    int nxsub;                 /* number of subcubes in x direction */
    int nysub;                 /* number of subcubes in y direction */
    int nzsub;                 /* number of subcubes in z direction */
#if 0
    MgpotAtomSender *asender;  /* 1 sender */
#endif

  } Mgpot;

#define MGPOTUSECUDA(emethod) (emethod & (MLATCUTMASK | MBINMASK | MDEVMASK))

#define MGPOTDEV(emethod) ((emethod & MDEVMASK) >> MDEVSHIFT)

  /* routines with external linkage */
  int mgpot_setup(Mgpot *mg, float h, float a,
      long int nx, long int ny, long int nz,
      int scalexp, int interp, int split,
      const float *atoms, float *grideners,
      long int numplane, long int numcol, long int numpt, long int numatoms,
      float gridspacing, const unsigned char *excludepos,
      int numprocs, int emethod);

  int mgpot_cleanup(Mgpot *mg);

  int mgpot_shortrng(Mgpot *mg, int threadid, int threadcount);
  int mgpot_shortrng_optimal(Mgpot *mg);
  int mgpot_shortrng_generic(Mgpot *mg);

  int mgpot_longrng(Mgpot *mg);
  int mgpot_longrng_cubic(Mgpot *mg);
  int mgpot_longrng_quintic1(Mgpot *mg);

  /* after threads join, sum long-range contribution into final result */
  int mgpot_longrng_finish(Mgpot *mg);

#ifdef __cplusplus
}
#endif

#endif /* MGPOT_DEFN_H */
