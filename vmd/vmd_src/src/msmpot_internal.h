/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: msmpot_internal.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.5 $      $Date: 2011/01/14 16:09:28 $
 *
 ***************************************************************************/

#ifndef MSMPOT_INTERNAL_H
#define MSMPOT_INTERNAL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "msmpot.h"

/* avoid parameter name collisions with AIX5 "hz" macro */
#undef hz

#ifdef __cplusplus
extern "C" {
#endif


  /* for 32-bit builds, IndexType is simply int */
  typedef int IndexType;


  /* create a 3D grid to access data of given TYPE */
#undef GRID_TEMPLATE
#define GRID_TEMPLATE(TYPE) \
  typedef struct TYPE##Grid_t { \
    TYPE *buffer;       /* raw buffer */ \
    TYPE *data;         /* data access offset from buffer */ \
    size_t numbytes;    /* number of bytes in use by buffer */ \
    size_t maxbytes;    /* actual allocation for buffer */ \
    int i0, j0, k0;     /* starting index value for each dimension */ \
    int ni, nj, nk;     /* number of elements in each dimension */ \
  } TYPE##Grid        /* expect closing ';' */

  /* initialize grid to empty */
#define GRID_INIT(a) \
  ((a)->buffer=NULL, (a)->data=NULL, (a)->numbytes=0, (a)->maxbytes=0, \
   (a)->i0=0, (a)->j0=0, (a)->k0=0, (a)->ni=0, (a)->nj=0, (a)->nk=0)  /* ; */

  /* finished with grid, free its memory */
#define GRID_DONE(a) \
  free((a)->buffer)  /* ; */

  /* determine the signed flattened index for 3D grid datum */
#define GRID_INDEX(a, _i, _j, _k) \
  (((_k)*((a)->nj) + (_j))*(IndexType)((a)->ni) + (_i))  /* ; */

  /* obtain pointer to 3D grid datum */
#define GRID_POINTER(a, _i, _j, _k) \
  ((a)->data + GRID_INDEX(a, _i, _j, _k))  /* ; */

  /* resize 3D grid buffer, setup its indexing */
  /* grab more memory when needed */
  /* (must be used within function returning int) */
#define GRID_RESIZE(a, __i0, __ni, __j0, __nj, __k0, __nk) \
  do { \
    int _i0=(__i0), _ni=(__ni); \
    int _j0=(__j0), _nj=(__nj); \
    int _k0=(__k0), _nk=(__nk); \
    size_t _numbytes = (_nk * _nj) * (size_t) _ni * sizeof((a)->buffer[0]); \
    if ((a)->maxbytes < _numbytes) { \
      void *_t = realloc((a)->buffer, _numbytes); \
      if (NULL == _t) return ERROR(MSMPOT_ERROR_MALLOC); \
      (a)->buffer = (float *) _t; \
      (a)->maxbytes = _numbytes; \
    } \
    (a)->numbytes = _numbytes; \
    (a)->i0 = _i0, (a)->ni = _ni; \
    (a)->j0 = _j0, (a)->nj = _nj; \
    (a)->k0 = _k0, (a)->nk = _nk; \
    (a)->data = (a)->buffer + GRID_INDEX((a), -_i0, -_j0, -_k0); \
  } while (0)  /* expect closing ';' */

  /* reset 3D grid data to 0 */
#define GRID_ZERO(a) \
  memset((a)->buffer, 0, (a)->numbytes)  /* ; */

  /* check 3D grid index range when debugging */
  /* (must be used within function returning int) */
#ifdef MSMPOT_DEBUG
#define GRID_INDEX_CHECK(a, _i, _j, _k) \
  do { \
    ASSERT((a)->i0 <= (_i) && (_i) < (a)->ni + (a)->i0); \
    ASSERT((a)->j0 <= (_j) && (_j) < (a)->nj + (a)->j0); \
    ASSERT((a)->k0 <= (_k) && (_k) < (a)->nk + (a)->k0); \
  } while (0)  /* expect closing ';' */
#else
#define GRID_INDEX_CHECK(a, _i, _j, _k)
#endif


  /* default MSM parameters */
#define DEFAULT_HMIN        2.f
#define DEFAULT_CUTOFF     12.f
#define DEFAULT_INTERP     MSMPOT_INTERP_CUBIC
#define DEFAULT_SPLIT      MSMPOT_SPLIT_TAYLOR2

#define DEFAULT_BINLENMAX   4.f

#define DEFAULT_BINDEPTH    8      /* set for CUDA hardware */
#define DEFAULT_BINFILL     0.75f  /* try to achieve average bin fill */
#define DEFAULT_DENSITY     0.1f   /* for atom biomolecule (units 1/A^3) */

#define DEFAULT_OVER       20

#define DEFAULT_ERRTOL      5e-3   /* for (1/2)% relative error */

#define ATOM_SIZE  4      /* number of floats per atom, stored x/y/z/q */

#define ATOM_X(i)  ((i)<<2)      /* index for x coordinate of atom i */
#define ATOM_Y(i)  (((i)<<2)+1)  /* index for y coordinate of atom i */
#define ATOM_Z(i)  (((i)<<2)+2)  /* index for z coordinate of atom i */
#define ATOM_Q(i)  (((i)<<2)+3)  /* index for q charge of atom i */

#define SET_X(flag)       ((flag) |= 0x01)
#define SET_Y(flag)       ((flag) |= 0x02)
#define SET_Z(flag)       ((flag) |= 0x04)
#define IS_SET_X(flag)    ((flag) & 0x01)
#define IS_SET_Y(flag)    ((flag) & 0x02)
#define IS_SET_Z(flag)    ((flag) & 0x04)
#define IS_SET_ANY(flag)  ((flag) & 0x07)
#define IS_SET_ALL(flag)  ((flag) == 0x07)


  GRID_TEMPLATE(float);   /* for MSM charge and potential grids */


#ifdef MSMPOT_CUDA
  struct MsmpotCuda_t;    /* forward definition of MsmpotCuda structure */
  typedef struct MsmpotCuda_t MsmpotCuda;

  MsmpotCuda *Msmpot_cuda_create(void);
  void Msmpot_cuda_destroy(MsmpotCuda *);
  int Msmpot_cuda_setup(MsmpotCuda *, Msmpot *);

  int Msmpot_cuda_compute_shortrng(MsmpotCuda *);

  int Msmpot_cuda_compute_latcut(MsmpotCuda *);
  int Msmpot_cuda_condense_qgrids(MsmpotCuda *);
  int Msmpot_cuda_expand_egrids(MsmpotCuda *);
#endif


  /*** Msmpot ****************************************************************/

  struct Msmpot_t {
    float *epotmap;       /* the map */
    int mx, my, mz;       /* map dimensions */
    float lx, ly, lz;     /* map lengths (rectangular box) */
    float lx0, ly0, lz0;  /* map origin */
    float dx, dy, dz;     /* map grid spacing, dx=lx/mx, etc. */

    const float *atom;    /* the original atom array stored x/y/z/q */
    int natoms;           /* number of atoms, atom array has 4*natoms floats */

    int isperiodic;       /* flag for periodicity in x, y, z */
    float px, py, pz;     /* atom domain lengths (whether or not periodic) */
    float px0, py0, pz0;  /* atom domain origin */

    float density;        /* expected density of system */

    float xmin, xmax;     /* max and min x-coordinates of atoms */
    float ymin, ymax;     /* max and min y-coordinates of atoms */
    float zmin, zmax;     /* max and min z-coordinates of atoms */

    /*
     * Short-range part:  spatial hashing of atoms into bins,
     * calculate short-range contribution to each grid point as
     * the sum of surrounding neighborhood of bins.
     *
     * Bins are accesed in a grid starting at 0:
     *   bin(i,j,k) == bin[ (k*nby + j)*nbx + i ]
     *
     * The bincount array corresponds to the bin array giving
     * number of atoms stored in each bin.
     *
     * The bins cover the atom domain starting at (px0,py0,pz0).
     *
     * The bin size is chosen to achieve a BINFILL*bindepth average
     * fill rate per bin.
     */

    float *bin;           /* length 4*bindepth*maxbins */
    int *bincount;        /* number of atoms in each bin */
    int bindepth;         /* number of atom slots per bin (from GPU hardware) */
    int nbx, nby, nbz;    /* number of bins in grid along x, y, z */
    int maxbin;           /* maximum allocation of bins >= nbx*nby*nbz */
    int isbinwrap;        /* flag for bin neighborhood wrapping in x, y, z */
    int islongcutoff;     /* flag for cutoff wrapping beyond nearest image */
    float binfill;        /* set bin size for this fill ratio (from user) */
    float bx, by, bz;     /* bx = px/nbx, by = py/nby, bz = py/nbz */
    float invbx, invby, invbz;  /* inverse lengths */

    float *over;          /* bin overflow list, length 4*maxoverflow */
    int nover;            /* how many atoms in overflow list */
    int maxover;          /* maximum allocation for list */

    int *boff;            /* neighborhood as bin index offsets */
    int nboff;            /* number of offsets, length is 3*nbdoff */
    int maxboff;          /* maximum allocation */

    /*
     * Fundamental MSM parameters:
     *
     * Default grid spacing hmin = 2A, cutoff a = 12A,
     * C1 cubic interpolation, C2 Taylor splitting,
     * all gives reasonable accuracy for atomistic systems.
     *
     * Find grid spacings in range hmin <= hx, hy, hz <= (1.5 * hmin)
     * such that along periodic dimensions the number of grid points is
     * some power of 2 times zero or one power of 3.
     *
     * Maintain ratio 4 <= a/hx, a/hy, a/hz <= 6 for accuracy,
     * and constants can fit into GPU.
     *
     *
     * for nonperiodic dimensions, the finest level lattice is chosen to be
     * smallest size aligned with epotmap containing both atoms and epotmap;
     * for periodic dimensions, the finest level lattice fits within cell
     * defined by epotmap parameters above;
     * the number of levels is determined to reduce coarsest level lattice
     * to be as small as possible for the given boundary conditions
     */
    float errtol;         /* error tolerance for convergence of "exact" PBC */

    float hmin;           /* smallest MSM grid spacing, hmax = 1.5 * hmin */
    float hx, hy, hz;     /* the finest level lattice spacings */
    float a;              /* the MSM cutoff between short- and long-range */
    int nx, ny, nz;       /* count number of h spacings that cover domain */

    int interp;           /* the interpolant MSMPOT_INTERP_ */
    int split;            /* the splitting MSMPOT_SPLIT_ */

    int nlevels;          /* number of lattice levels */

    /*
     * Grids for calculating long-range part:
     * q[0] is finest-spaced grid (hx, hy, hz),
     * grid level k has spacing 2^k * (hx, hy, hz).
     *
     * Use domain origin (px0, py0, pz0) for each grid origin, ao that
     * q[0](i,j,k) has position (i*hx + px0, j*hy + py0, k*hz + pz0)
     * the finest level lattice is 0, e.g. q0 = qh[0].
     *
     * Grid indexes can be negative for non-periodic dimensions,
     * the periodic dimensions wrap around edges of grid.
     */

    floatGrid *qh;        /* grids of charge, 0==finest */
    floatGrid *eh;        /* grids of potential, 0==finest */
    floatGrid *gc;        /* fixed-size grids of weights for convolution */
    int maxlevels;        /* maximum number of grid levels allocated */



    /* Interpolating from finest lattice to epotmap:
     * want ratio hx/dx to be rational 2^(px2)*3^(px3),
     * where px2 is unrestricted and px3=0 or px3=1.
     * The interpolation of epotmap from finest lattice then has
     * a fixed cycle of coefficients that can be precomputed.
     * The calculation steps through MSM lattice points and
     * adds their contribution to surrounding epotmap points. */

    int px2, py2, pz2;    /* powers of 2 */
    int px3, py3, pz3;    /* powers of 3 */
    float hx_dx, hy_dy, hz_dz;  /* scaling is integer for px2 >= 0 */

    int cycle_x, cycle_y, cycle_z;  /* counts MSM points between alignment */
    int rmap_x, rmap_y, rmap_z;     /* radius of map points about MSM point */

    int max_phi_x, max_phi_y, max_phi_z;  /* alloc length of phi arrays */
    float *phi_x;         /* coefficients, size cycle_x * (2*rmap_x + 1) */
    float *phi_y;         /* coefficients, size cycle_y * (2*rmap_y + 1) */
    float *phi_z;         /* coefficients, size cycle_z * (2*rmap_z + 1) */



    /* need these */

    int max_ezd, max_eyzd;  /* alloc length of interp temp buffer space */
    float *ezd;           /* interpolation temp row buffer, length mz */
    float *eyzd;          /* interpolation temp plane buffer, length my*mz */

    int max_lzd, max_lyzd;  /* alloc length of factor temp buffer space */
    float *lzd;           /* factor temp row buffer, length z-dim of h-level */
    float *lyzd;          /* factor temp row buffer, (y*z)-dim of h-level */


    /* cursor linked list implementation */
    int maxatoms;                      /* allocated number of atoms */
    int *first_atom_index;             /* length maxcells >= ncells */
    int *next_atom_index;              /* length maxatoms >= natoms */

#ifdef MSMPOT_CUDA
    MsmpotCuda *msmcuda;    /* handle to "MsmpotCuda" (CUDA-compute) object */
    const int *devlist;     /* list of devices, prioritized highest to lowest */
    int devlistlen;         /* length of devlist */
    int cuda_optional;      /* flag CUDA is optional, fall back on CPU */
    int use_cuda;           /* flag use of CUDA */
    int use_cuda_shortrng;  /* flag use of CUDA for short-range part */
    int use_cuda_latcut;    /* flag use of CUDA for lattice cutoff part */
#endif
  };


  /* for internal use only */
  void Msmpot_set_defaults(Msmpot *msm);
  int Msmpot_check_params(Msmpot *msm, const float *epotmap,
      int mx, int my, int mz, float lx, float ly, float lz,
      float vx, float vy, float vz, const float *atom, int natoms);

  int Msmpot_setup(Msmpot *msm);
  void Msmpot_cleanup(Msmpot *msm);

  /*
   * CPU implementation: hash atoms into bins,
   * evaluate grid points over neighborhood of bins
   */
  int Msmpot_compute_shortrng_bins(Msmpot *msm);

  /*
   * Determine the bin neighborhood for a given "region":
   * - for CPU, just give the bin lengths
   * - for GPU, give the region sizes, e.g., REGSIZE_X * dx
   */
  int Msmpot_compute_shortrng_bin_neighborhood(Msmpot *msm,
      float rx,  /* region length in x-dimension */
      float ry,  /* region length in y-dimension */
      float rz   /* region length in z-dimension */
      );

  /*
   * Perform the spatial hashing of atoms into bins,
   * place any extra atoms that overflow bins into overflow list,
   * can use for both GPU and CPU.
   */
  int Msmpot_compute_shortrng_bin_hashing(Msmpot *msm);

  /*
   * Use linklist data structure for spatial hashing of atoms:
   * - for CPU, when using by itself, give entire atom list
   * - for GPU or CPU, when using with bin hashing, give overflow atom list
   */
  int Msmpot_compute_shortrng_linklist(Msmpot *msm,
      const float *atom,    /* array of atoms stored x/y/z/q */
      int natoms            /* number of atoms in array */
      );

  int Msmpot_compute_longrng(Msmpot *msm);

  int Msmpot_compute_longrng_cubic(Msmpot *msm);


  /* exception handling:
   * MSMPOT_DEBUG turns on error reporting to stderr stream, 
   * in any case propagate error number back up the call stack */
#undef  ERROR
#ifndef MSMPOT_DEBUG
#define ERROR(err)       (err)
#define ERRMSG(err,msg)  (err)
#else
  /* report error to stderr stream, return error code "err" */
  int Msmpot_report_error(int err, const char *msg, const char *fn, int ln);
#define ERROR(err)       Msmpot_report_error(err, NULL, __FILE__, __LINE__)
#define ERRMSG(err,msg)  Msmpot_report_error(err, msg, __FILE__, __LINE__)
#endif


  /* check assertions when debugging, raise exception if failure */
#ifndef MSMPOT_DEBUG
#define ASSERT(expr)
#else
#define ASSERT(expr) \
  do { \
    if ( !(expr) ) { \
      return Msmpot_report_error(MSMPOT_ERROR_ASSERT, \
                                 #expr, __FILE__, __LINE__); \
    } \
  } while (0)
#endif


  /* MSMPOT_VERBOSE enables MSMPOT_REPORT */
#ifdef MSMPOT_VERBOSE
#undef MSMPOT_REPORT
#define MSMPOT_REPORT
#endif

  /* report status of MSMPOT calculation */
#ifndef MSMPOT_REPORT
#define REPORT(msg)
#else
#define REPORT(msg)  printf("MSMPOT: %s\n", (msg))
#endif


  /* SPOLY() calculates the polynomial part of the
   * normalized smoothing of 1/r, i.e. g_1((r/a)**2).
   *
   *   pg - float*, points to variable to receive the result
   *   s - (ra/)**2, assumed to be between 0 and 1
   *   split - identify the type of smoothing used to split the potential */
#undef  SPOLY
#define SPOLY(pg, s, split) \
  do { \
    float _s = s;  /* where s=(r/a)**2 */ \
    float _g = 0; \
    ASSERT(0 <= _s && _s <= 1); \
    switch (split) { \
      case MSMPOT_SPLIT_TAYLOR2: \
        _g = 1 + (_s-1)*(-1.f/2 + (_s-1)*(3.f/8)); \
        break; \
      case MSMPOT_SPLIT_TAYLOR3: \
        _g = 1 + (_s-1)*(-1.f/2 + (_s-1)*(3.f/8 + (_s-1)*(-5.f/16))); \
        break; \
      case MSMPOT_SPLIT_TAYLOR4: \
        _g = 1 + (_s-1)*(-1.f/2 + (_s-1)*(3.f/8 + (_s-1)*(-5.f/16 \
                + (_s-1)*(35.f/128)))); \
        break; \
      default: \
        return ERRMSG(MSMPOT_ERROR_SUPPORT, \
            "splitting function not implemented"); \
    } \
    *(pg) = _g; \
  } while (0)
  /* closing ';' from use as function call */


#ifdef __cplusplus
}
#endif

#endif /* MSMPOT_INTERNAL_H */
