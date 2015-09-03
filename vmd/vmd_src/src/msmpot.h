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
 *      $RCSfile: msmpot.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2010/06/03 20:07:09 $
 *
 ***************************************************************************/

/**@file    msmpot.h
 * @brief   Library interface file.
 * @author  David J. Hardy
 * @date    December 2009
 */

#ifndef MSMPOT_H
#define MSMPOT_H

#ifdef __cplusplus
extern "C" {
#endif

  /* struct Msmpot_t; */
  /** Private MSM data structure */
  typedef struct Msmpot_t Msmpot;

  /**@brief Constructor. */
  Msmpot *Msmpot_create(void);

  /**@brief Destructor. */
  void Msmpot_destroy(Msmpot *);


  /**@brief Calculate the electrostatic potential map for
   * the provided array of charged atoms.
   * The result is stored in @c epotmap.
   * Returns #MSMPOT_SUCCESS for success
   * or nonzero error code indicating failure.
   *
   * @param[in,out]  pm    Handle to #Msmpot
   * @param[out]  epotmap  Buffer space for electrostatic potential map
   *                         calculation, assumed to be at least length
   *                         @f$\mbox{mx}\times\mbox{my}\times\mbox{mz}@f$
   *                         stored "flat" in row-major order, i.e.,
   *                         @code &ep[i,j,k] == ep + ((k*my+j)*mx+i) @endcode
   * @param[in]   mx       Number of map lattice voxels in x-direction
   * @param[in]   my       Number of map lattice voxels in y-direction
   * @param[in]   mz       Number of map lattice voxels in z-direction
   * @param[in]   lx       Map lattice length in x-direction,
   *                         must have @c lx @f$> 0 @f$
   * @param[in]   ly       Map lattice length in y-direction,
   *                         must have @c ly @f$> 0 @f$
   * @param[in]   lz       Map lattice length in z-direction,
   *                         must have @c lz @f$> 0 @f$
   * @param[in]   x0       Origin of map, lowest x-coordinate
   * @param[in]   y0       Origin of map, lowest y-coordinate
   * @param[in]   z0       Origin of map, lowest z-coordinate
   * @param[in]   vx       Length of periodic cell side in x-direction,
   *                         must either have @c vx @f$\geq@f$ @c lx or
   *                         set @c vx @f$= 0 @f$ to indicate
   *                         x-direction is non-periodic
   * @param[in]   vy       Length of periodic cell side in y-direction,
   *                         must either have @c vy @f$\geq@f$ @c ly or
   *                         set @c vy @f$= 0 @f$ to indicate
   *                         y-direction is non-periodic
   * @param[in]   vz       Length of periodic cell side in z-direction,
   *                         must either have @c vz @f$\geq@f$ @c lz or
   *                         set @c vz @f$= 0 @f$ to indicate
   *                         z-direction is non-periodic
   * @param[in]   atom     Array of atoms, storing 4 floats x/y/z/q
   *                         for each atom giving position and charge
   * @param[in]   natoms   Number of atoms, must be positive
   *     
   * The map lattice spacings are
   *   @f$\mbox{dx}=\mbox{lx}/\mbox{mx}@f$,
   *   @f$\mbox{dy}=\mbox{ly}/\mbox{my}@f$,
   *   @f$\mbox{dz}=\mbox{lz}/\mbox{mz}@f$.
   * The lattice voxel ep[i,j,k] (i=0,..,mx-1, etc.)
   * is represented by rectangular coordinate:
   *   @f$ \mbox{origin} + [ i\times\mbox{dx}, j\times\mbox{dy},
   *                           k\times\mbox{dz} ]^T @f$.
   * This defines a cell-centered geometry, although the calculation
   * is at the lower-lefthand corner of each cell.
   *
   * There is some setup overhead on first call.  The overhead is reduced for
   * subsequent calls if map dimensions and number of atoms remain the same
   * and if atoms have the same bounding box.  Calls to #Msmpot_compute()
   * will allocate as much additional memory as needed, held until call to
   * #Msmpot_destroy().
   */
  int Msmpot_compute(Msmpot *pm,
      float *epotmap,               /* electrostatic potential map */
      int mx, int my, int mz,       /* map lattice dimensions */
      float lx, float ly, float lz, /* map lattice lengths */
      float x0, float y0, float z0, /* map origin (lower-left corner) */
      float vx, float vy, float vz, /* periodic cell lengths along x, y, z;
                                       set to 0 for non-periodic direction */
      const float *atom,            /* atoms stored x/y/z/q (length 4*natoms) */
      int natoms                    /* number of atoms */
      );


  /**@brief Calculate the exact electrostatic potential map for
   * the provided array of charged atoms.
   * The result is stored in @c epotmap.
   * Returns #MSMPOT_SUCCESS for success
   * or nonzero error code indicating failure.
   *
   * The parameters are identical to #Msmpot_compute().
   *
   * @param[in,out]  pm    Handle to #Msmpot
   * @param[out]  epotmap  Buffer space for electrostatic potential map
   *                         calculation, assumed to be at least length
   *                         @f$\mbox{mx}\times\mbox{my}\times\mbox{mz}@f$
   *                         stored "flat" in row-major order, i.e.,
   *                         @code &ep[i,j,k] == ep + ((k*my+j)*mx+i) @endcode
   * @param[in]   mx       Number of map lattice voxels in x-direction
   * @param[in]   my       Number of map lattice voxels in y-direction
   * @param[in]   mz       Number of map lattice voxels in z-direction
   * @param[in]   lx       Map lattice length in x-direction,
   *                         must have @c lx @f$> 0 @f$
   * @param[in]   ly       Map lattice length in y-direction,
   *                         must have @c ly @f$> 0 @f$
   * @param[in]   lz       Map lattice length in z-direction,
   *                         must have @c lz @f$> 0 @f$
   * @param[in]   x0       Origin of map, lowest x-coordinate
   * @param[in]   y0       Origin of map, lowest y-coordinate
   * @param[in]   z0       Origin of map, lowest z-coordinate
   * @param[in]   vx       Length of periodic cell side in x-direction,
   *                         must either have @c vx @f$\geq@f$ @c lx or
   *                         set @c vx @f$= 0 @f$ to indicate
   *                         x-direction is non-periodic
   * @param[in]   vy       Length of periodic cell side in y-direction,
   *                         must either have @c vy @f$\geq@f$ @c ly or
   *                         set @c vy @f$= 0 @f$ to indicate
   *                         y-direction is non-periodic
   * @param[in]   vz       Length of periodic cell side in z-direction,
   *                         must either have @c vz @f$\geq@f$ @c lz or
   *                         set @c vz @f$= 0 @f$ to indicate
   *                         z-direction is non-periodic
   * @param[in]   atom     Array of atoms, storing 4 floats x/y/z/q
   *                         for each atom giving position and charge
   * @param[in]   natoms   Number of atoms, must be positive
   *
   * This is meant only for testing accuracy of calculation and not
   * intended for production use.
   */
  int Msmpot_compute_exact(Msmpot *pm,
      float *epotmap,               /* electrostatic potential map */
      int mx, int my, int mz,       /* map lattice dimensions */
      float lx, float ly, float lz, /* map lattice lengths */
      float x0, float y0, float z0, /* map origin (lower-left corner) */
      float vx, float vy, float vz, /* periodic cell lengths along x, y, z;
                                       set to 0 for non-periodic direction */
      const float *atom,            /* atoms stored x/y/z/q (length 4*natoms) */
      int natoms                    /* number of atoms */
      );


  /*
   * Use CUDA GPU acceleration for Msmpot_compute().
   *   devlist - available devices, listed in decreasing order of preference
   *   listlen - length of devlist array
   *   cuda_optional - 1 indicates fall back on CPU if device or CUDA kernel
   *     can't be used to compute desired result, 0 indicates hard failure
   * No checking is done for actual device existence until Msmpot_compute().
   * However, an error is returned if the MSMPOT build lacks CUDA support
   * or if devlist and listlen do not together indicate at least one device.
   */
  int Msmpot_use_cuda(Msmpot *, const int *devlist, int listlen,
      int cuda_optional);


#if 0
  /*
   * Establish a callback indicating progress of Msmpot_compute().
   * Function "progress" is provided by user to accept four ints:
   *   numphases - number of phases
   *   phasecnt  - which phase
   *   numunits  - number of work units for this phase
   *   unitcnt   - which work unit
   * A nonzero return value from progress() will result in early
   * termination of the Msmpot_compute() call.
   */
  int Msmpot_callback_status(Msmpot *,
      int (*progress)(int numphases, int phasecnt, int numunits, int unitcnt));
#endif


  /*@brief Return the error message string for a particular return code. */
  const char *Msmpot_error_string(int retcode);

  /*
   * Configure the MSM parameters to improve accuracy or performance.
   *
   *   interp - choose interpolant (MSMPOT_INTERP_*);
   *     set less than 0 for default
   *   split - choose splitting (MSMPOT_SPLIT_*);
   *     set less than 0 for optimal choice based on interp
   *   nlevels - maximum number of levels for lattice hierarchy;
   *     set less than or equal to 0 to use maximum possible levels
   *   cutoff - length of short-range cutoff for atomic interactions;
   *     set less than or equal to 0 for default
   *   hmin - minimum spacing for MSM h-level lattice, hmax = (3/2)*hmin;
   *     for periodic boundaries, we can choose h, hmin <= h <= hmax;
   *     for non-periodic boundaries, h = hmin;
   *     set less than or equal to 0 for optimal choice based on cutoff
   *   binszmin - minimum bin size for geometric hashing;
   *     for periodic, binsize = L / floor(L / binszmin);
   *     for non-periodic boundaries, binsize = binszmin;
   *     set less than or equal to 0 for default
   *
   * Set any option less than 0 to 
   */
  int Msmpot_configure(Msmpot *,
      int interp,     /* which interpolant */
      int split,      /* which splitting */
      float cutoff,   /* cutoff distance (in Angstroms) */
      float hmin,     /* minimum spacing MSM h-level lattice */
      int nlevels,    /* maximum number of levels */
      float density,  /* expected density of system */
      float binfill,  /* ratio for bin fill, between 0 and 1 */
      float errtol,   /* error tolerance for convergence of periodic
                         #Msmpot_compute_exact() calculation */
      int usecuda     /* Use CUDA GPU acceleration? */
      );

  /*
   * MSM interpolation methods.  (Default is CUBIC.)
   */
  enum {
    MSMPOT_INTERP_CUBIC = 0,   /* C1 cubic (numerical Hermite) */
    MSMPOT_INTERP_QUINTIC,     /* C1 quintic (linear blend of quartics) */
    MSMPOT_INTERP_QUINTIC2,    /* C2 quintic */
    MSMPOT_INTERP_SEPTIC,      /* C1 septic (linear blend of sextics) */
    MSMPOT_INTERP_SEPTIC3,     /* C3 septic */
    MSMPOT_INTERP_NONIC,       /* C1 nonic (linear blend of octics) */
    MSMPOT_INTERP_NONIC4,      /* C4 nonic */
    MSMPOT_INTERPMAX           /* (for internal use) */
  };

  /*
   * MSM potential splitting methods.  (Default is TAYLOR2.)
   */
  enum {
    MSMPOT_SPLIT_TAYLOR2 = 0,  /* C2 Taylor splitting of s^(1/2), s = r^2 */
    MSMPOT_SPLIT_TAYLOR3,      /* C3 Taylor splitting of s^(1/2), s = r^2 */
    MSMPOT_SPLIT_TAYLOR4,      /* C4 Taylor splitting of s^(1/2), s = r^2 */
    MSMPOT_SPLIT_TAYLOR5,      /* C5 Taylor splitting of s^(1/2), s = r^2 */
    MSMPOT_SPLIT_TAYLOR6,      /* C6 Taylor splitting of s^(1/2), s = r^2 */
    MSMPOT_SPLIT_TAYLOR7,      /* C7 Taylor splitting of s^(1/2), s = r^2 */
    MSMPOT_SPLIT_TAYLOR8,      /* C8 Taylor splitting of s^(1/2), s = r^2 */
    MSMPOT_SPLITMAX            /* (for internal use) */
  };

  /**@enum MsmpotRetcode
   * @brief Return codes for Msmpot library calls.
   *
   * Zero is success, nonzero is failure.
   */
  typedef enum MsmpotRetcode_t {
    MSMPOT_SUCCESS = 0,        /**< success */
    MSMPOT_ERROR_ASSERT,       /**< assertion failed */
    MSMPOT_ERROR_MALLOC,       /**< unable to allocate memory */
    MSMPOT_ERROR_PARAM,        /**< illegal parameter */
    MSMPOT_ERROR_SUPPORT,      /**< unsupported request */
    MSMPOT_ERROR_CUDA_DEVREQ,  /**< request failed for CUDA device */
    MSMPOT_ERROR_CUDA_MALLOC,  /**< uable to allocate memory on CUDA device */
    MSMPOT_ERROR_CUDA_MEMCPY,  /**< memory copy failed between
                                 host and CUDA device */
    MSMPOT_ERROR_CUDA_KERNEL,  /**< CUDA kernel execution failed */
    MSMPOT_ERROR_CUDA_SUPPORT, /**< CUDA kernel does not support request */
    MSMPOT_ERROR_UNKNOWN       /**< unknown error number */
  } MsmpotRetcode;


#ifdef __cplusplus
}
#endif

#endif /* MSMPOT_H */
