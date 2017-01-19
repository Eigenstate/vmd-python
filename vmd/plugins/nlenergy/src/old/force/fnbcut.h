/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    force/fnbcut.h
 * @brief   Compute cutoff nonbonded interactions.
 * @author  David J. Hardy
 * @date    May 2008
 */

#ifndef FORCE_FNBCUT_H
#define FORCE_FNBCUT_H

#include "moltypes/moltypes.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct FnbcutPrm_t {
    dreal hgroupCutoff;  /**< twice max expected bond length to hydrogen,
                          *   (2.5 A is reasonable choice) */
    dreal pairlistDist;  /**< twice max atom move before pairlist regen */
    ivec cellDepth;      /**< grid cell depth in each dimension (max is 3) */
    int32 nbpairType;    /**< bitwise ORing of eval type flags below */
  } FnbcutPrm;

  enum {
    FNBCUT_ELEC          = 0x0001,  /**< standard electrostatics */
    FNBCUT_ELEC_INFTY    = 0x0002,  /**< infinite range, nonzero at cutoff */
    FNBCUT_ELEC_CUTOFF   = 0x0003,  /**< shifted, smoothly zero at cutoff */
    FNBCUT_ELEC_EWALD    = 0x0004,  /**< Ewald splitting (for PME) */
    FNBCUT_ELEC_TAYLOR2  = 0x0005,  /**< C2 Taylor splitting (for MSM) */
    FNBCUT_ELEC_TAYLOR3  = 0x0006,  /**< C3 Taylor splitting (for MSM) */
    FNBCUT_ELEC_TAYLOR4  = 0x0007,  /**< C4 Taylor splitting (for MSM) */
    FNBCUT_ELEC_TAYLOR5  = 0x0008,  /**< C5 Taylor splitting (for MSM) */
    FNBCUT_ELEC_END      = 0x0009,  /**< end of electrostatics selection */
    FNBCUT_ELEC_MASK     = 0x000F,  /**< mask electrostatics */
    FNBCUT_VDW           = 0x0010,  /**< van der Waals */
    FNBCUT_VDW_INFTY     = 0x0020,  /**< infinite range, nonzero at cutoff */
    FNBCUT_VDW_CUTOFF    = 0x0030,  /**< switched, smoothly zero at cutoff */
    FNBCUT_VDW_END       = 0x0040,  /**< end of van der Waals selection */
    FNBCUT_VDW_MASK      = 0x00F0,  /**< mask van der Waals */
    FNBCUT_BUCK          = 0x0100,  /**< Buckingham */
    FNBCUT_BUCK_INFTY    = 0x0200,  /**< infinite range, nonzero at cutoff */
    FNBCUT_BUCK_CUTOFF   = 0x0300,  /**< switched, smoothly zero at cutoff */
    FNBCUT_BUCK_END      = 0x0400,  /**< end of Buckingham selection */
    FNBCUT_BUCK_MASK     = 0x0F00,  /**< mask Buckingham */
    FNBCUT_MASK          = 0x0FFF   /**< overall mask of options */
  };

  enum {
    FNBCUT_CELL_MAXDEPTH = 3,       /**< max cell depth for hashing */
    FNBCUT_CELL_NBRLEN = ((2*FNBCUT_CELL_MAXDEPTH+1) *
        (2*FNBCUT_CELL_MAXDEPTH+1) * (2*FNBCUT_CELL_MAXDEPTH+1))/2 + 1
                                    /**< max number neighbors in half-shell */
  };

  typedef struct FnbcutCell_t {
    int32 head;    /**< list head, index of first atom in this cell */
    int32 cnt;     /**< count number of atoms in this cell */
    int32 nbrcnt;  /**< count number of neighbor cells in list */
    int32 nbr[FNBCUT_CELL_NBRLEN];  /**< array of neighbors in half-shell */
    char image[FNBCUT_CELL_NBRLEN]; /**< index into periodic image table */
  } FnbcutCell;

  typedef struct Fnbcut_t {
    FnbcutPrm nbprm;

    const ForcePrm *fprm;
    const Topology *topo;
    const Exclude *exclude;
    const Domain *domain;

    /* force field parameters */
    int32 exclpolicy;
    dreal cutoff;
    dreal extcutoff;
    dreal elec_const;
    dreal switchdist2;
    dreal inv_cutoff2;
    dreal inv_denom_switch;
    dreal ewald_coef;
    dreal ewald_grad_coef;
    dreal scaling14;

    /* for gridcell hashing */
    dvec hashorigin;  /**< in recip space */
    dvec hashfactor;  /**< in recip space */
    dvec cellsz;      /**< in real space */
    dvec mincellsz;   /**< in real space */
    ivec celldim;     /**< dimensions of 3D array of gridcells */
    int32 ncells;     /**< total number of gridcells */
    Array cell;       /**< array of FnbcutCell, the grid cells for hashing */
    Array next;       /**< array of int32, next atomID in cursor linked list */
    dvec imageTable[27];  /**< lookup translation for periodic image */

    /* for interaction energies */
    Array idmap;      /**< int32, gives 0..N-1 for all atom evaluation */
    Array alltrue;    /**< char, gives all TRUE for index set inclusion */
    Array isatomID1;  /**< char, set to map TRUE for atomID1 list */
    Array isatomID2;  /**< char, set to map TRUE for atomID2 list */
    Array atomID12;   /**< int32, set to union of atomID1 and atomID2 lists */
    Array fsubtr;     /**< dvec, subtract self-forces from atomID1 and 2 */

  } Fnbcut;

  int Fnbcut_init(Fnbcut *, const Exclude *);
  void Fnbcut_done(Fnbcut *);

  int Fnbcut_find_rminmax(const Fnbcut *, dvec *rmin, dvec *rmax,
      const dvec *pos, int32 n);

  int Fnbcut_setup(Fnbcut *, const FnbcutPrm *,
      const Domain *, const dvec *rmin, const dvec *rmax);
  int Fnbcut_setup_cellnbrs(Fnbcut *);
  int Fnbcut_check_gridcell(const Fnbcut *f);  /* for debugging */

  /**@brief Evaluate nonbonded interactions for all atoms.
   *
   * pos - input length N array (defined by topology) of atom positions,
   *   for periodic boundaries must be suitably wrapped, e.g. using
   *   Coord_update_pos()
   *
   * f - output length N array of atom forces, accumulate sum of requested
   *   potential function contributions
   *
   * e - output energies, accumulate separate sums of potential function
   *   contributions
   *
   * Note that f and e are not set to zero, so that this could be the
   * nonbonded contribution to the total MD forces or the short-range
   * contribution to the electrostatic potential, etc.
   */
  int Fnbcut_eval(Fnbcut *, const dvec *pos, dvec *f, Energy *e);

  /**@brief Evaluate nonbonded interactions for a subset of atoms.
   *
   * (first four arguments are identical to Fnbcut_eval())
   *
   * atomID - array of atom IDs that are interacting with each other,
   *   some nontrivial subset of {0..N-1}
   * len - length of atomID array (0 < len <= N)
   */
  int Fnbcut_eval_subset(Fnbcut *, const dvec *pos, dvec *f, Energy *en,
      const int32 *atomID, int32 len);

  /**@brief Evaluate nonbonded interactions between two subsets of atoms.
   *
   * (first four arguments are identical to Fnbcut_eval())
   *
   * atomID1 - array of atom IDs
   * len1 - length of atomID1 array (0 < len1 < N)
   * atomID2 - array of atom IDs, disjoint from atomID1
   * len2 - length of atomID2 array (0 < len2 < N, len1+len2 <= N)
   */
  int Fnbcut_eval_disjoint_subsets(Fnbcut *, const dvec *, dvec *f, Energy *,
      const int32 *atomID1, int32 len1, const int32 *atomID2, int32 len2);

  int Fnbcut_eval_cellhash(Fnbcut *, const dvec *pos,
      const int32 *atomID, int32 len);

  int Fnbcut_eval_cellpairs(Fnbcut *, const dvec *pos, dvec *f, Energy *e);

  int Fnbcut_eval_scaled14(Fnbcut *, const dvec *pos, dvec *f, Energy *e,
      const char *isatomID1, const char *isatomID2);

#ifdef __cplusplus
}
#endif

#endif /* FORCE_FNBCUT_H */
