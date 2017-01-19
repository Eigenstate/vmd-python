/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/forceprm.h
 * @brief   Force field parameter data container.
 * @author  David J. Hardy
 * @date    Mar. 2008
 */

#ifndef MOLTYPES_FORCEPRM_H
#define MOLTYPES_FORCEPRM_H

#include "nlbase/nlbase.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief Exclusion policies. */
  enum {
    EXCL_NONE     = 0,
    EXCL_12       = 1,
    EXCL_13       = 2,
    EXCL_14       = 3,
    EXCL_SCALED14 = 4
  };

  /**@brief Charge models. */
  enum {
    CHARGE_FIXED = 0,    /**< standard atomic fixed point-charge model */
    CHARGE_DRUDE = 1,    /**< Drude particle polarization */
    CHARGE_FLUCQ = 2     /**< charge equilibration (Brooks's fluc q) */
  };

  /**@brief Water models. */
  enum {
    WATER_TIP3 = 0,      /**< standard 3-point model */
    WATER_TIP4 = 1       /**< single lone pair charge site on oxygen */
  };

  /**@brief Nonbonded force field parameters. */
  typedef struct NonbPrm_t {
    dreal cutoff;        /**< cutoff distance (A, 8 <= cutoff <= 12 is good) */
    dreal switchdist;    /**< switching distance (A, switchdist <= cutoff) */
    dreal dielectric;    /**< dielectric constant for system */
    dreal scaling14;     /**< elec scaling factor for 1-4 interactions */
    boolean switching;   /**< Use a switching function for van der Waals? */
    int32 exclude;       /**< exclusion policy */
    int32 charge_model;  /**< charge model */
    int32 water_model;   /**< water model */
  } NonbPrm;

  /**@brief Stores short C-style string for atom types. */
  typedef char AtomType[8];

  enum {
    NUM_ATOMPRM_ATOMTYPE     = 1,
    NUM_BONDPRM_ATOMTYPE     = 2,
    NUM_ANGLEPRM_ATOMTYPE    = 3,
    NUM_DIHEDPRM_ATOMTYPE    = 4,
    NUM_IMPRPRM_ATOMTYPE     = 4,
    NUM_VDWPAIRPRM_ATOMTYPE  = 2,
    NUM_BUCKPAIRPRM_ATOMTYPE = 2
  };

  /**@brief Atom force field parameters (for now these are 
   * van der Waals parameters to be used with combination rules). */
  typedef struct AtomPrm_t {
    dreal emin;            /**< van der Waals energy min (AMU*(A/fs)^2) */
    dreal rmin;            /**< van der Waals distance for emin (A) */
    dreal emin14;          /**< modified 1-4 energy min (AMU*(A/fs)^2) */
    dreal rmin14;          /**< modified 1-4 distance for emin14 (A) */
    AtomType atomType[NUM_ATOMPRM_ATOMTYPE];  /**< identify atom types */
  } AtomPrm; 
  
  /**@brief Bond type force field parameters. */
  typedef struct BondPrm_t {
    dreal k;               /**< spring coefficient (AMU/fs^2) */
    dreal r0;              /**< equilibrium length (A) */
    AtomType atomType[NUM_BONDPRM_ATOMTYPE];  /**< identify atom types */
  } BondPrm;

  /**@brief Angle type force field parameters. */
  typedef struct AnglePrm_t {
    dreal k_theta;         /**< angle coefficient ((AMU*(A/fs)^2)/rad^2) */
    dreal theta0;          /**< equilibrium angle (radians) */
    dreal k_ub;            /**< coef for Urey-Bradley term (AMU/fs^2) */
    dreal r_ub;            /**< equil length for Urey-Bradley term (A) */
    AtomType atomType[NUM_ANGLEPRM_ATOMTYPE];  /**< identify atom types */
  } AnglePrm;

  /**@brief Dihedral force field terms. */
  typedef struct DihedTerm_t {
    dreal k_dihed;         /**< dihedral coefficient (AMU*(A/fs)^2) */
    dreal phi0;            /**< phase shift (radians) */
    int32 n;               /**< periodicity ( > 0) */
    int32 term;            /**< term number (is redundant) */
  } DihedTerm;

  /**@brief Dihedral type force field parameters.
   * Contains array of terms for dihedral multiplicity.
   * Since DihedPrm array elements are themselves objects,
   * we need to call constructor and destructor on each element.
   */
  typedef struct DihedPrm_t {
    AtomType atomType[NUM_DIHEDPRM_ATOMTYPE];  /**< identify atom types */
    Array dihedterm;       /**< array of DihedTerm */
  } DihedPrm;

  /**@brief DihedPrm constructor. */
  int DihedPrm_init(DihedPrm *);

  /**@brief DihedPrm destructor. */
  void DihedPrm_done(DihedPrm *);

  /**@brief DihedPrm copying. */
  int DihedPrm_copy(DihedPrm *dest, const DihedPrm *src);

  /**@brief Add terms to DihedTerm array. */
  int DihedPrm_setmaxnum_term(DihedPrm *, int32 n);
  int DihedPrm_add_term(DihedPrm *, const DihedTerm *);

  /**@brief Provide access to DihedTerm array. */
  const DihedTerm *DihedPrm_term_array(const DihedPrm *);
  int32            DihedPrm_term_array_length(const DihedPrm *);

  /**@brief DihedPrm erasing. */
  int DihedPrm_erase(DihedPrm *);

  /**@brief Improper type force field parameters. */
  typedef struct ImprPrm_t {
    dreal k_impr;          /**< improper coefficient ((AMU*(A/fs)^2)/rad^2) */
    dreal psi0;            /**< phase shift (radians) */
    AtomType atomType[NUM_IMPRPRM_ATOMTYPE];  /**< identify atom types */
  } ImprPrm;

  /**@brief Adjusted van der Waals parameters for indicated pairs. */
  typedef struct VdwpairPrm_t {
    dreal emin;            /**< van der Waals energy min (AMU*(A/fs)^2) */
    dreal rmin;            /**< van der Waals distance for emin (A) */
    dreal emin14;          /**< modified 1-4 energy min (AMU*(A/fs)^2) */
    dreal rmin14;          /**< modified 1-4 distance for emin14 (A) */
    AtomType atomType[NUM_VDWPAIRPRM_ATOMTYPE];  /**< identify atom types */
    //int32 atomParmID[2];   /**< index AtomParmArray */
  } VdwpairPrm;

  /**@brief Buckingham potential parameters for indicated pairs.
   * Functional form:  U = a*exp(-b*r) - c/r^6 */
  typedef struct BuckpairPrm_t {
    dreal a;               /**< energy (AMU*(A/fs)^2) */
    dreal b;               /**< inverse distance (1/A) */
    dreal c;               /**< units (AMU*(A/fs)^2)*(A^6) */
    AtomType atomType[NUM_BUCKPAIRPRM_ATOMTYPE];  /**< identify atom types */
    //int32 atomParmID[2];   /**< index AtomParmArray */
  } BuckpairPrm;

  /**@brief van der Waals pair parameters for nonbonded evaluation. */
  typedef struct VdwTableElem_t {
    dreal a, b;            /**< standard interactions */
    dreal a14, b14;        /**< modified 1-4 interactions */
  } VdwTableElem;

  /**@brief Buckingham pair parameters for nonbonded evaluation.
   * Safe extension removes non-physical well:  U = A/r^6 + B,  r^2 <= R^2 */
  typedef struct BuckTableElem_t {
    dreal a, b, c;         /**< standard interactions */
    dreal as, bs, rs2;     /**< safe extension, rs2 square of inner cutoff */
  } BuckTableElem;

  /**@brief Force field parameters container.
   *
   * This acts as a database of sets of force field parameters
   * indexed by atom types.
   *
   * Use of resize arrays and maps (hash tables) make any sequence of
   * operations have amortized cost O(1).  Any single _add() operation might
   * cost more if the array is resized.  All other operations are true O(1).
   */
  typedef struct ForcePrm_t {

    NonbPrm nonbprm;        /**< one copy of nonbonded parameters */

    /* tables (2D arrays) are size |atomprm|^2, indexed by two atomPrmIDs */
    Array vdwtable;         /**< 2D array of VdwTableElem */
    int32 vdwtablelen;      /**< save 1D length */
    Array bucktable;        /**< 2D array of BuckTableElem */
    int32 bucktablelen;     /**< save 1D length */

    /* force field parameter arrays */
    Array atomprm;          /**< array of AtomPrm */
    Array bondprm;          /**< array of BondPrm */
    Array angleprm;         /**< array of AnglePrm */
    Objarr dihedprm;        /**< array of DihedPrm (has array of DihedTerm) */
    Array imprprm;          /**< array of ImprPrm */
    Array vdwpairprm;       /**< array of VdwpairPrm */
    Array buckpairprm;      /**< array of BuckpairPrm */

    /* mapping atomType to index of corresponding FF parameter array */
    Arrmap atomprm_map;     /**< mapping of AtomPrm atomType to index */
    Arrmap bondprm_map;     /**< mapping of BondPrm atomType to index */
    Arrmap angleprm_map;    /**< mapping of AnglePrm atomType to index */
    Arrmap dihedprm_map;    /**< mapping of DihedPrm atomType to index */
    Arrmap imprprm_map;     /**< mapping of ImprPrm atomType to index */
    Arrmap vdwpairprm_map;  /**< mapping of VdwpairPrm atomType to index */
    Arrmap buckpairprm_map; /**< mapping of BuckpairPrm atomType to index */

    /*
     * whenever a parameter is removed, a hole opens up;
     * keep track of these indices to fill the hole on next add
     */
    Array atomprm_open;     /**< stack of open AtomPrm array index */
    Array bondprm_open;     /**< stack of open BondPrm array index */
    Array angleprm_open;    /**< stack of open AnglePrm array index */
    Array dihedprm_open;    /**< stack of open DihedPrm array index */
    Array imprprm_open;     /**< stack of open ImprPrm array index */
    Array vdwpairprm_open;  /**< stack of open VdwpairPrm array index */
    Array buckpairprm_open; /**< stack of open BuckpairPrm array index */

    int32 status;           /**< flags to indicate data modifications */

  } ForcePrm;

  /* constructor and destructor */
  int  ForcePrm_init(ForcePrm *);
  void ForcePrm_done(ForcePrm *);

  /* manage status flags */
  int32 ForcePrm_status(const ForcePrm *);
  void  ForcePrm_reset_status(ForcePrm *, int32 flags);

  /* manage nonbonded parameters */
  int ForcePrm_set_nonbprm(ForcePrm *, const NonbPrm *);
  const NonbPrm *ForcePrm_nonbprm(const ForcePrm *);

  /* manage van der Waals interaction table */
  int ForcePrm_setup_vdwtable(ForcePrm *);
  const VdwTableElem *ForcePrm_vdwtable(const ForcePrm *,
      int32 atomPrmID_0, int32 atomPrmID_1);
  const VdwTableElem *ForcePrm_vdwtable_array(const ForcePrm *);
  int32 ForcePrm_vdwtable_length(const ForcePrm *);

  /* manage safe switched Buckingham interaction table */
  int ForcePrm_setup_bucktable(ForcePrm *);
  const BuckTableElem *ForcePrm_bucktable(const ForcePrm *,
      int32 atomPrmID_0, int32 atomPrmID_1);
  const BuckTableElem *ForcePrm_bucktable_array(const ForcePrm *);
  int32 ForcePrm_bucktable_length(const ForcePrm *);
  int ForcePrm_calc_safebuck(ForcePrm *,
      dreal *as, dreal *bs, dreal *rs, dreal *urs, dreal *rmax, dreal *urmax,
      dreal a, dreal b, dreal c);

  /* manage atom parameters  */
  int32 ForcePrm_add_atomprm(ForcePrm *, const AtomPrm *);
  int32 ForcePrm_update_atomprm(ForcePrm *, const AtomPrm *);
  int32 ForcePrm_getid_atomprm(const ForcePrm *, const char *);
  int   ForcePrm_remove_atomprm(ForcePrm *, int32 id);

  const AtomPrm  *ForcePrm_atomprm(const ForcePrm *, int32 id);
  const AtomPrm  *ForcePrm_atomprm_array(const ForcePrm *);
  int32           ForcePrm_atomprm_array_length(const ForcePrm *);

  /* manage bond parameters  */
  int32 ForcePrm_add_bondprm(ForcePrm *, const BondPrm *);
  int32 ForcePrm_update_bondprm(ForcePrm *, const BondPrm *);
  int32 ForcePrm_getid_bondprm(const ForcePrm *,
      const char *, const char *);
  int   ForcePrm_remove_bondprm(ForcePrm *, int32 id);

  const BondPrm  *ForcePrm_bondprm(const ForcePrm *, int32 id);
  const BondPrm  *ForcePrm_bondprm_array(const ForcePrm *);
  int32           ForcePrm_bondprm_array_length(const ForcePrm *);

  /* manage angle parameters  */
  int32 ForcePrm_add_angleprm(ForcePrm *, const AnglePrm *);
  int32 ForcePrm_update_angleprm(ForcePrm *, const AnglePrm *);
  int32 ForcePrm_getid_angleprm(const ForcePrm *,
      const char *, const char *, const char *);
  int   ForcePrm_remove_angleprm(ForcePrm *, int32 id);

  const AnglePrm *ForcePrm_angleprm(const ForcePrm *, int32 id);
  const AnglePrm *ForcePrm_angleprm_array(const ForcePrm *);
  int32           ForcePrm_angleprm_array_length(const ForcePrm *);

  /* manage dihedral parameters  */
  int32 ForcePrm_add_dihedprm(ForcePrm *, const DihedPrm *);
  int32 ForcePrm_update_dihedprm(ForcePrm *, const DihedPrm *);
  int32 ForcePrm_getid_dihedprm(const ForcePrm *,
      const char *, const char *, const char *, const char *);
  int32 ForcePrm_matchid_dihedprm(const ForcePrm *,
      const char *, const char *, const char *, const char *);
  int   ForcePrm_remove_dihedprm(ForcePrm *, int32 id);

  const DihedPrm *ForcePrm_dihedprm(const ForcePrm *, int32 id);
  const DihedPrm *ForcePrm_dihedprm_array(const ForcePrm *);
  int32           ForcePrm_dihedprm_array_length(const ForcePrm *);

  /* manage improper parameters  */
  int32 ForcePrm_add_imprprm(ForcePrm *, const ImprPrm *);
  int32 ForcePrm_update_imprprm(ForcePrm *, const ImprPrm *);
  int32 ForcePrm_getid_imprprm(const ForcePrm *,
      const char *, const char *, const char *, const char *);
  int32 ForcePrm_matchid_imprprm(const ForcePrm *,
      const char *, const char *, const char *, const char *);
  int   ForcePrm_remove_imprprm(ForcePrm *, int32 id);

  const ImprPrm  *ForcePrm_imprprm(const ForcePrm *, int32 id);
  const ImprPrm  *ForcePrm_imprprm_array(const ForcePrm *);
  int32           ForcePrm_imprprm_array_length(const ForcePrm *);

  /* manage van der Waals pair parameters  */
  int32 ForcePrm_add_vdwpairprm(ForcePrm *, const VdwpairPrm *);
  int32 ForcePrm_update_vdwpairprm(ForcePrm *, const VdwpairPrm *);
  int32 ForcePrm_getid_vdwpairprm(const ForcePrm *,
      const char *, const char *);
  int   ForcePrm_remove_vdwpairprm(ForcePrm *, int32 id);

  const VdwpairPrm  *ForcePrm_vdwpairprm(const ForcePrm *, int32 id);
  const VdwpairPrm  *ForcePrm_vdwpairprm_array(const ForcePrm *);
  int32              ForcePrm_vdwpairprm_array_length(const ForcePrm *);

  /* manage Buckingham pair parameters  */
  int32 ForcePrm_add_buckpairprm(ForcePrm *, const BuckpairPrm *);
  int32 ForcePrm_update_buckpairprm(ForcePrm *, const BuckpairPrm *);
  int32 ForcePrm_getid_buckpairprm(const ForcePrm *,
      const char *, const char *);
  int   ForcePrm_remove_buckpairprm(ForcePrm *, int32 id);

  const BuckpairPrm *ForcePrm_buckpairprm(const ForcePrm *, int32 id);
  const BuckpairPrm *ForcePrm_buckpairprm_array(const ForcePrm *);
  int32              ForcePrm_buckpairprm_array_length(const ForcePrm *);


  /*
   * status flags indicate modifications to fundamental ForcePrm arrays
   */
  enum {
    FP_ATOMPRM_ADD        = 0x0000001,
    FP_ATOMPRM_UPDATE     = 0x0000002,
    FP_ATOMPRM_REMOVE     = 0x0000004,
    FP_ATOMPRM            =
      (FP_ATOMPRM_ADD | FP_ATOMPRM_UPDATE | FP_ATOMPRM_REMOVE),

    FP_BONDPRM_ADD        = 0x0000008,
    FP_BONDPRM_UPDATE     = 0x0000010,
    FP_BONDPRM_REMOVE     = 0x0000020,
    FP_BONDPRM            =
      (FP_BONDPRM_ADD | FP_BONDPRM_UPDATE | FP_BONDPRM_REMOVE),

    FP_ANGLEPRM_ADD       = 0x0000040,
    FP_ANGLEPRM_UPDATE    = 0x0000080,
    FP_ANGLEPRM_REMOVE    = 0x0000100,
    FP_ANGLEPRM           =
      (FP_ANGLEPRM_ADD | FP_ANGLEPRM_UPDATE | FP_ANGLEPRM_REMOVE),

    FP_DIHEDPRM_ADD       = 0x0000200,
    FP_DIHEDPRM_UPDATE    = 0x0000400,
    FP_DIHEDPRM_REMOVE    = 0x0000800,
    FP_DIHEDPRM           =
      (FP_DIHEDPRM_ADD | FP_DIHEDPRM_UPDATE | FP_DIHEDPRM_REMOVE),

    FP_IMPRPRM_ADD        = 0x0001000,
    FP_IMPRPRM_UPDATE     = 0x0002000,
    FP_IMPRPRM_REMOVE     = 0x0004000,
    FP_IMPRPRM            =
      (FP_IMPRPRM_ADD | FP_IMPRPRM_UPDATE | FP_IMPRPRM_REMOVE),

    FP_VDWPAIRPRM_ADD     = 0x0008000,
    FP_VDWPAIRPRM_UPDATE  = 0x0010000,
    FP_VDWPAIRPRM_REMOVE  = 0x0020000,
    FP_VDWPAIRPRM         =
      (FP_VDWPAIRPRM_ADD | FP_VDWPAIRPRM_UPDATE | FP_VDWPAIRPRM_REMOVE),

    FP_BUCKPAIRPRM_ADD    = 0x0040000,
    FP_BUCKPAIRPRM_UPDATE = 0x0080000,
    FP_BUCKPAIRPRM_REMOVE = 0x0100000,
    FP_BUCKPAIRPRM        =
      (FP_BUCKPAIRPRM_ADD | FP_BUCKPAIRPRM_UPDATE | FP_BUCKPAIRPRM_REMOVE),

    FP_ALLPRM = (FP_ATOMPRM | FP_BONDPRM | FP_ANGLEPRM |
        FP_DIHEDPRM | FP_IMPRPRM | FP_VDWPAIRPRM | FP_BUCKPAIRPRM)
  };


#ifdef __cplusplus
}
#endif

#endif /* MOLTYPES_FORCEPRM_H */
