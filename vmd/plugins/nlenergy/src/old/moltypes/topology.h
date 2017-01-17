/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/topology.h
 * @brief   Topology data container.
 * @author  David J. Hardy
 * @date    Mar. 2008
 */

/*
 * TODO:
 * Still need to provide sorting routines.
 */

#ifndef MOLTYPES_TOPOLOGY_H
#define MOLTYPES_TOPOLOGY_H

#include "moltypes/forceprm.h"

#ifdef __cplusplus
extern "C" {
#endif


  /**@brief Stores short C-style string for atom and residue names. */
  typedef char AtomName[8];
  typedef char ResName[12];

  enum {
    NUM_BOND_ATOM  = 2,
    NUM_ANGLE_ATOM = 3,
    NUM_DIHED_ATOM = 4,
    NUM_IMPR_ATOM  = 4,
    NUM_EXCL_ATOM  = 2
  };

  /**@brief Defines each individual atom in system. */
  typedef struct Atom_t {
    dreal m;             /**< mass (AMU) */
    dreal q;             /**< charge (e) */
    dreal inv_mass;      /**< 1/mass (set to 0 if mass is 0) */
    AtomName atomName;   /**< string to identify atom name */
    AtomType atomType;   /**< string to identify atom type */
    ResName resName;     /**< string to identify residue name */
    int32 residue;       /**< residue number */
    int32 atomPrmID;     /**< index ForcePrm array of AtomPrm */
    int32 clusterID;     /**< use smallest numbered atom in connected cluster */
    int32 clusterSize;   /**< total number of atoms in this cluster */
    int32 parentID;      /**< index of parent (equal to my index if _HEAVY) */
    int32 externID;      /**< external index never changes with sorting */
    int32 atomInfo;      /**< bitwise flags that denote special properties */
  } Atom;

  /**@brief atomInfo flags
   *
   * Take the atom type (one of HEAVY, HYDROGEN, DRUDE, LONEPAIR)
   * and OR it with zero or more secondary properties.
   *
   * Do AND with ATOM_TYPE to mask out just the type, ATOM_PROPERTY
   * to mask out secondary properties, and ATOM_MASK to mask out all
   * possible bit fields.
   *
   * The atom type is designated by looking at the mass of the particle:
   *   HEAVY:     mass > MASS_HYDROGEN_MAX
   *   HYDROGEN:  MASS_HYDROGEN_MIN <= mass <= MASS_HYDROGEN_MAX
   *   DRUDE:     0 < mass < MASS_HYDROGEN_MIN
   *   LONEPAIR:  mass = 0
   */
  enum {
    ATOM_HEAVY    = 0x00,  /**< atom is heavy (normal) */
    ATOM_DRUDE    = 0x01,  /**< atom is Drude particle (very light) */
    ATOM_LONEPAIR = 0x02,  /**< atom is massless charge site */
    ATOM_HYDROGEN = 0x03,  /**< atom is hydrogen (light) */
    ATOM_TYPE     = 0x03,  /**< mask to filter independent atom types */
    ATOM_WATER    = 0x10,  /**< atom belongs to water molecule */
    ATOM_FIXED    = 0x20,  /**< atom is fixed in space */
    ATOM_PROPERTY = 0x30,  /**< mask secondary property fields */
    ATOM_MASK     = 0x33   /**< mask out all possible bit fields */
  };

  /**@brief Defines each bond in system. */
  typedef struct Bond_t {
    int32 atomID[NUM_BOND_ATOM];   /**< index Topology array of Atom */
    int32 bondPrmID;               /**< index ForcePrm array of BondPrm */
  } Bond;

  /**@brief Defines each angle in system. */
  typedef struct Angle_t {
    int32 atomID[NUM_ANGLE_ATOM];  /**< index Topology array of Atom */
    int32 anglePrmID;              /**< index ForcePrm array of AnglePrm */
  } Angle;

  /**@brief Defines each dihedral in system. */
  typedef struct Dihed_t {
    int32 atomID[NUM_DIHED_ATOM];  /**< index Topology array of Atom */
    int32 dihedPrmID;              /**< index ForcePrm array of DihedPrm */
  } Dihed;

  /**@brief Defines each improper in system. */
  typedef struct Impr_t {
    int32 atomID[NUM_IMPR_ATOM];   /**< index Topology array of Atom */
    int32 imprPrmID;               /**< index ForcePrm array of ImprPrm */
  } Impr;

  /**@brief Defines explicit exclusions in system
   * (as opposed to implicitly defining exclusions by setting,
   * for instance, "exclude=scaled1-4" - use of Excl is deprecated). */
  typedef struct Excl_t {
    int32 atomID[NUM_EXCL_ATOM];   /**< index Topology array of Atom */
  } Excl;


  typedef struct Topology_t {

    Array atom;            /**< array of Atom */
    Array bond;            /**< array of Bond */
    Array angle;           /**< array of Dihed */
    Array dihed;           /**< array of Angle */
    Array impr;            /**< array of Impr */
    Array excl;            /**< array of Excl */

    Arrmap bond_map;       /**< map atom IDs of Bond to bond array index */
    Arrmap angle_map;      /**< map atom IDs of Angle to angle array index */
    Arrmap dihed_map;      /**< map atom IDs of Dihed to dihed array index */
    Arrmap impr_map;       /**< map atom IDs of Impr to impr array index */
    Arrmap excl_map;       /**< map atom IDs of Excl to excl array index */

    Array bond_open;       /**< stack of open indices from Bond removal */
    Array angle_open;      /**< stack of open indices from Angle removal */
    Array dihed_open;      /**< stack of open indices from Dihed removal */
    Array impr_open;       /**< stack of open indices from Impr removal */
    Array excl_open;       /**< stack of open indices from Excl removal */

    Objarr atom_bondlist;  /**< for each atom, Idlist of dependent bonds */
    Objarr atom_anglelist; /**< for each atom, Idlist of dependent angles */
    Objarr atom_dihedlist; /**< for each atom, Idlist of dependent dihedrals */
    Objarr atom_imprlist;  /**< for each atom, Idlist of dependent impropers */
    Array atom_indexmap;   /**< maps external atom ID to internal atom ID */

    Objarr bond_anglelist; /**< for each bond, Idlist of dependent angles */
    Objarr bond_dihedlist; /**< for each bond, Idlist of dependent dihedrals */
    Objarr bond_imprlist;  /**< for each bond, Idlist of dependent impropers */

    const ForcePrm *fprm;

    int32 status;

  } Topology;

  /* constructor and destructor */
  int  Topology_init(Topology *, const ForcePrm *);
  void Topology_done(Topology *);

  /* manage status flags */
  int32 Topology_status(const Topology *);
  void  Topology_reset_status(Topology *, int32 flags);

  /* manage atoms */
  int   Topology_setmaxnum_atom(Topology *, int32);
  int32 Topology_add_atom(Topology *, const Atom *);
  int   Topology_update_atom(Topology *, const Atom *, int32 id);

  int32 Topology_setprm_atom(Topology *, int32 id);
  int   Topology_setprm_atom_array(Topology *);

  const Atom   *Topology_atom(const Topology *, int32 id);
  const Atom   *Topology_atom_array(const Topology *);
  int32         Topology_atom_array_length(const Topology *);

  const Idlist *Topology_atom_bondlist(const Topology *, int32 id);
  const Idlist *Topology_atom_anglelist(const Topology *, int32 id);
  const Idlist *Topology_atom_dihedlist(const Topology *, int32 id);
  const Idlist *Topology_atom_imprlist(const Topology *, int32 id);

  int   Topology_sort_atom_array(Topology *);
  int   Topology_setup_atom_cluster(Topology *);
  int   Topology_setup_atom_parent(Topology *);

  const int32  *Topology_atom_indexmap(const Topology *);

  /* manage bonds */
  int   Topology_setmaxnum_bond(Topology *, int32);
  int32 Topology_add_bond(Topology *, const Bond *);
  int32 Topology_getid_bond(const Topology *, int32 a0, int32 a1);
  int   Topology_remove_bond(Topology *, int32 id);

  int32 Topology_setprm_bond(Topology *, int32 id);
  int   Topology_setprm_bond_array(Topology *);

  const Bond   *Topology_bond(const Topology *, int32 id);
  const Bond   *Topology_bond_array(const Topology *);
  int32         Topology_bond_array_length(const Topology *);

  const Idlist *Topology_bond_anglelist(const Topology *, int32 id);
  const Idlist *Topology_bond_dihedlist(const Topology *, int32 id);
  const Idlist *Topology_bond_imprlist(const Topology *, int32 id);

  int   Topology_compact_bond_array(Topology *);
  /*
  int   Topology_sort_bond_array(Topology *);
   */

  /* ALSO NEED TO TAKE PBC INTO ACCOUNT
  int32 Topology_generate_bond_array(Topology *,
      const dvec *pos, int32 natoms, dreal distance);
      */

  /* manage angles */
  int   Topology_setmaxnum_angle(Topology *, int32);
  int32 Topology_add_angle(Topology *, const Angle *);
  int32 Topology_getid_angle(const Topology *, int32 a0, int32 a1, int32 a2);
  int   Topology_remove_angle(Topology *, int32 id);

  int32 Topology_setprm_angle(Topology *, int32 id);
  int   Topology_setprm_angle_array(Topology *);

  const Angle  *Topology_angle(const Topology *, int32 id);
  const Angle  *Topology_angle_array(const Topology *);
  int32         Topology_angle_array_length(const Topology *);

  int   Topology_compact_angle_array(Topology *);
  /*
  int   Topology_sort_angle_array(Topology *);
   */

  /*
  int32 Topology_generate_angle_array(Topology *);
   */

  /* manage dihedrals */
  int   Topology_setmaxnum_dihed(Topology *, int32);
  int32 Topology_add_dihed(Topology *, const Dihed *);
  int32 Topology_getid_dihed(const Topology *,
      int32 a0, int32 a1, int32 a2, int32 a3);
  int   Topology_remove_dihed(Topology *, int32 id);

  int32 Topology_setprm_dihed(Topology *, int32 id);
  int   Topology_setprm_dihed_array(Topology *);

  const Dihed  *Topology_dihed(const Topology *, int32 id);
  const Dihed  *Topology_dihed_array(const Topology *);
  int32         Topology_dihed_array_length(const Topology *);

  int   Topology_compact_dihed_array(Topology *);
  /*
  int   Topology_sort_dihed_array(Topology *);
   */

  /*
  int32 Topology_generate_dihed_array(Topology *);
   */

  /* manage impropers */
  int   Topology_setmaxnum_impr(Topology *, int32);
  int32 Topology_add_impr(Topology *, const Impr *);
  int32 Topology_getid_impr(const Topology *,
      int32 a0, int32 a1, int32 a2, int32 a3);
  int   Topology_remove_impr(Topology *, int32 id);

  int32 Topology_setprm_impr(Topology *, int32 id);
  int   Topology_setprm_impr_array(Topology *);

  const Impr   *Topology_impr(const Topology *, int32 id);
  const Impr   *Topology_impr_array(const Topology *);
  int32         Topology_impr_array_length(const Topology *);

  int   Topology_compact_impr_array(Topology *);
  /*
  int   Topology_sort_impr_array(Topology *);
   */

  /* manage explicit exclusions */
  int   Topology_setmaxnum_excl(Topology *, int32);
  int32 Topology_add_excl(Topology *, const Excl *);
  int32 Topology_getid_excl(const Topology *, int32 a0, int32 a1);
  int   Topology_remove_excl(Topology *, int32 id);

  const Excl   *Topology_excl(const Topology *, int32 id);
  const Excl   *Topology_excl_array(const Topology *);
  int32         Topology_excl_array_length(const Topology *);

  int   Topology_compact_excl_array(Topology *);
  /*
  int   Topology_sort_excl_array(Topology *);
   */


  /*
   * status flags indicate modifications to Topology
   */
  enum {
    TOPO_ATOM_ADD      = 0x0000001,
    TOPO_ATOM_UPDATE   = 0x0000002,
    TOPO_ATOM_MISSPRM  = 0x0000004,
    TOPO_ATOM          =
      (TOPO_ATOM_ADD | TOPO_ATOM_UPDATE | TOPO_ATOM_MISSPRM),
    TOPO_BOND_ADD      = 0x0000008,
    TOPO_BOND_REMOVE   = 0x0000010,
    TOPO_BOND_MISSPRM  = 0x0000020,
    TOPO_BOND          =
      (TOPO_BOND_ADD | TOPO_BOND_REMOVE | TOPO_BOND_MISSPRM),
    TOPO_ANGLE_ADD     = 0x0000040,
    TOPO_ANGLE_REMOVE  = 0x0000080,
    TOPO_ANGLE_MISSPRM = 0x0000100,
    TOPO_ANGLE         =
      (TOPO_ANGLE_ADD | TOPO_ANGLE_REMOVE | TOPO_ANGLE_MISSPRM),
    TOPO_DIHED_ADD     = 0x0000200,
    TOPO_DIHED_REMOVE  = 0x0000400,
    TOPO_DIHED_MISSPRM = 0x0000800,
    TOPO_DIHED         =
      (TOPO_DIHED_ADD | TOPO_DIHED_REMOVE | TOPO_DIHED_MISSPRM),
    TOPO_IMPR_ADD      = 0x0001000,
    TOPO_IMPR_REMOVE   = 0x0002000,
    TOPO_IMPR_MISSPRM  = 0x0004000,
    TOPO_IMPR          =
      (TOPO_IMPR_ADD | TOPO_IMPR_REMOVE | TOPO_IMPR_MISSPRM),
    TOPO_EXCL_ADD      = 0x0008000,
    TOPO_EXCL_REMOVE   = 0x0010000,
    TOPO_EXCL          =
      (TOPO_EXCL_ADD | TOPO_EXCL_REMOVE)
  };


#ifdef __cplusplus
}
#endif

#endif /* MOLTYPES_TOPOLOGY_H */
