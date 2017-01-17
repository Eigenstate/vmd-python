/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#ifndef NLENERGY_H
#define NLENERGY_H

#include "tcl.h"
#include "nlbase/nlbase.h"
#include "moltypes/moltypes.h"
#include "molfiles/molfiles.h"
#include "force/force.h"
#if 0
#include "nlenergy/nlforce.h"
#endif


/*
 * for debugging, NLERR(errcode) invokes the ERROR() macro
 * but then returns the expected TCL_ERROR
 */
#ifdef DEBUG
#define NLERR(errcode) ( ERROR(errcode), TCL_ERROR )
#else
#define NLERR(errcode) TCL_ERROR
#endif


#ifdef __cplusplus
extern "C" {
#endif

  /* select contribution to nonbonded potential energy terms (atomsel),
   * the _B selections are for defining pair interaction energies
   * between disjoint sets of atoms
   */
  enum {
    ASEL_ELEC   = 0x01,
    ASEL_VDW    = 0x02,
    ASEL_NONB   = (ASEL_ELEC | ASEL_VDW),
    ASEL_ELEC_B = 0x04,
    ASEL_VDW_B  = 0x08,
    ASEL_NONB_B = (ASEL_ELEC_B | ASEL_VDW_B)
  };

  enum {
    EVAL_ENERGY,
    EVAL_FORCE,
    EVAL_MINIMIZE
  };

  typedef struct NLEnergy_t {
    int idnum;       /* store nlenergy counter number */
    int molid;       /* molecule ID used to initialize */
    char *aselname;  /* name of the atom selection */

    /* map external atom numbering to internal, and back again:
     *   extatomid[i]  gives external atom number for ith (internal) atom
     *   atomid[iext]  gives internal atom index for external atom number iext
     */
    Array extatomid; /* array of int32, length is natoms */
    Array atomid;    /* array of int32, length is (lastid - firstid + 1) */
    int32 firstid;   /* smallest external atom number */
    int32 lastid;    /* largest external atom number */

    /* data container objects */
    ForcePrm fprm;
    boolean fulldirect;
    boolean fulldirectvdw;
    Topology topo;
    Exclude exclude;
    PdbAtom pdbpos;
    Coord coord;

    Energy ener;
    Fbonded fbon;
    Fnbcut fnbcut;
    boolean nb_overlap;
    int32 fnbcut_all;
    int32 fnbcut_subset;
    int32 fnbcut_disjoint;

    /* arrays of char that select potential energy terms */
    Array atomsel;    /* elements can be FALSE, ASEL_* */
    Array nonbsel;    /* elements can be FALSE, ASEL_* */
    Array bondsel;    /* elements can be FALSE, TRUE */
    Array anglesel;
    Array dihedsel;
    Array imprsel;
    Array invsel;

    /* arrays of int32 used to store array indices for nonbonded subsets */
    Array idnonb;
    Array idnonb_b;
    Array idnbvdw;
    Array idnbvdw_b;

  } NLEnergy;


  int NLEnergy_init(NLEnergy *);
  void NLEnergy_done(NLEnergy *);

  int NLEnergy_setup(NLEnergy *, Tcl_Interp *, int, int, const char *);

  int NLEnergy_parse_get(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_parse_set(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_parse_add(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_parse_remove(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_parse_missing(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_parse_read(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_parse_write(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_parse_eval(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[], int evalType);

  int NLEnergy_get_coord(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_atom(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_bond(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_angle(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_dihed(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_impr(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_atomprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_bondprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_angleprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_dihedprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_imprprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_vdwpairprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_nonbprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_get_cellbasis(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);

  int NLEnergy_set_coord(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_atom(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_atomprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_bondprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_angleprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_dihedprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_imprprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_vdwpairprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_nonbprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_set_cellbasis(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);

  int NLEnergy_add_bond(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_angle(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_dihed(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_impr(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_atomprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_bondprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_angleprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_dihedprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_imprprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_add_vdwpairprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);

  int NLEnergy_remove_bond(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_angle(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_dihed(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_impr(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_atomprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_bondprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_angleprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_dihedprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_imprprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_remove_vdwpairprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);

  int NLEnergy_missing_atomprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_missing_bondprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_missing_angleprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_missing_dihedprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_missing_imprprm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);

  int NLEnergy_read_xplor(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_read_charmm(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_read_psf(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_read_pdb(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_read_namdbin(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);

  int NLEnergy_eval_force(NLEnergy *p);
#if 0
  int NLEnergy_energy_bond(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_energy_angle(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_energy_dihed(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_energy_impr(NLEnergy *, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[]);
  int NLEnergy_energy_nonbonded(NLEnergy *, int32 nbpairType,
      Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
#endif

#ifdef __cplusplus
}
#endif

#endif /* NLENERGY_H */
