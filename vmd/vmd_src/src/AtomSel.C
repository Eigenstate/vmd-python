/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: AtomSel.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.175 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Parse and maintain the data for selecting atoms.
 *
 ***************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#include "Atom.h"
#include "AtomSel.h"
#include "DrawMolecule.h"
#include "MoleculeList.h"
#include "VMDApp.h"
#include "Inform.h"
#include "SymbolTable.h"
#include "ParseTree.h"
#include "JRegex.h"
#include "VolumetricData.h"
#include "PeriodicTable.h"
#include "DrawRingsUtils.h"

// 'all'
static int atomsel_all(void * ,int, int *) {
  return 1;
}

// 'none'
static int atomsel_none(void *, int num, int *flgs) {
  for (int i=num-1; i>=0; i--) {
    flgs[i] = FALSE;
  }
  return 1;
}

#define generic_atom_data(fctnname, datatype, field)		       \
static int fctnname(void *v, int num, datatype *data, int *flgs) {     \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;      \
  for (int i=0; i<num; i++) {					       \
    if (flgs[i]) {						       \
      data[i] = atom_sel_mol->atom(i)->field;		               \
    }								       \
  }								       \
  return 1;							       \
}


#define generic_set_atom_data(fctnname, datatype, fieldtype, field)    \
static int fctnname(void *v, int num, datatype *data, int *flgs) {     \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;      \
  int i;							       \
  for (i=0; i<num; i++) {					       \
    if (flgs[i]) {						       \
      atom_sel_mol->atom(i)->field = (fieldtype) data[i];	       \
    }								       \
  }								       \
  return 1;							       \
}


// 'name'
static int atomsel_name(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i])
      data[i] = (atom_sel_mol->atomNames).name(
		  atom_sel_mol->atom(i)->nameindex);
  }
  return 1;
}


// 'type'
static int atomsel_type(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i])
      data[i] = (atom_sel_mol->atomTypes).name(
    		  atom_sel_mol->atom(i)->typeindex);
  }
  return 1;
}

// XXX probably only use this for internal testing
// 'backbonetype'
static int atomsel_backbonetype(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      switch (atom_sel_mol->atom(i)->atomType) {
        case ATOMNORMAL:
          data[i] = "normal";
          break;

        case ATOMPROTEINBACK:
          data[i] = "proteinback";
          break;

        case ATOMNUCLEICBACK:
          data[i] = "nucleicback";
          break;

        case ATOMHYDROGEN:
          data[i] = "hydrogen";
          break;

        default:
          data[i] = "unknown";
          break;
      }
    }
  }
  return 1;
}


// XXX probably only use this for internal testing
// 'residuetype'
static int atomsel_residuetype(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      switch (atom_sel_mol->atom(i)->residueType) {
        case RESNOTHING:
          data[i] = "nothing";
          break;

        case RESPROTEIN:
          data[i] = "protein";
          break;

        case RESNUCLEIC:
          data[i] = "nucleic";
          break;

        case RESWATERS:
          data[i] = "waters";
          break;

        default:
          data[i] = "unknown";
          break;
      }
    }
  }
  return 1;
}


// 'atomicnumber'
generic_atom_data(atomsel_atomicnumber, int, atomicnumber)

static int atomsel_set_atomicnumber(void *v, int num, int *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  int i;
  for (i=0; i<num; i++) {
    if (flgs[i]) {
      atom_sel_mol->atom(i)->atomicnumber = (int) data[i];
    }
  }
  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::ATOMICNUMBER);
  return 1;
}


// 'element'
static int atomsel_element(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i])
      data[i] = get_pte_label(atom_sel_mol->atom(i)->atomicnumber);
  }
  return 1;
}
static int atomsel_set_element(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      int idx = get_pte_idx_from_string((const char *)(data[i]));
      atom_sel_mol->atom(i)->atomicnumber = idx;
    }
  }
  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::ATOMICNUMBER);
  return 1;
}


// 'index'
static int atomsel_index(void *v, int num, int *data, int *flgs) { 
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      data[i] = i; // zero-based
    }
  }
  return 1;
}


// 'serial' (one-based atom index)
static int atomsel_serial(void *v, int num, int *data, int *flgs) { 
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      data[i] = i + 1; // one-based
    }
  }
  return 1;
}


// 'fragment'
static int atomsel_fragment(void *v, int num, int *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      int residue = atom_sel_mol->atom(i)->uniq_resid;
      data[i] = (atom_sel_mol->residue(residue))->fragment;
    }
  }
  return 1;
}


// 'numbonds'
generic_atom_data(atomsel_numbonds, int, bonds)


// 'residue'
generic_atom_data(atomsel_residue, int, uniq_resid)


// 'resname'
static int atomsel_resname(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i])
      data[i] = (atom_sel_mol->resNames).name(
                 atom_sel_mol->atom(i)->resnameindex);
  }
  return 1;
}


// 'altloc'
static int atomsel_altloc(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                 
  for (int i=0; i<num; i++) {
    if (flgs[i])
      data[i] = (atom_sel_mol->altlocNames).name(
        atom_sel_mol->atom(i)->altlocindex);
  }
  return 1;
}
static int atomsel_set_altloc(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                 
  NameList<int> &arr = atom_sel_mol->altlocNames;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      int ind = arr.add_name((const char *)(data[i]), arr.num());
      atom_sel_mol->atom(i)->altlocindex = ind;
    }
  }
  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::ALTLOC);
  return 1;
}


// 'insertion'
generic_atom_data(atomsel_insertion, const char *, insertionstr)


// 'chain'
static int atomsel_chain(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                 
  for (int i=0; i<num; i++) {
    if (flgs[i])
      data[i] = (atom_sel_mol->chainNames).name(
 	          atom_sel_mol->atom(i)->chainindex);
  }
  return 1;
}


// 'segname'
static int atomsel_segname(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i])
      data[i] = (atom_sel_mol->segNames).name(
                 atom_sel_mol->atom(i)->segnameindex);
  }
  return 1;
}


// The next few set functions affect the namelists kept in BaseMolecule
static int atomsel_set_name(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  NameList<int> &arr = atom_sel_mol->atomNames;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      int ind = arr.add_name((const char *)(data[i]), arr.num());
      atom_sel_mol->atom(i)->nameindex = ind;
    }
  }
  return 1;
}

static int atomsel_set_type(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  NameList<int> &arr = atom_sel_mol->atomTypes;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      int ind = arr.add_name((const char *)(data[i]), arr.num());
      atom_sel_mol->atom(i)->typeindex = ind;
    }
  }
  return 1;
}

static int atomsel_set_resname(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  NameList<int> &arr = atom_sel_mol->resNames;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      int ind = arr.add_name((const char *)(data[i]), arr.num());
      atom_sel_mol->atom(i)->resnameindex = ind;
    }
  }
  return 1;
}

static int atomsel_set_chain(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  NameList<int> &arr = atom_sel_mol->chainNames;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      int ind = arr.add_name((const char *)(data[i]), arr.num());
      atom_sel_mol->atom(i)->chainindex = ind;
    }
  }
  return 1;
}

static int atomsel_set_segname(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  NameList<int> &arr = atom_sel_mol->segNames;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      int ind = arr.add_name((const char *)(data[i]), arr.num());
      atom_sel_mol->atom(i)->segnameindex = ind;
    }
  }
  return 1;
}


// 'radius'
static int atomsel_radius(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *radius = atom_sel_mol->radius();
  for (int i=0; i<num; i++)
    if (flgs[i]) data[i] = radius[i];
  return 1;
}
static int atomsel_set_radius(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *radius = atom_sel_mol->radius();
  for (int i=0; i<num; i++)
    if (flgs[i]) radius[i] = (float) data[i];

  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::RADIUS);

  // Force min/max atom radii to be updated since we've updated the data
  atom_sel_mol->set_radii_changed();

  return 1;
}


// 'mass'
static int atomsel_mass(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *mass = atom_sel_mol->mass();
  for (int i=0; i<num; i++)
    if (flgs[i]) data[i] = mass[i];
  return 1;
}
static int atomsel_set_mass(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *mass = atom_sel_mol->mass();
  for (int i=0; i<num; i++)
    if (flgs[i]) mass[i] = (float) data[i];
  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::MASS);
  return 1;
}


// 'charge'
static int atomsel_charge(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *charge = atom_sel_mol->charge();
  for (int i=0; i<num; i++)
    if (flgs[i]) data[i] = charge[i];
  return 1;
}
static int atomsel_set_charge(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *charge = atom_sel_mol->charge();
  for (int i=0; i<num; i++)
    if (flgs[i]) charge[i] = (float) data[i];
  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::CHARGE);
  return 1;
}


// 'beta'
static int atomsel_beta(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *beta = atom_sel_mol->beta();
  for (int i=0; i<num; i++)
    if (flgs[i]) data[i] = beta[i];
  return 1;
}
static int atomsel_set_beta(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *beta = atom_sel_mol->beta();
  for (int i=0; i<num; i++)
    if (flgs[i]) beta[i] = (float) data[i];
  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::BFACTOR);
  return 1;
}


// 'occupancy?'
static int atomsel_occupancy(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *occupancy = atom_sel_mol->occupancy();
  for (int i=0; i<num; i++)
    if (flgs[i]) data[i] = occupancy[i];
  return 1;
}
static int atomsel_set_occupancy(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  float *occupancy = atom_sel_mol->occupancy();
  for (int i=0; i<num; i++)
    if (flgs[i]) occupancy[i] = (float) data[i];
  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::OCCUPANCY);
  return 1;
}


// 'flags'
static int atomsel_flags(void *v, int num, int *data, int *flgs, int idx) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  unsigned char *flags = atom_sel_mol->flags();

  printf("atomflags: %d\n", idx);
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      data[i] = (flags[i] >> idx) & 0x1U;
//      printf(" atom[%d] flags: %d  masked: %d\n", i, flags[i], data[i]);
    }
  }
  return 1;
}
static int atomsel_set_flags(void *v, int num, int *data, int *flgs, int idx) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  unsigned char *flags = atom_sel_mol->flags();
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      unsigned int val = (data[i] != 0);
      flags[i] = (flags[i] & (0xff ^ (0x1U << idx))) | (val << idx);
    }
  }

  // when user sets data fields they are marked as valid fields in BaseMolecule
  atom_sel_mol->set_dataset_flag(BaseMolecule::ATOMFLAGS);
  return 1;
}

//
// provide separate callbacks for the different fields due to limitations of
// the existing internal APIs
//
static int atomsel_flag0(void *v, int num, int *data, int *flgs) {
  return atomsel_flags(v, num, data, flgs, 0);
}
static int atomsel_set_flag0(void *v, int num, int *data, int *flgs) {
  return atomsel_set_flags(v, num, data, flgs, 0);
}

static int atomsel_flag1(void *v, int num, int *data, int *flgs) {
  return atomsel_flags(v, num, data, flgs, 1);
}
static int atomsel_set_flag1(void *v, int num, int *data, int *flgs) {
  return atomsel_set_flags(v, num, data, flgs, 1);
}

static int atomsel_flag2(void *v, int num, int *data, int *flgs) {
  return atomsel_flags(v, num, data, flgs, 2);
}
static int atomsel_set_flag2(void *v, int num, int *data, int *flgs) {
  return atomsel_set_flags(v, num, data, flgs, 2);
}

static int atomsel_flag3(void *v, int num, int *data, int *flgs) {
  return atomsel_flags(v, num, data, flgs, 3);
}
static int atomsel_set_flag3(void *v, int num, int *data, int *flgs) {
  return atomsel_set_flags(v, num, data, flgs, 3);
}

static int atomsel_flag4(void *v, int num, int *data, int *flgs) {
  return atomsel_flags(v, num, data, flgs, 4);
}
static int atomsel_set_flag4(void *v, int num, int *data, int *flgs) {
  return atomsel_set_flags(v, num, data, flgs, 4);
}

static int atomsel_flag5(void *v, int num, int *data, int *flgs) {
  return atomsel_flags(v, num, data, flgs, 5);
}
static int atomsel_set_flag5(void *v, int num, int *data, int *flgs) {
  return atomsel_set_flags(v, num, data, flgs, 5);
}

static int atomsel_flag6(void *v, int num, int *data, int *flgs) {
  return atomsel_flags(v, num, data, flgs, 6);
}
static int atomsel_set_flag6(void *v, int num, int *data, int *flgs) {
  return atomsel_set_flags(v, num, data, flgs, 6);
}

static int atomsel_flag7(void *v, int num, int *data, int *flgs) {
  return atomsel_flags(v, num, data, flgs, 7);
}
static int atomsel_set_flag7(void *v, int num, int *data, int *flgs) {
  return atomsel_set_flags(v, num, data, flgs, 7);
}



// 'resid'
static int atomsel_resid(void *v, int num, int *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      data[i] = atom_sel_mol->atom(i)->resid;
    }
  }
  return 1;
}
static int atomsel_set_resid(void *v, int num, int *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i])
      atom_sel_mol->atom(i)->resid = data[i];
  }
  return 1;
}



#define generic_atom_boolean(fctnname, comparison)		      \
static int fctnname(void *v, int num, int *flgs) {                    \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;     \
  for (int i=0; i<num; i++) {					      \
    if (flgs[i]) {						      \
      flgs[i] = atom_sel_mol->atom(i)->comparison;	              \
    }								      \
  }								      \
  return 1;							      \
}

// 'backbone'
static int atomsel_backbone(void *v, int num, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      const MolAtom *a = atom_sel_mol->atom(i);
      flgs[i] = (a->atomType == ATOMPROTEINBACK ||
                 a->atomType == ATOMNUCLEICBACK);
    }
  }
  return 1;
}

// 'h' (hydrogen)
generic_atom_boolean(atomsel_hydrogen, atomType == ATOMHYDROGEN)


// 'protein'
generic_atom_boolean(atomsel_protein, residueType == RESPROTEIN)


// 'nucleic'
generic_atom_boolean(atomsel_nucleic, residueType == RESNUCLEIC)


// 'water'
generic_atom_boolean(atomsel_water, residueType == RESWATERS)

#define generic_sstruct_boolean(fctnname, comparison)                    \
static int fctnname(void *v, int num, int *flgs) {                       \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;        \
  atom_sel_mol->need_secondary_structure(1);                             \
  for (int i=0; i<num; i++) {                                            \
    int s;                                                               \
    if (flgs[i]) {                                                       \
      s = atom_sel_mol->residue(                                         \
                                  atom_sel_mol->atom(i)->uniq_resid      \
                                  )->sstruct;                            \
      if (!comparison) {                                                 \
        flgs[i] = 0;                                                     \
      }                                                                  \
    }                                                                    \
  }                                                                      \
  return 1;                                                              \
}

// once I define a structure, I don't need to recompute; hence the 
//   need_secondary_structure(0);
#define generic_set_sstruct_boolean(fctnname, newval)                    \
static int fctnname(void *v, int num, int *data, int *flgs) {            \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;        \
  atom_sel_mol->need_secondary_structure(0);                             \
  for (int i=0; i<num; i++) {                                            \
    if (flgs[i] && data[i]) {                                            \
	atom_sel_mol->residue(atom_sel_mol->atom(i)->uniq_resid)         \
	  ->sstruct = newval;                                            \
    }                                                                    \
  }                                                                      \
  return 1;                                                              \
}


// XXX recursive routines should be replaced by an iterative version with
// it's own stack, so that huge molecules don't overflow the stack
static void recursive_find_sidechain_atoms(BaseMolecule *mol, int *sidechain,
					   int atom_index) {
  // Have I been here before?
  if (sidechain[atom_index] == 2) 
    return;

  // Is this a backbone atom
  MolAtom *atom = mol->atom(atom_index);
  if (atom->atomType == ATOMPROTEINBACK ||
      atom->atomType == ATOMNUCLEICBACK) return;
  
  // mark this atom
  sidechain[atom_index] = 2;

  // try the atoms to which this is bonded
  for (int i=0; i<atom->bonds; i++) {
    recursive_find_sidechain_atoms(mol, sidechain, atom->bondTo[i]);
  }
}

// give an array where 1 indicates an atom on the sidechain, find the
// connected atoms which are also on the sidechain
static void find_sidechain_atoms(BaseMolecule *mol, int *sidechain) {
  for (int i=0; i<mol->nAtoms; i++) {
    if (sidechain[i] == 1) {
      recursive_find_sidechain_atoms(mol, sidechain, i);
    }
  }
}


// 'sidechain' is tricky.  I start from a protein CA, pick a bond which
// isn't along the backbone (and discard it once if it is a hydrogen),
// and follow the atoms until I stop at another backbone atom or run out
// of atoms
static int atomsel_sidechain(void *v, int num, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;               
  const float *mass = atom_sel_mol->mass();
  int i;

  // generate a list of the "CB" atoms (or whatever they are)
  int *seed = new int[num];
  memset(seed, 0, num * sizeof(int));

  // get the CA and HA2 name index
  int CA = atom_sel_mol->atomNames.typecode((char *) "CA");

  // for each protein
  for (int pfrag=atom_sel_mol->pfragList.num()-1; pfrag>=0; pfrag--) {
    // get a residue
    Fragment &frag = *(atom_sel_mol->pfragList[pfrag]);
    for (int res = frag.num()-1; res >=0; res--) {
      // find the CA
      int atomid = atom_sel_mol->find_atom_in_residue(CA, frag[res]);
      if (atomid < 0) {
        msgErr << "atomselection: sidechain: cannot find CA" << sendmsg;
        continue;
      }
      // find at most two neighbors which are not on the backbone
      MolAtom *atom = atom_sel_mol->atom(atomid);
      int b1 = -1, b2 = -1;
      for (i=0; i<atom->bonds; i++) {
        int bondtype = atom_sel_mol->atom(atom->bondTo[i])->atomType;
        if (bondtype == ATOMNORMAL || bondtype == ATOMHYDROGEN) {
          if (b1 == -1) {
            b1 = atom->bondTo[i];
          } else {
            if (b2 == -1) {
              b2 = atom->bondTo[i];
            } else {
              msgErr << "atomselection: sidechain: protein residue index " 
                     << res << ", C-alpha index " << i << " has more than "
                     << "two non-backbone bonds; ignoring the others"
                     << sendmsg;
            }
          }
        }
      }
      if (b1 == -1) 
        continue;

      if (b2 != -1) {  // find the right one
        // first check the number of atoms and see if I have a lone H
        int c1 = atom_sel_mol->atom(b1)->bonds;
        int c2 = atom_sel_mol->atom(b2)->bonds;
        if (c1 == 1 && c2 > 1) {
          b1 = b2;
        } else if (c2 == 1 && c1 > 1) {
#if 1
          // XXX get rid of bogus self-assignment
          seed[b1] = 1; // found the right one; it is b1.
          continue; // b1 remains b1
#else
          b1 = b1;
#endif
        } else if (c1 ==1 && c2 == 1) {
          // check the masses
          float m1 = mass[b1];
          float m2 = mass[b2];
          if (m1 > 2.3 && m2 <= 2.3) {
            b1 = b2;
          } else if (m2 > 2.3 && m1 <= 2.3) {
#if 1
            // XXX get rid of bogus self-assignment
            seed[b1] = 1; // found the right one; it is b1.
            continue; // b1 remains b1
#else
            b1 = b1;
#endif
          } else if (m1 <= 2.0 && m2 <= 2.3) {
            // should have two H's, find the "first" of these
            if (strcmp(
              (atom_sel_mol->atomNames).name(atom_sel_mol->atom(b1)->nameindex),
              (atom_sel_mol->atomNames).name(atom_sel_mol->atom(b2)->nameindex)
              ) > 0) {
              b1 = b2;
            } // else b1 = b1
          } else {
            msgErr << "atomselect: sidechain:  protein residue index " 
                   << res << ", C-alpha index " << i << " has sidechain-like "
                   << "atom (indices " << b1 << " and " << b2 << "), and "
                   << "I cannot determine which to call a sidechain -- "
                   << "I'll guess" << sendmsg;
            if (strcmp(
              (atom_sel_mol->atomNames).name(atom_sel_mol->atom(b1)->nameindex),
              (atom_sel_mol->atomNames).name(atom_sel_mol->atom(b2)->nameindex)
              ) > 0) {
              b1 = b2;
            }
          } // punted
        } // checked connections and masses
      } // found the right one; it is b1.
      seed[b1] = 1;

    } // loop over residues
  } // loop over protein fragments

  // do the search for all the sidechain atoms (returned in seed)
  find_sidechain_atoms(atom_sel_mol, seed);

  // set the return values
  for (i=0; i<num; i++) {
    if (flgs[i]) 
      flgs[i] = (seed[i] != 0);
  }

  delete [] seed;

  return 1;
}


// which of these are helices?
generic_sstruct_boolean(atomsel_helix, (s == SS_HELIX_ALPHA ||
                                        s == SS_HELIX_3_10  ||
                                        s == SS_HELIX_PI))
generic_sstruct_boolean(atomsel_alpha_helix, (s == SS_HELIX_ALPHA))
generic_sstruct_boolean(atomsel_3_10_helix, (s == SS_HELIX_3_10))
generic_sstruct_boolean(atomsel_pi_helix, (s == SS_HELIX_PI))

// Makes the residue into a HELIX_ALPHA
generic_set_sstruct_boolean(atomsel_set_helix, SS_HELIX_ALPHA)
// Makes the residue into a HELIX_3_10
generic_set_sstruct_boolean(atomsel_set_3_10_helix, SS_HELIX_3_10)
// Makes the residue into a HELIX_PI
generic_set_sstruct_boolean(atomsel_set_pi_helix, SS_HELIX_PI)

// which of these are beta sheets?
generic_sstruct_boolean(atomsel_sheet, (s == SS_BETA ||
                                        s == SS_BRIDGE ))
generic_sstruct_boolean(atomsel_extended_sheet, (s == SS_BETA))
generic_sstruct_boolean(atomsel_bridge_sheet, (s == SS_BRIDGE ))

// Makes the residue into a BETA
generic_set_sstruct_boolean(atomsel_set_sheet, SS_BETA)
generic_set_sstruct_boolean(atomsel_set_bridge_sheet, SS_BRIDGE)

// which of these are coils?
generic_sstruct_boolean(atomsel_coil, (s == SS_COIL))

// Makes the residue into COIL
generic_set_sstruct_boolean(atomsel_set_coil, SS_COIL)

// which of these are TURNS?
generic_sstruct_boolean(atomsel_turn, (s == SS_TURN))

// Makes the residue into TURN
generic_set_sstruct_boolean(atomsel_set_turn, SS_TURN)

// return the sstruct information as a 1 character string
static int atomsel_sstruct(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  atom_sel_mol->need_secondary_structure(1);
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      switch(atom_sel_mol->residue(atom_sel_mol->atom(i)->uniq_resid)->sstruct) {
        case SS_HELIX_ALPHA: data[i] = "H"; break;
        case SS_HELIX_3_10 : data[i] = "G"; break;
        case SS_HELIX_PI   : data[i] = "I"; break;
        case SS_BETA       : data[i] = "E"; break;
        case SS_BRIDGE     : data[i] = "B"; break;
        case SS_TURN       : data[i] = "T"; break;
        default:
        case SS_COIL       : data[i] = "C"; break;
      }
    }
  }
  return 1;
}


// set the secondary structure based on a string value
static int atomsel_set_sstruct(void *v, int num, const char **data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  char c;
  // Defining it here so remind myself that I don't need STRIDE (or
  // whoever) to do it automatically
  atom_sel_mol->need_secondary_structure(0);
  for (int i=0; i<num; i++) {
    if (flgs[i]) {
      if (strlen(data[i]) == 0) {
        msgErr << "cannot set a 0 length secondary structure string"
               << sendmsg;
      } else {
        c = ((const char *) data[i])[0];
        if (strlen(data[i]) > 1) {
          while (1) {
 if (!strcasecmp((const char *) data[i], "helix")) { c = 'H'; break;}
 if (!strcasecmp((const char *) data[i], "alpha")) { c = 'H'; break;}
 if (!strcasecmp((const char *) data[i], "alpha_helix")) { c = 'H'; break;}
 if (!strcasecmp((const char *) data[i], "alphahelix"))  { c = 'H'; break;}
 if (!strcasecmp((const char *) data[i], "alpha helix")) { c = 'H'; break;}

 if (!strcasecmp((const char *) data[i], "pi"))        { c = 'I'; break;}
 if (!strcasecmp((const char *) data[i], "pi_helix"))  { c = 'I'; break;}
 if (!strcasecmp((const char *) data[i], "pihelix"))   { c = 'I'; break;}
 if (!strcasecmp((const char *) data[i], "pi helix"))  { c = 'I'; break;}

 if (!strcasecmp((const char *) data[i], "310"))       { c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "310_helix")) { c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "3_10"))      { c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "3"))         { c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "310 helix")) { c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "3_10_helix")){ c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "3_10 helix")){ c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "3 10 helix")){ c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "helix_3_10")){ c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "helix 3 10")){ c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "helix3_10")) { c = 'G'; break;}
 if (!strcasecmp((const char *) data[i], "helix310"))  { c = 'G'; break;}

 if (!strcasecmp((const char *) data[i], "beta"))      { c = 'E'; break;}
 if (!strcasecmp((const char *) data[i], "betasheet")) { c = 'E'; break;}
 if (!strcasecmp((const char *) data[i], "beta_sheet")){ c = 'E'; break;}
 if (!strcasecmp((const char *) data[i], "beta sheet")){ c = 'E'; break;}
 if (!strcasecmp((const char *) data[i], "sheet"))     { c = 'E'; break;}
 if (!strcasecmp((const char *) data[i], "strand"))    { c = 'E'; break;}
 if (!strcasecmp((const char *) data[i], "beta_strand"))  { c = 'E'; break;}
 if (!strcasecmp((const char *) data[i], "beta strand"))  { c = 'E'; break;}

 if (!strcasecmp((const char *) data[i], "turn"))  { c = 'T'; break;}

 if (!strcasecmp((const char *) data[i], "coil"))     { c = 'C'; break;}
 if (!strcasecmp((const char *) data[i], "unknown"))  { c = 'C'; break;}
 c = 'X';
 break;
          } // while (1)
        }
	// and set the value
	int s = SS_COIL;
	switch ( c ) {
	case 'H':
	case 'h': s = SS_HELIX_ALPHA; break;
	case '3':
	case 'G':
	case 'g': s = SS_HELIX_3_10; break;
	case 'P':  // so you can say 'pi'
	case 'p':
	case 'I':
	case 'i': s = SS_HELIX_PI; break;
	case 'S':  // for "sheet"
	case 's':
	case 'E':
	case 'e': s = SS_BETA; break;
	case 'B':
	case 'b': s = SS_BRIDGE; break;
	case 'T':
	case 't': s = SS_TURN; break;
	case 'L': // L is from PHD and may be an alternate form
	case 'l':
	case 'C':
	case 'c': s = SS_COIL; break;
	default: {
	  msgErr << "Unknown sstruct assignment: '"
	    << (const char *) data[i] << "'" << sendmsg;
	  s = SS_COIL; break;
	}
	}
	atom_sel_mol->residue(atom_sel_mol->atom(i)->uniq_resid)->sstruct = s;
      }
    }
  }
  return 1;
}


//// specialized function to turn on all atoms in a given residue
//      and leave the rest alone.
// It is slower this way, but easier to understand
static void mark_atoms_given_residue(DrawMolecule *mol, int residue, int *on)
{
  ResizeArray<int> *atoms = &(mol->residueList[residue]->atoms);
  for (int i= atoms->num()-1; i>=0; i--) {
     on[(*atoms)[i]] = TRUE;
  }
}


// macro for either protein or nucleic fragments
#define fragment_data(fctn, fragtype)					      \
static int fctn(void *v, int num, int *data, int *)			      \
{									      \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     \
   int *tmp = new int[num];						      \
   int i;								      \
   for (i=num-1; i>=0; i--) {  /* clear the arrays */		      \
      tmp[i] = 0;							      \
      data[i] = -1;  /* default is -1 for 'not a [np]frag' */		      \
   }									      \
   /* for each fragment */						      \
   for ( i=atom_sel_mol->fragtype.num()-1; i>=0; i--) {			      \
      /* for each residues of the fragment */				      \
      int j;								      \
      for (j=atom_sel_mol->fragtype[i]->num()-1; j>=0; j--) {		      \
	 /* mark the atoms in the fragment */				      \
	 mark_atoms_given_residue(atom_sel_mol,(*atom_sel_mol->fragtype[i])[j], tmp);      \
      }									      \
      /* and label them with the appropriate number */			      \
      for (j=num-1; j>=0; j--) {					      \
	 if (tmp[j]) {							      \
	    data[j] = i;						      \
	    tmp[j] = 0;							      \
	 }								      \
      }									      \
   }									      \
   delete [] tmp;							      \
   return 1;								      \
}

fragment_data(atomsel_pfrag, pfragList)
fragment_data(atomsel_nfrag, nfragList)

static Timestep *selframe(DrawMolecule *atom_sel_mol, int which_frame) {
  Timestep *r;
  switch (which_frame) {
   case AtomSel::TS_LAST: r = atom_sel_mol->get_last_frame(); break;

   case AtomSel::TS_NOW : r = atom_sel_mol->current(); break;
   default: {
     if (!atom_sel_mol->get_frame(which_frame)) {
       r = atom_sel_mol->get_last_frame();

     } else {
       r = atom_sel_mol->get_frame(which_frame);
     }
   }
  }
  return r;
}


#if defined(VMDWITHCARBS)
// 'pucker'
static int atomsel_pucker(void *v, int num, double *data, int *flgs) {
  int i;
  SmallRing *ring;
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  memset(data, 0, num*sizeof(double));
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts || !ts->pos) {
    return 1;
  }

  // XXX We're hijacking the ring list in BaseMolecule at present.
  //     It might be better to build our own independent one, but
  //     this way there's only one ring list in memory at a time.
  atom_sel_mol->find_small_rings_and_links(5, 6);

  for (i=0; i < atom_sel_mol->smallringList.num(); i++) {
    ring = atom_sel_mol->smallringList[i];
    int N = ring->num();

    // accept rings of size 5 or 6
    if (N >= 5 && N <= 6) {
      float pucker = hill_reilly_ring_pucker((*ring), ts->pos);
      
      int j;
      for (j=0; j<N; j++) {
        int ind = (*ring)[j];
        if (flgs[ind])
          data[ind] = pucker;
      }
    }
  }

  return 1;
}
#endif

static int atomsel_user(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts || !ts->user) {
    memset(data, 0, num*sizeof(double));
    return 1;                                                           
  }
  for (int i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->user[i];					      
    }								              
  }									      
  return 1;								     
}
static int atomsel_set_user(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts) return 0;
  if (!ts->user) {
    ts->user= new float[num];
    memset(ts->user, 0, num*sizeof(float));
  }
  for (int i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      ts->user[i] = (float)data[i];					      
    }								              
  }									      
  return 1;								     
}

static int atomsel_user2(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts || !ts->user2) {
    memset(data, 0, num*sizeof(double));
    return 1;                                                           
  }
  for (int i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->user2[i];					      
    }								              
  }									      
  return 1;								     
}
static int atomsel_set_user2(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts) return 0;
  if (!ts->user2) {
    ts->user2= new float[num];
    memset(ts->user2, 0, num*sizeof(float));
  }
  for (int i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      ts->user2[i] = (float)data[i];					      
    }								              
  }									      
  return 1;								     
}

static int atomsel_user3(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts || !ts->user3) {
    memset(data, 0, num*sizeof(double));
    return 1;                                                           
  }
  for (int i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->user3[i];					      
    }								              
  }									      
  return 1;								     
}
static int atomsel_set_user3(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts) return 0;
  if (!ts->user3) {
    ts->user3= new float[num];
    memset(ts->user3, 0, num*sizeof(float));
  }
  for (int i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      ts->user3[i] = (float)data[i];					      
    }								              
  }									      
  return 1;								     
}

static int atomsel_user4(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts || !ts->user4) {
    memset(data, 0, num*sizeof(double));
    return 1;                                                           
  }
  for (int i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->user4[i];					      
    }								              
  }									      
  return 1;								     
}
static int atomsel_set_user4(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts) return 0;
  if (!ts->user4) {
    ts->user4= new float[num];
    memset(ts->user4, 0, num*sizeof(float));
  }
  for (int i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      ts->user4[i] = (float)data[i];					      
    }								              
  }									      
  return 1;								     
}



static int atomsel_xpos(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts) {
    for (i=0; i<num; i++) 
      if (flgs[i]) data[i] = 0.0;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->pos[3L*i     ];					      
    }								              
  }									      
  return 1;								     
}

static int atomsel_ypos(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts) {
    for (i=0; i<num; i++) 
      if (flgs[i]) data[i] = 0.0;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->pos[3L*i + 1L];					      
    }								              
  }									      
  return 1;								     
}

static int atomsel_zpos(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts) {
    for (i=0; i<num; i++) 
      if (flgs[i]) data[i] = 0.0;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->pos[3L*i + 2L];					      
    }								              
  }									      
  return 1;								     
}

static int atomsel_ufx(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts || !ts->force) {
    for (i=0; i<num; i++) 
      if (flgs[i]) data[i] = 0.0;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->force[3L*i     ];
    }								              
  }									      
  return 1;								     
}

static int atomsel_ufy(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts || !ts->force) {
    for (i=0; i<num; i++) 
      if (flgs[i]) data[i] = 0.0;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->force[3L*i + 1L];
    }								              
  }									      
  return 1;								     
}

static int atomsel_ufz(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts || !ts->force) {
    for (i=0; i<num; i++) 
      if (flgs[i]) data[i] = 0.0;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) {							      
      data[i] = ts->force[3L*i + 2L];
    }								              
  }									      
  return 1;								     
}

#define atomsel_get_vel(name, offset) \
static int atomsel_##name(void *v, int num, double *data, int *flgs) {  \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;       \
  int which_frame = ((atomsel_ctxt *)v)->which_frame;  \
  Timestep *ts = selframe(atom_sel_mol, which_frame);  \
  int i;                                               \
  if (!ts || !ts->vel) {                               \
    for (i=0; i<num; i++)                              \
      if (flgs[i]) data[i] = 0.0;                      \
    return 0;                                          \
  }                                                    \
  for (i=0; i<num; i++) {	                       \
    if (flgs[i]) {		                       \
      data[i] = ts->vel[3L*i + (offset)];              \
    }                                                  \
  }                                                    \
  return 1;                                            \
}                                                      

atomsel_get_vel(vx, 0)
atomsel_get_vel(vy, 1)
atomsel_get_vel(vz, 2)

static int atomsel_phi(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts) {
    for (i=0; i<num; i++) data[i] = 0;
    return 0;  
  } 
  const float *r = ts->pos; 
  for (i=0; i<num; i++) {
    if (!flgs[i]) continue;
    MolAtom *atom = atom_sel_mol->atom(i);
    int resid = atom->uniq_resid;
    int N = atom_sel_mol->find_atom_in_residue("N", resid);
    int CA = atom_sel_mol->find_atom_in_residue("CA", resid);
    int C = atom_sel_mol->find_atom_in_residue("C", resid);

    // Find the index of the C atom from the previous residue by searching
    // the atoms bonded to N for an atom named "C".  
    if (N < 0) {
      data[i] = 0;
      continue;
    }
    MolAtom *atomN = atom_sel_mol->atom(N);
    int cprev = -1;
    for (int j=0; j<atomN->bonds; j++) {
      int aindex = atomN->bondTo[j];
      int nameindex = atom_sel_mol->atom(aindex)->nameindex;
      if (!strcmp(atom_sel_mol->atomNames.name(nameindex), "C")) {
        cprev = aindex;
        break;
      }
    }
    if (cprev >= 0 && CA >= 0 && C >= 0) 
      data[i] = dihedral(r+3L*cprev, r+3L*N, r+3L*CA, r+3L*C);
    else
      data[i] = 0.0; 
  }
  return 0;
}

static int atomsel_psi(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts) {
    for (i=0; i<num; i++) data[i] = 0;
    return 0;  
  } 
  const float *r = ts->pos; 
  for (i=0; i<num; i++) {
    if (!flgs[i]) continue;
    MolAtom *atom = atom_sel_mol->atom(i);
    int resid = atom->uniq_resid;
    int N = atom_sel_mol->find_atom_in_residue("N", resid);
    int CA = atom_sel_mol->find_atom_in_residue("CA", resid);
    int C = atom_sel_mol->find_atom_in_residue("C", resid);

    // Find the index of the N atom from the next residue by searching the
    // atoms bonded to C for an atom named "N".
    if (C < 0) {
      data[i] = 0;
      continue;
    }
    MolAtom *atomC = atom_sel_mol->atom(C);
    int nextn = -1;
    for (int j=0; j<atomC->bonds; j++) {
      int aindex = atomC->bondTo[j];
      int nameindex = atom_sel_mol->atom(aindex)->nameindex;
      if (!strcmp(atom_sel_mol->atomNames.name(nameindex), "N")) {
        nextn = aindex;
        break;
      }
    }
    if (nextn >= 0 && N >= 0 && CA >= 0) 
      data[i] = dihedral(r+3L*N, r+3L*CA, r+3L*C, r+3L*nextn);
    else
      data[i] = 0.0; 
  }
  return 0;
}

static int atomsel_set_phi(void *v, int num, double *data, int *flgs) {
  // We rotate about the N-CA bond
  // 0. Get the current value of phi
  // 1. Translate, putting CA at the origin
  // 2. Compute the axis along N-CA
  // 3. Rotate just the N-terminus about the given axis
  // 4. Undo the translation

  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  SymbolTable *table = ((atomsel_ctxt *)v)->table;
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  int i;
  if (!ts) {
    return 0;  
  } 
  float *r = ts->pos; 

  // Go through the residues; if the residue contains all the necessary atoms,
  // check to see if the CA of the residue is on.  If it is, proceed with the
  // rotation. 
  for (i=0; i<atom_sel_mol->fragList.num(); i++) {
    Fragment *frag = atom_sel_mol->fragList[i];
    int nfragres = frag->residues.num();
    if (nfragres < 2) continue;
    int C = atom_sel_mol->find_atom_in_residue("C", frag->residues[0]);
    // Start at second residue since I need the previous residue for phi
    for (int ires = 1; ires < nfragres; ires++) {
      int resid = frag->residues[ires];
      int cprev = C;
      int N = atom_sel_mol->find_atom_in_residue("N", resid);
      int CA = atom_sel_mol->find_atom_in_residue("CA", resid);
      C = atom_sel_mol->find_atom_in_residue("C", resid);
      if (cprev < 0 || N < 0 || CA < 0 || C < 0) continue;
      if (!flgs[CA]) continue;
      float CApos[3], Npos[3], axis[3];
      vec_copy(CApos, r+3L*CA);
      vec_copy(Npos, r+3L*N);
      vec_sub(axis, Npos, CApos);
      vec_normalize(axis); 
      double oldphi = dihedral(r+3L*cprev, Npos, CApos, r+3L*C);
      // Select the N-terminal part of the fragment, which includes all
      // residues up to the current one, plus N and NH of the curent one.
      // I'm just going to create a new atom selection for now.
      
      AtomSel *nterm = new AtomSel(table, atom_sel_mol->id());
      char buf[100];
      sprintf(buf, 
        "(fragment %d and residue < %d) or (residue %d and name N NH CA)",
        i, resid, resid);
      if (nterm->change(buf, atom_sel_mol) == AtomSel::NO_PARSE) {
        msgErr << "set phi: internal atom selection error" << sendmsg;
        msgErr << buf << sendmsg; 
        delete nterm;
        continue;
      }
      Matrix4 mat;
      mat.transvec(axis[0], axis[1], axis[2]);
      mat.rot((float) (data[CA]-oldphi), 'x');
      mat.transvecinv(axis[0], axis[1], axis[2]); 
        
      for (int j=0; j<num; j++) {
        if (nterm->on[j]) {
          float *pos = r+3L*j;
          vec_sub(pos, pos, CApos);
          mat.multpoint3d(pos, pos);
          vec_add(pos, pos, CApos); 
        }
      }
      delete nterm;
    }
  }
  return 0;
}

static int atomsel_set_psi(void *v, int num, double *data, int *flgs) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  SymbolTable *table = ((atomsel_ctxt *)v)->table;
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  if (!ts) {
    return 0; 
  }
  float *r = ts->pos;

  for (int i=0; i<atom_sel_mol->fragList.num(); i++) {
    Fragment *frag = atom_sel_mol->fragList[i];
    int nfragres = frag->residues.num();
    if (nfragres < 2) continue;
    int N = atom_sel_mol->find_atom_in_residue("N", frag->residues[nfragres-1]);
    for (int ires = nfragres-2; ires >= 0; ires--) {
      int resid = frag->residues[ires];
      int nextn = N;
      N = atom_sel_mol->find_atom_in_residue("N", resid);
      int CA = atom_sel_mol->find_atom_in_residue("CA", resid);
      int C = atom_sel_mol->find_atom_in_residue("C", resid);
      if (nextn < 0 || N < 0 || CA < 0 || C < 0) continue;
      if (!flgs[CA]) continue;
      float CApos[3], Cpos[3], axis[3];
      vec_copy(CApos, r+3L*CA);
      vec_copy(Cpos, r+3L*C);
      vec_sub(axis, Cpos, CApos);
      vec_normalize(axis);
      double oldpsi = dihedral(r+3L*N, CApos, Cpos, r+3L*nextn);

      AtomSel *cterm = new AtomSel(table, atom_sel_mol->id());
      char buf[100];
      sprintf(buf,
        "(fragment %d and residue > %d) or (residue %d and name CA C O)",
        i, resid, resid);
      if (cterm->change(buf, atom_sel_mol) == AtomSel::NO_PARSE) {
        msgErr << "set psi: internal atom selection error" << sendmsg;
        msgErr << buf << sendmsg;
        delete cterm;
        continue;
      }

      Matrix4 mat;
      mat.transvec(axis[0], axis[1], axis[2]);
      mat.rot((float) (data[CA]-oldpsi), 'x');
      mat.transvecinv(axis[0], axis[1], axis[2]);

      for (int j=0; j<num; j++) {
        if (cterm->on[j]) {
          float *pos = r+3L*j;
          vec_sub(pos, pos, CApos);
          mat.multpoint3d(pos, pos);
          vec_add(pos, pos, CApos);
        }
      }
      delete cterm;
    }
  }
  return 0;
}

#define set_position_data(fctn, offset)					      \
static int fctn(void *v, int num, double *data, int *flgs)		      \
{									      \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     \
  int which_frame = ((atomsel_ctxt *)v)->which_frame;                                 \
  Timestep *ts = selframe(atom_sel_mol, which_frame);                         \
  if (!ts) return 0;                                                          \
  for (int i=num-1; i>=0; i--) {					      \
    if (flgs[i]) {							      \
      ts->pos[3L*i + offset] = (float) data[i];				      \
    }								              \
  }									      \
  return 1;								      \
}


set_position_data(atomsel_set_xpos, 0)
set_position_data(atomsel_set_ypos, 1)
set_position_data(atomsel_set_zpos, 2)

#define set_force_data(fctn, offset)					      \
static int fctn(void *v, int num, double *data, int *flgs)		      \
{									      \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     \
  int which_frame = ((atomsel_ctxt *)v)->which_frame;                                 \
  Timestep *ts = selframe(atom_sel_mol, which_frame);                         \
  if (!ts) return 0;                                                          \
  if (!ts->force) {                                                           \
    ts->force = new float[3L*num];                                            \
    memset(ts->force, 0, 3L*num*sizeof(float));                               \
  }                                                                           \
  for (int i=num-1; i>=0; i--) {					      \
    if (flgs[i]) {							      \
      ts->force[3L*i + offset] = (float) data[i];                             \
    }								              \
  }									      \
  return 1;								      \
}

set_force_data(atomsel_set_ufx, 0)
set_force_data(atomsel_set_ufy, 1)
set_force_data(atomsel_set_ufz, 2)

#define set_vel_data(fctn, offset)					      \
static int fctn(void *v, int num, double *data, int *flgs)		      \
{									      \
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     \
  int which_frame = ((atomsel_ctxt *)v)->which_frame;                                 \
  Timestep *ts = selframe(atom_sel_mol, which_frame);                         \
  if (!ts) return 0;                                                          \
  if (!ts->vel) {                                                             \
    ts->vel= new float[3L*num];                                               \
    memset(ts->vel, 0, 3L*num*sizeof(float));                                 \
  }                                                                           \
  for (int i=num-1; i>=0; i--) {					      \
    if (flgs[i]) {							      \
      ts->vel[3L*i + offset] = (float) data[i];                               \
    }								              \
  }									      \
  return 1;								      \
}

set_vel_data(atomsel_set_vx, 0)
set_vel_data(atomsel_set_vy, 1)
set_vel_data(atomsel_set_vz, 2)

extern "C" {
  double atomsel_square(double x) { return x*x; }
}

// this is different than the previous.  It allows me to search for a
// given regex sequence.  For instace, given the protein sequence
//   WAPDTYLVAPDAQD
// the selection: sequence APDT
//  will select only the 2nd through 5th terms
// the selection: sequence APD
//  will select 2nd through 4th, and 9th to 11th.
// the selection: sequence "A.D"
//  will get 2-4 and 9-14
// and so on.
//   If a residue name is not normal, it becomes an X

// I am handed a list of strings for this selection
//  (eg, sequence APD "A.D" A to D)
// Since there are no non-standard residue names (ie, no '*')
// I'll interpret everything as a regex.  Also, phrases like
// "X to Y" are interpreted as "X.*Y"

static int atomsel_sequence(void *v, int argc, const char **argv, int *types,
			    int num, int *flgs)
{
   DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
   int i;
   // make a temporary array for marking the selected atoms
   int *selected = new int[num];
   for (i=0; i<num; i++) {
      selected[i] = FALSE;
   }
   // make the list of regex'es
   JRegex **regex = (JRegex **) malloc( argc * sizeof(JRegex *));
   int num_r = 0;
   {
      JString pattern;
      for (i=0; i<argc; i++) {
	 pattern  = argv[i];
	 if (types[i] >= 3) {  // get the next term (if a "to" element)
	    pattern += ".*";
	    pattern += argv[++i];
	 }
	 regex[num_r] = new JRegex(pattern);
	 num_r ++;
      }
   } // constructed the regex array
   if (num_r == 0) {
      return 0;
   }

   // construct a list of sequences from each protein (pfraglist)
   // and nucleic acid (nfragList)
   for (int fragcount=0; fragcount <2; fragcount++) {
      int pcount = atom_sel_mol->pfragList.num();
      int ncount = atom_sel_mol->nfragList.num();
      for (i=0; i< (fragcount == 0 ? pcount : ncount); i++) {
	 int size = (fragcount == 0 ? atom_sel_mol->pfragList[i]->num()
	                            : atom_sel_mol->nfragList[i]->num());
	 char *s = new char[size+1]; // so that it can be NULL-terminated
	 char *t = s;
	 int *mark = new int[size];
	 
	 for (int j=0; j<size; j++) {
	    int residuenum = ((fragcount == 0) ?
	          (*atom_sel_mol->pfragList[i])[j] :  
	          (*atom_sel_mol->nfragList[i])[j]);
	    int atomnum = (atom_sel_mol->residueList[residuenum]->atoms[0]);
	    MolAtom *atom = atom_sel_mol->atom(atomnum);
	    const char *resname = (atom_sel_mol->resNames).name(atom->resnameindex);
	    mark[j] = FALSE;
	    if (fragcount == 0) {
	       // protein translations
         // PDB names
	       if (!strcasecmp( resname, "GLY")) {*t++ = 'G'; continue;}
	       if (!strcasecmp( resname, "ALA")) {*t++ = 'A'; continue;}
	       if (!strcasecmp( resname, "VAL")) {*t++ = 'V'; continue;}
	       if (!strcasecmp( resname, "PHE")) {*t++ = 'F'; continue;}
	       if (!strcasecmp( resname, "PRO")) {*t++ = 'P'; continue;}
	       if (!strcasecmp( resname, "MET")) {*t++ = 'M'; continue;}
	       if (!strcasecmp( resname, "ILE")) {*t++ = 'I'; continue;}
	       if (!strcasecmp( resname, "LEU")) {*t++ = 'L'; continue;}
	       if (!strcasecmp( resname, "ASP")) {*t++ = 'D'; continue;}
	       if (!strcasecmp( resname, "GLU")) {*t++ = 'E'; continue;}
	       if (!strcasecmp( resname, "LYS")) {*t++ = 'K'; continue;}
	       if (!strcasecmp( resname, "ARG")) {*t++ = 'R'; continue;}
	       if (!strcasecmp( resname, "SER")) {*t++ = 'S'; continue;}
	       if (!strcasecmp( resname, "THR")) {*t++ = 'T'; continue;}
	       if (!strcasecmp( resname, "TYR")) {*t++ = 'Y'; continue;}
	       if (!strcasecmp( resname, "HIS")) {*t++ = 'H'; continue;}
	       if (!strcasecmp( resname, "CYS")) {*t++ = 'C'; continue;}
	       if (!strcasecmp( resname, "ASN")) {*t++ = 'N'; continue;}
	       if (!strcasecmp( resname, "GLN")) {*t++ = 'Q'; continue;}
	       if (!strcasecmp( resname, "TRP")) {*t++ = 'W'; continue;}
         // CHARMM names
         if (!strcasecmp( resname, "HSE")) {*t++ = 'H'; continue;}
         if (!strcasecmp( resname, "HSD")) {*t++ = 'H'; continue;}
         if (!strcasecmp( resname, "HSP")) {*t++ = 'H'; continue;}
         // AMBER names
         if (!strcasecmp( resname, "CYX")) {*t++ = 'C'; continue;}
	    } else {
	       // nucleic acid translations
	       if (!strcasecmp( resname, "ADE")) {*t++ = 'A'; continue;}
	       if (!strcasecmp( resname, "A")) {*t++ = 'A'; continue;}
	       if (!strcasecmp( resname, "THY")) {*t++ = 'T'; continue;}
	       if (!strcasecmp( resname, "T")) {*t++ = 'T'; continue;}
	       if (!strcasecmp( resname, "CYT")) {*t++ = 'C'; continue;}
	       if (!strcasecmp( resname, "C")) {*t++ = 'C'; continue;}
	       if (!strcasecmp( resname, "GUA")) {*t++ = 'G'; continue;}
	       if (!strcasecmp( resname, "G")) {*t++ = 'G'; continue;}
               if (!strcasecmp( resname, "URA")) {*t++ = 'U'; continue;}
               if (!strcasecmp( resname, "U")) {*t++ = 'U'; continue;}
	    }
	    // then I have no idea
	    *t++ = 'X';
	 }  // end loop 'j'; constructed the sequence for this protein
	 *t = 0; // terminate the string
	 
//	 msgInfo << "sequence " << i << " is: " << s << sendmsg;
	 // which of the residues match the regex(es)?
	 for (int r=0; r<num_r; r++) {
	    int len, start = 0, offset;
	    while ((offset = (regex[r]->search(s, strlen(s), len, start)))
		   != -1) {
	       // then there was a match from offset to offset+len
//	       msgInfo << "start " << start << " offset " << offset << " len "
//		       << len << sendmsg;
	       for (int loop=offset; loop<offset+len; loop++) {
		  mark[loop] = 1;
	       }
	       start = offset+len;
	    }
	 }
	 
	 // the list of selected residues is in mark
	 // turn on the right atoms
	 for (int marked=0; marked<size; marked++) {
	    if (mark[marked]) {
	       int residuenum = (fragcount == 0 ?
				 (*atom_sel_mol->pfragList[i])[marked] :
				 (*atom_sel_mol->nfragList[i])[marked]);
	       for (int atomloop=0;
		    atomloop< atom_sel_mol->residueList[residuenum]->
		    atoms.num(); atomloop++) {
		  selected[atom_sel_mol->residueList[residuenum]->
			   atoms[atomloop]]=TRUE;
	       }
	    }
	 }
	 delete [] mark;
	 delete [] s;
      } // end loop i over the fragments
   }  // end loop 'fragcount'


   // get rid of the compiled regex's
   for (i=0; i<num_r; i++) {
      delete regex[i];
   }
   // copy the 'selected' array into 'flgs'
   for (i=0; i<num; i++) {
      flgs[i] = flgs[i] && selected[i];
   }
   return 1;
}

/************ support for RasMol selections ******************/
//// parse the rasmol primitive
// the full form of a primitive is (seems to be)
//   {[<resname>]}{<resid>}{:<chain>}{.<atom name>}
// if resname is only alpha, the [] can be dropped
// if chain is alpha, the : can be dropped
// resname only contains * if it is the first one ?
// ? cannot go in the resid
// * can only replace the whole field
static int atomsel_rasmol_primitive(void *v, int argc, const char **argv, int *,
				    int num, int *flgs)
{
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  SymbolTable *table = ((atomsel_ctxt *)v)->table;

  // for each word, (ignoring the quote flags)
  for (int word=0; word<argc; word++) {
    const char *rnm0 = argv[word]; // resname start
    const char *rnm1;              // and end position
    const char *rid0, *rid1;
    if (*rnm0 == '*') {
      rnm1 = rnm0 + 1;
      rid0 = rnm1;
    } else if (*rnm0 == '[') {
      rnm0++;
      rnm1 = rnm0;
      while (*rnm1 && *rnm1 != ']') { // find trailing bracket
	rnm1 ++;
      }
      if (rnm1 == rnm0) {  // for cases like [] and "["
	rid0 = rnm1;
      } else {
	if (*rnm1==']') {  // for cases like [so4]
	  rid0 = rnm1+1;
	} else {           // for (incorrect) cases like [so4
	  rid0 = rnm1;
	}
      }
    } else { // then must be alpha or ?
      rnm1 = rnm0;
      while (isalpha(*rnm1) || *rnm1 == '?') {  // find first non-alpha
	rnm1++;
      }
      rid0 = rnm1;
    }
    // got the resname

    // parse the resid
    rid1 = rid0;
    if (*rid1 == '*') {
      rid1++;
    } else {
      while (isdigit(*rid1)) {
	rid1++;
      }
    }

    // if this is the : delimiter, skip over it
    const char *chn0, *chn1;
    if (*rid1 == ':') {
      chn0 = rid1 + 1;
    } else {
      chn0 = rid1;
    }

    // get the chain
    // seek the . or end of string
    chn1 = chn0;
    while (*chn1 && *chn1 != '.') {
      chn1++;
    }

    const char *nm0, *nm1;
    if (*chn1 == '.') {
      nm0 = chn1 + 1;
    } else {
      nm0 = chn1;
    }
    nm1 = nm0;
    // seek the end of string
    while (*nm1) {
      nm1++;
    }
    

    // save the info into strings
    JString resname, resid, chain, name;
    const char *s;
    for (s=rnm0; s<rnm1; s++) {
      resname += *s;
    }
    for (s=rid0; s<rid1; s++) {
      resid += *s;
    }
    for (s=chn0; s<chn1; s++) {
      chain += *s;
    }
    for (s=nm0; s<nm1; s++) {
      name += *s;
    }
    //    msgInfo << "resname: " << (const char *) resname << sendmsg;
    //    msgInfo << "resid: " << (const char *) resid << sendmsg;
    //    msgInfo << "chain: " << (const char *) chain << sendmsg;
    //    msgInfo << "name: " << (const char *) name << sendmsg;

    // convert to the VMD regex ( ? => .? and * => .*)
    //   (however, if there is a * for the whole field, delete the field)
    if (resname == "*") resname = "";
    if (resid == "*") resid = "";
    if (chain == "*") chain = "";
    if (name == "*") name = "";
    resname.gsub("?", ".?"); resname.gsub("*", ".*");
    resid.gsub("?", ".?"); resid.gsub("*", ".*");
    chain.gsub("?", ".?"); chain.gsub("*", ".*");
    name.gsub("?", ".?"); name.gsub("*", ".*");
    // make everything upcase
    resname.upcase();
    resid.upcase();
    chain.upcase();
    name.upcase();

    // construct a new search
    JString search;
    if (resname != "") {
      search = "resname ";
      search += '"';
      search += resname;
      search += '"';
    }
    if (resid != "") {
      if (search != "") {
	search += " and resid ";
      } else {
	search = "resid ";
      }
      search += '"';
      search += resid;
      search += '"';
    }
    // if the chain length > 1, it is a segname
    int is_segname = chain.length() > 1;
    if (chain != "") {
      if (search != "") {
	search += (is_segname ? " and segname " : " and chain ");
      } else {
	search = (is_segname ? "segname " : "chain ");
      }
      search += '"';
      search += chain;
      search += '"';
    }
    if (name != "") {
      if (search != "") {
	search += " and name ";
      } else {
	search = "name ";
      }
      search += '"';
      search += name;
      search += '"';
    }
    msgInfo << "Search = " << search << sendmsg;

    if (search == "") {
      search = "all";
    }
    // and do the search
    AtomSel *atomSel = new AtomSel(table, atom_sel_mol->id());
    if (atomSel->change((const char *) search, atom_sel_mol) ==
	AtomSel::NO_PARSE) {
      msgErr << "rasmol: cannot understand: " << argv[word] << sendmsg;
      delete atomSel;
      continue;
    }
    
    // save the results
    {
      for (int i=0; i<num; i++) {
	flgs[i] = flgs[i] && atomSel->on[i];
      }
    }
    delete atomSel;
  }
  return 1;
}

int atomsel_custom_singleword(void *v, int num, int *flgs) {
  SymbolTable *table = ((atomsel_ctxt *)v)->table;
  const char *singleword = ((atomsel_ctxt *)v)->singleword;
  if (!singleword) {
    msgErr << "Internal error, atom selection macro called without context"
           << sendmsg;
    return 0;
  }
  const char *macro = table->get_custom_singleword(singleword);
  if (!macro) {
    msgErr << "Internal error, atom selection macro has lost its defintion."
           << sendmsg;
    return 0;
  }
  // Create new AtomSel based on the macro
  DrawMolecule *mol = ((atomsel_ctxt *)v)->atom_sel_mol;
  AtomSel *atomSel = new AtomSel(table, mol->id());
  atomSel->which_frame = ((atomsel_ctxt *)v)->which_frame;
  if (atomSel->change(macro, mol) == AtomSel::NO_PARSE) {
    msgErr << "Unable to parse macro: " << macro << sendmsg;
    delete atomSel;
    return 0;
  }
  for (int i=0; i<num; i++) {
    flgs[i] = flgs[i] && atomSel->on[i];
  }
  delete atomSel;
  return 1;
}




// These functions allow voxel values from volumetric data loaded in a molecule
// to be used in atom selections. Three keywords are implemented: volN returns
// the value of the voxel under the selected atom, interpvolN returns the 
// voxel value interpolated from neighboring voxels, and volindexN returns the 
// numerical index of the underlying voxel (I found the latter useful, however
// its usefulness might be too marginal to be included into VMD?)

// This is a hack because the volume names are hardcoded. Ideally, VMD should
// invent an "array keyword" which would contain a parsable index (e.g. 
// "vol#2"). Right now, the first M keywords are hard-coded, and any volN with
// N greater than M will not work. This is because adding an array keyword
// involves a considerable amount of work, but perhaps this should be considered
// eventually...

// NOTE: if a coordinate lies outside the volmap range, a value of NAN is
// assigned. Since NAN != NAN, NAN atoms will automatically never be included 
// in selections (which is what we want!). Furthermore, "$sel get vol0" would
// return a Tcl-parsable NAN to easily allow the user to identify coords that
// lie outside the valid range.

#ifndef NAN //not a number
  const float NAN = sqrtf(-1.f); //need some kind of portable NAN definition
#endif


static int atomsel_volume_array(void *v, int num, double *data, int *flgs, int array_index) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  const VolumetricData *voldata = atom_sel_mol->get_volume_data(array_index);
  int i;
  if (!ts || !voldata) {
    if (!ts) msgErr << "atomselect: non-existent timestep." << sendmsg;
    if (!voldata) msgErr << "atomselect: volumetric map volume#" << array_index << " does not exist." << sendmsg;
    for (i=0; i<num; i++) if (flgs[i]) data[i] = NAN;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) 							      
      data[i] = voldata->voxel_value_from_coord(ts->pos[3L*i], ts->pos[3L*i+1], ts->pos[3L*i+2]);	        
  }									      
  return 1;								     
}


static int atomsel_interp_volume_array(void *v, int num, double *data, int *flgs, int array_index) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  const VolumetricData *voldata = atom_sel_mol->get_volume_data(array_index);
  int i;
  if (!ts || !voldata) {
    if (!ts) msgErr << "atomselect: non-existent timestep." << sendmsg;
    if (!voldata) msgErr << "atomselect: volumetric map volume#" << array_index << " does not exist." << sendmsg;
    for (i=0; i<num; i++) if (flgs[i]) data[i] = NAN;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) 							      
      data[i] = voldata->voxel_value_interpolate_from_coord(ts->pos[3L*i], ts->pos[3L*i+1], ts->pos[3L*i+2]);	        
  }									      
  return 1;								     
}


static int atomsel_gridindex_array(void *v, int num, double *data, int *flgs, int array_index) {
  DrawMolecule *atom_sel_mol = ((atomsel_ctxt *)v)->atom_sel_mol;                     
  int which_frame = ((atomsel_ctxt *)v)->which_frame;
  Timestep *ts = selframe(atom_sel_mol, which_frame);
  const VolumetricData *voldata = atom_sel_mol->get_volume_data(array_index);
  int i;
  if (!ts || !voldata) {
    if (!ts) msgErr << "atomselect: non-existent timestep." << sendmsg;
    if (!voldata) msgErr << "atomselect: volumetric map volume#" << array_index << " does not exist." << sendmsg;
    for (i=0; i<num; i++) if (flgs[i]) data[i] = NAN;
    return 0;                                                           
  }
  for (i=0; i<num; i++) {					      
    if (flgs[i]) 							      
      data[i] = voldata->voxel_index_from_coord(ts->pos[3L*i], ts->pos[3L*i+1], ts->pos[3L*i+2]);	        
  }									      
  return 1;								     
}


static int atomsel_gridindex0(void *v, int num, double *data, int *flgs) {							      
  return atomsel_gridindex_array(v, num, data, flgs, 0);								   
}
static int atomsel_gridindex1(void *v, int num, double *data, int *flgs) {							      
  return atomsel_gridindex_array(v, num, data, flgs, 1);								   
}
static int atomsel_gridindex2(void *v, int num, double *data, int *flgs) {							      
  return atomsel_gridindex_array(v, num, data, flgs, 2);								   
}
static int atomsel_gridindex3(void *v, int num, double *data, int *flgs) {							      
  return atomsel_gridindex_array(v, num, data, flgs, 3);								   
}
static int atomsel_gridindex4(void *v, int num, double *data, int *flgs) {							      
  return atomsel_gridindex_array(v, num, data, flgs, 4);								   
}
static int atomsel_gridindex5(void *v, int num, double *data, int *flgs) {							      
  return atomsel_gridindex_array(v, num, data, flgs, 5);								   
}
static int atomsel_gridindex6(void *v, int num, double *data, int *flgs) {							      
  return atomsel_gridindex_array(v, num, data, flgs, 6);								   
}
static int atomsel_gridindex7(void *v, int num, double *data, int *flgs) {							      
  return atomsel_gridindex_array(v, num, data, flgs, 7);								   
}


// 'volume0'
static int atomsel_volume0(void *v, int num, double *data, int *flgs) {							      
  return atomsel_volume_array(v, num, data, flgs, 0);								   
}
static int atomsel_volume1(void *v, int num, double *data, int *flgs) {							      
  return atomsel_volume_array(v, num, data, flgs, 1);								   
}
static int atomsel_volume2(void *v, int num, double *data, int *flgs) {							      
  return atomsel_volume_array(v, num, data, flgs, 2);								   
}
static int atomsel_volume3(void *v, int num, double *data, int *flgs) {							      
  return atomsel_volume_array(v, num, data, flgs, 3);								   
}
static int atomsel_volume4(void *v, int num, double *data, int *flgs) {							      
  return atomsel_volume_array(v, num, data, flgs, 4);								   
}
static int atomsel_volume5(void *v, int num, double *data, int *flgs) {							      
  return atomsel_volume_array(v, num, data, flgs, 5);								   
}
static int atomsel_volume6(void *v, int num, double *data, int *flgs) {							      
  return atomsel_volume_array(v, num, data, flgs, 6);								   
}
static int atomsel_volume7(void *v, int num, double *data, int *flgs) {							      
  return atomsel_volume_array(v, num, data, flgs, 7);								   
}


// 'interpolated_volume0'
static int atomsel_interp_volume0(void *v, int num, double *data, int *flgs) {							      
  return atomsel_interp_volume_array(v, num, data, flgs, 0);								   
}
static int atomsel_interp_volume1(void *v, int num, double *data, int *flgs) {							      
  return atomsel_interp_volume_array(v, num, data, flgs, 1);								   
}
static int atomsel_interp_volume2(void *v, int num, double *data, int *flgs) {							      
  return atomsel_interp_volume_array(v, num, data, flgs, 2);								   
}
static int atomsel_interp_volume3(void *v, int num, double *data, int *flgs) {							      
  return atomsel_interp_volume_array(v, num, data, flgs, 3);								   
}
static int atomsel_interp_volume4(void *v, int num, double *data, int *flgs) {							      
  return atomsel_interp_volume_array(v, num, data, flgs, 4);								   
}
static int atomsel_interp_volume5(void *v, int num, double *data, int *flgs) {							      
  return atomsel_interp_volume_array(v, num, data, flgs, 5);								   
}
static int atomsel_interp_volume6(void *v, int num, double *data, int *flgs) {							      
  return atomsel_interp_volume_array(v, num, data, flgs, 6);								   
}
static int atomsel_interp_volume7(void *v, int num, double *data, int *flgs) {							      
  return atomsel_interp_volume_array(v, num, data, flgs, 7);								   
}




void atomSelParser_init(SymbolTable *atomSelParser) {
  atomSelParser->add_keyword("name", atomsel_name, atomsel_set_name);
  atomSelParser->add_keyword("type", atomsel_type, atomsel_set_type);

  // XXX probably only use these two for testing of BaseMolecule::analyze()
  atomSelParser->add_keyword("backbonetype", atomsel_backbonetype, NULL);
  atomSelParser->add_keyword("residuetype", atomsel_residuetype, NULL);

  atomSelParser->add_keyword("index", atomsel_index, NULL);
  atomSelParser->add_keyword("serial", atomsel_serial, NULL);
  atomSelParser->add_keyword("atomicnumber", atomsel_atomicnumber, atomsel_set_atomicnumber);
  atomSelParser->add_keyword("element", atomsel_element, atomsel_set_element);
  atomSelParser->add_keyword("residue", atomsel_residue, NULL);
  atomSelParser->add_keyword("resname", atomsel_resname, atomsel_set_resname);
  atomSelParser->add_keyword("altloc", atomsel_altloc, atomsel_set_altloc);
  atomSelParser->add_keyword("resid", atomsel_resid, atomsel_set_resid);
  atomSelParser->add_keyword("insertion", atomsel_insertion, NULL);
  atomSelParser->add_keyword("chain", atomsel_chain, atomsel_set_chain);
  atomSelParser->add_keyword("segname", atomsel_segname, atomsel_set_segname);
  atomSelParser->add_keyword("segid", atomsel_segname, atomsel_set_segname);

  atomSelParser->add_singleword("all", atomsel_all, NULL);
  atomSelParser->add_singleword("none", atomsel_none, NULL);

  atomSelParser->add_keyword("fragment", atomsel_fragment, NULL);
  atomSelParser->add_keyword("pfrag", atomsel_pfrag, NULL);
  atomSelParser->add_keyword("nfrag", atomsel_nfrag, NULL);
  atomSelParser->add_keyword("numbonds", atomsel_numbonds, NULL);

  atomSelParser->add_singleword("backbone", atomsel_backbone, NULL);
  atomSelParser->add_singleword("sidechain", 
                               atomsel_sidechain, NULL);
  atomSelParser->add_singleword("protein", atomsel_protein, NULL);
  atomSelParser->add_singleword("nucleic", atomsel_nucleic, NULL);
  atomSelParser->add_singleword("water", atomsel_water, NULL);
  atomSelParser->add_singleword("waters", atomsel_water, NULL);
  atomSelParser->add_singleword("vmd_fast_hydrogen", atomsel_hydrogen, NULL);

  // secondary structure functions
  atomSelParser->add_singleword("helix", 
				atomsel_helix, atomsel_set_helix);
  atomSelParser->add_singleword("alpha_helix", 
				atomsel_alpha_helix, atomsel_set_helix);
  atomSelParser->add_singleword("helix_3_10", 
				atomsel_3_10_helix, atomsel_set_3_10_helix);
  atomSelParser->add_singleword("pi_helix", 
				atomsel_pi_helix, atomsel_set_pi_helix);
  atomSelParser->add_singleword("sheet", atomsel_sheet, atomsel_set_sheet);
  atomSelParser->add_singleword("betasheet", atomsel_sheet, atomsel_set_sheet);
  atomSelParser->add_singleword("beta_sheet",atomsel_sheet, atomsel_set_sheet);
  atomSelParser->add_singleword("extended_beta", atomsel_extended_sheet, 
				atomsel_set_sheet);
  atomSelParser->add_singleword("bridge_beta", atomsel_bridge_sheet, 
				atomsel_set_bridge_sheet);
  atomSelParser->add_singleword("turn", atomsel_turn, atomsel_set_turn);
  atomSelParser->add_singleword("coil", atomsel_coil, atomsel_set_coil);
  atomSelParser->add_keyword("structure",atomsel_sstruct, atomsel_set_sstruct);

#if defined(VMDWITHCARBS)
  atomSelParser->add_keyword("pucker", atomsel_pucker, NULL);
#endif

  atomSelParser->add_keyword("user",  atomsel_user,  atomsel_set_user);
  atomSelParser->add_keyword("user2", atomsel_user2, atomsel_set_user2);
  atomSelParser->add_keyword("user3", atomsel_user3, atomsel_set_user3);
  atomSelParser->add_keyword("user4", atomsel_user4, atomsel_set_user4);

  atomSelParser->add_keyword("x", atomsel_xpos, atomsel_set_xpos);
  atomSelParser->add_keyword("y", atomsel_ypos, atomsel_set_ypos);
  atomSelParser->add_keyword("z", atomsel_zpos, atomsel_set_zpos);
  atomSelParser->add_keyword("vx", atomsel_vx, atomsel_set_vx);
  atomSelParser->add_keyword("vy", atomsel_vy, atomsel_set_vy);
  atomSelParser->add_keyword("vz", atomsel_vz, atomsel_set_vz);
  atomSelParser->add_keyword("ufx", atomsel_ufx, atomsel_set_ufx);
  atomSelParser->add_keyword("ufy", atomsel_ufy, atomsel_set_ufy);
  atomSelParser->add_keyword("ufz", atomsel_ufz, atomsel_set_ufz);
  atomSelParser->add_keyword("phi", atomsel_phi, atomsel_set_phi);
  atomSelParser->add_keyword("psi", atomsel_psi, atomsel_set_psi);
  atomSelParser->add_keyword("radius", atomsel_radius, 
			     atomsel_set_radius);
  atomSelParser->add_keyword("mass", atomsel_mass, atomsel_set_mass);
  atomSelParser->add_keyword("charge", atomsel_charge, 
			     atomsel_set_charge);
  atomSelParser->add_keyword("beta", atomsel_beta, atomsel_set_beta);
  atomSelParser->add_keyword("occupancy", 
			     atomsel_occupancy, atomsel_set_occupancy);

  atomSelParser->add_keyword("flag0", atomsel_flag0, atomsel_set_flag0);
  atomSelParser->add_keyword("flag1", atomsel_flag1, atomsel_set_flag1);
  atomSelParser->add_keyword("flag2", atomsel_flag2, atomsel_set_flag2);
  atomSelParser->add_keyword("flag3", atomsel_flag3, atomsel_set_flag3);
  atomSelParser->add_keyword("flag4", atomsel_flag4, atomsel_set_flag4);
  atomSelParser->add_keyword("flag5", atomsel_flag5, atomsel_set_flag5);
  atomSelParser->add_keyword("flag6", atomsel_flag6, atomsel_set_flag6);
  atomSelParser->add_keyword("flag7", atomsel_flag7, atomsel_set_flag7);

  atomSelParser->add_stringfctn("sequence", atomsel_sequence);
  atomSelParser->add_stringfctn("rasmol", 
				atomsel_rasmol_primitive);

  // three letters for resname, 1 or more letters for resid
  ////  DOESN'T WORK WITH PARSER -- breaks 'segname PRO1'
  //   atomSelParser->add_singleword("[a-zA-Z][a-zA-Z][a-zA-Z][0-9]+",
  //				"resnameID", atomsel_resname_resid,
  //				NULL);

  // and a few functions for good measure
  // Note: These functions must return the same output for given input.
  //       Functions like rand() will break some optimizations in the 
  //       atom selection code otherwise.
  atomSelParser->add_cfunction("sqr", atomsel_square);
  atomSelParser->add_cfunction("sqrt", sqrt);
  atomSelParser->add_cfunction("abs", fabs);
  atomSelParser->add_cfunction("floor", floor);
  atomSelParser->add_cfunction("ceil", ceil);
  atomSelParser->add_cfunction("sin", sin);
  atomSelParser->add_cfunction("cos", cos);
  atomSelParser->add_cfunction("tan", tan);
  atomSelParser->add_cfunction("atan", atan);
  atomSelParser->add_cfunction("asin", asin);
  atomSelParser->add_cfunction("acos", acos);
  atomSelParser->add_cfunction("sinh", sinh);
  atomSelParser->add_cfunction("cosh", cosh);
  atomSelParser->add_cfunction("tanh", tanh);
  atomSelParser->add_cfunction("exp", exp);
  atomSelParser->add_cfunction("log", log);
  atomSelParser->add_cfunction("log10", log10);
  
  

  atomSelParser->add_keyword("volindex0", atomsel_gridindex0, NULL);
  atomSelParser->add_keyword("volindex1", atomsel_gridindex1, NULL);
  atomSelParser->add_keyword("volindex2", atomsel_gridindex2, NULL);
  atomSelParser->add_keyword("volindex3", atomsel_gridindex3, NULL);
  atomSelParser->add_keyword("volindex4", atomsel_gridindex4, NULL);
  atomSelParser->add_keyword("volindex5", atomsel_gridindex5, NULL);
  atomSelParser->add_keyword("volindex6", atomsel_gridindex6, NULL);
  atomSelParser->add_keyword("volindex7", atomsel_gridindex7, NULL);
  atomSelParser->add_keyword("vol0", atomsel_volume0, NULL);
  atomSelParser->add_keyword("vol1", atomsel_volume1, NULL);
  atomSelParser->add_keyword("vol2", atomsel_volume2, NULL);
  atomSelParser->add_keyword("vol3", atomsel_volume3, NULL);
  atomSelParser->add_keyword("vol4", atomsel_volume4, NULL);
  atomSelParser->add_keyword("vol5", atomsel_volume5, NULL);
  atomSelParser->add_keyword("vol6", atomsel_volume6, NULL);
  atomSelParser->add_keyword("vol7", atomsel_volume7, NULL);
  atomSelParser->add_keyword("interpvol0", atomsel_interp_volume0, NULL);
  atomSelParser->add_keyword("interpvol1", atomsel_interp_volume1, NULL);
  atomSelParser->add_keyword("interpvol2", atomsel_interp_volume2, NULL);
  atomSelParser->add_keyword("interpvol3", atomsel_interp_volume3, NULL);
  atomSelParser->add_keyword("interpvol4", atomsel_interp_volume4, NULL);
  atomSelParser->add_keyword("interpvol5", atomsel_interp_volume5, NULL);
  atomSelParser->add_keyword("interpvol6", atomsel_interp_volume6, NULL);
  atomSelParser->add_keyword("interpvol7", atomsel_interp_volume7, NULL);
}


//////////////////////////  constructor and destructor
// constructor; parse string and see if OK
AtomSel::AtomSel(SymbolTable *st, int id)
: ID(id) {
  
  // initialize variables
  table = st;
  selected = NO_PARSE;
  firstsel = 0;
  lastsel = NO_PARSE;
  on = NULL;
  cmdStr = NULL;
  tree = NULL;
  num_atoms = 0;
  which_frame = TS_NOW;
  do_update = 0;
}

// destructor; free up space and make all pointers invalid
AtomSel::~AtomSel(void) {
  table = NULL;
  num_atoms = 0;
  selected = NO_PARSE;
  firstsel = 0;
  lastsel = NO_PARSE;
  delete [] on;
  on = NULL;
  delete tree;
  delete [] cmdStr;
}

int AtomSel::change(const char *newcmd, DrawMolecule *mol) {
  if (newcmd) {
    ParseTree *newtree = table->parse(newcmd);
    if (!newtree) {
      return NO_PARSE;
    }
    delete [] cmdStr;
    cmdStr = stringdup(newcmd);
    delete tree;
    tree = newtree;
  }
  
  // and evaluate
  atomsel_ctxt context(table, mol, which_frame, NULL);

  // resize flags array if necessary
  if (num_atoms != mol->nAtoms || (on == NULL)) {
    if (on) delete [] on;
    on = new int[mol->nAtoms];
    num_atoms = mol->nAtoms;
  }

  tree->use_context(&context);
  int rc = tree->evaluate(mol->nAtoms, on);

  // store parse return code before we postprocess
  int ret_code = rc ? PARSE_SUCCESS : NO_EVAL;

  // find the first selected atom, the last selected atom,
  // and the total number of selected atoms
  selected = 0;
  firstsel = 0; // ensure that if nothing is selected, that we
  lastsel = -1; // trim subsequent loops to do no iterations

#if 1
  // use high performance vectorized select analysis routine
  if (analyze_selection_aligned(num_atoms, on, &firstsel, &lastsel, &selected))
    return ret_code;
#else
  // find the first selected atom, if any
  int i;
  for (i=0; i<num_atoms; i++) if (on[i]) break;

  // detect if no atoms were selected, 
  // otherwise we just found the first selected atom
  if (i == num_atoms)
    return ret_code;
  else
    firstsel = i;

  // count the number of selected atoms (there are only 0s and 1s)
  // and determine the index of the last selected atom
  for (i=firstsel; i<num_atoms; i++) {
    selected += on[i];
    if (on[i])
      lastsel = i;
  }
#endif

  return ret_code;
}


// return the current coordinates (or NULL if error)
float *AtomSel::coordinates(MoleculeList *mlist) const {
  Timestep *ts = timestep(mlist);
  if (ts) return ts->pos;
  return NULL;
}


// return the current timestep (or NULL if error)
Timestep *AtomSel::timestep(MoleculeList *mlist) const {
  DrawMolecule *mymol = mlist->mol_from_id(molid()); 

  if (!mymol) {
    msgErr << "No molecule" << sendmsg;
    return NULL;  // no molecules
  }

  switch (which_frame) {
    case TS_LAST: 
      return mymol->get_last_frame();

    case TS_NOW: 
      return mymol->current();

    default:
      // if past end of coords return last coord
      if (!mymol->get_frame(which_frame)) { 
        return mymol->get_last_frame(); 
      }
  }

  return mymol->get_frame(which_frame);
}


int AtomSel::get_frame_value(const char *s, int *val) {
  *val = 1;
  if (!strcmp(s, "last")) {
    *val = TS_LAST;
  }
  if (!strcmp(s, "first")) {
    *val = 0;
  }
  if (!strcmp(s, "now")) {
    *val = TS_NOW;
  }
  if (*val == 1) {
    int new_frame = atoi(s);
    *val = new_frame;
    if (new_frame < 0) {
      return -1;
    }
  }
  return 0;
}
