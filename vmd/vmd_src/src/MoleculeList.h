/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: MoleculeList.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.69 $	$Date: 2010/12/16 04:08:24 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The MoleculeList class, which is a list of the molecules being displayed.
 * This is a Displayable object, where each molecule is a child Displayable.
 *
 ***************************************************************************/
#ifndef MOLECULELIST_H
#define MOLECULELIST_H

#include "Molecule.h"
#include "AtomColor.h"
#include "AtomRep.h"
#include "ResizeArray.h"
#include "JString.h"
#include "utilities.h"
#include "inthash.h"

// use hash table to accelerate lookups
#define MLIST_USE_HASH 1

class VMDApp;

// number of color categories, and where they're found in the table
enum { MLCAT_NAMES, MLCAT_TYPES, MLCAT_ELEMENTS, 
       MLCAT_RESNAMES, MLCAT_RESTYPES,
       MLCAT_CHAINS, MLCAT_SEGNAMES, MLCAT_CONFORMATIONS, MLCAT_MOLECULES,
       MLCAT_SPECIAL, MLCAT_SSTRUCT, MLCAT_SURFACES, NUM_MLCAT};


/// Manages a list of the molecules being displayed.
class MoleculeList {
private:
  /// VMDApp handle, for scene, materialList, and atomSelParser
  VMDApp *app;

  /// cache the scene so we can update color entries as molecules are loaded.
  Scene *scene;

  /// the 'top' molecule, which determines what the centering and scaling of
  /// the molecules should be
  Molecule *topMol;

  /// molecules in this list.
  ResizeArray<Molecule *> molList;
#if defined(MLIST_USE_HASH)
  inthash_t indexIDhash;           ///< ID to index lookup accelerator
#endif

  /// current atom selection, representation, and coloring methods
  /// also the material
  AtomColor *currAtomColor;
  AtomRep *currAtomRep;
  char *currAtomSel;
  int currMaterial;
  float lastCenter[4];


  /// default atom rep methods for new molecules
  JString defaultAtomColor;
  JString defaultAtomRepresentation;
  JString defaultAtomSelection;
  JString defaultAtomMaterial;

  /// set the given Molecule as the top molecule.
  void set_top_molecule(Molecule *);

protected:
  /// do action when a new color list is provided
  void init_colors();

public:
  /// mapping of residue names <--> residue types (hydrophobic, neutral, etc.)
  NameList<const char *> resTypes;

  /// color category indices
  int colorCatIndex[NUM_MLCAT];

  /// put new names from given molecule into color lists
  void add_color_names(int);

public:
  MoleculeList(VMDApp *, Scene *);
  virtual ~MoleculeList(void);

  /// return the number of molecules in this list
  int num(void) { return molList.num(); }

  /// return the Nth molecule (index runs 0 ... (count-1))
  Molecule *molecule(int n) {
    Molecule *retval = NULL;
    if(n >= 0 && n < num())
      retval = molList[n];
    return retval;
  }

  /// return the index of the molecule with given ID (-1 if error)
  int mol_index_from_id(int id) {
#if defined(MLIST_USE_HASH)
    // HASH_FAIL == -1, so if we don't find the requested key 
    // the return code is still correct (-1 means no such molecule to VMD).
    return inthash_lookup(&indexIDhash, id);
#else
    // XXX slow linear search implementation, causes N^2 molecule load behavior
    for (int i=num() - 1; i >= 0; i--) {
      if (id == (molList[i])->id()) {
        return i;
      }
    }
    return -1;
#endif
  }

  /// return the molecule with given ID (NULL if error)
  Molecule *mol_from_id(int id) {
    return molecule(mol_index_from_id(id));
  }
  
  /// add a new molecule; return it's position in molList
  void add_molecule(Molecule *);

  /// remove the molecule from the list with given ID.  Return success.
  int del_molecule(int);

  /// delete all molecules (linear time in worst case)
  int del_all_molecules(void);


  //
  // routines to get/set characteristics for new graphics representations
  //
  
  /// get/set current atom coloring method
  int set_color(char *);
  char *color(void) { return currAtomColor->cmdStr; }
  
  /// get/set current atom representation method
  int set_representation(char *);
  char *representation(void) { return currAtomRep->cmdStr; }
  
  /// get/set current atom selection command. Return success.
  int set_selection(const char *);
  const char *selection() const { return currAtomSel; }
  
  /// get/set current atom material method
  int set_material(char *);
  const char *material(void);

  /// default values for new reps
  int set_default_color(const char *);
  int set_default_representation(const char *);
  int set_default_selection(const char *);
  int set_default_material(const char *);
  const char *default_color() const {
    return defaultAtomColor; 
  }
  const char *default_representation() const {
    return defaultAtomRepresentation;
  }
  const char *default_selection() const {
    return defaultAtomSelection;
  }
  const char *default_material() const {
    return defaultAtomMaterial;
  }

  /// add a new graphics representation to the specified molecule.
  /// uses the 'current' coloring, representation, and selection settings.
  int add_rep(int n);
  
  /// change the graphics representation m, for the specified molecule n.
  /// uses the 'current' coloring, representation, and selection settings.
  int change_rep(int m, int n);

  /// change just the coloring method for the mth rep in the nth molecule.
  int change_repcolor(int m, int n, char *);
  
  /// change just the representation for the mth rep in the nth molecule.
  int change_repmethod(int m, int n, char *);
  
  /// change just the selection for the mth rep in the nth molecule.  Return
  /// success.
  int change_repsel(int m, int n, const char *);
 
  /// change just the material for the mth rep in the nth molecule.
  int change_repmat(int m, int n, const char *);

  /// delete a graphics representation m, for the specified molecule n.
  /// return success.
  int del_rep(int m, int n);

  //
  //
  // routines to get/set top, active, displayed, fixed molecules
  //
  
  /// query or set the top molecule
  Molecule *top(void) { return topMol; }
  int is_top(int n) { return (molecule(n) == topMol); }
  int is_top(Molecule *m) { return topMol == m; }
  void make_top(int n) { make_top(molecule(n)); }
  void make_top(Molecule *m);

  /// query/set active status of Nth molecule
  int active(int n) { return molecule(n)->active; }
  int active(Molecule *m) { return (m && m->active); }
  void activate(int n) { molecule(n)->active = TRUE; }
  void inactivate(int n) { molecule(n)->active = FALSE; }
  
  /// query/set displayed status of Nth molecule
  int displayed(int n) { return molecule(n)->displayed(); }
  int displayed(Molecule *m) { return (m && m->displayed()); }
  void show(int n) { molecule(n)->on(); }
  void hide(int n) { molecule(n)->off(); }
  
  /// query/set fixed status of Nth molecule
  int fixed(int n) { return molecule(n)->fixed(); }
  int fixed(Molecule *m) { return (m && m->fixed()); }
  void fix(int n) { molecule(n)->fix(); }
  void unfix(int n) { molecule(n)->unfix(); }

  /// Sets molecules' tranformations to center and scale properly
  void center_from_top_molecule_reps(void);
  void center_top_molecule(void);
  void center_all_molecules(void);

  /// given a pickable, find it and return the Molecule*, or NULL
  Molecule *check_pickable(Pickable *pobj);
};

#endif

