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
 *	$RCSfile: MoleculeList.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.118 $	$Date: 2011/03/05 05:24:47 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The MoleculeList class, which is a list of the molecules being displayed.
 * This is a Displayable object, where each molecule is a child Displayable.
 *
 ***************************************************************************/

#include <stdio.h>
#include <ctype.h>

#include "MoleculeList.h"
#include "Inform.h"
#include "Scene.h"
#include "MaterialList.h"
#include "VMDApp.h"
#include "ParseTree.h"
#include "MoleculeGraphics.h" 
#include "CommandQueue.h"
#include "TextEvent.h"

// default atom selection
#define DEFAULT_ATOMSEL "all"

///////////////////////  constructor  
MoleculeList::MoleculeList(VMDApp *vmdapp, Scene *sc) 
        : app(vmdapp), scene (sc), molList(8) {

  topMol = NULL;
  defaultAtomColor = AtomColorName[DEFAULT_ATOMCOLOR];
  defaultAtomRepresentation = AtomRepInfo[DEFAULT_ATOMREP].name;
  defaultAtomSelection = DEFAULT_ATOMSEL;
  defaultAtomMaterial = app->materialList->material_name(0);

  currAtomRep = new AtomRep();
  currAtomSel = stringdup(DEFAULT_ATOMSEL);
  currMaterial = 0;
  lastCenter[0] = 0;
  lastCenter[1] = 0;
  lastCenter[2] = 0;
  lastCenter[3] = 1;

#if defined(MLIST_USE_HASH)
  inthash_init(&indexIDhash, 5003); // initialize ID to index hash table
#endif

  init_colors();
}


///////////////////////  destructor  
MoleculeList::~MoleculeList(void) {
  int i;

  // delete all added molecules
  for (i=0; i < num(); i++) {
    delete molecule(i);
  }

#if defined(MLIST_USE_HASH)
  inthash_destroy(&indexIDhash); // destroy ID to index hash table
#endif

  delete currAtomColor;
  delete currAtomRep;
  delete [] currAtomSel;
}

// do action when a new color list is provided
void MoleculeList::init_colors(void) {
  // create atom color object
  currAtomColor = new AtomColor(this, scene);
  
  // create new color name categories
  colorCatIndex[MLCAT_NAMES] = scene->add_color_category("Name");
  colorCatIndex[MLCAT_TYPES] = scene->add_color_category("Type");
  colorCatIndex[MLCAT_ELEMENTS] = scene->add_color_category("Element");
  colorCatIndex[MLCAT_RESNAMES] = scene->add_color_category("Resname");
  colorCatIndex[MLCAT_RESTYPES] = scene->add_color_category("Restype");
  colorCatIndex[MLCAT_CHAINS] = scene->add_color_category("Chain");
  colorCatIndex[MLCAT_SEGNAMES] = scene->add_color_category("Segname");
  colorCatIndex[MLCAT_CONFORMATIONS] = scene->add_color_category("Conformation");
  colorCatIndex[MLCAT_MOLECULES] = scene->add_color_category("Molecule");
  colorCatIndex[MLCAT_SPECIAL] = scene->add_color_category("Highlight");
  colorCatIndex[MLCAT_SSTRUCT] = scene->add_color_category("Structure");
  colorCatIndex[MLCAT_SURFACES] = scene->add_color_category("Surface");

  // ensure that Restype Unassigned has a color
  scene->add_color_item(
      colorCatIndex[MLCAT_RESTYPES], "Unassigned", scene->color_index("cyan"));
  scene->add_color_item(
      colorCatIndex[MLCAT_ELEMENTS], "X", scene->color_index("cyan"));
}


///////////////////////  public routines  

// put new names from given molecule into color lists
void MoleculeList::add_color_names(int molIndx) {
  int i, indx, catIndx;
  char newname[2];
  NameList<int> *molnames;
  Molecule *m;
  
  strcpy(newname, " "); // clear and add a NUL char

  // make sure this molecule is in the list already, otherwise just return
  if(!(m = molecule(molIndx))) 
    return;
  
  // for the given molecule, go through the NameList objects, and add them
  // to the color categories.  For new names, the color to use is the
  // color for the new index, mod # of total colors
  
  // atom names ... use upcased first non-numeric character as the name
  molnames = &(m->atomNames);
  int numatomnames = molnames->num();
  catIndx = colorCatIndex[MLCAT_NAMES];
  for (i=0; i < numatomnames; i++) {
    // get first non-numeric char of name, and convert to upper case
    const char *c = molnames->name(i);

    while (*c && isdigit(*c)) 
      c++;

    if (!(*c)) 
      c = molnames->name(i);

    newname[0] = (char)toupper(*c);
 
    // add this single-char name to color list; if it exists, get color
    indx = scene->add_color_item(catIndx, newname, 
             scene->num_category_items(catIndx) % VISCLRS);
    
    // for the molecule, set the color for this name
    molnames->set_data(i, indx);
  }
  
  // atom types ... use upcased first non-numeric character as the name
  molnames = &(m->atomTypes);
  int numatomtypes = molnames->num();
  catIndx = colorCatIndex[MLCAT_TYPES];
  for (i=0; i < numatomtypes; i++) {
    // get first non-numeric char of name, and convert to upper case
    const char *c = molnames->name(i);

    while (*c && isdigit(*c)) 
      c++;

    if (!(*c)) 
      c = molnames->name(i);

    newname[0] = (char)toupper(*c);
    
    // add this single-char name to color list; if it exists, get color
    indx = scene->add_color_item(catIndx, newname, 
             scene->num_category_items(catIndx) % VISCLRS);
    
    // for the molecule, set the color for this name
    molnames->set_data(i, indx);
  }
  
  // residue names and types ... use full name
  molnames = &(m->resNames);
  int numresnames = molnames->num();
  catIndx = colorCatIndex[MLCAT_RESNAMES];
  for (i=0; i < numresnames; i++) {
    indx = scene->add_color_item(catIndx, molnames->name(i), 
             scene->num_category_items(catIndx) % VISCLRS);
    molnames->set_data(i, indx);
    
    // check for residue types
    if (resTypes.typecode(molnames->name(i)) < 0) {
      // the residue has not been added to the restype list yet
      resTypes.add_name(molnames->name(i), "Unassigned");
    }
  }

  // chain names ... use full name
  molnames = &(m->chainNames);
  int numchainnames = molnames->num();
  catIndx = colorCatIndex[MLCAT_CHAINS];
  for (i=0; i < numchainnames; i++) {
    indx = scene->add_color_item(catIndx, molnames->name(i), 
             scene->num_category_items(catIndx) % VISCLRS);
    molnames->set_data(i, indx);
  }

  // segment names ... use full name
  molnames = &(m->segNames);
  int numsegnames = molnames->num();
  catIndx = colorCatIndex[MLCAT_SEGNAMES];
  for (i=0; i < numsegnames; i++) {
    indx = scene->add_color_item(catIndx, molnames->name(i), 
             scene->num_category_items(catIndx) % VISCLRS);
    molnames->set_data(i, indx);
  }

  // conformation names ... use full name, but special-case "" to be "all"
  molnames = &(m->altlocNames);
  int numconformations = molnames->num();
  catIndx = colorCatIndex[MLCAT_CONFORMATIONS];
  for (i=0; i < numconformations; i++) {
    const char *confname = molnames->name(i);
    if (confname[0] == '\0') {
      confname = "all";
    }
    indx = scene->add_color_item(catIndx, confname, 
             scene->num_category_items(catIndx) % VISCLRS);
    molnames->set_data(i, indx);
  }

  // molecule name ... use full name
  catIndx = colorCatIndex[MLCAT_MOLECULES];
  char buf[20];
  sprintf(buf, "%d", m->id());
  scene->add_color_item(catIndx, buf, 
        scene->num_category_items(catIndx) % VISCLRS);
}


// add a new molecule
void MoleculeList::add_molecule(Molecule *newmol) {
  int newmolindex;

  // add the molecule to our list of molecules
  molList.append(newmol);
  newmolindex = molList.num() - 1;

#if defined(MLIST_USE_HASH)
  // enhash the newly loaded molid for molid to molindex translation
  inthash_insert(&indexIDhash, newmol->id(), newmolindex);
#endif

  // make the newly loaded molecule top
  make_top(newmolindex);
}

// set the top molecule ...
// make sure the given molecule is in the list; if not, do not do anything
void MoleculeList::make_top(Molecule *m) {
#if 1
  if (!m) {
    topMol = m;
  } else if (m && mol_index_from_id(m->id()) >= 0) {
    topMol = m;
  }
#else
  topMol = m;
#endif
  app->commandQueue->runcommand(new MoleculeEvent(
                                m ? m->id() : -1, MoleculeEvent::MOL_TOP));
}


// delete the molecule by its id
int MoleculeList::del_molecule(int id) {
  Molecule *m, *newtopmol = NULL;

  // for this particular case, must make sure index is correct
  if (!(m = mol_from_id(id)))
    return FALSE;

  int n = mol_index_from_id(id);

  // must change the top molecule, if necessary
  if (is_top(n)) {
    if (n+1 < num()) {             // is there a molecule following this one?
      newtopmol = molecule(n+1);   // get it
    } else if (n-1 >= 0) {         // is there a molecule before this one?
      newtopmol = molecule(n-1);   // get it
    } else {
      newtopmol = topMol;          // signal there are NO molecules now
    }
  }
 
  delete m;                         // delete the molecule data structure
  molList.remove(n);                // remove the molecule from the list
#if defined(MLIST_USE_HASH)
  // completely rebuild the hash table since all of
  // the molindex values are different now.
  // XXX this can cause N^2 performance when deleting all molecules.
  inthash_destroy(&indexIDhash);    // destroy ID to index hash table
  int molcount = num();
  int ml;
  inthash_init(&indexIDhash, 5003); // initialize ID to index hash table
  for (ml=0; ml<molcount; ml++) {
    inthash_insert(&indexIDhash, molecule(ml)->id(), ml);
  }
#endif
 
  // now, change the top molecule if necessary
  if (newtopmol != NULL) {
    if (newtopmol != topMol) {
      make_top(newtopmol);
    } else {
      make_top((Molecule *)NULL);
    }
  }

  return TRUE;
}


// delete all molecules (linear time in worst case)
int MoleculeList::del_all_molecules(void) {
  int lastmol;
  while ((lastmol=num()) > 0) {
    int n = lastmol - 1;
    Molecule *m = molecule(n);      // retrieve last molecule in the list
    delete m;                       // delete the molecule data structure
    molList.remove(n);              // remove the molecule from the list
  }
  inthash_destroy(&indexIDhash);    // destroy ID to index hash table
  inthash_init(&indexIDhash, 5003); // initialize ID to index hash table
  make_top((Molecule *)NULL);       // signal there are NO molecules now

  return TRUE;
}

int MoleculeList::set_default_color(const char *s) {
  AtomColor ac(this, scene);
  int success = ac.change(s);
  if (success) {
    defaultAtomColor = s;
  }
  return success;
}

int MoleculeList::set_default_representation(const char *s) {
  AtomRep ar;
  int success = ar.change(s);
  if (success) {
    defaultAtomRepresentation = s;
  }
  return success;
}

int MoleculeList::set_default_selection(const char *s) {
  ParseTree *tree = app->atomSelParser->parse(s);
  if (!tree) return FALSE;
  delete tree;
  defaultAtomSelection = s;
  return TRUE;
}

int MoleculeList::set_default_material(const char *s) {
  if (app->materialList->material_index(s) < 0)
    return FALSE;
  defaultAtomMaterial = s;
  return TRUE;
}

// set current atom coloring method
int MoleculeList::set_color(char *s) {
  return (currAtomColor->change(s));
}


// set current atom representation method
int MoleculeList::set_representation(char *s) {
  return (currAtomRep->change(s));
}


// set current atom selection
int MoleculeList::set_selection(const char *s) {
  // check that the selection text can be parsed
  ParseTree *tree = app->atomSelParser->parse(s);
  if (!tree) return FALSE;
  delete tree;
  delete [] currAtomSel;
  currAtomSel = stringdup(s);
  return TRUE;
}


// set current material
int MoleculeList::set_material(char *s) {
  currMaterial = app->materialList->material_index(s);
  if (currMaterial < 0) {
    currMaterial = 0;
    msgErr << "Invalid material specified: " << s << sendmsg;
    msgErr << "Using default: " << app->materialList->material_name(0)
           << sendmsg;
    return 0;
  }
  return 1;
}


// return the current material
const char *MoleculeList::material(void) {
  return app->materialList->material_name(currMaterial);
}


// add a new representation to the nth molecule, copying from the current one
int MoleculeList::add_rep(int n) {
  Molecule *mol = molecule(n);
  if (!mol) return FALSE;
  AtomColor *ac = new AtomColor(*currAtomColor);
  AtomRep *ar = new AtomRep(*currAtomRep);
  AtomSel *as = new AtomSel(app->atomSelParser, mol->id());
  const Material *mat = app->materialList->material(currMaterial);
  as->change(currAtomSel, mol);
  ac->find(mol);
  mol->add_rep(ac, ar, as, mat);
  return TRUE;
}


// change the graphics representation m, for the specified molecule n, to
// the new settings.  Return success.
int MoleculeList::change_rep(int m, int n) {
  Molecule *mol = molecule(n);
  if (!mol) return FALSE;
  return (mol->change_rep(m, currAtomColor, currAtomRep, currAtomSel) &&
          change_repmat(m, n, material()));
}


// change just the coloring method for the mth rep in the nth molecule.
int MoleculeList::change_repcolor(int m, int n, char *s) {
  Molecule *mol = molecule(n);
  if (mol) {
    AtomColor ac(*currAtomColor);
    if (ac.change(s))
      return (mol->change_rep(m, &ac, NULL, NULL));
  }
  return FALSE;
}
  

// change just the representation for the mth rep in the nth molecule.
int MoleculeList::change_repmethod(int m, int n, char *s) {
  Molecule *mol = molecule(n);
  if (mol) {
    AtomRep ar(*currAtomRep);
    if (ar.change(s))
      return (mol->change_rep(m, NULL, &ar, NULL));
  }
  return FALSE;
}
  

// change just the selection for the mth rep in the nth molecule.
int MoleculeList::change_repsel(int m, int n, const char *s) {
  Molecule *mol = molecule(n);
  if (!mol) return FALSE;
  ParseTree *tree = app->atomSelParser->parse(s);
  if (!tree) return FALSE;
  delete tree;
  return (mol->change_rep(m, NULL, NULL, s));
}


// change the material for the mth rep in the nth molecule.
int MoleculeList::change_repmat(int m, int n, const char *s) {
  Molecule *mol = molecule(n);
  if (mol) {
    int ind = app->materialList->material_index(s);
    if (ind >= 0) {
      DrawMolItem *d = mol->component(m);
      if (d) {
        d->change_material(app->materialList->material(ind));  
        return TRUE;
      }
    }
  }
  return FALSE;
}


// delete a graphics representation m, for molecule n, return success.
int MoleculeList::del_rep(int m, int n) {
  Molecule *mol = molecule(n);
  if (mol != NULL)
    return (mol->del_rep(m) != 0);
  return 0;
}

// derive scaling and centering values from the current top molecule's
// proper centering location, and scaled to fit in (-1 ... 1) box
void MoleculeList::center_from_top_molecule_reps() {
  float x, y, z;
  
  if (topMol && topMol->cov(x,y,z)) {
    lastCenter[0] = x;
    lastCenter[1] = y;
    lastCenter[2] = z;
    lastCenter[3] = topMol->scale_factor();
  }
}

// apply the current centering transforms to the top molecule only
void MoleculeList::center_top_molecule(void) {
  if (topMol) {
    topMol->reset_transformation();
    topMol->mult_scale(lastCenter[3]);
    topMol->add_cent_trans(-lastCenter[0], -lastCenter[1], -lastCenter[2]);
  }
}
 
// apply the current centering transforms to all molecules
void MoleculeList::center_all_molecules(void) {
  int i;
  int n = num();
  for (i=0; i<n; i++) {
    molecule(i)->reset_transformation();
    molecule(i)->mult_scale(lastCenter[3]);
    molecule(i)->add_cent_trans(-lastCenter[0], -lastCenter[1], -lastCenter[2]);
  }
}


// For the given Pickable, determine if it is a molecule or a representation
// of a molecule, and return the proper pointer if it is (NULL otherwise)
Molecule *MoleculeList::check_pickable(Pickable *pobj) {
  int i, j, mnum, repnum;

  // loop over all molecules 
  mnum = num();
  for (i=0; i < mnum; i++) {
    // check this molecule to see if it matches
    Molecule *mol = molecule(i);
    if (pobj == mol)
      return mol;

    // and check each of its representations to see if they match
    repnum = mol->components();
    for (j=0; j < repnum; j++)
      if (pobj == mol->component(j))
        return mol;

    // check MoleculeGraphics pointer
    if (pobj == mol->moleculeGraphics()) 
      return mol;
  }

  return NULL; // if here, nothing found
}

