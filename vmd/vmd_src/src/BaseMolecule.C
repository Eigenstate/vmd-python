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
 *	$RCSfile: BaseMolecule.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.272 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Base class for all molecules, without display-specific information.  This
 * portion of a molecule contains the structural data, and all routines to
 * find the structure (backbone, residues, etc).  It does NOT contain the
 * animation list; that is maintained by Molecule (which is derived from
 * this class).
 *
 ***************************************************************************/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Inform.h"
#include "utilities.h"
#include "intstack.h"
#include "WKFUtils.h"

#include "BaseMolecule.h"
#include "VolumetricData.h"
#include "QMData.h"
#include "QMTimestep.h"

#ifdef VMDWITHCARBS
#include <vector> /* XXX this needs to go, gives trouble with MSVC */
#endif

#define MAXBONDERRORS 25

////////////////////////////  constructor

BaseMolecule::BaseMolecule(int myID) : residueList(10) , fragList(1),
#if defined(VMDWITHCARBS)
     pfragList(1), nfragList(1), smallringList(), smallringLinkages(), currentMaxRingSize(-1), currentMaxPathLength(-1), ID(myID) {
#else
     pfragList(1), nfragList(1), ID(myID) {
#endif

  // initialize all variables
  nAtoms = 0;
  cur_atom = 0;
  atomList = NULL;
  moleculename = NULL;
  lastbonderratomid=-1;
  bonderrorcount=0;
  datasetflags=NODATA;
  qm_data = NULL;
  radii_minmax_need_update = 1;
  radii_min = 0.0f;
  radii_max = 0.0f;
}


////////////////////////////   destructor

BaseMolecule::~BaseMolecule(void) {
  int i;

#if 0
  // XXX still need to track down ringlist leak(s)
  // delete carbohydrate ring data structures
  smallringList.clear();
  smallringLinkages.clear();
#endif

  // delete structural data
  delete [] atomList;
  for (i=0; i<residueList.num(); i++) {
    delete residueList[i];
  }
  for (i=0; i<nfragList.num(); i++) {
    delete nfragList[i];
  }
#if defined(VMDFASTRIBBONS)
  for (i=0; i<nfragCPList.num(); i++) {
    delete [] nfragCPList[i];
  }
#endif
  for (i=0; i<pfragList.num(); i++) {
    delete pfragList[i];
  }
#if defined(VMDFASTRIBBONS)
  for (i=0; i<pfragCPList.num(); i++) {
    delete [] pfragCPList[i];
  }
#endif
  for (i=0; i<fragList.num(); i++) {
    delete fragList[i];
  }
  for (i=0; i<volumeList.num(); i++) {
    delete volumeList[i];
  }

  // delete optional per-atom fields
  for (i=0; i<extraflt.num(); i++) {
    delete [] extraflt.data(i);
  }
  for (i=0; i<extraint.num(); i++) {
    delete [] extraint.data(i);
  }
  for (i=0; i<extraflg.num(); i++) {
    delete [] extraflg.data(i);
  }

  if (qm_data)
    delete qm_data;
}


///////////////////////  protected routines

// initialize the atom list ... should be called before adding any atoms
int BaseMolecule::init_atoms(int n) {
  if (n <= 0) {
    msgErr << "BaseMolecule: init_atoms called with invalid number of atoms: "
           << n << sendmsg;
    return FALSE;
  }
  if (cur_atom != 0 && nAtoms != n) {
    msgErr << "BaseMolecule: attempt to init atoms while structure building in progress!" << sendmsg;
    return FALSE;
  }

  if (!atomList) {
    int i;
    // first call to init_atoms
    nAtoms = n; // only place where nAtoms is set!
    atomList = new MolAtom[nAtoms];
    memset(atomList, 0, long(nAtoms)*sizeof(MolAtom));

    // initialize NULL extra data field, which is returned when
    // querying a non-existent field with extra*.data("fielddoesntexist")
    extraflt.add_name("NULL", NULL);
    extraint.add_name("NULL", NULL);
    extraflg.add_name("NULL", NULL);

    // initialize "default" extra data fields.
    extraflt.add_name("beta", new float[nAtoms]);
    extraflt.add_name("occupancy", new float[nAtoms]);
    extraflt.add_name("charge", new float[nAtoms]);
    extraflt.add_name("mass", new float[nAtoms]);
    extraflt.add_name("radius", new float[nAtoms]);

    // initialize default per-atom flags
    extraflg.add_name("flags", new unsigned char[nAtoms]);

    // initialize "default" extra floating point data fields.
    for (i=0; i<extraflt.num(); i++) {
      void *data = extraflt.data(i);
      if (data != NULL) 
        memset(data, 0, long(nAtoms)*sizeof(float));
    }

    // initialize "default" extra integer data fields.
    for (i=0; i<extraint.num(); i++) {
      void *data = extraint.data(i);
      if (data != NULL)
        memset(data, 0, long(nAtoms)*sizeof(int));
    }

    // initialize "default" extra flags data fields.
    for (i=0; i<extraflg.num(); i++) {
      void *data = extraflg.data(i);

      // 8 per-atom flags per unsigned char
      if (data != NULL)
        memset(data, 0, long(nAtoms)*sizeof(unsigned char));
    }

    return TRUE;
  }
  if (n != nAtoms) {
    msgErr << "The number of atoms in this molecule has already been assigned."
           << sendmsg;
    return FALSE;
  }
  return TRUE;
}

// add a new atom; return it's index, or (-1) if error.
int BaseMolecule::add_atoms(int addatomcount,
      const char *name, const char *atomtype, 
      int atomicnumber, const char *resname, int resid, 
      const char *chain, const char *segname, 
      const char *insertion, const char *altloc) {
  if (addatomcount < 1) {
    msgErr << "BaseMolecule: Cannot add negative atom count!" << sendmsg;
    return -1;
  }

  int newtotalcount = cur_atom + addatomcount;
  if (newtotalcount > nAtoms) {
    msgErr << "BaseMolecule: Cannot add more atoms to molecule!" << sendmsg;
    return -1;
  }

  if (!atomList || cur_atom >= nAtoms) {
    msgErr << "BaseMolecule: Cannot add new atom; currently " << nAtoms;
    msgErr << " atoms in structure." << sendmsg;
    return -1;
  }

  // add names to namelist, and put indices in MolAtom object
  int nameindex, typeindex, resnameindex, segnameindex, altlocindex, chainindex;
  nameindex = atomNames.add_name(name, atomNames.num());
  typeindex = atomTypes.add_name(atomtype, atomTypes.num());
  resnameindex = resNames.add_name(resname, resNames.num());
  segnameindex = segNames.add_name(segname, segNames.num());
  altlocindex = altlocNames.add_name(altloc, altlocNames.num());

  // use default of 'X' for chain if not given
  if(!chain || ! (*chain) || *chain == ' ')
    chainindex = chainNames.add_name("X", chainNames.num());
  else
    chainindex = chainNames.add_name(chain, chainNames.num());

  // create first atom  
  MolAtom *newatom = atom(cur_atom);
  newatom->init(cur_atom, resid, insertion);

  // set atom member variables
  newatom->nameindex = nameindex;
  newatom->typeindex = typeindex;
  newatom->atomicnumber = atomicnumber;
  newatom->resnameindex = resnameindex;
  newatom->segnameindex = segnameindex;
  newatom->altlocindex = altlocindex;
  newatom->chainindex = chainindex;

  // check for integer overflow/wraparound condition, which can occur
  // if an evil plugin defines 100,000 unique atom names, for example
  if (newatom->nameindex != nameindex ||
      newatom->typeindex != typeindex ||
      newatom->atomicnumber != atomicnumber ||
      newatom->resnameindex != resnameindex ||
      newatom->segnameindex != segnameindex ||
      newatom->altlocindex != altlocindex ||
      newatom->chainindex != chainindex) {
    msgErr << "BaseMolecule: Cannot add atom; namelist index value too large." << sendmsg;
    msgErr << "Recompile VMD with larger index types." << sendmsg;
    msgErr << "Atom namelist index values at time of overflow:" << sendmsg;
    msgErr << "  nameindex: " << nameindex << sendmsg;;
    msgErr << "  typeindex: " << typeindex << sendmsg;;
    msgErr << "  resnameindex: " << resnameindex << sendmsg;;
    msgErr << "  segnameindex: " << segnameindex << sendmsg;;
    msgErr << "  altlocindex: " << altlocindex << sendmsg;;
    msgErr << "  chainindex: " << chainindex << sendmsg;
    return -1;
  }

  cur_atom++; // first atom was added without difficulty

  // Now we can do the rest in a tight loop without
  // all of the per-atom checking we would do normally,
  // giving us a big speed boost with large counts.
  while (cur_atom < newtotalcount) {
    newatom = atom(cur_atom);
    newatom->init(cur_atom, resid, insertion);

    newatom->nameindex = nameindex;       // set atom member variables
    newatom->typeindex = typeindex;
    newatom->atomicnumber = atomicnumber;
    newatom->resnameindex = resnameindex;
    newatom->segnameindex = segnameindex;
    newatom->altlocindex = altlocindex;
    newatom->chainindex = chainindex;

    cur_atom++;
  } 

  return cur_atom;
}


// add a new bond; return 0 on success, or (-1) if error.
int BaseMolecule::add_bond(int a, int b, float bondorder, 
                           int bondtype, int backbonetype) {
  if (!nAtoms || a >= nAtoms || b >= nAtoms) {
    msgErr << "BaseMolecule: Atoms must be added before bonds." << sendmsg;
    return (-1);
  } 

  if (a == b) {
    msgErr << "BaseMolecule: Cannot bond atom " <<a<< " to itself." << sendmsg;
    return (-1);
  }

  // put the bond in the atom list
  if (atom(a)->add_bond(b, backbonetype)) {
    if (bonderrorcount < MAXBONDERRORS) {
      if (lastbonderratomid != a) {
        msgErr << "MolAtom " << a << ": Exceeded maximum number of bonds ("
               << atom(a)->bonds << ")." << sendmsg;
        lastbonderratomid=a;
        bonderrorcount++;
      }
    } else if (bonderrorcount == MAXBONDERRORS) {
      msgErr << "BaseMolecule: Excessive bonding errors encountered, perhaps atom coordinates are in the wrong units?" << sendmsg;
      msgErr << "BaseMolecule: Silencing bonding error messages." << sendmsg;
      bonderrorcount++;
    }
    return (-1);
  }

  if (atom(b)->add_bond(a, backbonetype)) {
    if (bonderrorcount < MAXBONDERRORS) {
      if (lastbonderratomid != b) {
        msgErr << "MolAtom " << b << ": Exceeded maximum number of bonds ("
               << atom(b)->bonds << ")." << sendmsg;
        lastbonderratomid=b;
        bonderrorcount++;
      }
    } else if (bonderrorcount == MAXBONDERRORS) {
      msgErr << "BaseMolecule: Excessive bonding errors encountered, perhaps atom coordinates are in the wrong units?" << sendmsg;
      msgErr << "BaseMolecule: Silencing bonding error messages." << sendmsg;
      bonderrorcount++;
    }
    return (-1);
  }

  // store bond orders and types
  setbondorder(a, atom(a)->bonds-1, bondorder);
  setbondorder(b, atom(b)->bonds-1, bondorder);

  setbondtype(a, atom(a)->bonds-1, bondtype);
  setbondtype(b, atom(b)->bonds-1, bondtype);

  return 0;
}

// Add a bond to a structure but check to make sure there isn't a 
// duplicate, as may be the case when merging bondlists from a file 
// and from a distance-based bond search
int BaseMolecule::add_bond_dupcheck(int a, int b, float bondorder, int bondtype) {
  int i;

  if (!nAtoms || a >= nAtoms || b >= nAtoms) {
    msgErr << "BaseMolecule: Atoms must be added before bonds." << sendmsg;
    return (-1);
  }

  MolAtom *atm = atom(a);
  int nbonds = atm->bonds;
  const int *bonds = &atm->bondTo[0];
  for (i=0; i<nbonds; i++) {
    if (bonds[i] == b) {
      return 0; // skip bond that already exists
    }
  }
  add_bond(a, b, bondorder, bondtype); // add it if it doesn't already exist

  return 0;
}

int BaseMolecule::add_angle(int a, int b, int c, int type) {
  int i,n;
  // make sure that a < c to make it easier to find duplicates later.
  if (a > c) { i = a; a = c; c = i; }
  
  angles.append3(a, b, c); 

  n = num_angles()-1;
  set_angletype(n, type);
  return n;
}


int BaseMolecule::set_angletype(int nangle, int type) {
  // type -1 is the anonymous type and is handled special w/o storage.
  // we also bail out in case we get an illegal angle index.
  if ((type < 0) || (nangle >= num_angles()))
    return -1;

  // fill array if not there
  if (angleTypes.num() <= nangle) {
    set_dataset_flag(BaseMolecule::ANGLETYPES);
    int addcnt = num_angles() - angleTypes.num();
    if (addcnt > 0)
      angleTypes.appendN(-1, addcnt);
  }
  
  angleTypes[nangle] = type;
  return type;
}

int BaseMolecule::get_angletype(int nangle) {
  if ((nangle < 0) || (nangle >= angleTypes.num()))
    return -1;

  return angleTypes[nangle];
}


int BaseMolecule::add_dihedral(int a, int b, int c, int d, int type) {
  int i, j, n;
  // make sure that b < c so that it is easier to find duplicates later.
  if (b > c) { i = a; j = b ; a = d ; b = c; d = i; c = j; }
  
  dihedrals.append4(a, b, c, d); 

  n = num_dihedrals()-1;
  set_dihedraltype(n, type);
  return n;
}


int BaseMolecule::set_dihedraltype(int ndihedral, int type) {
  // type -1 is the anonymous type and is handled special w/o storage.
  // we also bail out in case we get an illegal dihedral index.
  if ((type < 0) || (ndihedral >= num_dihedrals()))
    return -1;

  // fill array if not there
  if (dihedralTypes.num() <= ndihedral) {
    set_dataset_flag(BaseMolecule::ANGLETYPES);
    int addcnt = num_dihedrals() - dihedralTypes.num();
    if (addcnt > 0)
      dihedralTypes.appendN(-1, addcnt);
  }
  
  dihedralTypes[ndihedral] = type;
  return type;
}

int BaseMolecule::get_dihedraltype(int ndihedral) {
  if ((ndihedral < 0) || (ndihedral >= dihedralTypes.num()))
    return -1;

  return dihedralTypes[ndihedral];
}

int BaseMolecule::add_improper(int a, int b, int c, int d, int type) {
  int i, j, n;
  // make sure that b < c so that it is easier to find duplicates later.
  if (b > c) { i = a; j = b ; a = d ; b = c; d = i; c = j; }
  
  impropers.append4(a, b, c, d); 

  n = num_impropers()-1;
  set_impropertype(n, type);
  return n;
}


int BaseMolecule::set_impropertype(int nimproper, int type) {
  // type -1 is the anonymous type and is handled special w/o storage.
  // we also bail out in case we get an illegal improper index.
  if ((type < 0) || (nimproper >= num_impropers()))
    return -1;

  // fill array if not there
  if (improperTypes.num() <= nimproper) {
    set_dataset_flag(BaseMolecule::ANGLETYPES);
    int addcnt = num_impropers() - improperTypes.num();
    if (addcnt > 0)
      improperTypes.appendN(-1, addcnt);
  }
  
  improperTypes[nimproper] = type;
  return type;
}

int BaseMolecule::get_impropertype(int nimproper) {
  if ((nimproper < 0) || (nimproper >= improperTypes.num()))
    return -1;

  return improperTypes[nimproper];
}

///////////////////////////  public routines

void BaseMolecule::setbondorder(int atom, int bond, float order) {
  float *bondOrders = extraflt.data("bondorders");

  // if not already there, add it
  if (bondOrders == NULL) {
    if (order != 1) {
      int i;
      extraflt.add_name("bondorders", new float[nAtoms*MAXATOMBONDS]);
      bondOrders = extraflt.data("bondorders");
      for (i=0; i<nAtoms*MAXATOMBONDS; i++)
        bondOrders[i] = 1.0f;    

      bondOrders[atom * MAXATOMBONDS + bond] = order;
    } 
    return;
  }

  bondOrders[atom * MAXATOMBONDS + bond] = order;
}

float BaseMolecule::getbondorder(int atom, int bond) {
  float *bondOrders = extraflt.data("bondorders");

  // if not already there, add it
  if (bondOrders == NULL) { 
    return 1;
  }
   
  return bondOrders[atom * MAXATOMBONDS + bond];
}


void BaseMolecule::setbondtype(int atom, int bond, int type) {
  int *bondTypes = extraint.data("bondtypes");

  // if not already there, add it
  if (bondTypes == NULL) {
    if (type != -1) {
      int i;
      extraint.add_name("bondtypes", new int[nAtoms*MAXATOMBONDS]);
      bondTypes = extraint.data("bondtypes");
      for (i=0; i<nAtoms*MAXATOMBONDS; i++)
        bondTypes[i] = -1;    

      bondTypes[atom * MAXATOMBONDS + bond] = type;
    } 
    return;
  }

  bondTypes[atom * MAXATOMBONDS + bond] = type;
}

int BaseMolecule::getbondtype(int atom, int bond) {
  int *bondTypes = extraint.data("bondtypes");

  // if not already there, add it
  if (bondTypes == NULL) { 
    return -1;
  }
   
  return bondTypes[atom * MAXATOMBONDS + bond];
}


// return the Nth residue
Residue *BaseMolecule::residue(int n) {
  return residueList[n];
}


// return the Nth fragment
Fragment *BaseMolecule::fragment(int n) {
  return fragList[n];
}


// given an atom index, return the residue object for the residue it
// is in.  If it is not in a residue, return NULL.
Residue *BaseMolecule::atom_residue(int n) {
  MolAtom *atm = atom(n);
  if(atm->uniq_resid < 0)
    return NULL;
  else
    return residue(atm->uniq_resid);
}


// given an atom index, return the fragment object for the fragment it
// is in.  If it is not in a fragment, return NULL.
Fragment *BaseMolecule::atom_fragment(int n) {
  MolAtom *atm = atom(n);
  int frag = residue(atm->uniq_resid)->fragment;
  if(frag < 0)
    return NULL;
  else
    return fragment(frag);
}

// return a 'default' value for a given atom name
float BaseMolecule::default_radius(const char *nm) {
  float val = 1.5;
  // some names start with a number
  while (*nm && isdigit(*nm))
    nm++;
  if(nm) {
    switch(toupper(nm[0])) {
      // These are similar to the values used by X-PLOR with sigma=0.8
      // see page 50 of the X-PLOR 3.1 manual
      case 'H' : val = 1.00f; break;
      case 'C' : val = 1.50f; break;
      case 'N' : val = 1.40f; break;
      case 'O' : val = 1.30f; break;
      case 'F' : val = 1.20f; break;
      case 'S' : val = 1.90f; break;
    }
  }

  return val;
}


// return a 'default' value for a given atom name
float BaseMolecule::default_mass(const char *nm) {
  float val = 12.0;

  // some names start with a number
  while (*nm && isdigit(*nm)) nm++;
  if(nm) {
    switch(toupper(nm[0])) {
      case 'H' : val = 1.00800f; break;
      case 'C' : val = 12.01100f; break;
      case 'N' : val = 14.00700f; break;
      case 'O' : val = 15.99900f; break;
      case 'F' : val = 55.84700f; break;
      case 'P' : val = 30.97376f; break;
      case 'S' : val = 32.06000f; break;
    }
    if      ((toupper(nm[0]) == 'C') && (toupper(nm[1]) == 'L')) val = 35.453f;
    else if ((toupper(nm[0]) == 'N') && (toupper(nm[1]) == 'A')) val = 22.989770f;
    else if ((toupper(nm[0]) == 'M') && (toupper(nm[1]) == 'G')) val = 24.3050f;
  }

  return val;
}


float BaseMolecule::default_charge(const char *) {
  // return 0 for everything; later, when we put in a more reliable
  // system for determining the charge that's user-configurable,
  // we can start assigning more realistic charges.
  return 0.0f;
}


// count the number of unique bonds in the structure
int BaseMolecule::count_bonds(void) {
  int i, j;
  int count=0;

  for (i=0; i<nAtoms; i++) {
    int nbonds = atomList[i].bonds;
    const int *bonds = &atomList[i].bondTo[0];

    for (j=0; j<nbonds; j++) {
      if (bonds[j] > i)
        count++;
    }
  }

  return count;
}


void BaseMolecule::clear_bonds(void) {
  int i;
  for (i=0; i<nAtoms; i++)
    atomList[i].bonds = 0;
}


// analyze the molecule for more than just the atom/bond information
// This is here since it is called _after_ the molecule is added to
// the MoleculeList.  Thus, there is a Tcl callback to allow the
// user to update the bond information (or other fields?) before
// the actual search.
void BaseMolecule::analyze(void) {
  need_find_bonds = 0; // at this point it's too late

  // I have to let 0 atoms in because I want to be able to read things
  // like electron density maps, which have no atoms.
  // It is kinda wierd, then to make BaseMolecule be at the top of the
  // heirarchy.  Oh well.
  if (nAtoms < 1)
    return;

  wkf_timerhandle tm = wkf_timer_create();
  wkf_timer_start(tm);

  // call routines to find different characteristics of the molecule
  msgInfo << "Analyzing structure ..." << sendmsg;
  msgInfo << "   Atoms: " << nAtoms << sendmsg;

  // count unique bonds/angles/dihedrals/impropers/cterms
  int total_bondcount = count_bonds();
  double bondtime = wkf_timer_timenow(tm);

  msgInfo << "   Bonds: " << total_bondcount << sendmsg;
  msgInfo << "   Angles: " << num_angles()
          << "  Dihedrals: " << num_dihedrals()
          << "  Impropers: " << num_impropers()
          << "  Cross-terms: " << num_cterms()
          << sendmsg;

  msgInfo << "   Bondtypes: " << bondTypeNames.num()
          << "  Angletypes: " << angleTypeNames.num()
          << "  Dihedraltypes: " << dihedralTypeNames.num()
          << "  Impropertypes: " << improperTypeNames.num()
          << sendmsg;

  // restore residue and fragment lists to pristine state
  residueList.clear();
  fragList.clear();
  pfragList.clear();   ///< clear list of protein fragments
  pfragCyclic.clear(); ///< clear cyclic fragment flags
#if defined(VMDFASTRIBBONS)
  pfragCPList.clear(); ///< clear pre-computed control point lists
#endif
  nfragList.clear();   ///< clear list of nucleic fragments
  nfragCyclic.clear(); ///< clear cyclic fragment flags
#if defined(VMDFASTRIBBONS)
  nfragCPList.clear(); ///< clear pre-computed control point lists
#endif

  // assign per-atom backbone types
  find_backbone();
  double backbonetime = wkf_timer_timenow(tm);

  // find all the atoms in a resid connected to DNA/RNA/PROTEIN/WATER
  // also, assign a unique resid (uniq_resid) to each atom
  nResidues = find_residues();
  double findrestime = wkf_timer_timenow(tm);
  msgInfo << "   Residues: " << nResidues << sendmsg;

  nWaters = find_waters();
  double findwattime = wkf_timer_timenow(tm);
  msgInfo << "   Waters: " << nWaters << sendmsg;
  
  // determine which residues are connected to each other
  bonderrorcount=0; // reset error count before residue connectivity search
  find_connected_residues(nResidues); 
  double findconrestime = wkf_timer_timenow(tm);
 
  nSegments = find_segments(); 
  msgInfo << "   Segments: " << nSegments << sendmsg;

  nFragments = find_fragments();
  msgInfo << "   Fragments: " << nFragments;

  nProteinFragments = pfragList.num();
  msgInfo << "   Protein: " << nProteinFragments;

  nNucleicFragments = nfragList.num();
  msgInfo << "   Nucleic: " << nNucleicFragments << sendmsg;
  double findfragstime = wkf_timer_timenow(tm);
  
  // NOTE: The current procedure incorrectly identifies some lipid 
  // atoms as "ATOMNUCLEICBACK" (but not as "nucleic") as well
  // as some single water oxygens as "backbone". Here, we 
  // correct this by setting all atoms of non-polymeric residue types
  // to be "ATOMNORMAL" (i.e.: not backbone).
  int i;
  for (i=0; i<nAtoms; i++) {
    MolAtom *a = atom(i); 
    if ((a->residueType != RESNUCLEIC) && (a->residueType != RESPROTEIN)) 
      a->atomType = ATOMNORMAL;
  }
  double fixlipidtime = wkf_timer_timenow(tm);

  // Search for hydrogens
  // XXX Must be done after the rest of the structure finding routines,
  // because those routines assume that anything that isn't NORMAL is
  // a backbone atom.
  // We use the name-based definition used in the IS_HYDROGEN macro
  for (i=0; i<nAtoms; i++) {
    MolAtom *a = atom(i);
    const char *aname = atomNames.name(a->nameindex);
    if (aname != NULL && IS_HYDROGEN(aname))
      a->atomType = ATOMHYDROGEN;
  }
  double findhydrogentime = wkf_timer_timenow(tm);

#if defined(VMDFASTRIBBONS)
  calculate_ribbon_controlpoints();
#endif
  double calcribbontime = wkf_timer_timenow(tm);

#if 1
  if (getenv("VMDBASEMOLECULETIMING") != NULL) {
    printf("BaseMolecule::analyze() runtime breakdown:\n");
    printf("  %.2f bonds\n", bondtime);
    printf("  %.2f backbone\n", backbonetime - bondtime);
    printf("  %.2f findres\n", findrestime -  backbonetime);
    printf("  %.2f findwat\n", findwattime - findrestime);
    printf("  %.2f findconres\n", findconrestime - findwattime);
    printf("  %.2f findfrags\n", findfragstime - findconrestime);
    printf("  %.2f fixlipds\n", fixlipidtime - findfragstime);
    printf("  %.2f findH\n", findhydrogentime - fixlipidtime);
    printf("  %.2f calcribbons\n", calcribbontime - findhydrogentime);
  }
#endif
 
  wkf_timer_destroy(tm);
}


/// functions to find the backbone by matching atom names
int BaseMolecule::find_backbone(void) {
#if 1
  const char * protnames[] = { "CA", "C", "O", "N", NULL };
  const char * prottermnames[] = { "OT1", "OT2", "OXT", "O1", "O2", NULL };
  const char * nucnames[] = { "P", "O1P", "O2P", "OP1", "OP2", 
                              "C3*", "C3'", "O3*", "O3'", "C4*", "C4'", 
                              "C5*", "C5'", "O5*", "O5'", NULL };
#if 0
  // non-backbone nucleic acid atom names
  const char * nucnames2[] = { "C1*", "C1'", "C2*", "C2'", 
                               "O2*", "O2'", "O4*", "O4'", NULL };
#endif
  const char *nuctermnames[] = { "H5T", "H3T", NULL };

  ResizeArray<int> prottypecodes;
  ResizeArray<int> prottermtypecodes;
  ResizeArray<int> nuctypecodes;
  ResizeArray<int> nuctermtypecodes;
  const char ** str;
  int tc, i, j, k;

  // proteins
  for (str=protnames; *str!=NULL; str++) {
    if ((tc = atomNames.typecode((char *) *str)) >= 0) 
      prottypecodes.append(tc);
  }
  for (str=prottermnames; *str!=NULL; str++) {
    if ((tc = atomNames.typecode((char *) *str)) >= 0) 
      prottermtypecodes.append(tc);
  }

  // nucleic acids
  for (str=nucnames; *str!=NULL; str++) {
    if ((tc = atomNames.typecode((char *) *str)) >= 0) 
      nuctypecodes.append(tc);
  }
  for (str=nuctermnames; *str!=NULL; str++) {
    if ((tc = atomNames.typecode((char *) *str)) >= 0) 
      nuctermtypecodes.append(tc);
  }

  int numprotcodes = prottypecodes.num();
  int numprottermcodes = prottermtypecodes.num();
  int numnuccodes = nuctypecodes.num();
  int numnuctermcodes = nuctermtypecodes.num();
  
  int *protcodes = &prottypecodes[0];
  int *prottermcodes = &prottermtypecodes[0];
  int *nuccodes = &nuctypecodes[0];
  int *nuctermcodes = &nuctermtypecodes[0];

  // short-circuit protein and nucleic residue analysis if we 
  // don't find any atom names we recognize
  if ((numprotcodes + numprottermcodes + 
       numnuccodes + numnuctermcodes) == 0) {
    // initialize all atom types to non-backbone
    for (i=0; i<nAtoms; i++) {
      MolAtom *a = atom(i);
      a->atomType = ATOMNORMAL;
    }
  } else {
    // loop over all atoms assigning atom backbone type flags
    for (i=0; i<nAtoms; i++) {
      MolAtom *a = atom(i);
 
      // initialize atom type to non-backbone
      a->atomType = ATOMNORMAL;

      // check for protein backbone atom names
      for (j=0; j < numprotcodes; j++) {
        if (a->nameindex == protcodes[j]) {
          a->atomType = ATOMPROTEINBACK;
          break;
        }
      }

      // check terminal residue names as well
      for (j=0; j < numprottermcodes; j++) {
        if (a->nameindex == prottermcodes[j]) { // check if OT1, OT2
          for (k=0; k < a->bonds; k++) {
            if (atom(a->bondTo[k])->atomType == ATOMPROTEINBACK) {
              a->atomType = ATOMPROTEINBACK;
              break;
            }
          }
        }
      }

      // check if in nucleic backbone, if not already set
      if (!(a->atomType)) {
        for (j=0; j < numnuccodes; j++) {
          if (a->nameindex == nuccodes[j]) {
            a->atomType = ATOMNUCLEICBACK;
            break;
          }
        }
      }

      // check if nucleic terminal atom names
      for (j=0; j < numnuctermcodes; j++) {
        if (a->nameindex == nuctermcodes[j]) {
          for (k=0; k < a->bonds; k++) {
            if (atom(a->bondTo[k])->atomType == ATOMNUCLEICBACK) {
              a->atomType = ATOMNUCLEICBACK;
              break;
            }
          }
        }
      }
    }
  }
#else
  int i, j, k;

  // Search for the protein backbone
  int protypes[4];
  protypes[0] = atomNames.typecode((char *) "CA");
  protypes[1] = atomNames.typecode((char *) "C");
  protypes[2] = atomNames.typecode((char *) "O");
  protypes[3] = atomNames.typecode((char *) "N");

  // special case for terminal oxygens that miss the search for O
  // by looking for ones connected to a C
  int termtypes[5];
  termtypes[0] = atomNames.typecode((char *) "OT1"); // standard PDB names
  termtypes[1] = atomNames.typecode((char *) "OT2");
  termtypes[2] = atomNames.typecode((char *) "OXT"); // synonym for OT2
  termtypes[3] = atomNames.typecode((char *) "O1");  // Gromacs force field 
  termtypes[4] = atomNames.typecode((char *) "O2");  // atom names

  // search for the DNA/RNA backbone;  the atom names are:
  // for the phosphate:  P, O1P, O2P, OP1, OP2
  // for the rest: O3', C3', C4', C5', O5'
  // (or O3*, C3*, C4*, C5*, O5*)
  int nuctypes[15];
  nuctypes[ 0] = atomNames.typecode((char *) "P");
  nuctypes[ 1] = atomNames.typecode((char *) "O1P"); // old PDB files
  nuctypes[ 2] = atomNames.typecode((char *) "O2P"); // old PDB files
  nuctypes[ 3] = atomNames.typecode((char *) "OP1"); // new PDB files
  nuctypes[ 4] = atomNames.typecode((char *) "OP2"); // new PDB files
  nuctypes[ 5] = atomNames.typecode((char *) "C3*");
  nuctypes[ 6] = atomNames.typecode((char *) "C3'");
  nuctypes[ 7] = atomNames.typecode((char *) "O3*");
  nuctypes[ 8] = atomNames.typecode((char *) "O3'");
  nuctypes[ 9] = atomNames.typecode((char *) "C4*");
  nuctypes[10] = atomNames.typecode((char *) "C4'");
  nuctypes[11] = atomNames.typecode((char *) "C5*");
  nuctypes[12] = atomNames.typecode((char *) "C5'");
  nuctypes[13] = atomNames.typecode((char *) "O5*");
  nuctypes[14] = atomNames.typecode((char *) "O5'");

#if 0
  // non-backbone nucleic acid atom names
  nuctypes[  ] = atomNames.typecode((char *) "C1*");
  nuctypes[  ] = atomNames.typecode((char *) "C1'");
  nuctypes[  ] = atomNames.typecode((char *) "C2*");
  nuctypes[  ] = atomNames.typecode((char *) "C2'");
  nuctypes[  ] = atomNames.typecode((char *) "O2*");
  nuctypes[  ] = atomNames.typecode((char *) "O2'");
  nuctypes[  ] = atomNames.typecode((char *) "O4*");
  nuctypes[  ] = atomNames.typecode((char *) "O4'");
#endif

  // special case for terminal nucleic residues
  int nuctermtypes[2];
  nuctermtypes[0] = atomNames.typecode((char *) "H5T"); // standard names
  nuctermtypes[1] = atomNames.typecode((char *) "H3T");


  // loop over all atoms assigning atom backbone type flags
  for (i=0; i<nAtoms; i++) {
    MolAtom *a = atom(i);
 
    // initialize atom type to non-backbone
    a->atomType = ATOMNORMAL;

    // check for protein backbone atom names
    for (j=0; j < 4; j++) {
      if (a->nameindex == protypes[j]) {
        a->atomType = ATOMPROTEINBACK;
        break;
      }
    }

    // check terminal residue names as well
    for (j=0; j < 4; j++) {
      if (a->nameindex == termtypes[j]) { // check if OT1, OT2
        for (k=0; k < a->bonds; k++) {
          if (atom(a->bondTo[k])->atomType == ATOMPROTEINBACK) {
            a->atomType = ATOMPROTEINBACK;
            break;
          }
        }
      }
    }
  
    // check if in nucleic backbone, if not already set
    if (!(a->atomType)) {
      for (j=0; j < 15; j++) {
        if (a->nameindex == nuctypes[j]) {
          a->atomType = ATOMNUCLEICBACK;
          break;
        }
      }
    }

    // check if nucleic terminal atom names
    for (j=0; j < 2; j++) {
      if (a->nameindex == nuctermtypes[j]) {
        for (k=0; k < a->bonds; k++) {
          if (atom(a->bondTo[k])->atomType == ATOMNUCLEICBACK) {
            a->atomType = ATOMNUCLEICBACK;
            break;
          }
        }
      }
    }
  }
#endif

  return 0; 
}



// find water molecules based on the residue name
// from the documentation for molscript, these are possible
// waters:
// type H2O HH0 OHH HOH OH2 SOL WAT
// as well, I add TIP, TIP2, TIP3, and TIP4
// The count is the number of sets of connected RESWATERS
int BaseMolecule::find_waters(void) {
#if 1
  const char *watresnames[] = { "H2O", "HHO", "OHH", "HOH", "OH2", 
                                "SOL", "WAT", 
                                "TIP", "TIP2", "TIP3", "TIP4",
                                "SPC", NULL };

  // SPC conflicts with a PDB compound:
  //   http://minerva.roca.csic.es/hicup/SPC/spc_pdb.txt
  //
  // XXX we should add a check to make sure that there are only 
  //     three atoms in the residue when all is said and done. If there
  //     are more, then we should undo the assignment as a water and mark it
  //     as protein etc as appropriate.  This is tricky since at this stage
  //     we haven't done any connectivity tests etc, and we're only working
  //     with individual atoms.  Perhaps its time to re-think this logic.

  ResizeArray<int> watrestypecodes;
  const char ** str;
  int tc, i, j;

  for (str=watresnames; *str!=NULL; str++) {
    if ((tc = resNames.typecode((char *) *str)) >= 0)
      watrestypecodes.append(tc);
  }
  int numwatrescodes = watrestypecodes.num();
  int *watrescodes = &watrestypecodes[0];

  // Short-circuit the water search if no water residue name typecodes
  if (numwatrescodes == 0) {
    return 0;
  } else {
    for (i=0; i<nAtoms; i++) {
      MolAtom *a = atom(i);
      if (a->residueType == RESNOTHING) {  // make sure it isn't named yet
        for (j=0; j<numwatrescodes; j++) {
          if (a->resnameindex == watrescodes[j]) {
            a->residueType = RESWATERS;
            break;
          }
        }
      }
    }
  }
#else
  int i, j;
  int watertypes[12];
  watertypes[0] = resNames.typecode((char *) "H2O");
  watertypes[1] = resNames.typecode((char *) "HH0");
  watertypes[2] = resNames.typecode((char *) "OHH");
  watertypes[3] = resNames.typecode((char *) "HOH");
  watertypes[4] = resNames.typecode((char *) "OH2");
  watertypes[5] = resNames.typecode((char *) "SOL");
  watertypes[6] = resNames.typecode((char *) "WAT");
  watertypes[7] = resNames.typecode((char *) "TIP");
  watertypes[8] = resNames.typecode((char *) "TIP2");
  watertypes[9] = resNames.typecode((char *) "TIP3");
  watertypes[10] = resNames.typecode((char *) "TIP4");

  // this conflicts with a PDB compound:
  //   http://minerva.roca.csic.es/hicup/SPC/spc_pdb.txt
  //
  // XXX we should add a check to make sure that there are only 
  //     three atoms in the residue when all is said and done. If there
  //     are more, then we should undo the assignment as a water and mark it
  //     as protein etc as appropriate.  This is tricky since at this stage
  //     we haven't done any connectivity tests etc, and we're only working
  //     with individual atoms.  Perhaps its time to re-think this logic.
  watertypes[11] = resNames.typecode((char *) "SPC");

  // Short-circuit the water search if no water residue name typecodes
  // exist in the molecule yet.  This is a big performance gain in cases
  // when we're working with large structures containing billions of atoms
  // with no solvent added yet.
  int waterresnameexists=0;
  for (j=0; j<12; j++) {
    if (watertypes[j] != -1) {
      waterresnameexists=1;
    }
  }
  if (!waterresnameexists) {
    return 0;
  }

  for (i=0; i<nAtoms; i++) {
    MolAtom *a = atom(i);
    if (a->residueType == RESNOTHING) {  // make sure it isn't named yet
      for (j=0; j<12; j++) {
        if (watertypes[j] == a->resnameindex) {
          a->residueType = RESWATERS;
          break;
        }
      }
    }
  }
#endif
 
  int count = find_connected_waters2();

  return count;   
}


// if this is a RESWATERS with index idx, mark it and find if
// any of its neighbors are RESWATERS
// this does a depth-first search with RECURSION.
void BaseMolecule::find_connected_waters(int i, char *tmp) {
  MolAtom *a = atom(i);
  int j;
  if (a->residueType == RESWATERS && !tmp[i]) {
    tmp[i] = TRUE;
    for (j=0; j<a->bonds; j++) {
      find_connected_waters(a->bondTo[j], tmp);
    }
  }
}


// if this is a RESWATERS with index idx, mark it and find if
// any of its neighbors are RESWATERS
int BaseMolecule::find_connected_waters2(void) {
  MolAtom *a;
  int count, i;
  IntStackHandle s;

  // allocate cleared aray of tmp flags
  char *tmp = (char *) calloc(1, nAtoms * sizeof(char));

  s = intstack_create(nAtoms);

  for (count=0, i=0; i<nAtoms; i++) {
    if (atom(i)->residueType == RESWATERS && !tmp[i]) {
      int nextatom;

      count++;
      intstack_push(s, i);
    
      // find and mark all connected waters 
      while (!intstack_pop(s, &nextatom)) { 
        int j;

        a = atom(nextatom);
        tmp[nextatom] = TRUE;

        for (j=a->bonds - 1; j>=0; j--) {
          int bi = a->bondTo[j];
          MolAtom *b = atom(bi);
          if (b->residueType == RESWATERS && !tmp[bi])
            intstack_push(s, bi);
        }
      }
    }
  }

  intstack_destroy(s);
  free(tmp);

  return count;
}



// assign a uniq resid (uniq_resid) to each set of connected atoms
// with the same residue id.  There could be many residues with the
// same id, but not connected (the SSN problem - SSNs are not unique
// so don't use them as the primary key)
int BaseMolecule::make_uniq_resids(int *flgs) {
  int i;
  int num_residues = 0;
  IntStackHandle s = intstack_create(nAtoms);

  for (i=0; i<nAtoms; i++) {
    if (!flgs[i]) {  // not been numbered
      // find connected atoms to i with the same resid and label
      // it with the uniq_resid
      MolAtom *a = atom(i);
      int resid = a->resid;
//      char *insertion = a->insertionstr;
      char insertioncode = a->insertionstr[0];

      intstack_push(s, i);
      int nextatom;

      // Loop over all atoms we're bonded to in the same chain/segname
      while (!intstack_pop(s, &nextatom)) {
        MolAtom *a = atom(nextatom);
        a->uniq_resid = num_residues;  // give it the new resid number
        flgs[nextatom] = TRUE;         // mark this atom done
  
        int j;
        for (j=a->bonds - 1; j>=0; j--) {
          int bi = a->bondTo[j];
          if (flgs[bi] == 0) {
            MolAtom *b = atom(bi);
            if (a->chainindex == b->chainindex && 
                a->segnameindex == b->segnameindex &&
                b->resid == resid && b->insertionstr[0] == insertioncode)
//                b->resid == resid && !strcmp(b->insertionstr, insertion))
              intstack_push(s, bi);
          }
        }
      }

      num_residues++;
    }
  }

  intstack_destroy(s);

  return num_residues;
}


// find n backbone atoms connected together with the given residueid
// return the total count
// this assumes that the given atom (atomidx) is correct
int BaseMolecule::find_connected_backbone(IntStackHandle s, int backbone,
                         int atomidx, int residueid, int tmpid, int *flgs) {
  if (flgs[atomidx] != 0)
    return 0; // already done

  MolAtom *x = atom(atomidx);
  if (x->atomType != backbone || x->resid != residueid)
    return 0; // not a backbone atom, or resid doesn't match

  intstack_popall(s); // just in case
  intstack_push(s, atomidx);
  int nextatom;
  int count = 0;
   
  // find and mark connected backbone atoms
  while (!intstack_pop(s, &nextatom)) {
    MolAtom *a = atom(nextatom);
    flgs[nextatom] = tmpid;
    count++;

    int j;
    for (j=a->bonds - 1; j>=0; j--) {
      int bi = a->bondTo[j];
      if (flgs[bi] == 0) {
        MolAtom *b = atom(bi);

        // skip connections to atoms on different chains/segnames
        if (a->chainindex != b->chainindex || 
            a->segnameindex != b->segnameindex)
          continue;

        if (b->atomType == backbone && b->resid == residueid)
          intstack_push(s, bi);
      }
    }
  }

  return count;
}


// the find_connected_backbone left terms of flgs which need to be cleaned up
void BaseMolecule::clean_up_connection(IntStackHandle s, int i, int tmpid, int *flgs) {
  if (flgs[i] != tmpid)  // been here before
    return;

  intstack_popall(s); // just in case
  intstack_push(s, i);
  int nextatom;
 
  // find and null out non-matching atom flags
  while (!intstack_pop(s, &nextatom)) {
    flgs[nextatom] = 0;
    MolAtom *a = atom(nextatom);
    int j;
    for (j=a->bonds - 1; j>=0; j--) {
      int bi = a->bondTo[j];
      if (flgs[bi] == tmpid) {
        intstack_push(s, bi);
      }
    }
  }
}


// Find connected backbone atoms with the same resid
// if found, find all the atoms with the same resid
// which are connected to those backbone atoms only through
// atoms of the same resid
void BaseMolecule::find_and_mark(IntStackHandle s, int n, int backbone,
                                 int restype, int *tmpid, int *flgs) {
  int i;
  int residueid; // the real resid

  intstack_popall(s); // just in case
  for (i=0; i<nAtoms; i++) {
    MolAtom *a = atom(i);   // look for a new backbone atom
    if (a->atomType == backbone && flgs[i] == 0) {
      residueid = a->resid;
      if (find_connected_backbone(s, backbone, i, residueid, *tmpid, flgs) >= n) {
        // if find was successful, start all over again
        clean_up_connection(s, i, *tmpid, flgs);
        // but mark all the Atoms connected to here
        find_connected_atoms_in_resid(s, restype, i, residueid, *tmpid, flgs);
        // and one more was made
        (*tmpid)++;
      } else {
        // clean things up so I won't have problems later
        clean_up_connection(s, i, *tmpid, flgs);
      }
    }
  }
}


// now that I know this resid is okay, mark it so
void BaseMolecule::find_connected_atoms_in_resid(IntStackHandle s,
    int restype, int i, int residueid, int tmpid, int *flgs)
{
  if (flgs[i] != 0 || atom(i)->resid != residueid)
    return;

  intstack_popall(s); // just in case
  intstack_push(s, i);
  int nextatom;

  // find and mark all connected residues in the same chain/segname
  while (!intstack_pop(s, &nextatom)) {
    flgs[nextatom] = tmpid;
    MolAtom *a = atom(nextatom);
    a->residueType = restype;

    int j;
    for (j=a->bonds - 1; j>=0; j--) {
      int bi = a->bondTo[j];
      MolAtom *b = atom(bi);
      if (flgs[bi] == 0 &&
          a->chainindex == b->chainindex &&
          a->segnameindex == b->segnameindex &&
          b->resid == residueid) {
        intstack_push(s, bi);
      }
    }
  }
}


int BaseMolecule::find_residues(void) {
  // flags used for connected atom searches
  // zero cleared at allocation time..
  int *flgs = (int *) calloc(1, nAtoms * sizeof(int)); 
  
  // assign a uniq resid (uniq_resid) to each set of connected atoms
  // with the same residue id.  There could be many residues with the
  // same id, but not connected (the SSN problem - SSNs are not unique
  // so don't use them as the primary key)
  int num_residues = make_uniq_resids(flgs);
   
  int back_res_count = 1; // tmp count of number of residues on some backbone
  memset(flgs, 0, nAtoms * sizeof(int)); // clear flags array

  IntStackHandle s = intstack_create(nAtoms);
  
  //  hunt for the proteins
  // there must be 4 PROTEINBACK atoms connected and with the same resid
  // then all connected atoms will be marked as PROTEIN atoms
  // this gets everything except the terminals
  find_and_mark(s, 4, ATOMPROTEINBACK, RESPROTEIN, &back_res_count, flgs);
  
  // do the same for nucleic acids
  // XXX we might not want to check for the phosphate (P and 2 O's).  Here's
  // the quick way I can almost do that.  Unfortionately, that
  // messes up nfragList, since it needs a P to detect an end
  find_and_mark(s, 4, ATOMNUCLEICBACK, RESNUCLEIC, &back_res_count, flgs);

  intstack_destroy(s);
  
  free(flgs);
  return num_residues;
}

int BaseMolecule::find_atom_in_residue(const char *name, int residue) {
  int nametype = atomNames.typecode(name);
  if (nametype < 0)
    return -2;

  return find_atom_in_residue(nametype, residue);
}


// find which residues are connected to which
// remember, I already have the uniq_id for each atom
void BaseMolecule::find_connected_residues(int num_residues) {
  int i, j;
  residueList.appendN(NULL, num_residues); // init the list to NULLs
 
  for (i=nAtoms-1; i>=0; i--) {      // go through all the atoms
    MolAtom *a = atom(i);
    j = a->uniq_resid;
    if (residueList[j] == NULL) {    // then init the residue
      residueList[j] = new Residue(a->resid, a->residueType);
    }
    // Tell the residue that this atom is in it
    residueList[j]->add_atom(i);
  }


  // finally, check for unusual connections between residues, e.g. between
  // protein and water, or residues that have no atoms.
  int resmissingatomflag=0;

  // if we have more than 10M residues, skip the checks and text warnings
  if (num_residues > 10000000) {
    for (i=0; i<num_residues; i++) {
      Residue *res = residueList[i];

      // double check that everything was created
      if (res == NULL) {
        // no atom was found for this residue
        resmissingatomflag=1;
        res = new Residue((int) -1, RESNOTHING);
        residueList[i] = res;
      } 
    }
  } else {
    for (i=0; i<num_residues; i++) {
      Residue *res = residueList[i];

      // double check that everything was created
      if (res == NULL) {
        // no atom was found for this residue
        resmissingatomflag=1;
        res = new Residue((int) -1, RESNOTHING);
        residueList[i] = res;
      } 

      int bondfromtype = res->residueType;
      int numatoms = res->atoms.num();
      for (j=0; j<numatoms; j++) {
        MolAtom *a = atom(res->atoms[j]);

        // find off-residue bonds to residues of the same chain/segname
        int k;
        for (k=0; k<a->bonds; k++) {
          MolAtom *b = atom(a->bondTo[k]);

          // skip connections to atoms on different chains/segnames
          if (a->chainindex != b->chainindex || 
              a->segnameindex != b->segnameindex)
            continue;
         
          if (b->uniq_resid != i) {
            int bondtotype = residueList[b->uniq_resid]->residueType;
  
            if (bondfromtype != bondtotype) {
              if (i < b->uniq_resid) { // so that we only warn once
                msgWarn << "Unusual bond between residues:  ";
                msgWarn << residueList[i]->resid;
                switch (bondfromtype) {
                  case RESPROTEIN: msgWarn << " (protein)"; break;
                  case RESNUCLEIC: msgWarn << " (nucleic)"; break;
                  case RESWATERS:  msgWarn << " (waters)"; break;
                  default:
                  case RESNOTHING: msgWarn << " (none)"; break;
                }
                msgWarn << " and ";
                msgWarn << residueList[b->uniq_resid]->resid;
                switch (bondtotype) {
                  case RESPROTEIN: msgWarn << " (protein)"; break;
                  case RESNUCLEIC: msgWarn << " (nucleic)"; break;
                  case RESWATERS:  msgWarn << " (waters)"; break;
                  default:
                  case RESNOTHING: msgWarn << " (none)"; break;
                }
                msgWarn << sendmsg;
              }
            }
          }
        }
      }
    }
  }

  // emit any warnings here, only once
  if (resmissingatomflag) {
    msgErr << "Mysterious residue creation in " 
           << "BaseMolecule::find_connected_residues." << sendmsg;
  }
}


// find all the residues connected to a specific residue
int BaseMolecule::find_connected_fragments(void) {
  int i;
  int count = 0;
  // flags are all cleared to zeros initially
  char *flgs = (char *) calloc(1, residueList.num() * sizeof(char));
  IntStackHandle s = intstack_create(residueList.num());

  int atomsg = atomNames.typecode((char *) "SG"); // to find disulfide bonds

  int nextres;
  for (i=0; i<residueList.num(); i++) { // find unmarked fragment
    if (!flgs[i]) {
      fragList.append(new Fragment);
      intstack_push(s, i);

      // find and mark all connected residues with the same chain/segname
      while (!intstack_pop(s, &nextres)) {
        fragList[count]->append(nextres);
        Residue *res = residueList[nextres];
        res->fragment = count; // store residue's fragment

        int numatoms = res->atoms.num();
        int j;
        for (j=0; j<numatoms; j++) {
          MolAtom *a = atom(res->atoms[j]);

          // find all bonds to residues of the same chain/segname 
          int k;
          for (k=0; k<a->bonds; k++) {
            MolAtom *b = atom(a->bondTo[k]);
            int ri = b->uniq_resid;

            // skip connections to residues with different chains/segnames,
            // and don't follow disulfide bonds, as we want the order of
            // residue traversal to be correct so we can use it to build
            // subfragment lists later on
            if ((ri != i) &&
                (flgs[ri] == 0) &&
                (a->chainindex == b->chainindex) &&
                (a->segnameindex == b->segnameindex) &&
                ((a->nameindex != atomsg) || (b->nameindex != atomsg))) {
              flgs[ri] = TRUE;
              intstack_push(s, ri);
            }
          }
        }
      }

      count++;
    }
  }

  intstack_destroy(s);
  free(flgs);

  return count;
}


// find each collection of connected fragments
int BaseMolecule::find_fragments(void) {
  int count = find_connected_fragments();  // find and mark its neighbors

#if 1
  // find the protein subfragments
  find_subfragments(atomNames.typecode((char *) "N"), 
     -1,
     -1,
     atomNames.typecode((char *) "C"), 
     -1,
     -1,
     -1,
     RESPROTEIN, &pfragList);

#if 0
  // find the nucleic acid subfragments
  find_subfragments(atomNames.typecode((char *) "P"), 
     atomNames.typecode((char *) "H5T"),
     -1,
     atomNames.typecode((char *) "O3'"),
     atomNames.typecode((char *) "O3*"),
     atomNames.typecode((char *) "H3T"),
     -1,
     RESNUCLEIC, &nfragList);
#else
  // find the nucleic acid subfragments
  find_subfragments_topologically(
     RESNUCLEIC, &nfragList,
     atomNames.typecode((char *) "O3'"),
     atomNames.typecode((char *) "O3*"),
     atomNames.typecode((char *) "H3T"),
     -1);
#endif
#else
  find_subfragments_cyclic(&pfragList, RESPROTEIN);
  find_subfragments_cyclic(&nfragList, RESNUCLEIC);
#endif

  // determine whether fragments are cyclic or not
  find_cyclic_subfragments(&pfragList, &pfragCyclic);
  find_cyclic_subfragments(&nfragList, &nfragCyclic);

  return count;
}


void BaseMolecule::find_subfragments_cyclic(ResizeArray<Fragment *> *subfragList, int restype) {
  int numfrags = fragList.num();
  int i, frag;

  // test each fragment to see if it's a candidate for the subfraglist
  for (frag=0; frag<numfrags; frag++) {
    int numres = fragList[frag]->num();       // residues in this frag
    int match=1; // start true, and falsify

    // check each residue to see they are all the right restype
    for (i=0; i<numres; i++) {
      int resid = (*fragList[frag])[i];
      if (residueList[resid]->residueType != restype) {
        match=0;
        break;
      }
    }

    // if we found a matching fragment, add it to the subfraglist
    if (match) {
      Fragment *frg = new Fragment;

      // add all of the residues for this fragment to the subfraglist
      for (i=0; i<numres; i++) {
        int resid = (*fragList[frag])[i];
        frg->append(resid);
      } 

      subfragList->append(frg);
    }    
  }
}



void BaseMolecule::find_cyclic_subfragments(ResizeArray<Fragment *> *subfragList, ResizeArray<int> *subfragCyclic) {
  int i, j, frag;
  int numfrags = subfragList->num();

  // check each fragment for cycles
  for (frag=0; frag<numfrags; frag++) {
    int numres   = (*subfragList)[frag]->num();       // residues in this frag

    // skip testing fragments containing zero residues
    if (numres < 1) {
      // record that this fragment is not cyclic
      subfragCyclic->append(0);
      continue;
    }

    int startres = (*(*subfragList)[frag])[0];        // first residue
    int endres   = (*(*subfragList)[frag])[numres-1]; // last residue
    int cyclic   = 0;

    // check for bonds between startres and endres
    int numatoms = residueList[endres]->atoms.num();
    int done = 0;
    for (i=0; (i < numatoms) && (!done); i++) {
      MolAtom *a = atom(residueList[endres]->atoms[i]);
      int nbonds = a->bonds;
      for (j=0; j < nbonds; j++) {
        MolAtom *b = atom(a->bondTo[j]);

        if (b->uniq_resid == startres) {
          cyclic=1;
          done=1;
          break;
        }
      }  
    }

    // record whether this fragment is cyclic or not
    subfragCyclic->append(cyclic);
  }
}


// this adds the current residue type to the *subfragList,
// this finds the residue connected to the endatom atom type
// and calls this function recursively on that residue
// this will NOT work across NORMAL bonds
void BaseMolecule::find_connected_subfragment(int resnum, int fragnum, 
         char *flgs, int endatom,  int altendatom, 
         int alt2endatom, int alt3endatom,
         int restype, 
         ResizeArray<Fragment *> *subfragList)
{
  if (flgs[resnum] || residueList[resnum]->residueType != restype) 
      return;  // been here before, or this is no good
  (*subfragList)[fragnum]->append(resnum);    // add to the list
  flgs[resnum] = TRUE;                        // and prevent repeats

  // find the atom in this residue with the right type
  int i, j, nextres;
  MolAtom *a;
  for (i=residueList[resnum]->atoms.num() - 1; i>=0; i--) {
    a = atom(residueList[resnum]->atoms[i]);
    if (a->nameindex == endatom ||
        a->nameindex == altendatom ||
        a->nameindex == alt2endatom ||
        a->nameindex == alt3endatom) {   // found the end atom
      for (j=a->bonds-1; j>=0; j--) {    // look at the bonds
        // I can't look at if the residue "bond" is a PRO-PRO or NUC-NUC, since
        // that won't tell me if the atom involved is the endatom atom
        // This is important because I need to avoid things like S-S bonds
        // (note that I never checked that the end was bonded to a start on
        //  the next residue! - c'est la vie, or something like that
        if ((!(a->atomType == ATOMNORMAL && atom(a->bondTo[j])->atomType == ATOMNORMAL)) && // not backbone 
            (nextres = atom(a->bondTo[j])->uniq_resid) != resnum &&
            !flgs[nextres] ) { // found next residue, and unvisited
          find_connected_subfragment(nextres, fragnum, flgs, endatom,
              altendatom, alt2endatom, alt3endatom, restype, subfragList);
          return; // only find one; assume no branching
        }
      } // end of for
    } // end of found correct endtype
  } // searching atoms
} // end of finding connected subfragment


// find a class of fragments, and add them to the subfragment list
void BaseMolecule::find_subfragments(int startatom, 
          int altstartatom, int alt2startatom,
          int endatom, int altendatom, int alt2endatom, int alt3endatom,
          int restype, ResizeArray<Fragment *> *subfragList)
{
  // Short-circuit the fragment search if none of search typecodes exist
  if (startatom==-1 && altstartatom==-1 && alt2startatom==-1 && 
      endatom==-1 && altendatom==-1 && alt2endatom==-1 && alt3endatom==-1) {
    return;
  }

  int i, j, k;
  MolAtom *a;
  char *flgs = new char[residueList.num()];
  memset(flgs, 0, residueList.num() * sizeof(char));  // clear flags

  // Loop over all residues looking for candidate residues that start
  // a fragment.  A fragment starting residue must be an unvisited 
  // residue which has an startatom with no off residue bond to 
  // the same restype
  for (i=residueList.num()-1; i>=0; i--) {
    // test for previous visit, and whether it's the restype we want
    if (!flgs[i] && residueList[i]->residueType == restype) {
      // does this residue have a matching startatom
      for (j=residueList[i]->atoms.num()-1; j>=0; j--) { 
        int satom = (a=atom(residueList[i]->atoms[j]))->nameindex;
        if (satom == startatom || 
            satom == altstartatom || 
            satom == alt2startatom){
          for (k=a->bonds-1; k>=0; k--) {
            MolAtom *bondto = atom(a->bondTo[k]);
            // are there any off-residue bonds to the same restype
            if (bondto->uniq_resid != i && bondto->residueType == restype) {
              break; // if so then stop, so that k>=0
            }
          }

          // if we found a valid fragment start atom, find residues downchain
          if (k<0) { 
            subfragList->append(new Fragment);
            find_connected_subfragment(i, subfragList->num()-1, flgs, 
                  endatom, altendatom, alt2endatom, alt3endatom,
                  restype, subfragList);
          } // found starting residue
        } // found startatom
      } // going through atoms
    } // found restype
  } // going through residues

  // found 'em all
  delete [] flgs;
} 


// find a class of fragments, and add them to the subfragment list
void BaseMolecule::find_subfragments_topologically(int restype, 
  ResizeArray<Fragment *> *subfragList, 
  int endatom, int altendatom, int alt2endatom, int alt3endatom) {

  // Short-circuit the fragment search if none of search typecodes exist
  if (endatom==-1 && altendatom==-1 && alt2endatom==-1 && alt3endatom==-1) {
    return;
  }

  int i; 
  char *flgs = new char[residueList.num()];
  memset(flgs, 0, residueList.num() * sizeof(char));  // clear flags
  int numres = residueList.num();

  // Loop over all residues looking for candidate residues that start
  // a fragment.  A fragment starting residue must be an unvisited
  // residue which has an startatom with no off residue bond to
  // the same restype
  for (i=0; i<numres; i++) {
    Residue *res = residueList[i];

    // test for previous visit, and whether it's the restype we want
    if (!flgs[i] && res->residueType == restype) {
      // if this residue only has 1 bond to a residue of the same restype
      // it must be a terminal residue
      int offresbondcount = 0;
      int j, k;
      int numatoms = res->atoms.num();
      for (j=0; j<numatoms; j++) {
        MolAtom *a = atom(res->atoms[j]);

        // find off-residue bonds
        for (k=0; k<a->bonds; k++) {
          MolAtom *b = atom(a->bondTo[k]);
          if (b->uniq_resid != i && 
              residueList[b->uniq_resid]->residueType == restype) {
            offresbondcount++;
          }
        }
      }

      // if we found a valid fragment start atom, find residues downchain
      if (offresbondcount == 1) {
        subfragList->append(new Fragment);
        find_connected_subfragment(i, subfragList->num()-1, flgs,
              endatom, altendatom, alt2endatom, alt3endatom,
              restype, subfragList);
      }
    } // found restype
  } // going through residues

  // found 'em all
  delete [] flgs;
}


#if defined(VMDFASTRIBBONS)
// routine to pre-calculate lists of control point indices to
// be used by the ribbon/cartoon representations
void BaseMolecule::calculate_ribbon_controlpoints() {
  int onum, canum, frag, num, res;

  wkf_timerhandle tm = wkf_timer_create();

  msgInfo << "XXX Building pre-computed control point lists" << sendmsg;

  wkf_timer_start(tm);

  // Lookup atom typecodes ahead of time so we can use the most
  // efficient variant of find_atom_in_residue() in the main loop.
  // If we can't find the atom types we need, bail out immediately
  int CAtypecode  = atomNames.typecode("CA");
  int Otypecode   = atomNames.typecode("O");
  int OT1typecode = atomNames.typecode("OT1");
   
  // We can't draw a ribbon without CA and O atoms for guidance!
  if (CAtypecode < 0) {
    msgErr << "This structure has no identifiable CA atoms,"
           << "ribbon and cartoon representations will not be usable." 
           << sendmsg;
    return;
  }
  if ((Otypecode < 0) && (OT1typecode < 0)) {
    msgErr << "This structure has no identifiable Oxygen atoms,"
           << "ribbon and cartoon representations will not be usable." 
           << sendmsg;
    return;
  }

  //
  // calculate control point lists for the protein fragments
  //

  msgInfo << "XXX Building protein control point lists" << sendmsg;

  // go through each protein and find the CA and O atoms which are
  // eventually used to construct control points and perpendicular vectors
  ResizeArray<int> cpind;
  for (frag=0; frag<pfragList.num(); frag++) {
    num = pfragList[frag]->num();  // number of residues in this fragment
    if (num < 2) {
      // with less than two residues, we can't do anything useful, so skip
      nfragCPList.append(NULL);
      continue;
    }

    // check that we have a valid structure before continuing
      res = (*pfragList[frag])[0];
    canum = find_atom_in_residue(CAtypecode, res);
     onum = find_atom_in_residue(Otypecode, res);

    if (onum < 0 && OT1typecode >= 0) {
      onum = find_atom_in_residue(OT1typecode, res);
    }
    if (canum < 0 || onum < 0) {
      // can't find 1st CA or O of the protein fragment, so skip
      msgErr << "Fragment control point calc unable to identify target atoms" << sendmsg;
      nfragCPList.append(NULL);
      continue; 
    }

    // clear temp arrays before we begin
    cpind.clear();

    int loop;
    for (loop=0; loop<num; loop++) {
      res = (*pfragList[frag])[loop];

      // find next CA atom to be used as a spline control point
      canum = find_atom_in_residue(CAtypecode, res);

      // find next O atom to be used for spline orientation vector
      onum = find_atom_in_residue(Otypecode, res);
      if (onum < 0 && OT1typecode >= 0) {
        onum = find_atom_in_residue(OT1typecode, res);
      }

      // add the atom indices to the control point list
      cpind.append(canum);
      cpind.append(onum);
    }

    // copy this fragment's control point lists to permanent storage
    if (cpind.num() == 2L*num) {
      int *cpindices = new int[cpind.num()];
      memcpy(cpindices, &cpind[0], long(cpind.num()) * sizeof(int));

      // add control point list to the master list
      nfragCPList.append(cpindices);
    } else {
      msgErr << "Unexpected failure in control point calculation" << sendmsg;
      nfragCPList.append(NULL);
    }
  } 

  wkf_timer_stop(tm);

  msgInfo << "XXX Calc time for protein cplist: " 
          << wkf_timer_time(tm) << sendmsg;
  msgInfo << "XXX done building protein control point lists" << sendmsg;

  //
  // calculate control point lists for the nucleic fragments
  //

  wkf_timer_destroy(tm);
}
#endif

#if defined(VMDWITHCARBS)

// Routines for detecting carbohydrate rings
// contributed by Simon Cross and Michelle Kuttel
// XXX TODO:
//   don't create hashes as pointers, rather just pass pointers for them
//   into functions which need it.

// find all small rings and links between them
void BaseMolecule::find_small_rings_and_links(int maxpathlength, int maxringsize) {
  // skip ring finding if we've already done it
  if (maxpathlength == currentMaxPathLength && maxringsize == currentMaxRingSize)
    return;
  currentMaxPathLength = maxpathlength;
  currentMaxRingSize = maxringsize;

  // Clear smallringList, smallringLinkages
  smallringList.clear();
  smallringLinkages.clear();

  // find groups of atoms bonded into small rings  
  find_small_rings(maxringsize);
#if 0
  msgInfo << "   Rings: " << smallringList.num() << sendmsg;
#endif
  
  // orientate the rings if possible
  orientate_small_rings(maxringsize);
  int nOrientatedRings = 0;
  for (int i=0; i<smallringList.num(); i++) {
    if (smallringList[i]->orientated) 
      nOrientatedRings++;
  }
#if 0
  msgInfo << "   Rings orientated: " << nOrientatedRings << sendmsg;
#endif  
  // find paths between rings
  find_orientated_small_ring_linkages(maxpathlength, maxringsize);
#if 0
  msgInfo << "   Ring Paths: " << smallringLinkages.paths.num() << sendmsg;
#endif  
}

// find all loops less than a given size
int BaseMolecule::find_small_rings(int maxringsize) {
  int n_back_edges, n_rings;
  ResizeArray<int> back_edge_src, back_edge_dest;
    
  n_back_edges = find_back_edges(back_edge_src, back_edge_dest);
  n_rings = find_small_rings_from_back_edges(maxringsize, back_edge_src, back_edge_dest);

#if 1
  if (getenv("VMDFINDSMALLRINGSDEBUG") != NULL) {
    int i;

    msgInfo << "  BACK EDGES: " << n_back_edges << sendmsg;
    for (i=0; i < n_back_edges; i++) {
      msgInfo << "       SRC:" << back_edge_src[i] << ", DST:" << back_edge_dest[i] << sendmsg;
    }

    msgInfo << " SMALL RINGS: " << n_rings << sendmsg;
    for (i=0; i < n_rings; i++) {
      msgInfo << "    RING: " << *(smallringList[i]) << sendmsg;
    }
  }
#endif

  return n_rings; // number of rings found
}


#define INTREE_NOT      -1   // not in tree
#define INTREE_NOPARENT -2   // no parent 

// find the back edges of an arbitrary spanning tree (or set of trees if the molecule is disconnected)
int BaseMolecule::find_back_edges(ResizeArray<int> &back_edge_src, ResizeArray<int> &back_edge_dest) {
  int i;
  int n_back_edges = 0;  
  int *intree_parents = new int[nAtoms];
  memset(intree_parents, INTREE_NOT, nAtoms * sizeof(int));  // clear parents, -1 = not in tree, -2 = no parent
      
  for (i=0; i<nAtoms; i++) {
    if (intree_parents[i] == INTREE_NOT) {  // not been visited
      n_back_edges += find_connected_subgraph_back_edges(i,back_edge_src,back_edge_dest,intree_parents);
    }
  }

  delete [] intree_parents;
 
  return n_back_edges;
}


// find the back edges of a spanning tree (for a connected portion of the molecule)
int BaseMolecule::find_connected_subgraph_back_edges(int atomid, ResizeArray<int> &back_edge_src, ResizeArray<int> &back_edge_dest,
                                                     int *intree_parents) {
  int i, n_new_back_edges, cur_atom_id, child_atom_id, parent_atom_id;
  MolAtom *curatom;
  ResizeArray<int> node_stack;
  
  node_stack.append(atomid);
  intree_parents[atomid] = INTREE_NOPARENT; // in tree, but doesn't have parent
  n_new_back_edges = 0;

  while (node_stack.num() > 0) {
#if 1
    cur_atom_id = node_stack.pop();
#else
    cur_atom_id = node_stack[node_stack.num()-1];
    node_stack.remove(node_stack.num()-1);
#endif
    
    curatom = atom(cur_atom_id);
    parent_atom_id = intree_parents[cur_atom_id];
    
    for(i=0;i<curatom->bonds;i++) {
      child_atom_id = curatom->bondTo[i];
      if (intree_parents[child_atom_id] != INTREE_NOT) {
        // back-edge found
        if ((child_atom_id != parent_atom_id) && (child_atom_id > cur_atom_id)) {
            // we ignore edges back to the parent
            // and only add each back edge once
            // (it'll crop up twice since each bond is listed on both atoms)
            back_edge_src.append(cur_atom_id);
            back_edge_dest.append(child_atom_id);
            n_new_back_edges++;
        }
      } else {
        // extended tree
        intree_parents[child_atom_id] = cur_atom_id;
        node_stack.append(child_atom_id);
      }
    }
  }

  return n_new_back_edges;
}

// find rings smaller than maxringsize given list of back edges
int BaseMolecule::find_small_rings_from_back_edges(int maxringsize, ResizeArray<int> &back_edge_src, ResizeArray<int> &back_edge_dest) {
    int i, key, max_rings;
    int n_rings = 0;
    int n_back_edges = back_edge_src.num();
    SmallRing *ring;
    inthash_t *used_edges = new inthash_t; // back edges which have been dealt with 
    inthash_t *used_atoms = new inthash_t; // atoms (other than the first) which are used in the current path (i.e. possible loop)
    inthash_init(used_edges,n_back_edges);
    inthash_init(used_atoms,maxringsize);
   
    // cap the peak number of rings to find based on the size of the
    // input structure.  This should help prevent unusual structures 
    // with very high connectivity, such as silicon nanodevices from 
    // blowing up the ring search code
    max_rings = 2000 + (int) (100.0*sqrt((double) nAtoms));
 
    for(i=0;i<n_back_edges;i++) {
      ring = new SmallRing();
      ring->append(back_edge_src[i]);
      ring->append(back_edge_dest[i]);
      
      // first atom is not marked used, since we're allowed to re-use it.
      inthash_insert(used_atoms,back_edge_dest[i],1);
      
      n_rings += find_small_rings_from_partial(ring,maxringsize,used_edges,used_atoms);
      delete ring;
      
      // abort if there are too many rings
      if (n_rings > max_rings) {
        msgWarn << "Maximum number of rings (" << max_rings << ") exceed."
                << " Stopped looking for rings after " << n_rings << " rings found." << sendmsg;
        break;
      }
      
      key = get_edge_key(back_edge_src[i],back_edge_dest[i]);
      inthash_insert(used_edges,key,1);
      
      // remove last atom from used_atoms
      inthash_delete(used_atoms,back_edge_dest[i]);
    }
    
    inthash_destroy(used_edges);
    delete used_edges;
    inthash_destroy(used_atoms);
    delete used_atoms;

    return n_rings;
}


// find rings smaller than maxringsize from the given partial ring (don't reuse used_edges)
int BaseMolecule::find_small_rings_from_partial(SmallRing *ring, int maxringsize, inthash_t *used_edges, inthash_t *used_atoms) {
    int i, next_bond_pos, cur_atom_id, child_atom_id, bond_key, barred, closes, do_pop;
    int n_rings = 0;
    MolAtom *curatom;

    IntStackHandle atom_id_stack = intstack_create(maxringsize);
    IntStackHandle bond_pos_stack = intstack_create(maxringsize);
    intstack_push(atom_id_stack,ring->last_atom());
    intstack_push(bond_pos_stack,0);

    while (!intstack_pop(atom_id_stack,&cur_atom_id)) {
      intstack_pop(bond_pos_stack,&next_bond_pos);
      curatom = atom(cur_atom_id);
      do_pop = 0; // whether we need to pop the current atom

      // if the ring is already maxringsize, just pop the current atom off.
      if (ring->num() > maxringsize)
         do_pop = 1;

      if (next_bond_pos == 0 && !do_pop) {
        // ignore barred rings (by checking for links back to earlier parts of the ring
        // before exploring further)
        barred = closes = 0;
        for(i=0;i<curatom->bonds;i++) {
          child_atom_id = curatom->bondTo[i];

          // check that this is not an edge immediately back to the previous atom
          if (child_atom_id == ring->atoms[ring->atoms.num()-2]) continue;

          // check that this is not an atom we've included
          // (an exception is the first atom, which we're allowed to try add, obviously :)
          if (inthash_lookup(used_atoms, child_atom_id) != HASH_FAIL) {
            barred = 1;
            continue;
          }

          // check is this not a back edge which has already been used
          bond_key = get_edge_key(cur_atom_id,child_atom_id);
          if (inthash_lookup(used_edges, bond_key) != HASH_FAIL) {
            if (child_atom_id == ring->first_atom())
              barred = 1; // used back-edge which closes the ring counts as barred
            continue;
          }

          // check whether ring closes
          if (child_atom_id == ring->first_atom()) {
            closes = 1;
            continue;
          }
        }

        if (closes && !barred) {
          smallringList.append(ring->copy());
          n_rings += 1;
        }
        if (closes || barred) {
          // adding this atom would create a barred ring, so skip to next atom on stack
          do_pop = 1;
        }
      }

      if (!do_pop) {
        for(i=next_bond_pos;i<curatom->bonds;i++) {
          child_atom_id = curatom->bondTo[i];

          // check that this is not an edge immediately back to the previous atom
          if (child_atom_id == ring->atoms[ring->atoms.num()-2]) continue;
                  
          // check is this not a back edge which has already been used
          bond_key = get_edge_key(cur_atom_id,child_atom_id);
          if (inthash_lookup(used_edges, bond_key) != HASH_FAIL) continue;
          
          // Append child atom and go deeper
          ring->append(child_atom_id);
          inthash_insert(used_atoms,child_atom_id,1);
          
          // push current state and new state onto stack and recurse
          intstack_push(atom_id_stack,cur_atom_id);
          intstack_push(bond_pos_stack,i+1);
          intstack_push(atom_id_stack,child_atom_id);
          intstack_push(bond_pos_stack,0);
          break;     
        }

        // we finished the children, pop one parent
        if(i>=curatom->bonds)
            do_pop = 1;        
      }

      if (do_pop && !intstack_empty(atom_id_stack)) {
          // finished with cur_atom_id and all its children      
          // clean up before returning from recurse
          ring->remove_last();
          inthash_delete(used_atoms,cur_atom_id);      
      }
    }
    
    intstack_destroy(atom_id_stack);
    intstack_destroy(bond_pos_stack);
    return n_rings;
}


// construct edge key
int BaseMolecule::get_edge_key(int edge_src, int edge_dest) {
    int t;
    if (edge_dest > edge_src) {
       t = edge_src;
       edge_src = edge_dest;
       edge_dest = t;
    }
    return edge_src * nAtoms + edge_dest;
}

// Routines for orientating rings
// contributed by Simon Cross and Michelle Kuttel

void BaseMolecule::orientate_small_rings(int maxringsize) {
  for (int i=0;i<smallringList.num();i++)
    orientate_small_ring(*smallringList[i],maxringsize);
}

void BaseMolecule::orientate_small_ring(SmallRing &ring,int maxringsize) {
  short int oxygen = -1;
  int oxygen_typeindex, c1_nameindex, c1p_nameindex, cu1_nameindex;
  int c2_nameindex, c2p_nameindex, cu2_nameindex;
  MolAtom *curatom, *atom_before_O, *atom_after_O;

  // lookup atom name indices in advance
  oxygen_typeindex = atomTypes.typecode("O");
  c1_nameindex = atomNames.typecode("C1");
  c1p_nameindex = atomNames.typecode("C1'");
  cu1_nameindex = atomNames.typecode("C_1");
  c2_nameindex = atomNames.typecode("C2");
  c2p_nameindex = atomNames.typecode("C2'");
  cu2_nameindex = atomNames.typecode("C_2");
    
  // Find an oxygen (or something with two bonds)
  for (int i=0; i<ring.num(); i++) {
    curatom = atom(ring[i]);
    if (curatom->atomicnumber == 8 || curatom->typeindex == oxygen_typeindex || curatom->bonds == 2) {
      oxygen = i;
      break;
    }
  }

  // leave ring unorientated if no oxygen found
  if (oxygen == -1) return;

  // find atoms before and after oxygen (taking into account wrapping)
  if (oxygen == 0) atom_before_O = atom(ring.last_atom());
  else atom_before_O = atom(ring[oxygen-1]);

  if (oxygen == ring.num()-1) atom_after_O = atom(ring.first_atom());
  else atom_after_O = atom(ring[oxygen+1]);
    
  // ensure C1 carbon is after the oxygen
  // leave unorientated if the C1 carbon can't be found
  if (atom_before_O->nameindex == c1_nameindex || atom_before_O->nameindex == c1p_nameindex
      || atom_before_O->nameindex == cu1_nameindex
      || atom_before_O->nameindex == c2_nameindex || atom_before_O->nameindex == c2p_nameindex
      || atom_before_O->nameindex == cu2_nameindex) {
    ring.reverse();
    ring.orientated = 1;
  }
  else if (atom_after_O->nameindex == c1_nameindex || atom_after_O->nameindex == c1p_nameindex
           || atom_after_O->nameindex == cu1_nameindex
           || atom_after_O->nameindex == c2_nameindex || atom_after_O->nameindex == c2p_nameindex
           || atom_after_O->nameindex == cu2_nameindex) {
    ring.orientated = 1;
  }
}


// Routines for finding links between rings
// contributed by Simon Cross and Michelle Kuttel

// find all paths shorter than a given length between the small rings
// listed in smallringList
int BaseMolecule::find_orientated_small_ring_linkages(int maxpathlength, int maxringsize) {
  SmallRing *sr;
  LinkagePath *lp;
  int i, j, atom_id;
  int n_paths = 0;
  inthash_t *atom_to_ring = new inthash_t;
  inthash_t *multi_ring_atoms = new inthash_t;
  inthash_t *used_atoms = new inthash_t;
  inthash_init(atom_to_ring,smallringList.num()*maxringsize/2);
  inthash_init(multi_ring_atoms,smallringList.num());
  inthash_init(used_atoms,maxpathlength);
   
  // create lookup from atom id to ring id
  for (i=0;i<smallringList.num();i++) {
    sr = smallringList[i];
    // XXX: Uncomment this if we want to include non-orientated rings
    //      in linkage paths
    // if (!sr->orientated) continue;
    for (j=0;j<sr->num();j++) {
      atom_id = (*sr)[j];
      if (inthash_lookup(atom_to_ring,atom_id) == HASH_FAIL)
        inthash_insert(atom_to_ring,atom_id,i);
      else
        inthash_insert(multi_ring_atoms,atom_id,1);
    }
  }

  // look for ring linkage paths starting from each atom
  // of each orientated ring
  for (i=0;i<smallringList.num();i++) {
    sr = smallringList[i];
    if (!sr->orientated) continue;
    for (j=0;j<sr->num();j++) {
      if (inthash_lookup(multi_ring_atoms,(*sr)[j]) != HASH_FAIL) continue;
      lp = new LinkagePath();
      lp->start_ring = i;
      lp->path.append((*sr)[j]);    
      n_paths += find_linkages_for_ring_from_partial(*lp,maxpathlength,atom_to_ring,multi_ring_atoms,used_atoms);
      delete lp;
    }
  }
  
  inthash_destroy(atom_to_ring);
  delete atom_to_ring;
  inthash_destroy(multi_ring_atoms);
  delete multi_ring_atoms;
  inthash_destroy(used_atoms);
  delete used_atoms;

#if 0
  msgInfo << smallringLinkages << sendmsg;
#endif
  
  return n_paths; // number of paths found.
}

int BaseMolecule::find_linkages_for_ring_from_partial(LinkagePath &lp, int maxpathlength, inthash_t *atom_to_ring, inthash_t *multi_ring_atoms, inthash_t *used_atoms) {
  int i, cur_atom_id, next_bond_pos, child_atom_id, ringidx;
  int n_paths = 0;
  MolAtom *curatom;

  IntStackHandle atom_id_stack = intstack_create(maxpathlength);
  IntStackHandle bond_pos_stack = intstack_create(maxpathlength);
  intstack_push(atom_id_stack,lp.path.last_atom());
  intstack_push(bond_pos_stack,0);

  while (!intstack_pop(atom_id_stack,&cur_atom_id)) {
    intstack_pop(bond_pos_stack,&next_bond_pos);
    curatom = atom(cur_atom_id);

    if (next_bond_pos == 0)
      inthash_insert(used_atoms, cur_atom_id, 1);

    for(i=next_bond_pos;i<curatom->bonds;i++) {
      child_atom_id = curatom->bondTo[i];

      // check that this isn't an atom that belongs to multiple rings
      if (inthash_lookup(multi_ring_atoms,child_atom_id) != HASH_FAIL) continue;

      // check that this is not an edge immediately back to the previous atom
      // (when there is only one atom in the path, it can't be a link back)
      if (lp.num() > 1 && child_atom_id == lp[lp.num()-2]) continue;
      
      // check that we haven't arrived at a non-orientated ring
      ringidx = inthash_lookup(atom_to_ring, child_atom_id);
      if (ringidx != HASH_FAIL && !smallringList[ringidx]->orientated) continue;

      // only store paths from smaller ringidx to larger ringidx (to avoid getting a copy of each orientation of the path)
      // ignore paths which return to the same ring
      // check that we're leaving the starting ring
      if (ringidx != HASH_FAIL && ringidx <= lp.start_ring) continue;
      
      // see if this takes us to another ring
      if (ringidx != HASH_FAIL && (ringidx > lp.start_ring)) {
          lp.append(child_atom_id);
          lp.end_ring = ringidx;
          smallringLinkages.addLinkagePath(*lp.copy());
          n_paths++;
          lp.end_ring = -1;
          lp.remove_last();
          continue;
      }
      
      // check that this is not an atom we've included
      // (an exception is the first atom, which we're allowed to try add, obviously :)
      if (inthash_lookup(used_atoms, child_atom_id) != HASH_FAIL) continue;
              
      if (lp.num() < maxpathlength) {
         lp.append(child_atom_id);

         // push current state and new state onto stack and recurse
         intstack_push(atom_id_stack,cur_atom_id);
         intstack_push(bond_pos_stack,i+1);
         intstack_push(atom_id_stack,child_atom_id);
         intstack_push(bond_pos_stack,0);
         break;
      }
    }

    if ((i>=curatom->bonds)&&(!intstack_empty(atom_id_stack))) {      
      // clean up before returning from recurse
      lp.remove_last();
      inthash_delete(used_atoms,cur_atom_id);
    }
  }
  
  intstack_destroy(atom_id_stack);
  intstack_destroy(bond_pos_stack);
  return n_paths;
}

#endif  // end of carbohydrate related stuff



void BaseMolecule::add_volume_data(const char *name, const float *o,
    const float *xa, const float *ya, const float *za, int x, int y, int z,
    float *scalar, float *grad, float *variance) {
  msgInfo << "Analyzing Volume..." << sendmsg;

  VolumetricData *vdata = new VolumetricData(name, o, xa, ya, za,
                                             x, y, z, scalar);
  
  // Print out grid size along with memory use for the grid itself plus
  // the memory required for the volume gradients (4x the scalar grid memory)
  // Color texture maps require another 0.75x the original scalar grid size.
  msgInfo << "   Grid size: " << x << "x" << y << "x" << z << "  (" 
          << (int) (4.0 * (vdata->gridsize() * sizeof(float)) / (1024.0 * 1024.0)) << " MB)" 
          << sendmsg;

  msgInfo << "   Total voxels: " << vdata->gridsize() << sendmsg;

  float datamin, datamax;
  vdata->datarange(datamin, datamax);
#if 1
  char minstr[1024];
  char maxstr[1024];
  char rangestr[1024];
  sprintf(minstr, "%g", datamin);
  sprintf(maxstr, "%g", datamax);
  sprintf(rangestr, "%g", (datamax - datamin));
  msgInfo << "   Min: " << minstr << "  Max: " << maxstr 
          << "  Range: " << rangestr << sendmsg;
#else
  msgInfo << "   Min: " << datamin << "  Max: " << datamax 
          << "  Range: " << (datamax - datamin) << sendmsg;
#endif

  if (grad) {
    msgInfo << "   Assigning volume gradient and normal map" << sendmsg;
    vdata->set_volume_gradient(grad); // set gradients for smooth normals
  } else {
    msgInfo << "   Computing volume gradient map for smooth shading" << sendmsg;
    vdata->compute_volume_gradient(); // calc gradients for smooth normals
  }

  volumeList.append(vdata);

  msgInfo << "Added volume data, name=" << vdata->name << sendmsg;
}


void BaseMolecule::add_volume_data(const char *name, const double *o,
    const double *xa, const double *ya, const double *za, int x, int y, int z,
    float *scalar) {
  msgInfo << "Analyzing Volume..." << sendmsg;

  VolumetricData *vdata = new VolumetricData(name, o, xa, ya, za,
                                             x, y, z, scalar);
  
  // Print out grid size along with memory use for the grid itself plus
  // the memory required for the volume gradients (4x the scalar grid memory)
  // Color texture maps require another 0.75x the original scalar grid size.
  msgInfo << "   Grid size: " << x << "x" << y << "x" << z << "  (" 
          << (int) (4.0 * (vdata->gridsize() * sizeof(float)) / (1024.0 * 1024.0)) << " MB)" 
          << sendmsg;

  msgInfo << "   Total voxels: " << vdata->gridsize() << sendmsg;

  float datamin, datamax;
  vdata->datarange(datamin, datamax);
  msgInfo << "   Min: " << datamin << "  Max: " << datamax 
          << "  Range: " << (datamax - datamin) << sendmsg;

  msgInfo << "   Computing volume gradient map for smooth shading" << sendmsg;
  vdata->compute_volume_gradient(); // calc gradients for smooth vertex normals

  volumeList.append(vdata);

  msgInfo << "Added volume data, name=" << vdata->name << sendmsg;
}


void BaseMolecule::remove_volume_data(int idx) { 
  if (idx >= 0 && idx < volumeList.num()) {
    delete volumeList[idx];    
    volumeList.remove(idx);
  }
}


int BaseMolecule::num_volume_data() {
  return volumeList.num();
}

VolumetricData *BaseMolecule::modify_volume_data(int id) {
  if (id >= 0 && id < volumeList.num())
    return volumeList[id];
  return NULL;
}

const VolumetricData *BaseMolecule::get_volume_data(int id) {
  return modify_volume_data(id);
}

