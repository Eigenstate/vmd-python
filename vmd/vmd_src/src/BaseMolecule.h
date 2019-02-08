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
 *	$RCSfile: BaseMolecule.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.148 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Base class for all molecules, without display-specific information.  This
 * portion of a molecule contains the structural data, and all routines to
 * find the structure (backbone, residues, etc).  It contains the
 * animation list as well.
 *
 ***************************************************************************/
#ifndef BASEMOLECULE_H
#define BASEMOLECULE_H

#ifndef NAMELIST_TEMPLATE_H
#include "NameList.h"
#endif
#ifndef RESIZEARRAY_TEMPLATE_H
#include "ResizeArray.h"
#endif
#include "Atom.h"
#include "Residue.h"
#include "Timestep.h"
#include "Fragment.h"
#include "Matrix4.h"
#include "intstack.h"

#ifdef VMDWITHCARBS
#include "SmallRing.h"
#include "SmallRingLinkages.h"
#include "inthash.h"
#endif

#if 0
#define VMDFASTRIBBONS
#endif

class VolumetricData;
class QMData;

/// Base class for all molecules, without display-specific information.  This
/// portion of a molecule contains the structural data, and all routines to
/// find the structure (backbone, residues, etc).  It contains the
/// animation list as well.
class BaseMolecule {
public:
  //
  // public molecular structure data (for ease of access):
  //
  int nAtoms;                         ///< number of atoms
  int nResidues;                      ///< number of residues
  int nWaters;                        ///< number of waters
  int nSegments;                      ///< number of segments
  int nFragments;                     ///< total number of fragments
  int nProteinFragments;              ///< number of protein fragments
  int nNucleicFragments;              ///< number of nucleic fragments

  NameList<int> atomNames;            ///< list of unique atom names
  NameList<int> atomTypes;            ///< list of unique atom types
  NameList<int> resNames;             ///< list of unique residue names
  NameList<int> chainNames;           ///< list of unique chain names
  NameList<int> segNames;             ///< list of unique segment names
  NameList<int> altlocNames;          ///< list of alternate location names

  ResizeArray<Residue *> residueList; ///< residue connectivity list
  ResizeArray<Fragment *> fragList;   ///< list of connected residues

  ResizeArray<Fragment *> pfragList;  ///< list of connected protein residues
  				      ///< this is a single chain from N to C
  ResizeArray<int> pfragCyclic;       ///< flag indicating cyclic fragment
#if defined(VMDFASTRIBBONS)
  ResizeArray<int *> pfragCPList;     ///< pre-computed control point lists
                                      ///< for ribbon/cartoon rendering
#endif


  ResizeArray<Fragment *> nfragList;  ///< ditto for nuc; from 5' to 3'
  ResizeArray<int> nfragCyclic;       ///< flag indicating cyclic fragment
#if defined(VMDFASTRIBBONS)
  ResizeArray<int *> nfragCPList;     ///< pre-computed control point lists
                                      ///< for ribbon/cartoon rendering
#endif

#ifdef VMDWITHCARBS
  ResizeArray<SmallRing *> smallringList; ///< list of small rings
                      ///< each ring is a single orientated chain
  SmallRingLinkages smallringLinkages; ///< paths joining small rings
  int currentMaxRingSize;             ///< limit on size of small rings
  int currentMaxPathLength;           ///< limit on length of paths joining rings
#endif

  // Molecule instance transformation matrices, not including self
  ResizeArray<Matrix4> instances;     ///< list of instance transform matrices

  // Extra molecular dynamics structure info not normally needed for
  // anything but structure building tasks
  ResizeArray<int> angles;            ///< list of angles
  ResizeArray<int> dihedrals;         ///< list of dihedrals
  ResizeArray<int> impropers;         ///< list of impropers
  ResizeArray<int> cterms;            ///< list of cross-terms 

  // Extra (optional) data for this molecule.
  // If the name is unassigned then the values can be taken to be zero.
  NameList<float *> extraflt;         ///< optional floating point data
  NameList<int *> extraint;           ///< optional integer data
  NameList<unsigned char *> extraflg; ///< optional bitwise flags

  // more structure topology handling info. assign names to FF types.
  ResizeArray<int> angleTypes;        ///< type of angle
  ResizeArray<int> dihedralTypes;     ///< type of dihedral
  ResizeArray<int> improperTypes;     ///< type of improper
  NameList<int> bondTypeNames;        ///< list of unique bond type names
  NameList<int> angleTypeNames;       ///< list of unique angle type names
  NameList<int> dihedralTypeNames;    ///< list of unique dihedral type names
  NameList<int> improperTypeNames;    ///< list of unique improper type names

  // Data related to quantum mechanics simulations.
  QMData *qm_data;

  // Interface to standard extra data.  These will exist and be initialized
  // to zero when init_atoms is called.

  /// flags indicating what data was loaded or user-specified vs. guessed
  enum dataset_flag {       
               NODATA=0x0000,
            INSERTION=0x0001, 
            OCCUPANCY=0x0002, 
              BFACTOR=0x0004, 
                 MASS=0x0008, 
               CHARGE=0x0010, 
               RADIUS=0x0020, 
               ALTLOC=0x0040, 
         ATOMICNUMBER=0x0080, 
                BONDS=0x0100, 
           BONDORDERS=0x0200,
               ANGLES=0x0400,
               CTERMS=0x0800, 
            BONDTYPES=0x1000,
           ANGLETYPES=0x2000,
            ATOMFLAGS=0x4000 };
  
  int datasetflags;
  void set_dataset_flag(int flag) { 
    datasetflags |= flag;
  }
  void unset_dataset_flag(int flag) {
    datasetflags &= (~flag);
  }
  int test_dataset_flag(int flag) {
    return (datasetflags & flag) != 0;
  }

  /// various standard per-atom fields
  float *radius() { return extraflt.data("radius"); }
  float *mass() { return extraflt.data("mass");   }
  float *charge() { return extraflt.data("charge"); }
  float *beta() { return extraflt.data("beta");   }
  float *occupancy() { return extraflt.data("occupancy"); }

  // per-atom flags array
  unsigned char *flags() { return extraflg.data("flags"); }

  /// methods for querying the min/max atom radii, used by the
  /// the QuickSurf representation
  int radii_minmax_need_update;
  float radii_min;
  float radii_max;
  void set_radii_changed() { radii_minmax_need_update = 1; }

  void get_radii_minmax(float &min, float &max) {
    const float *r = NULL;
    if (radii_minmax_need_update && nAtoms > 0 &&
        (r = extraflt.data("radius")) != NULL) {

#if 1
      // use fast 16-byte-aligned min/max routine
      minmax_1fv_aligned(r, nAtoms, &radii_min, &radii_max);
#else
      // scalar min/max loop
      radii_min = r[0];
      radii_max = r[0];
      for (int i=0; i<nAtoms; i++) {
        if (r[i] < radii_min) radii_min = r[i];
        if (r[i] > radii_max) radii_max = r[i];
      }
#endif
      radii_minmax_need_update = 0;
    }

    min = radii_min;  
    max = radii_max;  
  }

  /// number of electron pairs, also fractional
  float *bondorders() { return extraflt.data("bondorders"); }
  void setbondorder(int atom, int bond, float order);
  float getbondorder(int atom, int bond);

  /// per atom/bond bond type list.
  int *bondtypes() { return extraint.data("bondtypes"); }

  /// set new bond type
  void setbondtype(int atom, int bond, int type);

  /// quest bond type
  int  getbondtype(int atom, int bond);

  /// has structure information already been loaded for this molecule?
  int has_structure() const { return cur_atom > 0; }

  /// clear the entire bond list for all atoms
  void clear_bonds(void);

  /// return the number of unique bonds in the molecule 
  int count_bonds(void);

#ifdef VMDWITHCARBS
  /// locate small rings and paths between them.
  void find_small_rings_and_links(int maxpathlength, int maxringsize);
#endif

private:
  const int ID;          ///< unique mol. ID number

  //
  // molecular structure data:
  //
  int cur_atom;          ///< index of next atom added
  MolAtom *atomList;     ///< atom data
  int lastbonderratomid; ///< last atom id that generated a bonding error msg
  int bonderrorcount;    ///< number of bonds-exceeded errors we've printed
 
  //
  // routines to determine components of molecular structure
  //
 
  /// Stage 1 of structure building.
  /// (a) find_backbone: assign atomType to atoms based on their
  /// backbone type.  This is the only place where these types get assigned.
  /// protein backbone = name CA C O N or (name OT1 OT2 and bondedto backbone)
  /// nucleic backbone = name P O1P O2P OP1 OP2 
  ///                         O3' C3' C4' C5' O5' O3* C3* C4* C5* O5*
  /// XXX Might be nice for the user to be able to override these definitions.
  int find_backbone(void);

  // find the residues in the molecule; return number found.
  // I look for atoms with the same resid connected to 
  // backbone atoms with the same resid (find enough backbone
  // atoms, then find atoms connected to them)
  int find_connected_backbone(IntStackHandle, int, int, int, int, int *);
  void clean_up_connection(IntStackHandle, int, int, int *);
  void find_connected_atoms_in_resid(IntStackHandle, int, int, 
     int, int, int *);
  void find_and_mark(IntStackHandle, int, int, int, int *, int *);
  int make_uniq_resids(int *flgs); ///< give each residue a uniq resid

  /// Stage 2 of structure building.
  /// Called after atom types and bond types have been assigned to each atom.
  /// (a) Assign uniq_resid to each atom by finding atoms that are bonded to
  ///     each other and have the the same resid string and insertion string.
  /// (b) Assign residueType to each atom by checking for 4 atoms of a given
  ///     atomType in the sets of bonded atoms.  
  int find_residues(void);
  
  /// Find the waters, based on resname, and return number.
  // This should take place after find_residues to keep
  // from mistaking a protein resname as a water resname, maybe
  void find_connected_waters(int i, char *tmp);
  int find_connected_waters2(void);

  /// Stage 2b of structure building.  This is essentially a continuation
  /// of Stage 2.  For atoms that do not yet have a residue type, their resname
  /// is matched against a list of water residue names.  residueType is 
  /// assigned if there is match.
  int find_waters(void);

  /// Stage 3 of structure building: 
  ///   (a) Create new residues.  The residue type is determined by the first
  ///       atom added to the residue.
  ///   (b) assign atoms to residues;
  ///   (c) assign bonds to residues (i.e. which residues are bonded to which)
  void find_connected_residues(int num_residues);
  
  /// find the segments in the molecule; return number found.
  int find_segments(void) { return segNames.num(); }
  
  /// find the connected residues and put the info in fragList
  int find_connected_fragments();

  /// Stage 4 of structure building.
  ///    (a) Create new fragments, and assign residues to them.
  ///    (b) Assign fragment to each atom
  ///    (c) Sort fragments into protein framgents and nucleic fragments,
  ///        far more complex than one might guess, see the code for details.
  ///        This creates pfragList and nfragList for the molecule.
  int find_fragments(void);

  void find_subfragments_cyclic(ResizeArray<Fragment *> *subfragList, int restype);  
  void find_cyclic_subfragments(ResizeArray<Fragment *> *subfragList, ResizeArray<int> *subfragCyclic);
 
  /// find ordered protein and nucleic subfragments
  void find_connected_subfragment(int resnum, int fragnum, char *flgs, 
         int endatom, int altendatom, int alt2endatom, int alt3endatom,
         int restype, 
         ResizeArray<Fragment *> *subfragList);

  void find_subfragments(int startatom, int altstartatom, int alt2startatom,
    int endatom, int altendatom, int alt2endatom, int alt3endatom,
    int restype, ResizeArray<Fragment *> *subfragList);

  void find_subfragments_topologically(int restype, ResizeArray<Fragment *> *subfragList, int endatom, int altendatom, int alt2endatom, int alt3endatom);

#if defined(VMDFASTRIBBONS)
  /// pre-calculate lists of control points used by ribbon/cartoon reps,
  /// greatly accelerates trajectory rendering for large structures.
  void calculate_ribbon_controlpoints();
#endif

#ifdef VMDWITHCARBS
   /// find small rings
   int find_small_rings(int maxringsize);
   int find_back_edges(ResizeArray<int> &back_edge_src, ResizeArray<int> &back_edge_dest);
   int find_connected_subgraph_back_edges(int atomid, ResizeArray<int> &back_edge_src, ResizeArray<int> &back_edge_dest,
                                          int *intree_parents);
   int find_small_rings_from_back_edges(int maxringsize, ResizeArray<int> &back_edge_src, ResizeArray<int> &back_edge_dest);
   int find_small_rings_from_partial(SmallRing *ring, int maxringsize, inthash_t *used_edges, inthash_t *used_atoms);
   int get_edge_key(int edge_src, int edge_dest);
   
   // orientate small rings
   void orientate_small_rings(int maxringsize);
   void orientate_small_ring(SmallRing &ring,int maxringsize);
   
   // find links between small rings
   int find_orientated_small_ring_linkages(int maxpathlength,int maxringsize);
   int find_linkages_for_ring_from_partial(LinkagePath &lp, int maxpathlength, inthash_t *atom_to_ring, inthash_t *multi_ring_atoms, inthash_t *used_atoms);
#endif


protected:
  char *moleculename;  ///< name of the molcule
  int need_find_bonds; ///< whether to compute bonds from the first timestep

public:
  // constructor; just sets most things to null.  Derived classes must put
  // in structure in 'create' routine.  Typical sequence of creating a
  // molecule should be:
  //	mol = new Molecule(....)
  //	( specify parameters for creation )
  //	mol->create();	... return success
  //    mol->analyze(); ... find information about the structure
  BaseMolecule(int);      ///< constructor takes molecule ID
  virtual ~BaseMolecule(void); ///< destructor

  //
  // routines to develop molecular structure
  //
  
  /// Try to set the number of atoms to n.  n must be positive.  May be called
  /// more than once with the same n.  Return success. 
  int init_atoms(int n); 

  /// compute molecule's bonds using distance bond search from 1st timestep
  void find_bonds_from_timestep() { need_find_bonds = 1; }
  void find_unique_bonds_from_timestep() { need_find_bonds = 2; }

  /// add a new atom; return it's index.
  int add_atoms(int natoms, 
                const char *name, const char *type, int atomicnumber, 
                const char *resname, int resid,
	        const char *chainid, const char *segname,
	        const char *insertion = (char *) " ", 
                const char *altloc = "");

  /// add a new bond from a to b; return total number of bonds added so far.
  int add_bond(int, int, float, int, int = ATOMNORMAL);

  /// add a bond after checking for duplicates
  int add_bond_dupcheck(int, int, float, int);

  /// clear list of angles and types.
  void clear_angles(void) { angles.clear(); angleTypes.clear(); }

  /// count angle list entries.
  int num_angles() { return angles.num() / 3; }

  /// add an angle definition to existing list with optional angletype.
  /// checks for duplicates and returns angle index number.
  int add_angle(int a, int b, int c, int type=-1);

  /// set new angle type.
  int set_angletype(int a, int type);

  /// query angle type.
  int get_angletype(int a);

  /// clear list of dihedrals and types.
  void clear_dihedrals(void) { dihedrals.clear(); dihedralTypes.clear(); }

  /// count dihedral list entries.
  int num_dihedrals() { return dihedrals.num() / 4; }

  /// append a dihedral definition to existing list with optional dihedraltype.
  /// checks for duplicates. return dihedral index number.
  int add_dihedral(int a, int b, int c, int d, int type=-1);
    
  /// set new dihedral type. return type index number
  int set_dihedraltype(int d, int type);

  /// query dihedral type for number.
  int get_dihedraltype(int d);

  /// clear list of impropers and types.
  void clear_impropers(void) { impropers.clear(); improperTypes.clear(); }

  /// count improper list entries.
  int num_impropers() { return impropers.num() / 4; }

  /// append a improper definition to existing list with optional impropertype.
  /// checks for duplicates. return improper index number.
  int add_improper(int a, int b, int c, int d, int type=-1);
  
  /// set new improper type. returns type index number.
  int set_impropertype(int i, int type);
  /// query improper type for number;
  int get_impropertype(int i);

  /// clear list of improper definitions
  void clear_cterms() {cterms.clear();}

  /// count number of crossterm list entries
  int num_cterms() { return cterms.num() / 8; }

  /// add cross term definition.
  void add_cterm(int a, int b, int c, int d, 
                 int e, int f, int g, int h)  {
    cterms.append(a); cterms.append(b); cterms.append(c); cterms.append(d); 
    cterms.append(e); cterms.append(f); cterms.append(g); cterms.append(h); 
  }
  
  /// find higher level constructs given the atom/bond information
  // (By this time, the molecule is on the MoleculeList!)
  void analyze(void);

  //
  // interfaces to add/query/delete instance transformation matrices
  //
  void add_instance(Matrix4 & inst) { instances.append(inst); }
  int num_instances(void) { return instances.num(); }
  void clear_instances(void) { instances.clear(); }


  //
  // query info about the molecule
  //
  int id(void) const { return ID; } ///< return id number of this molecule
  const char *molname() const {return moleculename; } ///< return molecule name

  // Return the Nth atom, residue, and fragment.  All assume correct index 
  // and that the structure has been initialized (for speed).
  MolAtom *atom(int n) { return atomList+n; } ///< return Nth atom
  Residue *residue(int);                      ///< return Nth residue
  Fragment *fragment(int);                    ///< return Nth fragment

  // return the residue or fragment in which the given atom is located.
  Residue *atom_residue(int);       ///< return residue atom is in
  Fragment *atom_fragment(int);     ///< return fragment atom is in

  //@{
  /// find first occurance of an atom name in the residue, returns -3 not found
  int find_atom_in_residue(int atomnameindex, int residue) {
    const ResizeArray<int> &atoms = residueList[residue]->atoms;
    int num = atoms.num();
    for (int i=0; i<num; i++) {
      if (atom(atoms[i])->nameindex == atomnameindex) return atoms[i];
    }
    return -3;
  }

  int find_atom_in_residue(const char *atomname, int residue);
  //@}

  //@{ 
  /// return 'default' charge, mass, occupancy value and beta value
  float default_charge(const char *);
  float default_mass(const char *);
  float default_radius(const char *);
  float default_occup(void) { return 1.0; }
  float default_beta(void) { return 0.0; }
  //@}

  /// add volumetric data to a molecule
  void add_volume_data(const char *name, const float *o, 
    const float *xa, const float *ya, const float *za, int x, int y, int z,
    float *voldata, float *gradient=NULL, float *variance=NULL); 

  void add_volume_data(const char *name, const double *o, 
    const double *xa, const double *ya, const double *za, int x, int y, int z,
    float *voldata); 

  /// remove a volumetric data object from a molecule
  void remove_volume_data(int idx);

  int num_volume_data(); ///< return number of volume data sets in molecule 
  const VolumetricData *get_volume_data(int); ///< return requested data set
  VolumetricData *modify_volume_data(int); ///< return requested data set
  void compute_volume_gradient(VolumetricData *);  ///< compute negated normalized volume gradient map

protected:
  ResizeArray<VolumetricData *>volumeList;    ///< array of volume data sets
};

// Hydrogen atom name detection macro
#define IS_HYDROGEN(s) (s[0] == 'H' || (isdigit(s[0]) && s[1] == 'H' ))

#endif

