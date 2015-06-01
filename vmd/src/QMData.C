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
 *	$RCSfile: QMData.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.102 $	$Date: 2012/08/10 14:44:47 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The QMData class, which stores all QM calculation data that
 * are not dependent on the timestep (the latter are handled by
 * the QMTimestep class).
 * These data are, for instance, basis set and calculation input
 * parameters such as method and multiplicity.
 * Note that the wavefunctions are stored in QMTimestep but that
 * a signature of all wavefunctions that occur in the trajectory
 * is kept here in QMData. The signature is needed for setting up
 * the Orbital representation GUI.
 *
 * The basis set data are stored in hierarchical data structures
 * for convenient access and better readability of the non 
 * performance critical code. However, the actual orbital computation
 * (performed by the Orbital class) needs simple contiguous
 * arrays which is why we keep those too.
 *
 ***************************************************************************/

#include <math.h>
#include <stdio.h>
#include "Inform.h"
#include "Timestep.h"
#include "QMData.h"
#include "QMTimestep.h"
#include "Orbital.h"
#include "Molecule.h"

//#define DEBUGGING 1

///  constructor  
QMData::QMData(int natoms, int nbasis, int nshells, int nwave) :
  num_wave_f(nwave),
  num_basis(nbasis),
  num_atoms(natoms),
  num_shells(nshells)
{
  num_wavef_signa = 0;
  wavef_signa = NULL;
  num_shells_per_atom = NULL;
  num_prim_per_shell = NULL;
  wave_offset = NULL;
  atom_types = NULL;
  atom_basis = NULL;
  basis_array = NULL;
  basis_set = NULL;
  shell_types = NULL;
  angular_momentum = NULL;
  norm_factors = NULL;
  carthessian = NULL;
  inthessian  = NULL;
  wavenumbers = NULL;
  intensities = NULL;
  normalmodes = NULL;
  imagmodes   = NULL;
  runtype = RUNTYPE_UNKNOWN;
  scftype = SCFTYPE_UNKNOWN;
  status = QMSTATUS_UNKNOWN;
};


QMData::~QMData() {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    free(wavef_signa[i].orbids);
    free(wavef_signa[i].orbocc);
  }
  free(wavef_signa);
  delete [] basis_array;
  delete [] shell_types;
  delete [] num_shells_per_atom;
  delete [] num_prim_per_shell;
  delete [] atom_types;
  delete [] wave_offset;
  delete [] angular_momentum;
  delete [] carthessian;
  delete [] inthessian;
  delete [] wavenumbers;
  delete [] intensities;
  delete [] normalmodes;
  delete [] imagmodes;
  delete [] basis_string;
  delete [] atom_basis;
  if (norm_factors) {
    for (i=0; i<=highest_shell; i++) {
      if (norm_factors[i]) delete [] norm_factors[i];
    }
    delete [] norm_factors;
  }
  if (basis_set)
    delete_basis_set();
}

// Free memory of the basis set
void QMData::delete_basis_set() {
  int i, j;
  for (i=0; i<num_types; i++) {
    for (j=0; j<basis_set[i].numshells; j++) {
      delete [] basis_set[i].shell[j].prim;
    }
    delete [] basis_set[i].shell;
  }
  delete [] basis_set;

  basis_set = NULL;
}



//! Set the total molecular charge, multiplicity and compute
//! the corresponding number of alpha/beta and total electrons.
//! XXX: this may be rather deduced from the occupations if available.
void QMData::init_electrons(Molecule *mol, int totcharge) {

  int i, nuclear_charge = 0;
  for (i=0; i<num_atoms; i++) {
    nuclear_charge += mol->atom(i)->atomicnumber;
  }
  
  totalcharge   = totcharge;
  num_electrons = nuclear_charge - totalcharge;
  //multiplicity  = mult;

#if 0
  if (scftype == SCFTYPE_RHF) {
    if (mult!=1) {
      msgErr << "For RHF calculations the multiplicity has to be 1, but it is "
             << multiplicity << "!"
             << sendmsg;
    }
    if (num_electrons%2) {
      msgErr << "Unpaired electron(s) in RHF calculation!"
             << sendmsg;
    }
    num_orbitals_A = num_orbitals_B = num_electrons/2;
  }
  else if ( (scftype == SCFTYPE_ROHF) ||
            (scftype == SCFTYPE_UHF) ) {
    num_orbitals_B = (num_electrons-multiplicity+1)/2;
    num_orbitals_A = num_electrons-num_orbitals_B;
  }
#endif
}



//   ======================================
//   Functions for basis set initialization
//   ======================================


// Populate basis set data and organize them into
// hierarcical data structures.
int QMData::init_basis(Molecule *mol, int num_basis_atoms,
                       const char *bstring,
                       const float *basis,
                       const int *atomic_numbers,
                       const int *nshells,
                       const int *nprims,
                       const int *shelltypes) {
  num_types = num_basis_atoms;

  basis_string = new char[1+strlen(bstring)];
  strcpy(basis_string, bstring);

  if (!basis && (!strcmp(basis_string, "MNDO") ||
                 !strcmp(basis_string, "AM1")  ||
                 !strcmp(basis_string, "PM3"))) {
    // Semiempirical methods are based on STOs.
    // The only parameter we need for orbital rendering
    // are the exponents zeta for S, P, D,... shells for
    // each atom. Since most QM packages don't print these
    // values we have to generate the basis set here using
    // hardcoded table values.

    // generate_sto_basis(basis_string);

    return 1;
  }

  int i, j;


  // Copy the basis set arrays over.
  if (!basis || !num_basis) return 1;
  basis_array = new float[2*num_basis];
  memcpy(basis_array, basis, 2*num_basis*sizeof(float));

  if (!nshells || !num_basis_atoms) return 0;
  num_shells_per_atom = new int[num_basis_atoms];
  memcpy(num_shells_per_atom, nshells, num_basis_atoms*sizeof(int));

  if (!nprims || !num_shells) return 0;
  num_prim_per_shell = new int[num_shells];
  memcpy(num_prim_per_shell, nprims, num_shells*sizeof(int));

  if (!shelltypes || !num_shells) return 0;
  shell_types = new int[num_shells];
  highest_shell = 0;
  for (i=0; i<num_shells; i++) {
    // copy shell types ({0, 1, 2, ...} meaning {S, P, D, ...})
    shell_types[i] = shelltypes[i];

    // Translate the combined shell types that have negative
    // codes into their corresponding basic types in order
    // to be able to determine the highest shell.
    // The highest shell is needed by init_angular_norm_factors().
    int basictype = shell_types[i];
    switch (basictype) {
      case SP_S_SHELL:  basictype = S_SHELL; break;
      case SP_P_SHELL:  basictype = P_SHELL; break;
      case SPD_S_SHELL: basictype = S_SHELL; break;
      case SPD_P_SHELL: basictype = P_SHELL; break;
      case SPD_D_SHELL: basictype = D_SHELL; break;
    }
    if (basictype>highest_shell) highest_shell = basictype;
  }
#ifdef DEBUGGING
  printf("highest shell = %d\n", highest_shell);
#endif

  // Create table of angular normalization constants
  init_angular_norm_factors();

  // Organize basis set data hierarchically
  int boffset = 0;
  int shell_counter = 0;
  int numcartpershell[14] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 93, 107}; 
  basis_set  = new basis_atom_t[num_basis_atoms];

  for (i=0; i<num_basis_atoms; i++) {
    basis_set[i].atomicnum = atomic_numbers[i];
    basis_set[i].numshells = num_shells_per_atom[i];
    basis_set[i].shell = new shell_t[basis_set[i].numshells];

    for (j=0; j<basis_set[i].numshells; j++) {
      // We keep the info about the origin of a shell from a
      // combined shell (e.g. SP shell having common exponents
      // for S and P) in an extra flag in the basis_set structure.
      // In shell_types we just store S, P, D ... (0, 1, 2, ...)
      // because this is the relevant info.
      switch (shell_types[shell_counter]) {
      case SP_S_SHELL:
        shell_types[shell_counter] = S_SHELL;
        basis_set[i].shell[j].combo = 1;
	break;
      case SP_P_SHELL:
	shell_types[shell_counter] = P_SHELL;
        basis_set[i].shell[j].combo = 1;
	break;
      case SPD_S_SHELL: 
        shell_types[shell_counter] = S_SHELL;
        basis_set[i].shell[j].combo = 2;
	break;
      case SPD_P_SHELL:
	shell_types[shell_counter] = P_SHELL;
        basis_set[i].shell[j].combo = 2;
	break;
      case SPD_D_SHELL:
	shell_types[shell_counter] = D_SHELL;
        basis_set[i].shell[j].combo = 2;
	break;
      }

      basis_set[i].shell[j].type = shell_types[shell_counter];

      int shelltype = shell_types[shell_counter];
      basis_set[i].shell[j].num_cart_func = numcartpershell[shelltype];
      basis_set[i].shell[j].basis = basis_array+2*boffset;
      basis_set[i].shell[j].norm_fac = norm_factors[shelltype];
      basis_set[i].shell[j].numprims = num_prim_per_shell[shell_counter];

      basis_set[i].shell[j].prim = new prim_t[basis_set[i].shell[j].numprims];
#ifdef DEBUGGING
      //printf("atom %i shell %i %s\n", i, j, get_shell_type_str(&basis_set[i].shell[j]));
#endif

      int k;
      for (k=0; k<basis_set[i].shell[j].numprims; k++) {
        float expon = basis_array[2*(boffset+k)  ];
        float coeff = basis_array[2*(boffset+k)+1];
        basis_set[i].shell[j].prim[k].expon = expon;
        basis_set[i].shell[j].prim[k].coeff = coeff;
     }
  
      // Offsets to get to this shell in the basis array.
      boffset += basis_set[i].shell[j].numprims;

      shell_counter++;
    }
  }



  // Collapse basis set so that we have one basis set
  // per atom type.
  if (!create_unique_basis(mol, num_basis_atoms)) {
    return 0;
  }

  // Multiply the contraction coefficients with
  // the shell dependent part of the normalization factor.
  normalize_basis();

  return 1;
}


// =================================================
// Helper functions for building the list of unique
// basis set atoms
// =================================================

// Return 1 if the two given shell basis sets are identical,
// otherwise return 0.
static int compare_shells(const shell_t *s1, const shell_t *s2) {
  if (s1->type     != s2->type)     return 0;
  if (s1->numprims != s2->numprims) return 0;
  int i;
  for (i=0; i<s1->numprims; i++) {
    if (s1->prim[i].expon != s2->prim[i].expon) return 0;
    if (s1->prim[i].coeff != s2->prim[i].coeff) return 0;
  }
  return 1;
}

// Return 1 if the two given atomic basis sets are identical,
// otherwise return 0.
static int compare_atomic_basis(const basis_atom_t *a1, const basis_atom_t *a2) {
  if (a2->atomicnum != a1->atomicnum) return 0;
  if (a1->numshells != a2->numshells) return 0;
  int i;
  for (i=0; i<a1->numshells; i++) {
    if (!compare_shells(&a1->shell[i], &a2->shell[i])) return 0;
  }
  return 1;
}

static void copy_shell_basis(const shell_t *s1, shell_t *s2) {
  s2->numprims = s1->numprims;
  s2->type     = s1->type;
  s2->combo    = s1->combo;
  s2->norm_fac = s1->norm_fac;
  s2->num_cart_func = s1->num_cart_func;
  s2->prim = new prim_t[s2->numprims];
  int i;
  for (i=0; i<s2->numprims; i++) {
    s2->prim[i].expon = s1->prim[i].expon;
    s2->prim[i].coeff = s1->prim[i].coeff;
  }
}

static void copy_atomic_basis(const basis_atom_t *a1, basis_atom_t *a2) {
  a2->atomicnum = a1->atomicnum;
  a2->numshells = a1->numshells;
  a2->shell = new shell_t[a2->numshells];
  int i;
  for (i=0; i<a2->numshells; i++) {
    copy_shell_basis(&a1->shell[i], &a2->shell[i]);
  }
}

// Collapse basis set so that we have one basis set per
// atom type rather that per atom. In most cases an atom
// type is a chemical element. Create an array that maps
// individual atoms to their corresponding atomic basis.
int QMData::create_unique_basis(Molecule *mol, int num_basis_atoms) {
  basis_atom_t *unique_basis = new basis_atom_t[num_basis_atoms];
  copy_atomic_basis(&basis_set[0], &unique_basis[0]);
  int num_unique_atoms = 1;
  int i, j, k;
  for (i=1; i<num_basis_atoms; i++) {
    int found = 0;
    for (j=0; j<num_unique_atoms; j++) {
      if (compare_atomic_basis(&basis_set[i], &unique_basis[j])) {
        found = 1;
        break;
      }
    }
    if (!found) {
      copy_atomic_basis(&basis_set[i], &unique_basis[j]);
      num_unique_atoms++;
    }
  }

  msgInfo << "Number of unique atomic basis sets = "
          << num_unique_atoms <<"/"<< num_atoms << sendmsg;


  // Free memory of the basis set
  delete_basis_set();
  delete [] basis_array;

  num_types = num_unique_atoms;
  basis_set = unique_basis;

  // Count the new number of basis functions
  num_basis = 0;
  for (i=0; i<num_types; i++) {
    for (j=0; j<basis_set[i].numshells; j++) {
      num_basis += basis_set[i].shell[j].numprims;
    }
  }

  basis_array       = new float[2*num_basis];
  int *basis_offset = new int[num_types];

  int ishell = 0;
  int iprim  = 0;
  for (i=0; i<num_types; i++) {
     basis_offset[i] = iprim;

    for (j=0; j<basis_set[i].numshells; j++) {
      basis_set[i].shell[j].basis = basis_array+iprim;
#ifdef DEBUGGING
      printf("atom type %i shell %i %s\n", i, j, get_shell_type_str(&basis_set[i].shell[j]));
#endif
      for (k=0; k<basis_set[i].shell[j].numprims; k++) {
        basis_array[iprim  ] = basis_set[i].shell[j].prim[k].expon;
        basis_array[iprim+1] = basis_set[i].shell[j].prim[k].coeff;
#ifdef DEBUGGING 
        printf("prim %i: % 9.2f % 9.6f \n", k, basis_array[iprim], basis_array[iprim+1]);
#endif
        iprim += 2;
      }
      ishell++;
    }
  }

  atom_types = new int[num_atoms];

  // Assign basis set type to each atom and
  // create array of offsets into basis_array.
  for (i=0; i<num_atoms; i++) {
    int found = 0;
    for (j=0; j<num_types; j++) {
      //printf("atomicnum %d--%d\n", basis_set[j].atomicnum, mol->atom(i)->atomicnumber);
      if (basis_set[j].atomicnum == mol->atom(i)->atomicnumber) {
        found = 1;
        break;
      }
    }
    if (!found) {
      msgErr << "Error reading QM data: Could not assign basis set type to atom "
             << i << "." << sendmsg;
      delete_basis_set();
      delete [] basis_offset;
      return 0;
    }
    atom_types[i] = j;
#ifdef DEBUGGING 
    printf("atom_types[%d]=%d\n", i, j);
#endif
  }

  // Count the new number of shells
  num_shells = 0;
  for (i=0; i<num_atoms; i++) {
    num_shells += basis_set[atom_types[i]].numshells;
  }

  // Reallocate symmetry expanded arrays
  delete [] shell_types;
  delete [] num_prim_per_shell;
  delete [] num_shells_per_atom;
  shell_types      = new int[num_shells];
  num_prim_per_shell  = new int[num_shells];
  num_shells_per_atom = new int[num_atoms];
  atom_basis          = new int[num_atoms];
  wave_offset         = new int[num_atoms];
  int shell_counter = 0;
  int woffset = 0;

  // Populate the arrays again.
  for (i=0; i<num_atoms; i++) {
    int type = atom_types[i];

    // Offsets into wavefunction array
    wave_offset[i] = woffset;

    for (j=0; j<basis_set[type].numshells; j++) {
      shell_t *shell = &basis_set[type].shell[j];

      woffset += shell->num_cart_func;

      shell_types[shell_counter]        = shell->type;
      num_prim_per_shell[shell_counter] = shell->numprims;
      shell_counter++;
    }

    num_shells_per_atom[i] = basis_set[type].numshells;

    // Offsets into basis_array
    atom_basis[i] = basis_offset[type];
  }

  delete [] basis_offset;

  return 1;
}



// Multiply the contraction coefficients with
// the shell dependent part of the normalization factor
// N = (2a/pi)^(3/4) * sqrt[(8a)^L].
// Here L denotes the shell type with L={0,1,2,3,...}
// for {S,P,D,F,...} shells.
//
// Note that the angular momentum dependent part of the
// normalization factor is
// n = sqrt[i!j!k!/((2i)!(2j)!(2k)!)]
// These values are stored in a table by 
// init_angular_norm_factors().
// 
void QMData::normalize_basis() {
  int i, j, k;
  for (i=0; i<num_types; i++) {
    for (j=0; j<basis_set[i].numshells; j++) {
      shell_t *shell = &basis_set[i].shell[j];
      int shelltype = shell->type;
      for (k=0; k<shell->numprims; k++) {
        float expon = shell->prim[k].expon;
        float norm = (float) (pow(2.0*expon/VMD_PI, 0.75)*sqrt(pow(8*expon, shelltype)));
#ifdef DEBUGGING
        //printf("prim %i: % 9.2f % 9.6f  norm=%f\n", k, expon, coeff, norm);
#endif
        shell->basis[2*k+1] = norm*shell->prim[k].coeff;
      }
    }
  }
}

// Computes the factorial of n
static int fac(int n) {
  if (n==0) return 1;
  int i, x=1;
  for (i=1; i<=n; i++) x*=i;
  return x;
}

// Computes the factorial of n
// Caution: Recursive function! Don't use with large n.
// (For the overlap integrals we need only very small n.)
static int doublefac(int n) {
  if (n<=1) return 1;
  return n*doublefac(n-2);
}

// Initialize table of angular momentum dependent normalization
// factors containing different factors for each shell and its
// cartesian functions.
void QMData::init_angular_norm_factors() {
  int shell;
  norm_factors = new float*[highest_shell+1];
  for (shell=0; shell<=highest_shell; shell++) {
    int i, j, k;
    int numcart = 0;
    for (i=0; i<=shell; i++) numcart += i+1;

    norm_factors[shell] = new float[numcart];
    int count = 0;
    for (k=0; k<=shell; k++) {
      for (j=0; j<=shell; j++) {
        for (i=0; i<=shell; i++) {
          if (i+j+k==shell) {
#ifdef DEBUGGING
            printf("count=%i (%i%i%i) %f\n", count, i, j, k, sqrt(((float)(fac(i)*fac(j)*fac(k))) / (fac(2*i)*fac(2*j)*fac(2*k))));
#endif
            norm_factors[shell][count++] = (float) sqrt(((float)(fac(i)*fac(j)*fac(k))) / (fac(2*i)*fac(2*j)*fac(2*k)));
          }
        }
      }
    }
  } 
}



//   =================
//   Basis set acccess
//   =================


// Get basis set for an atom
const basis_atom_t* QMData::get_basis(int atom) const {
  if (!basis_set || !num_types || atom<0 || atom>=num_atoms)
    return NULL;
  return &(basis_set[atom_types[atom]]);
}


// Get basis set for a shell
const shell_t* QMData::get_basis(int atom, int shell) const {
  if (!basis_set || !num_types || atom<0 || atom>=num_atoms ||
      shell<0 || shell>=basis_set[atom_types[atom]].numshells)
    return NULL;
  return &(basis_set[atom_types[atom]].shell[shell]);
}


int QMData::get_num_wavecoeff_per_atom(int atom) const {
  if (atom<0 || atom>num_atoms) {
    msgErr << "Atom "<<atom<<" does not exist!"<<sendmsg;
    return -1;
  }
  int i;
  int a = atom_types[atom];
  int num_cart_func = 0;
  for (i=0; i<basis_set[a].numshells; i++) {
    num_cart_func += basis_set[a].shell[i].num_cart_func;
  }
  return num_cart_func;
}

// Get the offset in the wavefunction array for a specified
// shell in an atom.
int QMData::get_wave_offset(int atom, int shell) const {
  if (atom<0 || atom>num_atoms) {
    msgErr << "Atom "<<atom<<" does not exist!"<<sendmsg;
    return -1;
  }
  if (shell<0 || shell>=basis_set[atom_types[atom]].numshells) {
    msgErr << "Shell "<<shell<<" in atom "<<atom
           << " does not exist!"<<sendmsg;
    return -1;
  }
  int i;
  int numcart = 0;
  for (i=0; i<shell; i++) {
    numcart += basis_set[atom_types[atom]].shell[i].num_cart_func;
  }
  return wave_offset[atom]+numcart;
}


/// Get shell type letter (S, P, D, F, ...) followed by '\0'
const char* QMData::get_shell_type_str(const shell_t *shell) {
  const char* map[14] = {"S\0", "P\0", "D\0", "F\0", "G\0", "H\0",
                  "I\0", "K\0", "L\0", "M\0", "N\0", "O\0", "Q\0", "R\0"};

  return map[shell->type];
}



int QMData::set_angular_momenta(const int *angmom) {
  if (!angmom || !num_wave_f) return 0;
  angular_momentum = new int[3*num_wave_f];
  memcpy(angular_momentum, angmom, 3*num_wave_f*sizeof(int));
  return 1;
}

void QMData::set_angular_momentum(int atom, int shell, int mom,
                                  int *array) {
  if (!array || !angular_momentum) return;
  int offset = get_wave_offset(atom, shell);
  if (offset<0) return;
  memcpy(&angular_momentum[3*(offset+mom)], array, 3*sizeof(int));
}


// For a certain atom and shell return the exponent of the
// requested cartesian component of the angular momentum
// (specified by comp=0,1,2 for x,y,z resp.).
// Example:
// For XYYYZZ the exponents of the angular momentum are
// X (comp 0): 1
// Y (comp 1): 3
// Y (comp 2): 2
int QMData::get_angular_momentum(int atom, int shell, int mom, int comp) {
  if (!angular_momentum) return -1;
  int offset = get_wave_offset(atom, shell);
  if (offset<0 ||
      mom>=get_basis(atom, shell)->num_cart_func) return -1;
  //printf("atom=%d, shell=%d, mom=%d, comp=%d\n", atom, shell, mom, comp);
  return angular_momentum[3*(offset+mom)+comp];
}


// Set the angular momentum from a string
void QMData::set_angular_momentum_str(int atom, int shell, int mom,
                                  const char *tag) {
  unsigned int j;
  int offset = get_wave_offset(atom, shell);
  if (offset<0) return;

  int xexp=0, yexp=0, zexp=0;

  for (j=0; j<strlen(tag); j++) {
    switch (tag[j]) {
      case 'X':
        xexp++;
        break;
      case 'Y':
        yexp++;
        break;
      case 'Z':
        zexp++;
        break;
    }
  }
  angular_momentum[3*(offset+mom)  ] = xexp;
  angular_momentum[3*(offset+mom)+1] = yexp;
  angular_momentum[3*(offset+mom)+2] = zexp;
}


// Returns a pointer to a string representing the angular
// momentum of a certain cartesian basis function.
// The strings for an F-shell would be for instance
// XX YY ZZ XY XZ YZ.
// The necessary memory is automatically allocated.
// Caller is responsible delete the string!
char* QMData::get_angular_momentum_str(int atom, int shell, int mom) const {
  int offset = get_wave_offset(atom, shell);
  if (offset<0) return NULL;

  char *s = new char[2+basis_set[atom_types[atom]].shell[shell].type];
  int i, j=0;
  for (i=0; i<angular_momentum[3*(offset+mom)  ]; i++) s[j++]='X';
  for (i=0; i<angular_momentum[3*(offset+mom)+1]; i++) s[j++]='Y';
  for (i=0; i<angular_momentum[3*(offset+mom)+2]; i++) s[j++]='Z';
  s[j] = '\0';
  if (!strlen(s)) strcpy(s, "S");

  return s;
}



//   ========================
//   Hessian and normal modes
//   ========================


void QMData::set_carthessian(int numcart, double *array) {
  if (!array || !numcart || numcart!=3*num_atoms) return;
  carthessian = new double[numcart*numcart];
  memcpy(carthessian, array, numcart*numcart*sizeof(double));
}

void QMData::set_inthessian(int numint, double *array) {
  if (!array || !numint) return;
  nintcoords = numint;
  inthessian = new double[numint*numint];
  memcpy(inthessian, array, numint*numint*sizeof(double));
}

void QMData::set_normalmodes(int numcart, float *array) {
  if (!array || !numcart || numcart!=3*num_atoms) return;
  normalmodes = new float[numcart*numcart];
  memcpy(normalmodes, array, numcart*numcart*sizeof(float));
}

void QMData::set_wavenumbers(int numcart, float *array) {
  if (!array || !numcart || numcart!=3*num_atoms) return;
  wavenumbers = new float[numcart];
  memcpy(wavenumbers, array, numcart*sizeof(float));
}

void QMData::set_intensities(int numcart, float *array) {
  if (!array || !numcart || numcart!=3*num_atoms) return;
  intensities = new float[numcart];
  memcpy(intensities, array, numcart*sizeof(float));
}

void QMData::set_imagmodes(int numimag, int *array) {
  if (!array || !numimag) return;
  nimag = numimag;
  imagmodes = new int[nimag];
  memcpy(imagmodes, array, nimag*sizeof(int));
}



//   =====================
//   Calculation meta data
//   =====================


// Translate the runtype constant into a string
const char *QMData::get_runtype_string(void) const
{
  switch (runtype) {
  case RUNTYPE_ENERGY:     return "single point energy";   break;
  case RUNTYPE_OPTIMIZE:   return "geometry optimization"; break;
  case RUNTYPE_SADPOINT:   return "saddle point search";   break;
  case RUNTYPE_HESSIAN:    return "Hessian/frequency";     break;
  case RUNTYPE_SURFACE:    return "potential surface scan"; break;
  case RUNTYPE_GRADIENT:   return "energy gradient";        break;
  case RUNTYPE_MEX:        return "minimum energy crossing"; break;
  case RUNTYPE_DYNAMICS:   return "molecular dynamics";    break;
  case RUNTYPE_PROPERTIES: return "properties"; break;
  default:                 return "(unknown)";  break;
  }
}


// Translate the scftype constant into a string
const char *QMData::get_scftype_string(void) const
{
  switch (scftype) {
  case SCFTYPE_NONE:  return "NONE";     break;
  case SCFTYPE_RHF:   return "RHF";      break;
  case SCFTYPE_UHF:   return "UHF";      break;
  case SCFTYPE_ROHF:  return "ROHF";     break;
  case SCFTYPE_GVB:   return "GVB";      break;
  case SCFTYPE_MCSCF: return "MCSCF";    break;
  case SCFTYPE_FF:    return "force field"; break;
  default:            return "(unknown)";   break;
  }
}


// Get status of SCF and optimization convergence
const char* QMData::get_status_string() {
  if      (status==QMSTATUS_OPT_CONV)
    return "Optimization converged";
  else if (status==QMSTATUS_OPT_NOT_CONV)
    return "Optimization not converged";
  else if (status==QMSTATUS_SCF_NOT_CONV)
    return "SCF not converged";
  else if (status==QMSTATUS_FILE_TRUNCATED)
    return "File truncated";
  else
    return "Unknown";
}



//   =======================
//   Wavefunction signatures
//   =======================


/// Determine a unique ID for each wavefuntion based on it's signature
/// (type, spin, excitation, info)
/// If for a given timestep there are more than one wavefunctions
/// with the same signature the we assume these are different and
/// we assign a different IDs. This can happen if the wavefunctions
/// cannot be sufficiently distinguished by the existing descriptors
/// and the plugin didn't make use of the info string to set them
/// apart.
/// signa_ts is the list of numsig signatures for the wavefunctions
/// already processed for this timestep. 
int QMData::assign_wavef_id(int type, int spin, int exci, char *info,
                            wavef_signa_t *(&signa_curts),
                            int &num_signa_curts) {
  int j, idtag=-1;

  // Check if a wavefunction with the same signature exists
  // already in the global signature list. In this case we
  // return the corresponding idtag (which will cause
  // QMTimestep::add_wavefunction to overwrite the existing
  // wavefunction with that idtag) .
  for (j=0; j<num_wavef_signa; j++) {
    if (wavef_signa[j].type==type &&
        wavef_signa[j].spin==spin &&
        wavef_signa[j].exci==exci &&
        (info && !strncmp(wavef_signa[j].info, info, QMDATA_BUFSIZ))) {
      idtag = j;
    }
  }

  // Check if we have the same signature in the current timestep
  int duplicate = 0;
  for (j=0; j<num_signa_curts; j++) {
    if (signa_curts[j].type==type &&
        signa_curts[j].spin==spin &&
        signa_curts[j].exci==exci &&
        (info && !strncmp(signa_curts[j].info, info, QMDATA_BUFSIZ))) {
      duplicate = 1;
    }
  }
  
  // Add a new signature for the current timestep
  if (!signa_curts) {
    signa_curts = (wavef_signa_t *)calloc(1, sizeof(wavef_signa_t));
  } else {
    signa_curts = (wavef_signa_t *)realloc(signa_curts,
                         (num_signa_curts+1)*sizeof(wavef_signa_t));
  }
  signa_curts[num_signa_curts].type = type;
  signa_curts[num_signa_curts].spin = spin;
  signa_curts[num_signa_curts].exci = exci;
  if (!info)
    signa_curts[num_signa_curts].info[0] = '\0';
  else
    strncpy(signa_curts[num_signa_curts].info, info, QMDATA_BUFSIZ);
  num_signa_curts++;

  // Add new wavefunction ID tag in case this signature wasn't
  // found at all or in case we have a duplicate in the current
  // timestep.
  // If in a single timestep two wavefunction with the same type
  // are sent by the molfile_plugin then we assume that the plugin
  // considers them different wavefunctions. Our categories are
  // just not sufficient to distinguish them.
  if (idtag<0 || duplicate) {
    if (!wavef_signa) {
      wavef_signa = (wavef_signa_t *)calloc(1, sizeof(wavef_signa_t));
    } else {
      wavef_signa = (wavef_signa_t *)realloc(wavef_signa,
                           (num_wavef_signa+1)*sizeof(wavef_signa_t));
    }
    wavef_signa[num_wavef_signa].type = type;
    wavef_signa[num_wavef_signa].spin = spin;
    wavef_signa[num_wavef_signa].exci = exci;
    wavef_signa[num_wavef_signa].max_avail_orbs = 0;
    wavef_signa[num_wavef_signa].orbids = NULL;
    if (!info)
      wavef_signa[num_wavef_signa].info[0] = '\0';
    else
      strncpy(wavef_signa[num_wavef_signa].info, info, QMDATA_BUFSIZ);
    idtag = num_wavef_signa;
    num_wavef_signa++;
  }

  //printf("idtag=%d (%d, %d, %d, %s)\n", idtag, type, spin, exci, info);

  return idtag;  
}


// Find the wavefunction ID tag by comparing
// type, spin, and excitation with the signatures
// of existing wavefunctions
// Returns -1 if no such wavefunction exists.
int QMData::find_wavef_id_from_gui_specs(int guitype, int spin, int exci) {
  int i, idtag = -1;
  for (i=0; i<num_wavef_signa; i++) {
    if (spin==wavef_signa[i].spin &&
        exci==wavef_signa[i].exci) {
      if (compare_wavef_guitype_to_type(guitype, wavef_signa[i].type)) {
        idtag = i;
      }
      
    }
  }
  //if (idtag<0) { printf("Couldn't find_wavef_id_from_gui_specs: guitype=%d \n", guitype); }
  return idtag;
}


/// Return 1 if we have a wavefunction with the given GUI_WAVEF_TYPE_*
int QMData::has_wavef_guitype(int guitype) {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    if (compare_wavef_guitype_to_type(guitype, wavef_signa[i].type)) {
      return 1;
    }
  }
  return 0;
}

int QMData::compare_wavef_guitype_to_type(int guitype, int type) {
   if ((guitype==GUI_WAVEF_TYPE_CANON    && type==WAVE_CANON)    ||
       (guitype==GUI_WAVEF_TYPE_GEMINAL  && type==WAVE_GEMINAL)  ||
       (guitype==GUI_WAVEF_TYPE_MCSCFNAT && type==WAVE_MCSCFNAT) ||
       (guitype==GUI_WAVEF_TYPE_MCSCFOPT && type==WAVE_MCSCFOPT) ||
       (guitype==GUI_WAVEF_TYPE_CINAT    && type==WAVE_CINATUR)  ||
       (guitype==GUI_WAVEF_TYPE_OTHER    && type==WAVE_UNKNOWN)  ||
       (guitype==GUI_WAVEF_TYPE_LOCAL    && 
        (type==WAVE_BOYS || type==WAVE_RUEDEN || type==WAVE_PIPEK))) {
     return 1;
   }
   return 0;
}


/// Return 1 if we have any wavefunction with the given spin
int QMData::has_wavef_spin(int spin) {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].spin==spin) return 1;
  }
  return 0;
}


/// Return 1 if we have any wavefunction with the given spin
int QMData::has_wavef_exci(int exci) {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].exci==exci) return 1;
  }
  return 0;
}


/// Return 1 if we have any wavefunction with the given
/// signature (type, spin, and excitation).
int QMData::has_wavef_signa(int type, int spin, int exci) {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].type==type &&
        wavef_signa[i].exci==exci &&
        wavef_signa[i].spin==spin) return 1;
  }
  return 0;
}


/// Get the highest excitation for any wavefunction 
/// with the given type.
int QMData::get_highest_excitation(int guitype) {
  int i, highest=0;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].exci>highest &&
        compare_wavef_guitype_to_type(guitype, wavef_signa[i].type)) {
      highest = wavef_signa[i].exci;
    }
  }
  return highest;
}


//   =====================================================
//   Functions dealing with the list of orbitals available 
//   for a given wavefunction. Needed for the GUI.
//   =====================================================


// Merge the provided list of orbital IDs with the existing
// list of available orbitals. Available orbitals are the union
// of all orbital IDs for the wavefunction with ID iwavesig
// occuring throughout the trajectory.
void QMData::update_avail_orbs(int iwavesig, int norbitals,
                               const int *orbids, const float *orbocc) {
  int i, j;

  // Signature of wavefunction
  wavef_signa_t *cursig = &wavef_signa[iwavesig];

  for (i=0; i<norbitals; i++) {
    int found = 0;
    for (j=0; j<cursig->max_avail_orbs; j++) {
      if (cursig->orbids[j]==orbids[i]) {
        found = 1;
        break;
      }
    }
    if (!found) {
      if (!cursig->orbids) {
        cursig->orbids = (int  *)calloc(1, sizeof(int));
        cursig->orbocc = (float*)calloc(1, sizeof(float));
      } else {
        cursig->orbids = (int  *)realloc(cursig->orbids,
                                  (cursig->max_avail_orbs+1)*sizeof(int));
        cursig->orbocc = (float*)realloc(cursig->orbocc,
                                  (cursig->max_avail_orbs+1)*sizeof(float));
      }
      cursig->orbids[cursig->max_avail_orbs] = orbids[i];
      cursig->orbocc[cursig->max_avail_orbs] = orbocc[i];
      cursig->max_avail_orbs++;
    }
  }
//   printf("iwavesig=%d, ", iwavesig);
//   for (j=0; j<cursig->max_avail_orbs; j++) {
//     printf("%d %.2f\n",cursig->orbids[j], cursig->orbocc[j]);
//   }
//   printf("\n");
}


/// Return the maximum number of available orbitals
/// for the given wavefunction over all frames
/// Can be used to determine the number of orbitals
/// to be displayed in the GUI.
/// Returns -1 if requested wavefunction doesn't exist.
int QMData::get_max_avail_orbitals(int iwavesig) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa) return -1;
  return wavef_signa[iwavesig].max_avail_orbs;
}


/// Get IDs of all orbitals available for the given wavefunction.
/// iwavesig is the index of the wavefunction signature.
/// Returns 1 upon success, 0 otherwise.
int QMData::get_avail_orbitals(int iwavesig, int *(&orbids)) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa) return 0;

  int i;
  for (i=0; i<wavef_signa[iwavesig].max_avail_orbs; i++) {
    orbids[i] = wavef_signa[iwavesig].orbids[i];
  }
  return 1;
}


/// Get occupancies of all orbitals available for the given wavefunction.
/// iwavesig is the index of the wavefunction signature.
/// Returns 1 upon success, 0 otherwise.
int QMData::get_avail_occupancies(int iwavesig, float *(&orbocc)) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa) return 0;

  int i;
  for (i=0; i<wavef_signa[iwavesig].max_avail_orbs; i++) {
    orbocc[i] = wavef_signa[iwavesig].orbocc[i];
  }
  return 1;
}


/// For the given wavefunction signature return
/// the iorb-th orbital ID. Used to translate from the
/// GUI list of available orbitals the unique orbital label.
/// Returns -1 if requested wavefunction doesn't exist or
/// the orbital index is out of range.
int QMData::get_orbital_label_from_gui_index(int iwavesig, int iorb) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa ||
      iorb<0 ||iorb>=wavef_signa[iwavesig].max_avail_orbs)
    return -1;
  return wavef_signa[iwavesig].orbids[iorb];
}

/// Return 1 if the given wavefunction has an orbital with
/// ID orbid in any frame.
int QMData::has_orbital(int iwavesig, int orbid) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa) return 0;

  int i;
  for (i=0; i<wavef_signa[iwavesig].max_avail_orbs; i++) {
    if (orbid==wavef_signa[iwavesig].orbids[i]) return 1;
  }
  return 0;

}

int QMData::expand_atompos(const float *atompos,
                           float *(&expandedpos)) {
  int i, at;
  expandedpos = new float[3*num_wave_f];

  int t = 0;
  // loop over all the QM atoms
  for (at=0; at<num_atoms; at++) {
    int a = atom_types[at];
    float x = atompos[3*at  ]*ANGS_TO_BOHR;
    float y = atompos[3*at+1]*ANGS_TO_BOHR;
    float z = atompos[3*at+2]*ANGS_TO_BOHR;
    printf("{%.2f %.2f %.2f}\n", x, y, z);
    for (i=0; i<basis_set[a].numshells; i++) {
      // Loop over the Gaussian primitives of this contracted 
      // basis function to build the atomic orbital
      shell_t *shell = &basis_set[a].shell[i];
      int shelltype = shell->type;
      printf("shelltype = %d\n", shelltype);
      int l, m, n;
      for (n=0; n<=shelltype; n++) {
        int mmax = shelltype - n; 
        for (m=0, l=mmax; m<=mmax; m++, l--) {
          expandedpos[3*t  ] = x;
          expandedpos[3*t+1] = y;
          expandedpos[3*t+2] = z;
          t++;
        }
      }
    }
  }
  return 0;
}


int QMData::expand_basis_array(float *(&expandedbasis), int *(&numprims)) {
  int i, at;
  int num_prim_total = 0;
  for (at=0; at<num_atoms; at++) {
    int a = atom_types[at];
    for (i=0; i<basis_set[a].numshells; i++) {
      num_prim_total += basis_set[a].shell[i].numprims *
        basis_set[a].shell[i].num_cart_func;
    }
  }

  numprims = new int[num_wave_f];
  expandedbasis = new float[2*num_prim_total];
  int t=0, ifunc=0;
  // loop over all the QM atoms
  for (at=0; at<num_atoms; at++) {
    int a = atom_types[at];
    printf("atom %d\n", at);

    for (i=0; i<basis_set[a].numshells; i++) {
      // Loop over the Gaussian primitives of this contracted 
      // basis function to build the atomic orbital
      shell_t *shell = &basis_set[a].shell[i];
      int maxprim   = shell->numprims;
      int shelltype = shell->type;
      printf("shelltype = %d\n", shelltype);
      int l, m, n, icart=0;
      for (n=0; n<=shelltype; n++) {
        int mmax = shelltype - n; 
        for (m=0, l=mmax; m<=mmax; m++, l--) {
          printf("lmn=%d%d%d %d%d%d\n", l, m, n,
                 angular_momentum[3*ifunc],
                 angular_momentum[3*ifunc+1],
                 angular_momentum[3*ifunc+2]);
          numprims[ifunc++] = maxprim;
          float normfac = shell->norm_fac[icart];
          icart++;
          int prim;
          for (prim=0; prim<maxprim; prim++) {
            expandedbasis[2*t  ] = shell->prim[prim].expon;
            //expandedbasis[2*t+1] = normfac*shell->prim[prim].coeff;
            expandedbasis[2*t+1] = normfac*shell->basis[2*prim+1];
            printf("expon=%f coeff=%f numprims=%d\n", expandedbasis[2*t], expandedbasis[2*t+1], numprims[ifunc-1]);
            t++;
          }
        }
      }
    }
  }
  return 1;
}


#define MIN(X,Y) (((X)<(Y))? (X) : (Y))
#define MAX(X,Y) (((X)>(Y))? (X) : (Y))


// 1-electron integral evaluation
// ==============================
//
// Below are function for the evaluation of the overlap
// integrals which can be used as a template for developing
// other 1e-integral such as the kinetic energy integrals
// or the nuclear attraction integrals.
// The implemetation for the overlap matrix closely follows
// "Fundamentals of Molecular Integrals Evaluation" by 
// Fermann & Valeev (lecture notes):
// http://www.files.chem.vt.edu/chem-dept/valeev/docs/ints.pdf
// See also "Molecular Integrals", Huzinaga 1967, p.68.
// and "Modern Quantum Chemistry", Szabo & Ostlund 1989,
// Appendix A.
// 
// The molecular integral scheme used below represents the
// direct computation of the analytical integrals and is not
// optimized.
// Improved algorithms for molecular integrals include
// * the Obara-Saika scheme (exploiting the translational
//   and horizontal recurrence relation for cartesian overlap
//   integrals)
// * the McMurchie-Davidson scheme (expanding the overlap
//   distribution in Hermite Gaussians).


// Binomial coefficient
static int binomial(int n, int k) {
  return fac(n)/(fac(k)*fac(n-k));
}


// Expression (2.46) for f_k on page 13 of Fermann & Valeev.
float overlap_f(int k, int l1, int l2, float PAx, float PBx) {
  int q;
  float f = 0.f;
  for (q=MAX(-k, k-2*l2); q<=MIN(k, 2*l1-k); q+=2) {
    int i = (k+q)/2;
    int j = (k-q)/2;
    f += (float) (binomial(l1, i)*binomial(l2, j)*pow(PAx, l1-i)*pow(PBx, l2-j));
  }
  return f;
}


// Expression (3.15) for Ix on page 16 of Fermann & Valeev.
float overlap_I(int l1, int l2, float PAx, float PBx, float gamma) {
  int i;
  float Ix = 0.f;
  for (i=0; i<=(l1+l2)/2; i++) {
    Ix += (float) (overlap_f(2*i, l1, l2, PAx, PBx) * doublefac(2*i-1)/pow(2*gamma,i) * sqrtf(float(VMD_PI)/gamma));
  }
  return Ix;
}


// Expression (3.12) for S12 on page 16 of Fermann & Valeev:
// Overlap of primitive functions with arbitrary angular momentum 
float overlap_S12(int l1, int m1, int n1, int l2, int m2, int n2,
                  float alpha, float beta, float rAB2,
                  const float *PA, const float *PB) {
  float gamma = alpha+beta;

  float Ix = overlap_I(l1, l2, PA[0], PB[0], gamma);
  float Iy = overlap_I(m1, m2, PA[1], PB[1], gamma);
  float Iz = overlap_I(n1, n2, PA[2], PB[2], gamma);
  //printf("   I = {%f %f %f}\n", Ix, Iy, Iz);
  return (float) exp(-alpha*beta*rAB2/gamma)*Ix*Iy*Iz;
}


// Overlap of contracted functions with arbitrary angular momentum
// Summing up contributions from oververlap integrals between all
// combinations of primitives of the two contrated functions. 
float overlap_S12_contracted(const float *basis1, const float *basis2,
                             int numprim1, int numprim2,
                             int l1, int m1, int n1, int l2, int m2, int n2,
                             const float *A, const float *B) {
  int i, j;
  float S12_contracted = 0.f;
  float sqrtpigamma3 = 1.0f;
  float rAB2 = distance2(A, B);
  int SS = 0;

  if (l1+m1+n1==0 && l2+m2+n2==0) SS = 1;
  
  for (i=0; i<numprim1; i++) {
    float alpha = basis1[2*i];
    for (j=0; j<numprim2; j++) {
      float beta  = basis2[2*j];
      float gamma = alpha+beta;

      // P = (alpha*A + beta*B)/gamma
      float P[3], PA[3], PB[3];
      vec_scale(P, alpha, A);     // P = alpha*A
      vec_scaled_add(P, beta, B); // P += beta*B
      vec_scale(P, 1/gamma, P);   // P = P/gamma
      vec_sub(PA, P, A);
      vec_sub(PB, P, B);

      switch (SS) {
      case 1:
        // We can use a simpler formula for S-S overlap
        sqrtpigamma3 = sqrtf(float(VMD_PI)/gamma);
        sqrtpigamma3 = sqrtpigamma3*sqrtpigamma3*sqrtpigamma3;
        S12_contracted += float(basis1[2*i+1]*basis2[2*j+1]*exp(-alpha*beta*rAB2/gamma)*sqrtpigamma3);
        break;
      default:
        float S = basis1[2*i+1]*basis2[2*j+1]*
          overlap_S12(l1, m1, n1, l2, m2, n2,
                      alpha, beta, rAB2, PA, PB);
        //printf("  prim %d,%d: co %f %f ex %f %f\n",
        //           i+1, j+1, 
        //           basis1[2*i+1], basis2[2*j+1],
        //           basis1[2*i],   basis2[2*j]);
        S12_contracted += S;
      }
    }
  }
  return S12_contracted;
}

int get_overlap_matrix(int num_wave_f, const float *expandedbasis,
                          const int *numprims,
                          const float *atompos,
                          const int *lmn,
                          float *overlap_matrix) {
  int i, j;
  int t1 = 0;
  for (i=0; i<num_wave_f; i++) {
    const float *basis1 = &expandedbasis[2*t1];
    int numprim1 = numprims[i];
    const float *pos1 = &atompos[3*i];
    int l1 = lmn[3*i  ];
    int m1 = lmn[3*i+1];
    int n1 = lmn[3*i+2];
    int t2 = t1;
    for (j=i; j<num_wave_f; j++) {
      int l2 = lmn[3*j  ];
      int m2 = lmn[3*j+1];
      int n2 = lmn[3*j+2];
      const float *basis2 = &expandedbasis[2*t2];
      int numprim2 = numprims[j];
      const float *pos2 = &atompos[3*j];
      printf("%d,%d %d%d%d--%d%d%d %d-%d {%.2f %.2f %.2f} {%.2f %.2f %.2f}\n",
             i, j, l1, m1, n1, l2, m2, n2, numprim1, numprim2,
             pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]);

      float Sij = overlap_S12_contracted(basis1, basis2, numprim1, numprim2, 
                               l1, m1, n1, l2, m2, n2, pos1, pos2);
      overlap_matrix[i*num_wave_f+j] = Sij;
      overlap_matrix[j*num_wave_f+i] = Sij;
      printf("  S12 = %f\n", Sij);
      t2 += numprim2;
    }
    t1 += numprim1;
  }
  return 0;
}

#if 0
/// Debris I might still need...
float* QMData::get_overlap_integrals(const float *atompos) {
  float *ao_overlap_integrals=NULL;
  if (!ao_overlap_integrals) {
    ao_overlap_integrals = new float[num_wave_f*num_wave_f];
    int i;
    for (i=0; i<num_wave_f*num_wave_f; i++) {
      ao_overlap_integrals[i] = 1.f;
    }

    int at1;
    int shell_counter = 0;
    int prim_counter = 0;
    // loop over all the QM atoms
    for (at1=0; at1<num_atoms; at1++) {
      int maxshell1 = num_shells_per_atom[at1];
      float x1 = atompos[3*at1  ]*ANGS_TO_BOHR;
      float y1 = atompos[3*at1+1]*ANGS_TO_BOHR;
      float z1 = atompos[3*at1+2]*ANGS_TO_BOHR;
      int shell1;
      for (shell1=0; shell1 < maxshell1; shell1++) {
        // Loop over the Gaussian primitives of this contracted 
        // basis function to build the atomic orbital
        int numprims = num_prim_per_shell[shell_counter];
        int shelltype = shell_types[shell_counter];
        int prim1;
        for (prim1=0; prim1 < numprims;  prim1++) {
          float exponent1       = basis_array[prim_counter    ];
          float contract_coeff1 = basis_array[prim_counter + 1];
          int at2;
          for (at2=0; at2<num_atoms; at2++) {
            int maxshell2 = num_shells_per_atom[at2];
            float x2 = atompos[3*at2  ]*ANGS_TO_BOHR;
            float y2 = atompos[3*at2+1]*ANGS_TO_BOHR;
            float z2 = atompos[3*at2+2]*ANGS_TO_BOHR;
            float dx = x2-x1;
            float dy = y2-y1;
            float dz = z2-z1;
            float dist2 = dx*dx + dy*dy + dz*dz;
            int shell2;
            for (shell2=0; shell2 < maxshell2; shell2++) {
              int numprims2 = num_prim_per_shell[shell_counter];
              int shelltype = shell_types[shell_counter];
              int prim2;
              for (prim2=0; prim2 < numprims;  prim2++) {
                float exponent2       = basis_array[prim_counter    ];
                float contract_coeff2 = basis_array[prim_counter + 1];
                prim_counter += 2;
              }
            }
          }
        }
      }
    }
  }
  return ao_overlap_integrals;
}
#endif


/// Compute electronic overlap integrals Sij.
/// Memory for overlap_matrix will be allocated.
void QMData::compute_overlap_integrals(Timestep *ts,
                                      const float *expandedbasis,
                                      const int *numprims,
                                      float *(&overlap_matrix)) {
  float *expandedpos = NULL;
  expand_atompos(ts->pos, expandedpos);


  overlap_matrix = new float[num_wave_f*num_wave_f];
  memset(overlap_matrix, 0, num_wave_f*num_wave_f*sizeof(float));

  get_overlap_matrix(num_wave_f, expandedbasis, numprims, 
                     expandedpos,
                     angular_momentum, overlap_matrix);
  delete [] expandedpos;
}


// matrix multiplication
static void mm_mul(const float *a, int awidth, int aheight,
            const float *b, int bwidth, int bheight, 
            float *(&c)) {
  if (awidth!=bheight)
    printf("mm_mul(): incompatible sizes %d,%d\n", awidth, bheight);
  c = new float[aheight*bwidth];
  for (int i=0; i<aheight; i++) {
    for (int j=0; j<bwidth; j++) {
      float cc = 0.f;
      for (int k=0; k<awidth; k++)
        cc += a[i*awidth+k]*b[k*bwidth+j];
      c[i*bwidth+j] = cc;
    }
  }
}



int QMData::mullikenpop(Timestep *ts, int iwavesig, 
                        const float *expandedbasis,
                        const int *numprims) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa || !ts) return 0;

  int iwave = ts->qm_timestep->get_wavef_index(iwavesig);
  const Wavefunction *wave = ts->qm_timestep->get_wavefunction(iwave);


  float *S;
  compute_overlap_integrals(ts, expandedbasis, numprims, S);

  int numcoeffs = wave->get_num_coeffs();

  float *P;
  wave->population_matrix(S, P);

  int i,j;
  for (i=0; i<numcoeffs; i++) {
    for (j=0; j<numcoeffs; j++) {
      printf("P[%d,%d]=%f\n", i, j, P[i*numcoeffs+j]);
    }
  }

  float *PA;
  wave->population_matrix(this, 2, S, PA);
  //wave->density_matrix(this, 0, PA);

  for (i=0; i<get_num_wavecoeff_per_atom(2); i++) {
    for (j=0; j<numcoeffs; j++) {
      printf("PA[%d,%d]=%f\n", i, j, PA[i*numcoeffs+j]);
    }
  }
  delete [] PA;

  float *GOP = new float[numcoeffs];
  for (i=0; i<numcoeffs; i++) {
    GOP[i] = 0.f;
    for (j=0; j<numcoeffs; j++) {
      GOP[i] += P[i*numcoeffs+j];
    }
    printf("GOP[%d] = %f\n", i, GOP[i]);
  }

  float *GAP = new float[num_atoms];
  int coeff_count = 0;
  int a;
  for (a=0; a<num_atoms; a++) {
    int num_cart_func = get_num_wavecoeff_per_atom(a);
    GAP[a] = 0.f;
    for (i=0; i<num_cart_func; i++) {
      GAP[a] += GOP[coeff_count++];
    }
    printf("GAP[%d] = %f\n", a, GAP[a]);
  }

  float *D;
  wave->density_matrix(D);
  float *DS = new float[numcoeffs*numcoeffs];

  mm_mul(D, numcoeffs, numcoeffs, S, numcoeffs, numcoeffs, DS);

  float Nelec = 0.f;
  for (i=0; i<numcoeffs; i++) {
    printf("DS[%d,%d] = %f\n", i, i, DS[i*numcoeffs+i]);
    Nelec += DS[i*numcoeffs+i];
  }

  printf("Nelec=%f\n", Nelec);

  delete [] S;
  delete [] P;
  delete [] D;
  delete [] DS;

  return 1;
}


// ========================================================
// Orbital localization
// ========================================================

/// Create a new wavefunction object based on existing
/// wavefunction <waveid> with orbitals localized using
/// the Pipek-Mezey algorithm:
/// "A fast intrinsic localization procedure applicable
/// for ab initio and semiempirical linear combination
/// of atomic orbital wave functions"
/// J. Pipek & G. Mezey (1989), J. Chem. Phys. 90 (9),
/// 4916-4926.
///
/// However, the algorithm is described in a more
/// computational manner in
/// "Comparison of the Boys and Pipek-Mezey Localizations
/// in the Local Correlation Approach and Automatic
/// Virtual Basis Selection"
/// J. W. Boughton & P. Pulay (1993), J. Comp. Chem. 14 (6),
/// 736-740.
///
/// The functions below follow the terminology of this
/// latter paper.
///
int QMData::orblocalize(Timestep *ts, int iwavesig, 
                        const float *expandedbasis,
                        const int *numprims) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa || !ts) return 0;

  int iwave = ts->qm_timestep->get_wavef_index(iwavesig);
  Wavefunction *wave = ts->qm_timestep->get_wavefunction(iwave);


  float *S;
  compute_overlap_integrals(ts, expandedbasis, numprims, S);

  int i, j;
  for (i=0; i<num_wave_f; i++) {
    for (j=i; j<num_wave_f; j++) {
      printf("S12[%d,%d] = %f\n", i, j, S[i*num_wave_f+j]);
    }
  }
  int numoccorbs = wave->get_num_occupied_double();
  //  int numorbs = wave->get_num_orbitals();
  float *C = new float[numoccorbs*num_wave_f];
  const float *Ccanon = wave->get_coeffs();
  //  for (i=0; i<num_wave_f; i++) {
  //  memcpy(&C[i*numoccorbs], &Ccanon[i*numorbs],
  //         numoccorbs*sizeof(float));
  // }
  memcpy(C, Ccanon, numoccorbs*num_wave_f*sizeof(float));
  double D = mean_localization_measure(numoccorbs, C, S);
  printf("Delocalization D=%f \n", D);

  double Dold = D;
  int iter;
  for (iter=0; iter<20; iter++) {
    // Find that orbital pair for which 2x2 maximization of D
    // yields the greatest increase in total delocalization D:
    // deltaD = Dmax(ui,uj) - D(phii,phij)
    //        = Aij + sqrt(Aij^2 + Bij^2)
    // where 
    // ui, uj     = rotated orbital pair
    // phii, phij = non-rotated orbital pair 
    // Aij = gross Mulliken population of orbital pair i,j
    // Bij = see localization_rotation_angle()
    
    double deloc, maxchange = 0.0;
    int maxdelocorb1 = 0;
    int maxdelocorb2 = 0;
    for (i=0; i<numoccorbs; i++) {
      for (j=i+1; j<numoccorbs; j++) {
        deloc = pair_localization_measure(numoccorbs, i, j, C, S);
        double change = localization_orbital_change(i, j, C, S);
        printf("deloc[%d,%d] = %f change = %.7f\n", i, j, deloc, change);
        if (change>maxchange) {
          maxchange = change;
          maxdelocorb1 = i;
          maxdelocorb2 = j;
        }
      }
    }
    if (maxchange<0.000001) {
      printf("maxchange = %f\n",maxchange);
      break;
    }

    double gamma = localization_rotation_angle(C, S, maxdelocorb1,
                                              maxdelocorb2);

    printf("Rotating orbitals %d,%d by %f\n", maxdelocorb1, maxdelocorb2, gamma);
    
    rotate_2x2_orbitals(C, maxdelocorb1, maxdelocorb2, gamma);
    
    D = mean_localization_measure(numoccorbs, C, S);
    printf("Delocalization after rot[%d] D=%f \n", iter, D);

    if (fabs(D-Dold)<0.000001) break;

    Dold = D;
  }

  int b;
  int ncol = 5;
  for (b=0; b<=(numoccorbs-1)/ncol; b++) {
    for (j=0; j<num_wave_f; j++) {
      printf("%4d   ", j);
      for (i=0; i<ncol && b*ncol+i<numoccorbs; i++) {
        printf("% f  ", C[(b*ncol+i)*num_wave_f+j]);
      }
      printf("\n");
    }
    printf("\n");
  }

//  XXX dead code?
//  float occupancies[numoccorbs];
//  for (i=0; i<numoccorbs; i++) occupancies[i] = 2;

  int num_signa_ts = 0;
  wavef_signa_t *signa_ts = NULL;
  ts->qm_timestep->add_wavefunction(this, num_wave_f, numoccorbs,
                                    C, NULL, NULL, NULL, 0.0,
                                    WAVE_PIPEK, wave->get_spin(), 
                                    wave->get_excitation(),
                                    wave->get_multiplicity(),
                                    "Pipek-Mezey localization",
                                    signa_ts, num_signa_ts);
  free(signa_ts);

  delete [] S;
  delete [] C;

  return 1;
}

double QMData::localization_orbital_change(int orbid1, int orbid2,
                                          const float *C,
                                          const float *S) {
  double Ast = 0.0;
  double Bst = 0.0;
  int a;
  for (a=0; a<num_atoms; a++) {
    double QAs = gross_atom_mulliken_pop(C, S, a, orbid1);
    double QAt = gross_atom_mulliken_pop(C, S, a, orbid2);
    double QAst = orb_pair_gross_atom_mulliken_pop(C, S, a, orbid1, orbid2);
    double QAdiff = (QAs-QAt);
    Ast += QAst*QAst - 0.25*QAdiff*QAdiff;
    Bst += QAst*QAdiff;
  }

  return Ast + sqrtf(float(Ast*Ast+Bst*Bst));
}
 
// Return the delocalization of an orbital pair.
float QMData::pair_localization_measure(int num_orbitals,
                                        int orbid1, int orbid2,
                                        const float *C,
                                        const float *S) {
  int a;
  float deloc1 = 0.0;
  float deloc2 = 0.0;
  for (a=0; a<num_atoms; a++) {
    // gross Mulliken population of orbital <orbid1> on atom <a>
    float mullpop = float(gross_atom_mulliken_pop(C, S, a, orbid1));
    //printf("mullpop[%d,%d] = %f\n", a, orbid1, mullpop);
    deloc1 += mullpop*mullpop;
  }
  for (a=0; a<num_atoms; a++) {
    // gross Mulliken population of orbital <orbid2> on atom <a>
    float mullpop = float(gross_atom_mulliken_pop(C, S, a, orbid2));
    //printf("mullpop[%d,%d] = %f\n", a, orbid2, mullpop);
    deloc2 += mullpop*mullpop;
  }
  return deloc1+deloc2;
}


// Return the mean over-all localization.
double QMData::mean_localization_measure(int num_orbitals,
                                        const float *C,
                                        const float *S) {
  double Dinv = 0.0;
  int a, orbid=0;
  for (orbid=0; orbid<num_orbitals; orbid++) {
    double deloc = 0.0;

    for (a=0; a<num_atoms; a++) {
      // gross Mulliken population of orbital <orbid> on atom <a>
      float mullpop = float(gross_atom_mulliken_pop(C, S, a, orbid));

      //printf("mullpop[%d,%d] = %f\n", a, orbid, 2*mullpop);
      deloc += mullpop*mullpop;
    }
    //deloc[orbid] = deloc;
    //printf("deloc[%d] = %f\n", orbid, deloc);
    Dinv += deloc;
  }
  //Dinv /= num_orbitals;

  return Dinv;
}

/// Perform 2x2 rotation of orbital pair.
/// Equations (10a) and (10b) in Pipek & Mezey (1989).
void QMData::rotate_2x2_orbitals(float *C, int orbid1, int orbid2,
                         double gamma) {
  int num_coeffs = num_wave_f;
  float *Corb1 = &C[orbid1*num_coeffs];
  float *Corb2 = &C[orbid2*num_coeffs];
  double singamma = sin(gamma);
  double cosgamma = cos(gamma);
  int i;
  for (i=0; i<num_coeffs; i++) {
    double tmp =  cosgamma*Corb1[i] + singamma*Corb2[i];
    Corb2[i]   = float(-singamma*Corb1[i] + cosgamma*Corb2[i]);
    Corb1[i]   = float(tmp);
  }
}

/// Rotation angle gamma for the 2 by 2 orbital rotations used
/// in the orbital localization.
/// Equation (9) in Boughton & Pulay (1993).
///
/// Ast = sum_atoms(QAst^2 - 0.25*(QAs-QAt))
/// Bst = sum_atoms(QAst*(QAs-QAt))
double QMData::localization_rotation_angle(const float *C, const float *S,
                                          int orbid1, int orbid2) {
  double Ast = 0.0;
  double Bst = 0.0;
  int a;
  for (a=0; a<num_atoms; a++) {
    double QAs = gross_atom_mulliken_pop(C, S, a, orbid1);
    double QAt = gross_atom_mulliken_pop(C, S, a, orbid2);
    double QAst = orb_pair_gross_atom_mulliken_pop(C, S, a, orbid1, orbid2);
    double QAdiff = (QAs-QAt);
    Ast += QAst*QAst - 0.25*QAdiff*QAdiff;
    Bst += QAst*QAdiff;
  }
#if 0
  double T = 0.25*atan2(4.0*Bst, 4.0*Ast);
  while (T>0) {
  double sig = 1.0;
  if (T>0) sig = -1.0;
  T += sig*0.25*VMD_PI;
  printf("atan(Bst,Ast) = %f\n", T);
  if (fabs(T)<0.00001 || fabs(fabs(T)-0.25*VMD_PI)<0.00001) break;
  }
  return T;
#endif

  double sign = 1.0;
  if (Bst<0.f) sign = -1.0;

  return sign*0.25*acos(-Ast/sqrt(Ast*Ast+Bst*Bst));
}


// Return the gross Mulliken population of orbital <orbid>
// on atom <atomid>.
//
// QA = sum_i(sum_j(C_orb,i * C_orb,j * S_ij))
//
// where i runs over all coefficients related to the atom <atomid>
// and j runs over all coefficients.

// Input arrays:
// C = wavefunction coefficients
// S = overlap matrix
double QMData::gross_atom_mulliken_pop(const float *C, const float *S,
                                      int atomid, int orbid) const {
  int num_coeffs = num_wave_f;
  int first_coeff = get_wave_offset(atomid, 0);
  const float *Corb = &C[orbid*num_coeffs];
  int num_atom_coeffs = get_num_wavecoeff_per_atom(atomid);
  double QA = 0.0;
  int i, j;
  for (i=first_coeff; i<first_coeff+num_atom_coeffs; i++) {
    double Corbi = Corb[i];
    for (j=0; j<num_coeffs; j++) {
      QA += Corbi*Corb[j]*S[i*num_coeffs+j];
    }
  }

  return QA;
}


// Return the gross Mulliken population of orbital pair 
//<orbid1>/<orbid2> on atom <atomid>.
// Input arrays:
// C = wavefunction coefficients
// S = overlap matrix
double QMData::orb_pair_gross_atom_mulliken_pop(const float *C,
                                               const float *S,
                                               int atomid,
                                               int orbid1, int orbid2) {
  int num_coeffs = num_wave_f;
  int first_coeff = get_wave_offset(atomid, 0);
  const float *Corb1 = &C[orbid1*num_coeffs];
  const float *Corb2 = &C[orbid2*num_coeffs];
  int num_atom_coeffs = get_num_wavecoeff_per_atom(atomid);
  double QA = 0.f;
  int i, j;
  for (i=first_coeff; i<first_coeff+num_atom_coeffs; i++) {
    double C1mu = Corb1[i];
    double C2mu = Corb2[i];
    //printf("Cmu[%d]=%f\n", i, Cmu);
    for (j=0; j<num_coeffs; j++) {
      double C1nu = Corb1[j];
      double C2nu = Corb2[j];
      //printf("  Cnu=%f S[%d,%d]=%f\n", orbital[j], i, j, S[wave_offset+i*num_coeffs+j]);
      QA += (C1nu*C2mu+C1mu*C2nu)*S[i*num_coeffs+j];
    }
  }
  return 0.5*QA;
}

#if 0
void QMData::gross_atom_mulliken_pop_matrix(const float *C,
                                            const float *S,
                                            int atomid) {

}
#endif

// ========================================================
// Orbital rendering
// ========================================================

// Create a new Orbital object and return the pointer.
// User is responsible for deleting.
// The orbital object can be used to compute the wavefunction
// amplitude or electron density for rendering.
Orbital* QMData::create_orbital(int iwave, int orbid, float *pos,
                         QMTimestep *qmts) {
  Orbital *orbital = new Orbital(pos,
                  qmts->get_wavecoeffs(iwave),
                  basis_array, basis_set, atom_types,
                  atom_sort, atom_basis,
                  (const float**)norm_factors,
                  num_shells_per_atom,
                  num_prim_per_shell, shell_types,
                  num_atoms, num_types, num_wave_f, num_basis,
                  orbid);
  return orbital;
}



// =========================================
// Currently unused stuff I might need later
// =========================================


#if 0                           // XXX: unused
// XXX these quicksort routines are duplicates of the ones
// in QMTimestep.
static void quicksort(const int *tag, int *A, int p, int r);
static int  quickpart(const int *tag, int *A, int p, int r);

// Create an index array *atom_sort that sorts the atoms
// by basis set type (usually that means by atomic number).
void QMData::sort_atoms_by_type() {
  int i;
  if (atom_sort) delete [] atom_sort;

  // Initialize the index array;
  atom_sort = new int[num_atoms];
  for (i=0; i<num_atoms; i++) {
    atom_sort[i] = i;
  }

  // Sort index array according to the atom_types
  quicksort(atom_types, atom_sort, 0, num_atoms-1);

  //int *sorted_types = new int[num_atoms];

  // Copy data over into sorted arrays
  //for (i=0; i<num_atoms; i++) {
  //  sorted_types[i] = atom_types[atom_sort[i]];
  //}
}

// The standard quicksort algorithm except for it doesn't
// sort the data itself but rather sorts array of ints *A
// in the same order as it would sort the integers in array
// *tag. Array *A can then be used to reorder any array
// according to the string tags.
// Example:
// tag:   BC DD BB AA  -->  AA BB BC DD
// index:  0  1  2  3  -->   3  2  0  1
//
static void quicksort(const int* tag, int *idx, int p, int r) {
  int q;
  if (p < r) {
    q=quickpart(tag, idx, p, r);
    quicksort(tag, idx, p, q);
    quicksort(tag, idx, q+1, r);
  }
}

// Partitioning for quicksort.
static int quickpart(const int *tag, int *idx, int p, int r) {
  int i, j;
  int tmp;
  int x = tag[idx[p]];
  i = p-1;
  j = r+1;

  while (1) {
    // Find highest element smaller than idx[p]
    do j--; while (tag[idx[j]] > x);

    // Find lowest element larger than idx[p]
    do i++; while (tag[idx[i]] < x);

    if (i < j) {
      tmp    = idx[i];
      idx[i] = idx[j];
      idx[j] = tmp;
    }
    else {
      return j;
    }
  }
}
#endif
