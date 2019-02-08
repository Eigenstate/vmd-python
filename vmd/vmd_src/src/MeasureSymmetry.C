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
 *	$RCSfile: MeasureSymmetry.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.65 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * The Symmetry class is the work horse behind the "measure symmetry"
 * command which guesses the pointgroup of a selection and returns the
 * according symmetry elements (mirror planes rotary axes, rotary reflection
 * axes). The algorithm is is fairly forgiving about molecules where atoms
 * are perturbed from the ideal position and tries its best to guess the
 * correct point group anyway. 
 * The tolerance can be controlled with the sigma parameter which is
 * the average allowed deviation from the ideal position.
 * Works nice on my 30 non-patholocical test cases. A pathological case
 * would for instance be a system with more than only a few atoms (say 15)
 * where only one atom distinguishes between two point groups.
 * If your selection contains more than a certain number of atoms then
 * only that much randomly chosen atoms are used to find planes and axes
 * in order to save time.
 *
 ***************************************************************************/

#include <stdio.h>
#include <math.h>
#include "utilities.h"
#include "Measure.h"
#include "MeasureSymmetry.h"
#include "AtomSel.h"
#include "Matrix4.h"
#include "MoleculeList.h"
#include "SpatialSearch.h"     // for vmd_gridsearch3()
#include "MoleculeGraphics.h"  // needed only for debugging
#include "Inform.h"
#include "WKFUtils.h"

#define POINTGROUP_UNKNOWN 0
#define POINTGROUP_C1      1
#define POINTGROUP_CN      2
#define POINTGROUP_CNV     3
#define POINTGROUP_CNH     4
#define POINTGROUP_CINFV   5
#define POINTGROUP_DN      6
#define POINTGROUP_DND     7
#define POINTGROUP_DNH     8
#define POINTGROUP_DINFH   9
#define POINTGROUP_CI     10
#define POINTGROUP_CS     11
#define POINTGROUP_S2N    12
#define POINTGROUP_T      13
#define POINTGROUP_TD     14  
#define POINTGROUP_TH     15
#define POINTGROUP_O      16
#define POINTGROUP_OH     17
#define POINTGROUP_I      18
#define POINTGROUP_IH     19
#define POINTGROUP_KH     20

#define OVERLAPCUTOFF 0.4

// The bondsum is the sum of all normalized bond vectors
// for an atom. If the endpoints of the bondsum vectors
// of two atoms are further apart than BONDSUMTOL, then
// The two atoms are not considered images of each other
// with respect to the transformation that was tested.
#define BONDSUMTOL    1.5

// Minimum atom distance before the sanity check for
// idealized coordinates fails.
#define MINATOMDIST   0.6

// Mirror plane classification
#define VERTICALPLANE   1
#define DIHEDRALPLANE   3
#define HORIZONTALPLANE 4

// Special axis order values
#define INFINITE_ORDER -1
#define PRELIMINARY_C2 -2

// Flags for requesting special geometries for idealize_angle()
#define TETRAHEDRON  -4
#define OCTAHEDRON   -8
#define DODECAHEDRON -12



#if defined(_MSC_VER)
// Microsoft's compiler lacks erfc() so we make our own here:
static double myerfc(double x) {
  double p, a1, a2, a3, a4, a5;
  double t, erfcx;

  p  =  0.3275911;
  a1 =  0.254829592;
  a2 = -0.284496736;
  a3 =  1.421413741;
  a4 = -1.453152027;
  a5 =  1.061405429;

  t = 1.0 / (1.0 + p*x);
  erfcx = ((a1 + (a2 + (a3 + (a4 + a5*t)*t)*t)*t)*t) * exp(-pow(x,2.0));
  return erfcx;
}
#endif


// Forward declarations of helper functions
// static inline bool coplanar(const float *normal1, const float *normal2, float tol);
static inline bool collinear(const float *axis1, const float *axis2, float tol);
static inline bool orthogonal(const float *axis1, const float *axis2, float tol);
static inline bool behind_plane(const float *normal, const float *coor);
static int isprime(int x);
static int numprimefactors(int x);
static void align_plane_with_axis(const float *normal, const float *axis, float *alignedplane);
static void assign_atoms(AtomSel *sel, MoleculeList *mlist, float *(&mycoor), int *(&atomtype));
static float trans_overlap(int *atomtype, float *(&coor), int numcoor,
                           const Matrix4 *trans, float sigma,
                           bool skipident, int maxnatoms, float &overlappermatch);
static inline float trans_overlap(int *atomtype, float *(&coor), int numcoor,
                                  const Matrix4 *trans, float sigma,
                                  bool skipident, int maxnatoms);


// Store the symmetry characteristics for a structure
typedef struct {
  int pointgroup;
  int pointgrouporder;
  float rmsd; 
  float *idealcoor;
  Plane *planes;
  Axis  *axes;
  Axis  *rotreflections;
  bool linear;        ///< is molecule linear?
  bool planar;        ///< is molecule planar?
  bool inversion;     ///< do we have an inversion center?
  bool sphericaltop;  ///< is it a spherical top?
  bool symmetrictop;  ///< is it a symmetrical top?
} Best;


// Symmetry elements defining a pointgroup
typedef struct {
  char  name[6]; // name string
  ElementSummary summary;   
} PointGroupDefinition;

PointGroupDefinition pgdefinitions[] = {
//        inv sig Cinf      C2 C3 C4 C5 C6 C7 C8   S3        S8              S16
  { "C1",  {0, 0, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "Cs",  {0, 1, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "Ci",  {1, 0, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C2",  {0, 0, 0, {0, 0, 1, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C3",  {0, 0, 0, {0, 0, 0, 1, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C4",  {0, 0, 0, {0, 0, 1, 0, 1, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C5",  {0, 0, 0, {0, 0, 0, 0, 0, 1, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C6",  {0, 0, 0, {0, 0, 1, 1, 0, 0, 1, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C7",  {0, 0, 0, {0, 0, 0, 0, 0, 0, 0, 1, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C8",  {0, 0, 0, {0, 0, 1, 0, 1, 0, 0, 0, 1}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D2",  {0, 0, 0, {0, 0, 3, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D3",  {0, 0, 0, {0, 0, 3, 1, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D4",  {0, 0, 0, {0, 0, 5, 0, 1, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D5",  {0, 0, 0, {0, 0, 5, 0, 0, 1, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D6",  {0, 0, 0, {0, 0, 7, 1, 0, 0, 1, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D7",  {0, 0, 0, {0, 0, 7, 0, 0, 0, 0, 1, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D8",  {0, 0, 0, {0, 0, 9, 0, 1, 0, 0, 0, 1}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C2v", {0, 2, 0, {0, 0, 1, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C3v", {0, 3, 0, {0, 0, 0, 1, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C4v", {0, 4, 0, {0, 0, 1, 0, 1, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C5v", {0, 5, 0, {0, 0, 0, 0, 0, 1, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C6v", {0, 6, 0, {0, 0, 1, 1, 0, 0, 1, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C7v", {0, 7, 0, {0, 0, 0, 0, 0, 0, 0, 1, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C8v", {0, 8, 0, {0, 0, 1, 0, 1, 0, 0, 0, 1}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C2h", {1, 1, 0, {0, 0, 1, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C3h", {0, 1, 0, {0, 0, 0, 1, 0, 0, 0, 0, 0}, {1,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C4h", {1, 1, 0, {0, 0, 1, 0, 1, 0, 0, 0, 0}, {0,1,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C5h", {0, 1, 0, {0, 0, 0, 0, 0, 1, 0, 0, 0}, {0,0,1,0,0,0,0,0,0,0,0,0,0,0}} },
  { "C6h", {1, 1, 0, {0, 0, 1, 1, 0, 0, 1, 0, 0}, {0,0,0,1,0,0,0,0,0,0,0,0,0,0}} },
  { "C7h", {0, 1, 0, {0, 0, 0, 0, 0, 0, 0, 1, 0}, {0,0,0,0,1,0,0,0,0,0,0,0,0,0}} },
  { "C8h", {1, 1, 0, {0, 0, 1, 0, 1, 0, 0, 0, 1}, {0,1,0,0,0,1,0,0,0,0,0,0,0,0}} },
  { "D2h", {1, 3, 0, {0, 0, 3, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D3h", {0, 4, 0, {0, 0, 3, 1, 0, 0, 0, 0, 0}, {1,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D4h", {1, 5, 0, {0, 0, 5, 0, 1, 0, 0, 0, 0}, {0,1,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D5h", {0, 6, 0, {0, 0, 5, 0, 0, 1, 0, 0, 0}, {0,0,1,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D6h", {1, 7, 0, {0, 0, 7, 1, 0, 0, 1, 0, 0}, {1,0,0,1,0,0,0,0,0,0,0,0,0,0}} },
  { "D7h", {0, 8, 0, {0, 0, 7, 0, 0, 0, 0, 1, 0}, {0,0,0,0,1,0,0,0,0,0,0,0,0,0}} },
  { "D8h", {1, 9, 0, {0, 0, 9, 0, 1, 0, 0, 0, 1}, {0,1,0,0,0,1,0,0,0,0,0,0,0,0}} },
  { "D2d", {0, 2, 0, {0, 0, 3, 0, 0, 0, 0, 0, 0}, {0,1,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "D3d", {1, 3, 0, {0, 0, 3, 1, 0, 0, 0, 0, 0}, {0,0,0,1,0,0,0,0,0,0,0,0,0,0}} },
  { "D4d", {0, 4, 0, {0, 0, 5, 0, 1, 0, 0, 0, 0}, {0,0,0,0,0,1,0,0,0,0,0,0,0,0}} },
  { "D5d", {1, 5, 0, {0, 0, 5, 0, 0, 1, 0, 0, 0}, {0,0,0,0,0,0,0,1,0,0,0,0,0,0}} },
  { "D6d", {0, 6, 0, {0, 0, 7, 1, 0, 0, 1, 0, 0}, {0,1,0,0,0,0,0,0,0,1,0,0,0,0}} },
  { "D7d", {1, 7, 0, {0, 0, 7, 0, 0, 0, 0, 1, 0}, {0,0,0,0,0,0,0,0,0,0,0,1,0,0}} },
  { "D8d", {0, 8, 0, {0, 0, 9, 0, 1, 0, 0, 0, 1}, {0,0,0,0,0,0,0,0,0,0,0,0,0,1}} },
  { "S4",  {0, 0, 0, {0, 0, 1, 0, 0, 0, 0, 0, 0}, {0,1,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "S6",  {1, 0, 0, {0, 0, 0, 1, 0, 0, 0, 0, 0}, {0,0,0,1,0,0,0,0,0,0,0,0,0,0}} },
  { "S8",  {0, 0, 0, {0, 0, 1, 0, 1, 0, 0, 0, 0}, {0,0,0,0,0,1,0,0,0,0,0,0,0,0}} },
  { "T",   {0, 0, 0, {0, 0, 3, 4, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "Th",  {1, 3, 0, {0, 0, 3, 4, 0, 0, 0, 0, 0}, {0,0,0,4,0,0,0,0,0,0,0,0,0,0}} },
  { "Td",  {0, 6, 0, {0, 0, 3, 4, 0, 0, 0, 0, 0}, {0,3,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "O",   {0, 0, 0, {0, 0, 9, 4, 3, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "Oh",  {1, 9, 0, {0, 0, 9, 4, 3, 0, 0, 0, 0}, {0,3,0,4,0,0,0,0,0,0,0,0,0,0}} },
  { "I",   {0, 0, 0, {0, 0,15,10, 0, 6, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "Ih",  {1,15, 0, {0, 0,15,10, 0, 6, 0, 0, 0}, {0,0,0,10,0,0,0,6,0,0,0,0,0,0}} },
  { "Cinfv", {0, 1, 1, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "Dinfh", {1, 2, 1, {0, 0, 1, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
  { "Kh",    {1, 1, 1, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0}} },
};

#define NUMPOINTGROUPS (sizeof(pgdefinitions)/sizeof(PointGroupDefinition))


Symmetry::Symmetry(AtomSel *mysel, MoleculeList *mymlist, int verbosity) :
  sel(mysel),
  mlist(mymlist),
  verbose(verbosity)
{
  sigma = 0.1f;
  sphericaltop = 0;
  symmetrictop = 0;
  linear = 0;
  planar = 0;
  inversion = 0;
  maxaxisorder = 0;
  maxrotreflorder = 0;
  maxweight  = 0;
  maxoverlap = 0.0;
  pointgroup = POINTGROUP_UNKNOWN;
  pointgrouporder = 0;
  numdihedralplanes = 0;
  numverticalplanes = 0;
  horizontalplane = -1;
  maxnatoms = 200;
  rmsd = 0.0;
  idealcoor = NULL;
  coor = NULL;
  checkbonds = 0;
  bondsum = NULL;
  bondsperatom = NULL;
  atomtype = NULL;
  atomindex = NULL;
  uniqueatoms = NULL;
  elementsummarystring = NULL;
  missingelementstring    = new char[2 + 10L*(3+MAXORDERCN+2*MAXORDERCN)];
  additionalelementstring = new char[2 + 10L*(3+MAXORDERCN+2*MAXORDERCN)];
  missingelementstring[0]    = '\0';
  additionalelementstring[0] = '\0';

  memset(&elementsummary, 0, sizeof(ElementSummary));

  if (sel->selected) {
    coor      = new float[3L*sel->selected];
    idealcoor = new float[3L*sel->selected];
    bondsum   = new float[3L*sel->selected];
    bondsperatom = new Bondlist[sel->selected];
    atomtype    = new int[sel->selected];
    atomindex   = new int[sel->selected];
    uniqueatoms = new int[sel->selected];

    memset(bondsperatom, 0, sel->selected*sizeof(Bondlist));
  }

  memset(&(inertiaaxes[0][0]),  0, 9L*sizeof(float));
  memset(&(inertiaeigenval[0]), 0, 3L*sizeof(float));
  memset(&(uniqueprimary[0]),   0, 3L*sizeof(int));
  memset(&(rcom[0]), 0, 3L*sizeof(float));

  // Copy coordinates of selected atoms into local array and assign
  // and atomtypes based on chemial element and topology.
  assign_atoms(sel, mlist, coor, atomtype);

   // Determine the bond topology
  assign_bonds();


  // default collinearity tolerance ~0.9848
  collintol = cosf(float(DEGTORAD(10.0f)));

  // default coplanarity tolerance ~0.1736
  orthogtol = cosf(float(DEGTORAD(80.0f)));

  if (sel->selected <= 10) {
    collintol = cosf(float(DEGTORAD(15.0f)));
    orthogtol = cosf(float(DEGTORAD(75.0f)));
  }
}



Symmetry::~Symmetry(void) {
  if (coor)           delete [] coor;
  if (idealcoor)      delete [] idealcoor;
  if (bondsum)        delete [] bondsum;
  if (atomtype)       delete [] atomtype;
  if (atomindex)      delete [] atomindex;
  if (uniqueatoms)    delete [] uniqueatoms;
  if (elementsummarystring) delete [] elementsummarystring;
  delete [] missingelementstring;
  delete [] additionalelementstring;
  if (bondsperatom) {
    int i;
    for (i=0; i<sel->selected; i++) {
      if (bondsperatom[i].bondto) delete [] bondsperatom[i].bondto;
      if (bondsperatom[i].length) delete [] bondsperatom[i].length;
    }
    delete [] bondsperatom;
  }
}

/// Get the guessed point group and order in form of a nice
/// string like C3, D2d, C4v, Dinfh, Cs, T, Oh, ...
void Symmetry::get_pointgroup(char pg[8], int *order) {
  char n[3];
  if (order==NULL) sprintf(n, "%i", pointgrouporder);
  else {
    strcpy(n, "n");
    *order = pointgrouporder;
  }

  switch (pointgroup) {
  case POINTGROUP_UNKNOWN:
    strcpy(pg, "Unknown");
    break;
  case POINTGROUP_C1:
    strcpy(pg, "C1");
    break;
  case POINTGROUP_CN:
    sprintf(pg, "C%s", n);
    break;
  case POINTGROUP_CNV:
    sprintf(pg, "C%sv", n);
    break;
  case POINTGROUP_CNH:
    sprintf(pg, "C%sh", n);
    break;
  case POINTGROUP_CINFV:
    strcpy(pg, "Cinfv");
    break;
  case POINTGROUP_DN:
    sprintf(pg, "D%s", n);
    break;
  case POINTGROUP_DND: 
    sprintf(pg, "D%sd", n);
    break;
  case POINTGROUP_DNH:
    sprintf(pg, "D%sh", n);
    break;
  case POINTGROUP_DINFH:
    strcpy(pg, "Dinfh");
    break;
  case POINTGROUP_CI:
    strcpy(pg, "Ci");
    break;
  case POINTGROUP_CS:
    strcpy(pg, "Cs");
    break;
  case POINTGROUP_S2N:
    if (!strcmp(n, "n")) 
      strcpy(pg, "S2n");
    else
      sprintf(pg, "S%i", 2*pointgrouporder);
    break;
  case POINTGROUP_T:
    strcpy(pg, "T");
    break;
  case POINTGROUP_TD:
    strcpy(pg, "Td");
    break;
  case POINTGROUP_TH:
    strcpy(pg, "Th");
    break;
  case POINTGROUP_O:
    strcpy(pg, "O");
    break;
  case POINTGROUP_OH: 
    strcpy(pg, "Oh");
    break;
  case POINTGROUP_I:
    strcpy(pg, "I");
    break;
  case POINTGROUP_IH:
    strcpy(pg, "Ih");
    break;
  case POINTGROUP_KH:
    strcpy(pg, "Kh");
    break;
  }
}


/// Get order of specified axis
int Symmetry::get_axisorder(int n) {
  if (n<numaxes()) return axes[n].order;
  return 0;
}

/// Get order of specified rotary reflection
int Symmetry::get_rotreflectorder(int n) {
  if (n<numrotreflect()) return rotreflections[n].order;
  return 0;
}

/// Return the unmber of S2N rotary reflection (Sn with even n)
int Symmetry::numS2n() {
  int i=0, count=0;
  for (i=0; i<numrotreflect(); i++) {
    if (rotreflections[i].order % 2 == 0) count++;
  }
  return count;
}

/// Count the number of primary axes
int Symmetry::numprimaryaxes() {
  int i;
  int numprimary = 0;
  for (i=0; i<numaxes(); i++) {
    if (axes[i].order==maxaxisorder) numprimary++;
  }
  return numprimary;
}


// Impose certain symmetry elements on structure
// by wrapping coordinates around and averaging them.
// This creates symmetry, if there was one before or not.
// If mutually incompatible elements are specified the
// result is hard to predict:
// Planes are idealized first, then rotary axes, then
// rotary reflections and last an inversion.
void Symmetry::impose(int have_inversion, 
                      int nplanes, const float *planev,
                      int naxes, const float *axisv, const int *axisorder,
                      int nrotrefl, const float* rotreflv,
                      const int *rotreflorder) {
  int i;
  char buf[256];
  if (have_inversion) {
    inversion = 1;
    if (verbose>1) {
      sprintf(buf, "imposing inversion\n");
      msgInfo << buf << sendmsg;
    }
  }
  planes.clear();
  for (i=0; i<nplanes; i++) {
    Plane p;
    vec_copy(p.v, &planev[3L*i]);
    vec_normalize(p.v);
    if (norm(p.v)==0.f) continue;
    p.overlap = 1.f;
    p.weight = 1;
    p.type = 0;
    planes.append(p);
    if (verbose>1) {
      sprintf(buf, "imposing plane {% .2f % .2f % .2f}\n", p.v[0], p.v[1], p.v[2]);
      msgInfo << buf << sendmsg;
    }
  }
  axes.clear();
  for (i=0; i<naxes; i++) {
    if (axisorder[i]<=1) continue;
    Axis a;
    vec_copy(a.v, &axisv[3L*i]);
    vec_normalize(a.v);
    if (norm(a.v)==0.f) continue;
    a.order = axisorder[i];
    a.overlap = 1.f;
    a.weight = 1;
    a.type = 0;
    axes.append(a);
    if (verbose>1) {
      sprintf(buf, "imposing axis {% .2f % .2f % .2f}\n", a.v[0], a.v[1], a.v[2]);
      msgInfo << buf << sendmsg;
    }
  }
  rotreflections.clear();
  for (i=0; i<nrotrefl; i++) {
    if (rotreflorder[i]<=1) continue;
    Axis a;
    vec_copy(a.v, &rotreflv[3L*i]);
    vec_normalize(a.v);
    if (norm(a.v)==0.f) continue;
    a.order = rotreflorder[i];
    a.overlap = 1.f;
    a.weight = 1;
    a.type = 0;
    rotreflections.append(a);
    if (verbose>1) {
      sprintf(buf, "imposing rraxis {% .2f % .2f % .2f}\n", a.v[0], a.v[1], a.v[2]);
      msgInfo << buf << sendmsg;
    }
  }

  // Abuse measure_inertia() to update the center of mass
  float itensor[4][4];
  int ret = measure_inertia(sel, mlist, coor, rcom, inertiaaxes,
                            itensor, inertiaeigenval);
  if (ret < 0) msgErr << "measure inertia failed with code " << ret << sendmsg;
  
  //printf("rcom={%.2f %.2f %.2f}\n", rcom[0], rcom[1], rcom[2]);

  // Assign the bondsum vectors
  assign_bondvectors();


  for (i=0; i<2; i++) {

  // During idealization the coordinates are wrapped and
  // averaged.
  idealize_coordinates();

  // Use improved coordinates
  memcpy(coor, idealcoor, 3L*sel->selected*sizeof(float));
  }
}


/// Determine the symmetry point group and all symmetry
/// elements.
int Symmetry::guess(float mysigma) {
  if (!sel)               return MEASURE_ERR_NOSEL;
  if (sel->selected == 0) return MEASURE_ERR_NOATOMS;

  if (sel->selected == 1) {
    pointgroup = POINTGROUP_KH;
    elementsummarystring = new char[1];
    elementsummarystring[0] = '\0';
    uniqueatoms[0] = 1;
    memcpy(idealcoor, coor, 3L*sel->selected*sizeof(float));

    return MEASURE_NOERR;
  }
   

  float maxsigma = mysigma;

  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);

  // We don't know beforehand what is the ideal (sigma) tolerance
  // value to use in oder to find the right point group. If we
  // choose it too low we might find only a subgroup of the real 
  // pointgroup (e.g. Cs instead of D2h). However, if we choose
  // it too high we will find impossible combinations of symmetry
  // elements caused by false positives.
  // Thus we slowly ramp up the tolerance in steps of 0.05 until
  // we reach the user-specified maximum. This has some advantages:
  // After each step where we find a new non-C1 symmetry we'll
  // idealize the coordinatesand get a more symmetric structure
  // even if we just found a subgroup of the real one.
  // At some point we will find the higher symmetry point group.
  // But how do we know when to stop?
  // After each step the sanity of the idealized coordinates is
  // checked. Wrong point groups (i.e. wrong symmetry elements)
  // will screw up the idealized coordinates and bondsums which
  // can be easily detected. When this occcurs our search has
  // finished and we go back to the last sane symmetry.
  // The step where this symmetry occured the first time is
  // an indicator of the quality of the guess. The earlier we
  // found the right point group the less we had to bend the
  // coodinates to make them symmetric and the more trustworthy
  // is the guess.
  int beststep = -1;
  Best best;
  best.pointgroup = POINTGROUP_UNKNOWN;
  best.pointgrouporder = 0;
  best.idealcoor = NULL;
  int step, nstep;
  float stepsize = 0.05f;
  nstep = int(0.5f+(float)maxsigma/stepsize);

  for (step=0; step<nstep; step++) {
    sigma = (1+step)*stepsize;
    if (verbose>1) {
      msgInfo << sendmsg;
      msgInfo << "  STEP " << step << "  sigma=" << sigma << sendmsg
              << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << sendmsg << sendmsg;
    }


    // Self consistent iteration of symmetry element search
    if (!iterate_element_search()) continue;

    wkf_timer_stop(timer);
    if (verbose>0) {
      char buf[256];
      sprintf(buf, "Guess %i completed in %f seconds, RMSD=%.3f", step, wkf_timer_time(timer), rmsd);
      msgInfo << buf << sendmsg;
    }


    // If the idealized coordinates are not sane, i.e.
    // there are atoms closer than MINATOMDIST we stop.
    // Coordinates typically collapse during idealization
    // when false symmetry elements were found.
    // We can stop the search because with a higher tolerance
    // in the next step we are going to find the same bad
    // symmetry element again (and maybe even more).
    if (!ideal_coordinate_sanity()) {
      if (verbose>1) {
        msgInfo << "INSANE IDEAL COORDINATES" << sendmsg;
      }
      break;
    }

    if (verbose>1) {
      print_statistics();
    }

    // Determine the point group from symmetry elements
    determine_pointgroup();

    char pgname[8];
    get_pointgroup(pgname, NULL);
    compare_element_summary(pgname);

    if (verbose>1) {
      msgInfo << "Point group:     " << pgname << sendmsg << sendmsg;
      print_element_summary(pgname);
      //msgInfo << "Level:           " << pointgroup_rank(pointgroup, pointgrouporder)
      //        << sendmsg;
    }

    // Try to decide wether the current step yielded a
    // better guess:
    // Whenever we detect a new pointgroup we have to
    // compare the ranks in order to determine which one
    // has higher symmetry. Often we find a subgroup
    // of the actual point group first. When the tolerance 
    // has increased we find the real point group which
    // will have a higher rank than the subgroup.
    if (pointgroup!=best.pointgroup &&
        !strlen(additionalelementstring) &&
        !strlen(missingelementstring) &&
        pointgroup_rank(pointgroup, pointgrouporder) >
        pointgroup_rank(best.pointgroup, best.pointgrouporder)) {
      beststep = step;
      best.pointgroup = pointgroup;
      best.pointgrouporder = pointgrouporder;
      if (!best.idealcoor) best.idealcoor = new float[3L*sel->selected];
      memcpy(best.idealcoor, idealcoor, 3L*sel->selected*sizeof(float));
      best.linear = linear;
      best.planar = planar;
      best.inversion = inversion;
      best.sphericaltop = sphericaltop;
    }
  }

  // Reset sigma and idealized coordinates to the values of 
  // the best step and recompute the symmetry elements.
  sigma = (1+beststep)*stepsize;
  if (best.idealcoor) {
    memcpy(coor, best.idealcoor, 3L*sel->selected*sizeof(float));
    delete [] best.idealcoor;
  }
  iterate_element_search();

  // Determine the point group from symmetry elements
  determine_pointgroup();

  if (verbose>0) {
    msgInfo << sendmsg
            << "RESULT:" << sendmsg
            << "=======" << sendmsg << sendmsg;

    msgInfo << "Needed tol = " << sigma << " to detect symmetry."
            << sendmsg;

    print_statistics();
    char pgname[8];
    get_pointgroup(pgname, NULL);
    compare_element_summary(pgname);
    msgInfo << "Point group:     " << pgname << sendmsg << sendmsg;
    print_element_summary(pgname);
  }


  // Sort planes in the order horizontal, dihedral, vertical, rest
  sort_planes();

  // Determine the unique coordinates of the molecule
  unique_coordinates();

  // Try to get a transformation matrix that orients the
  // molecule according to the GAMESS standard orientation.
  orient_molecule();


  wkf_timer_destroy(timer);

  return MEASURE_NOERR;
}


// Self consistent iteration of symmetry element search
// ----------------------------------------------------
// If we would base our point group guess on elements that were
// found in a single pass, then the biggest problem is to decide
// which symmetry elements to keep and which to discard, based
// on their overlap score.
// While a point group guess, relying only on certain key
// features, can be quite good, the number of found elements would
// often be wrong. In case additional elements were found this
// could totally screw up the idealized coordinates.
// To avoid these problems we are iterate the element guess.
// In each iteration we only keep the elements with a very good
// score. This ensures that we don't get false positives. We then
// idealize the symmetry elements which means we correct the
// angles between then (i.e. an 87deg angle between two planes
// becomes 90deg). Now all atoms will be wrapped around each
// symmetry element and their positions averaged with their images
// which gives us an updated set of idealized coordinates that will
// be "more symmetric" than the coordinates from the previous
// generation. In the next pass we use these improved coordinates
// for guessing the elements and will probably find a few more this
// time. This is repeated until the number of found symmetry elements
// converges.
int Symmetry::iterate_element_search() {
  int iteration;
  char oldelementsummarystring[2 + 10*(3+MAXORDERCN+2*MAXORDERCN)];
  oldelementsummarystring[0] = '\0';
  float oldrmsd = 999999.9f;
  int converged = 0;

  for (iteration=0; iteration<20; iteration++) {
    // In each iteration we start with a clean slate
    pointgroup = POINTGROUP_UNKNOWN;
    inversion = 0;
    linear = 0;
    planar = 0;
    sphericaltop = 0;
    symmetrictop = 0;
    axes.clear();
    planes.clear();
    rotreflections.clear();
    bonds.clear();
    numdihedralplanes = 0;
    numverticalplanes = 0;
    horizontalplane   = -1;

    if (verbose>2) {
      msgInfo << sendmsg 
              << "###################" << sendmsg;
      msgInfo << "Iteration " << iteration << sendmsg;
      msgInfo << "###################" << sendmsg << sendmsg;
    }

    // Reassign the bondsum vectors
    assign_bondvectors();

    // Determine primary axes of inertia
    float itensor[4][4];
    int ret = measure_inertia(sel, mlist, coor, rcom, inertiaaxes,
                              itensor, inertiaeigenval);
    if (ret < 0) msgErr << "measure inertia failed with code " << ret << sendmsg;


    if (verbose>2) {
      char buf[256];
      sprintf(buf, "eigenvalues: %.2f  %.2f  %.2f", inertiaeigenval[0], inertiaeigenval[1], inertiaeigenval[2]);
      msgInfo << buf << sendmsg;
    }


    // Find all rotary axes, mirror planes and rotary reflection axes
    find_symmetry_elements();

    if (verbose>2) {
      if (linear)
        msgInfo << "Linear molecule" << sendmsg;
      if (planar)
        msgInfo << "Planar molecule" << sendmsg;
      if (sphericaltop)
        msgInfo << "Molecule is a spherical top." << sendmsg;
      if (symmetrictop)
        msgInfo << "Molecule is a symmetric top." << sendmsg;

      int i, numuniqueprimary = 0;
      for (i=0; i<3; i++) {
        if (uniqueprimary[i]) numuniqueprimary++;
      }
      msgInfo << "Number of unique primary axes = " << numuniqueprimary << sendmsg;

    }

    // Now we have a set of symmetry elements but since they were
    // constructed from the possibly inacccurate molecular coordinates
    // their relationship is not ideal. For example the angle between
    // mirror planes parallel to a 3-fold rotation axis will only
    // approximately be 120 degree. We have to idealize the symmetry
    // elements so that we have a basis to idealize the coordinates
    // and determine the symmetry unique atoms.
    idealize_elements();

    // Generate symmetricized coordinates by wrapping atoms around
    // the symmetry elements and average with these images with the 
    // original positions.
    idealize_coordinates();

    // Should check the sanity of the idealized coordinates
    // here. I.e. check if there are atoms closer than, 0.6 A
    // and check if the bondsums match with the original geometry.

    // Compute RMSD between original and idealized coordinates
    rmsd = ideal_coordinate_rmsd();

    // Update the symmetry element statistics
    build_element_summary();

    // Create a string representation of the summary
    // Example: (inv) 2*(sigma) (Cinf) (C2)
    build_element_summary_string(elementsummary, elementsummarystring);

    // Iterations have converged when the rmsd difference between this and
    // the last structure is below a threshold and the symmetry elements
    // haven't changed
    converged = 0;    
    if (fabs(rmsd-oldrmsd)<0.01 && 
        !strcmp(elementsummarystring, oldelementsummarystring)) {
      if (verbose>1) {
        msgInfo << "Symmetry search converged after " << iteration+1 
                << " iterations" << sendmsg;
      }
      converged = 1;
    }
    

    if (verbose>3 && (converged || verbose>2)) {
      print_statistics();
    }
    if (converged) break;

    // Use improved coordinates for the next pass
    memcpy(coor, idealcoor, 3L*sel->selected*sizeof(float));

    oldrmsd = rmsd;
    strcpy(oldelementsummarystring, elementsummarystring);
  }

  if (!converged) return 0;
  return 1;
}


// Find all mirror planes, rotary axes, rotary reflections.
void Symmetry::find_symmetry_elements() {

  // ----- Planes -----------------

  // Check if the unique primary axes of inertia correspond to rotary axes
  find_elements_from_inertia();

  // Find all mirror planes
  find_planes();

  // Remove planes with too bad overlap
  purge_planes(0.5);

  // Are there horizontal/vertical/dihedral mirror planes?
  classify_planes();

  // Check for an inversion center
  check_add_inversion();
    

  // ------ Axes ------------------

  // Find all rotary axes that are intersections of planes.
  find_axes_from_plane_intersections();

  // Determine the order of each axis
  assign_axis_orders();

  // Some C2 axes cannot be found from principle axes of inertia
  // or through plane intersection. We must get these from the
  // atomic coordinates.
  find_C2axes();

  assign_prelimaxis_orders(2);

  if (!axes.num() && !planes.num()) {
    int j;
    for (j=3; j<=8; j++) {
      if (isprime(j) && !axes.num()) {
        find_axes(j);
        assign_prelimaxis_orders(j);
      }
    }
  }

  // Sort axes by decreasing order
  sort_axes();


  // ----- Rotary reflections -----

  // Since all rotary reflections are collinear with a rotation axis
  // we can simply go through the list of axes and check if which
  // axes correspond to improper rotations.
  int i;
  for (i=0; i<numaxes(); i++) {
    check_add_rotary_reflection(axes[i].v, 2*axes[i].order);
  }

  // Normalize the summed up rotary reflection vectors
  maxrotreflorder = 0;
  for (i=0; i<numrotreflect(); i++) {
    vec_normalize(rotreflections[i].v);

    if (rotreflections[i].order>maxrotreflorder)
      maxrotreflorder = rotreflections[i].order;     
  }


  // Remove axes with too bad overlap
  purge_axes(0.7f);
  purge_rotreflections(0.75f);

  // Must classify planes again because of dihedral planes depending
  // on the new axes:
  classify_planes();

  // Classify perpendicular axes
  classify_axes();
}


// Determine principle axes of inertia, check them for uniqueness
// by comparing the according eigenvalues. Then, see if the unique
// axes correspond to rotary axes or rotary reflections and, in
// case, add axes to the list and assing the order.
// We also check if there are horizontal mirror planes for the
// principal axes.
void Symmetry::find_elements_from_inertia() {
  int i;
  int numuniqueprimary = 0;
  memset(uniqueprimary, 0, 3L*sizeof(int));

  // Normalize eigenvalues
  float eigenval[3];
  float e0 = inertiaeigenval[0];
  for (i=0; i<3; i++) {
    eigenval[i] = inertiaeigenval[i]/e0;
  }

  // Check if the molecule is linear
  if (fabs(eigenval[0]-eigenval[1])<0.05 && eigenval[2]<0.05) {
    linear = 1;
  }

  // Get provisional info about the rotational order of the inertia axes
  float overlap[3];
  int order[3], primaryCn[3];
  for (i=0; i<3; i++) {
    order[i] = axis_order(inertiaaxes[i], overlap+i);
    //printf("primary axis of inertia %i: order=%i overlap=%.2f\n", i, order[i], overlap[i]);
    if ((order[i]>1 && overlap[i]>1.5*OVERLAPCUTOFF) ||
        (linear && eigenval[i]<0.05)) {
      primaryCn[i] = 1;
    } else {
      primaryCn[i] = 0;
    }
  }

  float tol = 0.25f*sigma + 1.0f/(powf(float(sel->selected+1), 1.5f)); // empirical;

  // If all three moments of inertia are the same, the molecule is a spherical top,
  // i.e. the possible point groups are T, Td, Th, O, Oh, I, Ih.
  if (fabs(eigenval[0]-eigenval[1])<tol &&
      fabs(eigenval[1]-eigenval[2])<tol) {

    sphericaltop = 1;

  } else {
    // If two moments of inertia are the same, the molecule is a
    // symmetric top, i.e. the possible points groups are C1, Cn,
    // Cnv, Cnh, Sn, Dn, Dnh, Dnd.
    if (fabs(eigenval[0]-eigenval[1])<tol ||
        fabs(eigenval[1]-eigenval[2])<tol ||
        fabs(eigenval[0]-eigenval[2])<tol) {

      // Determine which is the unique primary axis. Also make sure
      // that this axis corresponds to a Cn rotation, otherwise we
      // have a false positive, i.e. two eigenvalues are identical
      // or similar only by incident.
      if (fabs(eigenval[1]-eigenval[2])<tol && primaryCn[0]) {
        uniqueprimary[0] = 1;
        numuniqueprimary++;
      }
      if (fabs(eigenval[0]-eigenval[2])<tol && primaryCn[1]) {
        uniqueprimary[1] = 1;
        numuniqueprimary++;
      }
      if (fabs(eigenval[0]-eigenval[1])<tol && primaryCn[2]) {
        uniqueprimary[2] = 1;
        numuniqueprimary++;
      }

      if (numuniqueprimary==1) {
        symmetrictop = 1;
      }
      else {
        uniqueprimary[0] = 1;
        uniqueprimary[1] = 1;
        uniqueprimary[2] = 1;
        numuniqueprimary = 3;	
      }
    } 
    else {
      uniqueprimary[0] = 1;
      uniqueprimary[1] = 1;
      uniqueprimary[2] = 1;
      numuniqueprimary = 3;
    }
  }


  for (i=0; i<3; i++) {
    // If the molecule is planar (but not linear) with the current
    // primary axis of inertia corresaponding to the plane normal
    // then we want to add that plane to the list.
    int planarmol = is_planar(inertiaaxes[i]);
    if (planarmol && !linear) {
      if (verbose>3) {
        msgInfo << "Planar mol: primary axes " << i << " defines plane" << sendmsg;
      }
      check_add_plane(inertiaaxes[i]);
      planar = 1;
    }


    if (!uniqueprimary[i]) continue;

    if (linear) {
      // Cinfv and Dinfh symmetry:
      // Use the unique primary axis of inertia as Cinf
      // For Dinfh an arbitrary C2 axis perpendicular to Cinf will be
      // automatically generated later by plane intersection.
      Axis a;
      vec_copy(a.v, inertiaaxes[i]);
      a.order = INFINITE_ORDER;
      float overlap1 = score_axis(a.v, 2);
      float overlap2 = score_axis(a.v, 4);
      a.overlap = 0.5f*(overlap1+overlap2);
      a.weight = sel->num_atoms/2;
      a.type = 0;
      axes.append(a);

      // We have to add an arbitrary vertical plane.
      Plane p;
      vec_copy(p.v, inertiaaxes[0]);
      p.overlap = eigenval[0]*eigenval[1] - eigenval[2];
      p.weight = sel->num_atoms/2;
      p.type = 0;
      planes.append(p);

    } else {
      //printf("primary axis of inertia %i: order=%i overlap=%.2f\n", i, order[i], overlap[i]);

      if (order[i] > 1 && overlap[i]>OVERLAPCUTOFF) {
        //printf("unique primary axes = %i planarmol = %i\n", i, planarmol);
        Axis a;
        vec_copy(a.v, inertiaaxes[i]);
        a.order = order[i];
        a.overlap = overlap[i];
        a.weight = 1;
        a.type = 0;
        axes.append(a);
      }

      // Add a possible horizontal mirror plane:
      // In case the molecule is planar the general plane finding algorithm
      // won't recognize the horizontal plane because identical atom transformations
      // are skipped. Thus we boost the weight here and provide it to 
      // check_add_plane().
      int weight = 1;
      if (planarmol) weight = sel->num_atoms/2;

      check_add_plane(inertiaaxes[i], weight);
    }
  }
}


// Find atoms with same distance to the center of mass and check if the 
// plane that is orthogonal to their connecting vector is a mirror plane.
void Symmetry::find_planes() {
  int i,j;
  float posA[3], posB[3];

  // Loop over all atoms
  for (i=0; i<sel->selected; i++) {

    // If we have more than maxnatoms atoms in the selection then pick 
    // only approximately maxnatoms random atoms  for the comparison.
    if (sel->selected > maxnatoms && vmd_random() % sel->selected > maxnatoms)
      continue;

    vec_sub(posA, coor+3L*i, rcom);

    float rA = sqrtf(posA[0]*posA[0] + posA[1]*posA[1] + posA[2]*posA[2]); 

    // Exclude atoms that are close to COM:
    // Closer than sigma is too close in any case, otherwise exclude
    // more the larger the molecule is.
    if (rA < sigma || rA < sel->num_atoms/7.0*sigma) continue;

    for (j=0; j<i; j++) {
      vec_sub(posB, coor+3L*j, rcom);

      float rB = sqrtf(posB[0]*posB[0] + posB[1]*posB[1] + posB[2]*posB[2]);
      
      if (fabs(rA-rB) > sigma) continue;

      // consider only pairs with identical atom types
      if (atomtype[i]==atomtype[j]) {

        // If the atoms are too close to the plane, the error gets
        // too big so we skip.
        float dist = distance(posA, posB);
        if (dist<0.25) continue;

        // We have found a hypothetical rotation axis or mirror plane
	
        // Subtracting the two position vectors yields a vector that
        // defines the normal of a plane that bisects the angle between
        // the two atoms, i.e. the plane is a potential mirror plane.
        float normal[3];
        vec_sub(normal, posA, posB);
        vec_normalize(normal);

        // Check plane and possibly add it to the list;
        check_add_plane(normal);
      }
    }
  }

  // Normalize the summed up normal vectors
  for (i=0; i<numplanes(); i++) {
    vec_normalize(planes[i].v);
    //printf("plane[%i]: weight=%3i, overlap=%.2f\n", i, planes[i].weight, planes[i].overlap);
  }
}


// Find C2 rotation axes from the vectors bisecting the angle between
// two atoms and the center of mass.
void Symmetry::find_C2axes() {
  int i,j;
  float posA[3], posB[3];
  float rA, rB;
  // Loop over all atoms
  for (i=0; i<sel->selected; i++) {
    // If we have more than maxnatoms atoms in the selection then pick 
    // only approximately maxnatoms random atoms  for the comparison.
    if (sel->selected > maxnatoms && vmd_random() % sel->selected > maxnatoms)
      continue;

    vec_sub(posA, coor+3L*i, rcom);

    rA = sqrtf(posA[0]*posA[0] + posA[1]*posA[1] + posA[2]*posA[2]); 

    // Exclude atoms that are close to COM:
    // Closer than sigma is too close in any case, otherwise exclude
    // more the larger the molecule is.
    if (rA < sigma || rA < sel->num_atoms/7.0*sigma) continue;

    for (j=i+1; j<sel->selected; j++) {
      vec_sub(posB, coor+3L*j, rcom);

      rB = sqrtf(posB[0]*posB[0] + posB[1]*posB[1] + posB[2]*posB[2]);
      
      if (fabs(rA-rB) > sigma) continue;

      // consider only pairs with identical atom types
      if (atomtype[i]==atomtype[j]) {
        // We have found a hypothetical rotation axis or mirror plane
        float alpha = angle(posA, posB);
	
        // See if the vector bisecting the angle between the two atoms
        // and the center of mass corresponds to a C2 axis. We must
        // exclude alpha==180 because in this case the bisection is
        // not uniquely defined.
        if (fabs(fabs(alpha)-180) > 10) {
          float testaxis[3];
          vec_add(testaxis, posA, posB);
          vec_normalize(testaxis);

          // Check axis and possibly add it to the list;
          if (!planes.num() || numverticalplanes) {
            //printf("Checking C2 axis\n"); 
            check_add_axis(testaxis, 2);
          }
        }
      }
    }
  }

  // Normalize the summed up axis vectors
  for (i=0; i<numaxes(); i++) {
    vec_normalize(axes[i].v);
  }
}

// Computes the factorial of n
static int fac(int n) {
  if (n==0) return 1;
  int i, x=1;
  for (i=1; i<=n; i++) x*=i;
  return x;
}


// Generate all n!/(k!(n-k)!) combinations of k different
// elements drawn from a total of n elements.
class Combinations {
public:
  int *combolist;
  int *combo;
  int num;
  int n;
  int k;

  Combinations(int N, int K);
  ~Combinations();

  // Create the combinations (i.e. populate combolist)
  void create() { recurse(0, -1); }

  void recurse(int begin, int level);
  void print();

  // Get pointer to the i-th combination
  int* get(int i) { return &combolist[i*k]; }
};

Combinations::Combinations(int N, int K) : n(N), k(K) {
  if (n>10) n = 10; // prevent overflow of float by fac(n)
  combo = new int[k];
  combolist = new int[k*fac(n)/(fac(k)*fac(n-k))];
  num = 0;
}

Combinations::~Combinations() {
  delete [] combo;
  delete [] combolist;
}

// Recursive function to generate combinations
void Combinations::recurse(int begin, int level) {
  int i;
  level++;
  if (level>=k) {
    for (i=0; i<k; i++) {
      combolist[num*k+i] = combo[i];
    }
    num++;
    return;
  }
  
  for (i=begin; i<n; i++) {
    combo[level] = i;
    recurse(i+1, level);
  }  
}

// Print the combinations
void Combinations::print() {
  int i, j;
  for (i=0; i<fac(n)/(fac(k)*fac(n-k)); i++) {
    printf("combo %d/%d {", i, num);
    for (j=0; j<k; j++) {
      printf(" %d", combolist[i*k+j]);
    }
    printf("}\n");
  }

}


// Find Cn rotation axes from the atoms that lie in the
// same plane. The plane normal defines the hypothetical
// axis
void Symmetry::find_axes(int order) {
  int i,j;
  float posA[3], posB[3];
  float rA, rB;
  int *atomtuple = new int[sel->selected];

  // Loop over all atoms
  for (i=0; i<sel->selected; i++) {
    // If we have more than maxnatoms atoms in the selection then pick 
    // only approximately maxnatoms random atoms  for the comparison.
    if (sel->selected > maxnatoms && vmd_random() % sel->selected > maxnatoms)
      continue;

    vec_sub(posA, coor+3L*i, rcom);

    rA = sqrtf(posA[0]*posA[0] + posA[1]*posA[1] + posA[2]*posA[2]); 

    // Exclude atoms that are close to COM:
    // Closer than sigma is too close in any case, otherwise exclude
    // more the larger the molecule is.
    if (rA < sigma || rA < sel->num_atoms/7.0*sigma) continue;

    atomtuple[0] = i;
    int ntup = 1;

    for (j=i+1; j<sel->selected; j++) {
      // If we have more than maxnatoms atoms in the selection
      // then pick only approximately maxnatoms random atoms
      // for the comparison.
      if (sel->selected > maxnatoms && vmd_random() % sel->selected > maxnatoms)
        continue;
      
      // Consider only pairs with identical atom types
      if (atomtype[j]!=atomtype[i]) continue;
      
      vec_sub(posB, coor+3L*j, rcom);
      
      rB = sqrtf(posB[0]*posB[0] + posB[1]*posB[1] + posB[2]*posB[2]);
      
      // Atoms should have approximately same distance from COM
      if (fabs(rA-rB) > sigma) continue;
      
      atomtuple[ntup++] = j;

      // We don't consider more then 10 equivalent atoms
      // to save resources.
      if (ntup>=10) break;
    }
    if (ntup<order) continue;

    //printf("equiv. atoms: ");
    //for (j=0; j<ntup; j++) { printf("%d ",atomtuple[j]); }
    //printf("\n");

    // Generate all combinations of tuples with order different
    // elements drawn from a total of ntup elements.
    Combinations combo(ntup, order);
    combo.create();


    int m;
    for (j=0; j<combo.num; j++) {
      float testaxis[3], pos[3], pos2[3], cross[3], normal[3];
      vec_zero(testaxis);
      vec_zero(normal);
      //printf("combi %d: {", j);

      // Test wether all atoms in the tuple lie aproximately
      // within one plane
      int inplane = 1, anyinplane = 0;
      for (m=0; m<order; m++) {
        int atom = atomtuple[combo.get(j)[m]];
        //printf(" %d", atom); //combo.get(j)[m]);
        vec_sub(pos, coor+3L*atom, rcom);
        vec_add(testaxis, testaxis, pos);

        // Find a second atom from the tuple that encloses
        // an angle of >45 deg with the first one.
        // If we don't find one then we ignore the in-plane
        // testing for the current atom.
        int n, found = 0;
        for (n=0; n<order; n++) {
          if (n==m) continue;
          int atom2 = atomtuple[combo.get(j)[m]];
          vec_sub(pos2, coor+3L*atom2, rcom);
          if (angle(pos, pos2)>45.f) {
            found = 1;
            break;
          }
        }
        if (!found) continue;
        
        cross_prod(cross, pos2, pos);
        
        // Check if the cross product of the two atom positions
        // is collinear with the plane normal (which was summed
        // up from the cross products of the previous pairs).
        if (collinear(normal, cross, collintol)) {
          vec_add(normal, normal, cross);
          anyinplane = 1;
        } else {
          // This atom is not in plane with the others
          inplane = 0;
          break;
        }
      }

      //printf("}\n");
      
      // If the atoms of the current tuple aren't in one
      // plane then the positions cannot be used to define
      // a Cn axis and we skip it.
      if (!inplane || !anyinplane) continue;

      vec_normalize(normal);

      // Check axis and possibly add it to the list;
      printf("Checking C%d axis\n", order); 
      check_add_axis(normal, order);
    }

  }
   
  delete [] atomtuple;

  // Normalize the summed up axis vectors
  for (i=0; i<numaxes(); i++) {
    vec_normalize(axes[i].v);
  }
}


// Get rotation axes.
// Each intersection of two mirror planes is itself a rotation
// axis. Computing the intersection is easy in this case because
// we know that any two mirror planes go through the center of
// mass. The direction of the intersection line is just the 
// cross product of the two plane normals.

void Symmetry::find_axes_from_plane_intersections() {
  // If there aren't at least 2 planes we won't find any axes.
  if (numplanes()<2) return;

  int i,j;

  for (i=0; i<numplanes(); i++) {
    for (j=i+1; j<numplanes(); j++) {
      float newaxis[3];
      cross_prod(newaxis, plane(i), plane(j));

      vec_normalize(newaxis);

      // Ignore intersections of parallel planes
      if (norm(newaxis)<0.05) continue;

      // Loop over already existing axes and check if the new axis is
      // collinear with one of them. In this case we add the two axes
      // in order to obtain an average (after normalizing at the end).
      int k;
      bool found = 0;
      for (k=0; k<numaxes(); k++) {
        float avgaxis[3];
        vec_copy(avgaxis, axis(k));
        vec_normalize(avgaxis);

        float dot = dot_prod(avgaxis, newaxis);
        //printf("dot=% .4f fabs(dot)=%.4f, 1-collintol=%.2f\n", dot, fabs(dot), collintol);
        if (fabs(dot) > collintol) {
          // We are summing up the collinear axes to get the average
          // of the equivalent axes.
          if (dot>0) {
            vec_incr(axis(k), newaxis);         // axes[k] += normal
          } else {
            vec_sub(axis(k), axis(k), newaxis); // axes[k] -= newaxis
          }
          axes[k].weight++;
          found = 1;
          break;
        }
      }

      if (!found) {
        // We found no existing collinear axis, so add a new one to the list.
        Axis a;
        vec_copy(a.v, newaxis);
        a.type    = 0;
        a.order   = 0;
        a.overlap = 0.0;
        if (planes[i].weight<planes[j].weight)
          a.weight = planes[i].weight;
        else
          a.weight = planes[j].weight;
        axes.append(a);
      }
    }
  }

  // Normalize the summed up axis vectors
  for (i=0; i<numaxes(); i++) {
    vec_normalize(axis(i));
  }
}


// Determine which of the existing mirror planes are horizontal,
// vertical, or dihedral planes with respect to the first axis.
// The axes must have been sorted already so that the axes with
// the highest order (primary axis) is the first one.
// A horizontal plane is perpendicular to the primary axis.
// A vertical plane includes the primary axis.
// A dihedral plane is vertical to the corresponding axis and 
// bisects the angle formed by a pair of C2 axes.
void Symmetry::classify_planes() {
  if (!numplanes()) return;
  if (!numaxes())   return;

  numdihedralplanes = 0;
  numverticalplanes = 0;
  horizontalplane = -1;

  int i;

  // Is there a plane perpendicular to the highest axis?
  for (i=0; i<numplanes(); i++) {
    if (collinear(axis(0), plane(i), collintol)) {
      horizontalplane = i;
      planes[i].type = HORIZONTALPLANE;
      break;
    }
  }


  for (i=0; i<numplanes(); i++) {
    if (!orthogonal(axis(0), plane(i), orthogtol)) continue;
    
    // The current plane is vertical to the first axis.
    numverticalplanes++;
    planes[i].type = VERTICALPLANE;

    // Now loop over pairs of C2 axes that are in that plane:
    int j, k;
    for (j=1; j<numaxes(); j++) {
      if (axes[j].order != 2) continue;
      if (!orthogonal(axis(j), axis(0), orthogtol)) continue;
      
      for (k=j+1; k<numaxes(); k++) {
        if (axes[k].order != 2) continue;
        if (!orthogonal(axis(k), axis(0), orthogtol)) continue;
	
        // Check if the plane bisects the pair of C2 axes
        float bisect[3];
        vec_add(bisect, axis(j), axis(k));
        vec_normalize(bisect);
        if (orthogonal(bisect, plane(i), orthogtol)) {
          planes[i].type = DIHEDRALPLANE;
          numdihedralplanes++;
          j=numaxes(); // stop looping axis pairs
          break;
        }

        vec_sub(bisect, axis(j), axis(k));
        vec_normalize(bisect);
        if (orthogonal(bisect, plane(i), orthogtol)) {
          planes[i].type = DIHEDRALPLANE;
          numdihedralplanes++;
          j=numaxes(); // stop looping axis pairs
          break;
        }
	
      }
    }
  }
}

void Symmetry::classify_axes() {
  if (!numaxes()) return;

  // If n is the higest Cn axis order, are there n C2 axes
  // perpendicular to Cn?
  int numprimary = 0;
  int i;
  for (i=0; i<numaxes(); i++) {
    if (axes[i].order==maxaxisorder) {
      axes[i].type = PRINCIPAL_AXIS;
      numprimary++;
    }
  }
  for (i=1; i<numaxes(); i++) {
    if (orthogonal(axis(0), axis(i), orthogtol))
      axes[i].type |= PERPENDICULAR_AXIS;
  }
}

// Check if center of mass is an inversion center.
float Symmetry::score_inversion() {
  // Construct inversion matrix
  Matrix4 inv;
  inv.translate(rcom[0], rcom[1], rcom[2]);
  inv.scale(-1.0);  // inversion
  inv.translate(-rcom[0], -rcom[1], -rcom[2]);

  // Return score	
  return trans_overlap(atomtype, coor, sel->selected, &inv, 1.5f*sigma,
                       NOSKIP_IDENTICAL, maxnatoms);
}

// Check if center of mass is an inversion center
// and set the inversion flag accordingly.
void Symmetry::check_add_inversion() {
  // Verify inversion center
  float overlap = score_inversion();

  //printf("inversion overlap = %.2f\n", overlap);

  // We trust planes and axes more than the inversion since 
  // they are averages over multiple detected elements.
  // Hencewe make the cutoff range from 0.3 to 0.5, depending
  // on the number of other found symmetry elements.
  if (overlap>0.5-0.2/(1+(planes.num()*axes.num()))) {
    inversion = 1;
  }
}

// Check if the given normal represents a mirror plane.
float Symmetry::score_plane(const float *normal) {
  // A mirror symmetry operation is an inversion + a rotation
  // by 180 deg about the normal of the mirror plane.
  // We also need to shift to origin and back.
  Matrix4 mirror;
  mirror.translate(rcom[0], rcom[1], rcom[2]);
  mirror.scale(-1.0);  // inversion
  mirror.rotate_axis(normal, float(VMD_PI));
  mirror.translate(-rcom[0], -rcom[1], -rcom[2]);
	  
  // Return score
  return trans_overlap(atomtype, coor, sel->selected, &mirror, 1.5f*sigma, NOSKIP_IDENTICAL, maxnatoms);
}

// Append the given plane to the list.
// If a approximately coplanar plane exists already then add the new
// plane normal to the existing one thus getting an average direction.
// The input plane normal must have a length of 1.
// Note: After the list of planes is complete you want to normalize all
// plane vectors.
void Symmetry::check_add_plane(const float *normal, int weight) {
	  
  // Verify the mirror plane
  float overlap = score_plane(normal);

  // In case of significant overlap between the original and the mirror
  if (overlap<OVERLAPCUTOFF) return;


  // We have to loop over all existing planes and see if the current
  // plane is coplanar with one of them. If so, we add to that existing
  // one. The resulting plane normal vector can later be normalized so
  // that we get the average plane normal of the equivalent planes.
  // If no existing coplanar plane is found a new plane is added to the
  // array.
  int k;
  bool found=0;
  for (k=0; k<numplanes(); k++) {
    float avgnormal[3];
    vec_copy(avgnormal, plane(k));
    vec_normalize(avgnormal);

    // If two planes are coplanar we can use their average.
    // The planes are parallel if cross(n1,n2)==0.
    // However it is faster to use 
    // |cross(n1,n2)|^2 = |n1|*|n2| - dot(n1,n2)^2
    //                  =     1     - dot(n1,n2)^2
    // The latter is true because the normals have length 1.
    // Hence |cross(n1,n2)| == 0 is equivalent to 
    // |dot(n1,n2)| == 1.
    float dot = dot_prod(avgnormal, normal);
    if (fabs(dot) > collintol) {

      // We are summing up the coplanar planes to get the average
      // of the equivalent planes.
      if (dot>0) { 
        vec_incr(plane(k), normal);           // planes[k] += normal
      } else {
        vec_scaled_add(plane(k), -1, normal); // planes[k] -= normal
      }
      planes[k].overlap = (planes[k].weight*planes[k].overlap + overlap)/(planes[k].weight+1);
      (planes[k].weight)++;
      found = 1;
      break;
    }
  }
  if (!found) {
    // We found no existing coplanar plane, so add a new one to the list.
    Plane p;
    vec_copy(p.v, normal);
    p.overlap = overlap;
    p.weight  = weight;
    p.type    = 0;
    planes.append(p);
  }
}


// Check if vector testaxis defines a rotary axis of the given order.
float Symmetry::score_axis(const float *testaxis, int order) {
  // Construct rotation matrix.
  Matrix4 rot;
  rot.translate(rcom[0], rcom[1], rcom[2]);
  rot.rotate_axis(testaxis, float(VMD_TWOPI)/order);
  rot.translate(-rcom[0], -rcom[1], -rcom[2]);
	
  // Verify symmetry axis
  return trans_overlap(atomtype, coor, sel->selected, &rot, 2*sigma,
                       NOSKIP_IDENTICAL, maxnatoms);
}


// Append a given axis to the list in case it is C2.
// We look for C2 axes only, because these are the only ones
// that cannot always be generated by intersecting mirror planes.
// Examples are the three C2 axes perpendicular to the primary C3 in 
// the D3 pointgroup.
void Symmetry::check_add_axis(const float *testaxis, int order) {
  int k;
  bool found = 0;
  for (k=0; k<numaxes(); k++) {
    float avgaxis[3];
    vec_copy(avgaxis, axis(k));
    vec_normalize(avgaxis);
    
    
    float dot = dot_prod(avgaxis, testaxis);
    if (fabs(dot) > collintol) {
      // Preliminary axes (order<0) haven't been averaged and
      // collapsed yet. For these cases we are summing up the
      // collinear axes to get the average of the equivalent
      // axes.
      if (axes[k].order==-order) {
        if (dot>0) { 
          vec_incr(axis(k), testaxis);           // axes[k] += testaxis
        } else {
          vec_scaled_add(axis(k), -1, testaxis); // axes[k] -= testaxis
        }
        axes[k].weight++;
      }
      found = 1;
      break;
    }
  }
  if (!found) {
    // We found no existing collinear axis, so add a new one to the list.
    float overlap = score_axis(testaxis, order);
    if (overlap>OVERLAPCUTOFF) {
      Axis a;
      vec_copy(a.v, testaxis);
      // We tag the order as preliminary C2 in order to distinguish it 
      // from C2 axes found by plane intersection. The latter have a higher
      // accuracy since they are based on averaged planes.
      a.order  = -order; //PRELIMINARY_C2;
      a.overlap = overlap;
      a.weight = 1;
      a.type   = 0;
      axes.append(a);
    }
  }
}

// Get the score for given rotary reflection.
float Symmetry::score_rotary_reflection(const float *testaxis, int order) {
  // Construct transformation matrix. An n-fold rotary
  // reflection is a rotation by 360/n deg followed by
  // a reflection about the plane perpendicular to the axis.
  Matrix4 rot;
  rot.translate(rcom[0], rcom[1], rcom[2]);
  rot.rotate_axis(testaxis, float(VMD_TWOPI)/order);
  rot.scale(-1.0);  // inversion
  rot.rotate_axis(testaxis, float(VMD_PI));
  rot.translate(-rcom[0], -rcom[1], -rcom[2]);
  
  // Return score
  return trans_overlap(atomtype, coor, sel->selected, &rot, sigma,
                       NOSKIP_IDENTICAL, maxnatoms);
}


// testaxis must be normalized.
void Symmetry::check_add_rotary_reflection(const float *testaxis, int maxorder) {
  if (maxorder<4) return;

  //msgInfo << "checking improper: maxorder=" << maxorder << sendmsg;

  int n;
  for (n=3; n<=maxorder; n++) {
    if (n>=9 && n%2) continue;
    if (maxorder%n)  continue;

    // Get the overlap for each rotary reflection:
    float overlap = score_rotary_reflection(testaxis, n);
    //printf("rotrefl: n=%i, axis angle = %.2f, overlap = %.2f\n", n, 360.0/n, overlap);
  

    if (overlap>OVERLAPCUTOFF) {
      int k;
      bool found = 0;
      for (k=0; k<numrotreflect(); k++) {
        float avgaxis[3];
        vec_copy(avgaxis, rotreflect(k));
        vec_normalize(avgaxis);
      
        if (n!=rotreflections[k].order) continue;

        float dot = dot_prod(avgaxis, testaxis);
        if (fabs(dot) > collintol) {
          // We are summing up the collinear axes to get the average
          // of the equivalent axes.
          if (dot>0) { 
            vec_incr(rotreflect(k), testaxis);           // axes[k] += testaxis
          } else {
            vec_scaled_add(rotreflect(k), -1, testaxis); // axes[k] -= testaxis
          }
          rotreflections[k].weight++;
          found = 1;
          break;
        }
      }
      if (!found) {
        // We found no existing collinear rr-axis, so add a new one to the list.
        Axis a;
        vec_copy(a.v, testaxis);
        a.order   = n;
        a.overlap = overlap;
        a.weight  = 1;
        a.type    = 0;
        rotreflections.append(a);
      }
    }
  }
}


// Purge planes with an overlap below half of the maximum overlap
// that occurs in planes and axes.
void Symmetry::purge_planes(float cutoff) {
  maxweight  = 0;
  maxoverlap = 0.0;
  if (!numplanes()) return;

  maxweight  = planes[0].weight;
  maxoverlap = planes[0].overlap;
  int i;
  for (i=1; i<numplanes(); i++) {
    if (planes[i].overlap>maxoverlap) maxoverlap = planes[i].overlap;
    if (planes[i].weight >maxweight)  maxweight  = planes[i].weight;
  }

  float tol = cutoff*maxoverlap;
  int *keep = new int[planes.num()];
  memset(keep, 0, planes.num()*sizeof(int));

  for (i=0; i<numplanes(); i++) {
    if (planes[i].overlap>tol) {
      // keep this plane
      keep[i] = 1;
    }
  }

  prune_planes(keep);

  delete [] keep;
}


// Purge axes with an overlap below half of the maximum overlap
// that occurs in planes and axes.
void Symmetry::purge_axes(float cutoff) {
  if (!numaxes()) return;

  int i;
  for (i=0; i<numaxes(); i++) {
    if (axes[i].overlap>maxoverlap) maxoverlap = axes[i].overlap;
    if (axes[i].weight >maxweight)  maxweight  = axes[i].weight;
  }

  float tol = cutoff*maxoverlap;
  int *keep = new int[axes.num()];
  memset(keep, 0, axes.num()*sizeof(int));

  for (i=0; i<numaxes(); i++) {
    if (axes[i].overlap>tol) {
      // keep this axis
      keep[i] = 1;
    }
  }

  prune_axes(keep);

  delete [] keep;
}


// Purge rotary reflections with an overlap below half of the maximum overlap
// that occurs in planes and axes.
void Symmetry::purge_rotreflections(float cutoff) {
  if (!numrotreflect()) return;

  int i;
  for (i=0; i<numrotreflect(); i++) {
    if (rotreflections[i].overlap>maxoverlap) maxoverlap = rotreflections[i].overlap;
    if (rotreflections[i].weight >maxweight)  maxweight  = rotreflections[i].weight;
  }

  float tol = cutoff*maxoverlap;
  int *keep = new int[rotreflections.num()];
  memset(keep, 0, rotreflections.num()*sizeof(int));

  for (i=0; i<numrotreflect(); i++) {
    if (rotreflections[i].overlap>tol) {
      // keep this rotary reflection
      keep[i] = 1;
    }
    else if (verbose>4) {
      // purge this rotary reflection
      char buf[512];
      sprintf(buf, "purged S%i rotary reflection %i (weight=%i, overlap=%.2f tol=%.2f)",
              rotreflections[i].order, i, rotreflections[i].weight, rotreflections[i].overlap, tol);
      msgInfo << buf << sendmsg;
    }
  }

  prune_rotreflections(keep);

  delete [] keep;
}


// For each plane i prune from the list if keep[i]==0.
void Symmetry::prune_planes(int *keep) {
  if (!planes.num()) return;

  numverticalplanes = 0;
  numdihedralplanes = 0;

  int numkept = 0;
  Plane *keptplanes = new Plane[numplanes()];
  int i;
  for (i=0; i<planes.num(); i++) {
    //printf("keep plane[%i] = %i\n", i, keep[i]);
    if (keep[i]) {
      // keep this plane
      keptplanes[numkept] = planes[i];
      if (planes[i].type==HORIZONTALPLANE) horizontalplane=numkept;
      else if (planes[i].type==VERTICALPLANE) numverticalplanes++;
      else if (planes[i].type==DIHEDRALPLANE) {
        numdihedralplanes++;
        numverticalplanes++;
      }
      numkept++;
      
    } else {
      // purge this plane
      if (planes[i].type==HORIZONTALPLANE) horizontalplane=-1;
      if (verbose>3) {
        char buf[256];
        sprintf(buf, "removed %s%s%s plane %i (weight=%i, overlap=%.2f)\n", 
                (planes[i].type==HORIZONTALPLANE?"horiz":""),
                (planes[i].type==VERTICALPLANE?"vert":""),
                (planes[i].type==DIHEDRALPLANE?"dihed":""),
                i, planes[i].weight, planes[i].overlap);
        msgInfo << buf << sendmsg;
      }
    }
  }
  memcpy(&(planes[0]), keptplanes, numkept*sizeof(Plane));
  planes.truncatelastn(numplanes()-numkept);

  delete [] keptplanes;
}


// For each axis i prune from the list if keep[i]==0.
void Symmetry::prune_axes(int *keep) {
  if (!axes.num()) return;

  int numkept = 0;
  Axis *keptaxes = new Axis[numaxes()];

  maxaxisorder = 0;
  int i;
  for (i=0; i<axes.num(); i++) {
    //printf("keep axis[%i] = %i\n", i, keep[i]);
    if (keep[i]) {
      // keep this axis
      keptaxes[numkept++] = axes[i];
      
      if (axes[i].order>maxaxisorder) 
        maxaxisorder = axes[i].order;
    }
  }
  memcpy(&(axes[0]), keptaxes, numkept*sizeof(Axis));
  axes.truncatelastn(numaxes()-numkept);
  
  delete [] keptaxes;
}


// For each axis i prune from the list if keep[i]==0.
void Symmetry::prune_rotreflections(int *keep) {
  if (!rotreflections.num()) return;

  int numkept = 0;
  Axis *keptrotrefl = new Axis[numrotreflect()];

  maxrotreflorder = 0;
  int i;
  for (i=0; i<numrotreflect(); i++) {
    if (keep[i]) {
      // keep this rotary reflection
      keptrotrefl[numkept++] = rotreflections[i];

      if (rotreflections[i].order>maxrotreflorder)
        maxrotreflorder = rotreflections[i].order;
    }
  }

  memcpy(&(rotreflections[0]), keptrotrefl, numkept*sizeof(Axis));
  rotreflections.truncatelastn(numrotreflect()-numkept);

  delete [] keptrotrefl;
}


// Assign orders to rotational symmetry axes.
void Symmetry::assign_axis_orders() {
  if (!numaxes()) return;

  maxaxisorder = axes[0].order;

  // Compute all axis orders
  int i;
  for (i=0; i<numaxes(); i++) {
    if (!axes[i].order) axes[i].order = axis_order(axis(i), &(axes[i].overlap));

    //printf("axis[%i] {%f %f %f} order = %i, weight = %i overlap = %.2f\n", i,
    //     axes[i].v[0], axes[i].v[1], axes[i].v[2], axes[i].order, axes[i].weight, axes[i].overlap);
    if (axes[i].order>maxaxisorder) {
      maxaxisorder = axes[i].order;
    }
  }
}


// Assign orders to preliminary C2 rotational symmetry axes.
void Symmetry::assign_prelimaxis_orders(int order) {
  int i;
  for (i=0; i<numaxes(); i++) {
    if (axes[i].order==-order) {
      axes[i].order=order; 
      if (axes[i].weight>1) {
        // We have computed the overlap only for
        // the first instance not for the average
        // so we want to re-score it here.
        axes[i].overlap = score_axis(axis(i), order);
      }
    }

    if (axes[i].order>maxaxisorder) maxaxisorder = axes[i].order;
  }
}

// Sort the axes according to decreasing order
// Also eliminate C1 axes and Cn axes that are parallel
// to a Cinf axes.
void Symmetry::sort_axes() {
  int i;

  // count number of sorted axes.
  int numsortedaxes = 0;
  for (i=0; i<numaxes(); i++) {
    if (axes[i].order>1 || axes[i].order==INFINITE_ORDER) numsortedaxes++;
  }

  Axis *sortedaxes = new Axis[numsortedaxes*sizeof(Axis)];
  int j, k=0, have_inf=0;
  for (j=0; j<numaxes(); j++) {
    if (axes[j].order == INFINITE_ORDER) {
      sortedaxes[k] = axes[j];
      k++;
      have_inf = 1;
    }
  }

  for (i=maxaxisorder; i>1; i--) {
    for (j=0; j<numaxes(); j++) {
      if (axes[j].order == i) {
        if (have_inf && 
            collinear(sortedaxes[0].v, axes[j].v, collintol)) {
          continue;
        }
        sortedaxes[k] = axes[j];
        k++;
      }
    }
  }

  memcpy(&(axes[0]), sortedaxes, numsortedaxes*sizeof(Axis));
  axes.truncatelastn(numaxes()-numsortedaxes);

  delete [] sortedaxes;
}


// Sort the planes: horizontal plane first, then dihedral,
// then vertical, then the rest
void Symmetry::sort_planes() {
  Plane *sortedplanes = new Plane[numplanes()*sizeof(Plane)];

  int k = 0;
  if (horizontalplane>=0) {
    sortedplanes[k] = planes[horizontalplane];
    horizontalplane = k++;
  }

  int j;
  for (j=0; j<numplanes(); j++) {
    if (planes[j].type == DIHEDRALPLANE) {
      sortedplanes[k++] = planes[j];
    }
  }
  for (j=0; j<numplanes(); j++) {
    if (planes[j].type == VERTICALPLANE) {
      sortedplanes[k++] = planes[j];
    }
  }
  for (j=0; j<numplanes(); j++) {
    if (planes[j].type == 0) {
      sortedplanes[k++] = planes[j];
    }
  }

  memcpy(&(planes[0]), sortedplanes, numplanes()*sizeof(Plane));

  delete [] sortedplanes;
}


// Find the highest rotational symmetry order (up to 8) for the given axis.
// Rotates the selection by 360/i degrees (i=2...8) and checks the structural
// overlap.
int Symmetry::axis_order(const float *axis, float *overlap) {
  int i;

  // Get the overlap for each rotation
  float overlaparray[MAXORDERCN+1];
  float overlappermarray[MAXORDERCN+1];
  overlappermarray[1] = 0.0;
  for (i=2; i<=MAXORDERCN; i++) {
    Matrix4 rot;
    // need to shift to origin and back
    rot.translate(rcom[0], rcom[1], rcom[2]);
    rot.rotate_axis(axis, float(DEGTORAD(360.0f/i)));
    rot.translate(-rcom[0], -rcom[1], -rcom[2]);
    overlaparray[i] = trans_overlap(atomtype, coor, sel->selected, &rot, 1.5f*sigma, SKIP_IDENTICAL, maxnatoms, overlappermarray[i]);
  }

  // Get the maximum overlap
  float maxover = overlaparray[2];
  //printf("orders: %.2f ", overlaparray[2]);
  for (i=3; i<=MAXORDERCN; i++) {
    //printf("%.2f ", overlaparray[i]);
    if (overlaparray[i]>maxover) {
      maxover = overlaparray[i];
    }
  }

  // If overlap for a certain rotation is greater than half maxover
  // we assume that the axis has the according order or a multiple of that.
  float maxfalseov = 0.0f;
  int maxorder = 1;
  short int orders[MAXORDERCN+1];
  for (i=2; i<=MAXORDERCN; i++) {
    if (overlaparray[i]>0.5f*maxover) {
      orders[i] = 1;
      maxorder = i;
    } else {
      orders[i] = 0;
      if (overlaparray[i]>maxfalseov) maxfalseov = overlaparray[i];
    }
  }

#if defined(_MSC_VER)
  float tol1 = 0.25f + 0.15f*float(myerfc(0.05f*float(sel->selected-50)));
#elif defined(__irix)
  float tol1 = 0.25f + 0.15f*erfc(0.05f*float(sel->selected-50));
#else
  float tol1 = 0.25f + 0.15f*erfcf(0.05f*float(sel->selected-50));
#endif
  //printf(" tol1=%.2f tol=%.2f\n", tol1, tol1-tol2+pow(0.09*maxorder,6));

  // Make the tolerance dependent on the maximum order.
  // For maxorder=2 tol=0.2 for maxorder=8 tol=0.27 (empirical)
  if (maxover<tol1+powf(0.09f*maxorder, 6)) {
    *overlap = maxover;
    return 1;
  }

  // We return the arithmetic mean of the relative and the maximum overlap.
  // The relative overlap is the quality of the matching axis orders wrt
  // the best non-matching one.
  if (orders[2]) {
    if (orders[3] && orders[6]) {
      float avgov = (overlaparray[2]+overlaparray[3]+overlaparray[6])/3.0f;
      *overlap = 0.5f*(maxover + (avgov-maxfalseov)/avgov);
      return 6;
    } else if (orders[4]) {
      if (MAXORDERCN>=8 && orders[8]) {
        float avgov = (overlaparray[2]+overlaparray[4]+overlaparray[8])/3.0f;
        *overlap = 0.5f*(maxover + (avgov-maxfalseov)/avgov);
        return 8;
      }
      *overlap = 0.5f*(maxover + (overlaparray[4]-maxfalseov)/overlaparray[4]);
      return 4;
    }

    *overlap = 0.5f*(maxover + (overlaparray[2]-maxfalseov)/overlaparray[2]);
    return 2;      
  } else if (orders[3]) {
    *overlap = 0.5f*(maxover + (overlaparray[3]-maxfalseov)/overlaparray[3]);
    return 3;
  } else if (orders[5]) {
    *overlap = 0.5f*(maxover + (overlaparray[5]-maxfalseov)/overlaparray[5]);
    return 5;
  } else if (orders[7]) {
    *overlap = 0.5f*(maxover + (overlaparray[7]-maxfalseov)/overlaparray[7]);
    return 7;
  }

  *overlap = maxover;
  return 1;
}


// Generate symmetricized coordinates by wrapping atoms around
// the symmetry elements and average these images with the 
// original positions.
void Symmetry::idealize_coordinates() {
  int i;

  // copy atom coordinates into idealcoor array
  memcpy(idealcoor, coor, 3L*sel->selected*sizeof(float));

  // -----------Planes-------------
  int *keep = new int[planes.num()];
  memset(keep, 0, planes.num()*sizeof(int));

  for (i=0; i<numplanes(); i++) {
    Matrix4 mirror;
    mirror.translate(rcom[0], rcom[1], rcom[2]);
    mirror.scale(-1.0);  // inversion
    mirror.rotate_axis(planes[i].v, float(VMD_PI));
    mirror.translate(-rcom[0], -rcom[1], -rcom[2]);

    //printf("PLANE %d\n", i);
    if (average_coordinates(&mirror)) keep[i] = 1;
  }

  prune_planes(keep);


  // -----------Axes-----------
  int   *weight  = new int[sel->selected];
  float *avgcoor = new float[3L*sel->selected];

  delete [] keep;
  keep = new int[axes.num()];
  memset(keep, 0, axes.num()*sizeof(int));

  for (i=0; i<numaxes(); i++) {
    memset(weight,  0,   sel->selected*sizeof(int));
    memcpy(avgcoor, idealcoor, 3L*sel->selected*sizeof(int));
    int order = axes[i].order;
    //printf("AXIS %i\n", i);

    // In case of a Cinf axis just use order 4 to idealize in the two
    // dimensions perpendicular to the axis
    if (order==INFINITE_ORDER) order = 4;

    // Average the coordinates over all rotary orientations
    int success = 1;
    int n, j;
    for (n=1; n<order; n++) {
      Matrix4 rot;
      rot.translate(rcom[0], rcom[1], rcom[2]);
      rot.rotate_axis(axis(i), n*float(VMD_TWOPI)/order);
      rot.translate(-rcom[0], -rcom[1], -rcom[2]);

      // For each atom find the closest transformed image atom with
      // the same chemical element. The array matchlist will contain
      // the closest match atom index (into the collapsed list of
      // selected atoms) for each selected atom.
      int nmatches;
      int *matchlist = NULL;
      identify_transformed_atoms(&rot, nmatches, matchlist);

      for(j=0; j<sel->selected; j++) {
        int m = matchlist[j];
        if (m<0) continue; // no match found;

        if (checkbonds) {
          // Rotate the bondsum vector according to the Cn axis
          // and compare
          if (!check_bondsum(j, m, &rot)) {
            success = 0;
            break;
          }
        }

        float tmpcoor[3];
        rot.multpoint3d(idealcoor+3L*m, tmpcoor);
        vec_incr(avgcoor+3L*j, tmpcoor);
        weight[j]++;
      }
      if (matchlist) delete [] matchlist;

      if (!success) break;
      //printf("average coor for angle=%.2f\n", n*360.f/order);
    }

    if (success) {
      // Combine the averaged coordinates for this axis with the existing
      // ideal coordinates
      for(j=0; j<sel->selected; j++) {
        vec_scale(idealcoor+3L*j, 1.0f/(1+weight[j]), avgcoor+3L*j);
      }

      keep[i] = 1;
    }
  }

  prune_axes(keep);

  delete [] weight;
  delete [] avgcoor;


  // -----------Rotary reflections-----------
  delete [] keep;
  keep = new int[rotreflections.num()];
  memset(keep, 0, rotreflections.num()*sizeof(int));

  for (i=0; i<numrotreflect(); i++) {
    Matrix4 rot;
    rot.translate(rcom[0], rcom[1], rcom[2]);
    rot.rotate_axis(rotreflect(i), float(VMD_TWOPI)/rotreflections[i].order);
    rot.scale(-1.0f);  // inversion
    rot.rotate_axis(rotreflect(i), float(VMD_PI));
    rot.translate(-rcom[0], -rcom[1], -rcom[2]);

    //printf("ROTREF %i\n", i);
    if (average_coordinates(&rot)) keep[i] = 1;
  }

  prune_rotreflections(keep);

  delete [] keep;

  // -----------Inversion-----------
  if (inversion) {
    // Construct inversion matrix
    Matrix4 inv;
    inv.translate(rcom[0], rcom[1], rcom[2]);
    inv.scale(-1.0f);  // inversion
    inv.translate(-rcom[0], -rcom[1], -rcom[2]);

    if (!average_coordinates(&inv)) inversion = 0;
  }
}


// Averages between original and transformed coordinates.
int Symmetry::average_coordinates(Matrix4 *trans) {
  int j;
  int nmatches;
  int *matchlist = NULL;
  identify_transformed_atoms(trans, nmatches, matchlist);

  if (checkbonds) {
    int success = 1;
    for(j=0; j<sel->selected; j++) {
      int m = matchlist[j];
      
      if (m<0) continue; // no match found;
      
      if (!check_bondsum(j, m, trans)) {
        success = 0;
        break;
      }
    }
    if (!success) {
      if (verbose>3) {
        msgInfo << "Transformation messes up bonds!" << sendmsg;
      }
      if (matchlist) delete [] matchlist;
      return 0;
    }
  }

  for(j=0; j<sel->selected; j++) {
    int m = matchlist[j];

    if (m<0) continue; // no match found;

    // Average between original and image coordinates    
    float avgcoor[3];
    trans->multpoint3d(idealcoor+3L*m, avgcoor);
    vec_incr(avgcoor, idealcoor+3L*j);
    vec_scale(idealcoor+3L*j, 0.5, avgcoor);
  }
  if (matchlist) delete [] matchlist;

  return 1;
}

// Check the bondsum for atom j and its image m generated by 
// transformation trans.
int Symmetry::check_bondsum(int j, int m, Matrix4 *trans) {
  float tmp[3];
  vec_add(tmp, bondsum+3L*m, rcom);
  trans->multpoint3d(tmp, tmp);
  vec_sub(tmp, tmp, rcom);

  if (distance(bondsum+3L*j, tmp)>BONDSUMTOL) {
    if (verbose>4) {
      char buf[256];
      sprintf(buf, "Bond mismatch %i-->%i, bondsum distance = %.2f",
              atomindex[j], atomindex[m], 
              distance(bondsum+3L*j, tmp));
      msgInfo << buf << sendmsg;
    }
    //printf("bondsum 1:        {%.2f %.2f %.2f}\n", bondsum[3L*j], bondsum[3L*j+1], bondsum[3L*j+2]);
    //printf("bondsum 2:        {%.2f %.2f %.2f}\n", bondsum[3L*m], bondsum[3L*m+1], bondsum[3L*m+2]);
    //printf("bondsum 1 transf: {%.2f %.2f %.2f}\n", tmp[0], tmp[1], tmp[2]);
    return 0;
  }
  return 1;
}


// For each atom find the closest transformed image atom
// with the same chemical element. The result is an array
// containing the closest match atom index for each
// selected atom.
// XXX maybe put bondsum check in here?!
void Symmetry::identify_transformed_atoms(Matrix4 *trans,
                                          int &nmatches,
                                          int *(&matchlist)) {
  // get atom coordinates
  const float *posA = coor;

  float *posB = new float[3L*sel->selected];

  if (matchlist) delete [] matchlist;
  matchlist = new int[sel->selected];

  // generate transformed coordinates
  int i;
  for(i=0; i<sel->selected; i++) {
    trans->multpoint3d(posA+3L*i, posB+3L*i);
  }

  nmatches = 0;
  for(i=0; i<sel->selected; i++) {
    float minr2=999999.f;
    int k, kmatch = -1;
    for(k=0; k<sel->selected; k++) {
      // consider only pairs with identical atom types 
      if (atomtype[i]==atomtype[k]) {
#if 0
        if (checkbonds) {
          float imagebondsum[3];
          vec_add(imagebondsum, bondsum+3L*k, rcom);
          trans->multpoint3d(imagebondsum, imagebondsum);
          vec_sub(imagebondsum, imagebondsum, rcom);

          if (distance(bondsum+3L*i, imagebondsum)>BONDSUMTOL) {
            continue;
          }
        }
#endif
        float r2 = distance2(posA+3L*i, posB+3L*k);

        if (r2<minr2) { minr2 = r2; kmatch = k; }
      }
    }

    if (kmatch>=0) {
      matchlist[i] = kmatch;
      nmatches++;
      //printf("atom %i matches %i (atomtype %i)\n",
      //   atomindex[i], atomindex[kmatch], atomtype[i]);
    }
    else {
      // no match found
      matchlist[i] = i;

      if (verbose>3) {
        char buf[256];
        sprintf(buf, "No match for atom %i (atomtype %i)\n", atomindex[i], atomtype[i]);
        msgInfo << buf << sendmsg;
      }
    }
  }

  delete [] posB;
}


// Compute the RMSD between original and idealized coordinates
float Symmetry::ideal_coordinate_rmsd () {
  // get original atom coordinates
  const float *pos = sel->coordinates(mlist);

  float rmsdsum = 0.0;
  int i, j=0;
  for (i=0; i<sel->num_atoms; i++) {
    if (sel->on[i]) {
      rmsdsum += distance2(pos+3L*i, idealcoor+3L*j);
      j++;
    }
  }

  return sqrtf(rmsdsum / sel->selected);
}

// Compute the RMSD between original and idealized coordinates
int Symmetry::ideal_coordinate_sanity() {
  int i, j;
  float mindist2 = float(MINATOMDIST*MINATOMDIST);
  for (i=0; i<sel->selected; i++) {
    for (j=i+1; j<sel->selected; j++) {
      if (distance2(idealcoor+3l*i, idealcoor+3L*j)<mindist2)
        return 0;
    }
  }

  return 1;
}


// Idealize all symmetry elements so that they have certain geometries
// with respect to each other. This means, for instance, to make sure
// that the planes vertical to a C3 axis are evenly spaced by 60 degree
// angles and that their intersection is exactly the C3 axis.
//
// We start with idealizing horizontal and vertical planes with respect
// to the primary axis. If there is no unique primary axis (e.g. for
// Oh, Th, Ih) we take each of the highest order axes, idealize it wrt
// to the first one and then idealize the planes vertical and horizontal
// to them. Since all plajnes should be idealized by now we next
// idealize axes that are intersections of two planes. Then we adjust
// axes that are ideally bisectors of two planes sch as they occur in
// Dnd point groups. Finally we idealize axes that are not related to
// any planes such as in Dn groups. They should be perpedicular to the
// highest order axis and evenly spaced around it.
// Rotary reflections are simply brought to collinearity with their
// accompanying rotary axis (An improper axis never comes alone, e.g.
// S4 implies a C2 axis).
// Symmetry elements that could not be idealized are removed from the
// list.
void Symmetry::idealize_elements() {
  int i;

  // Array of flags indicating if a plane has been idealized
  int *idealp = NULL;
  if (planes.num()) {
    idealp = new int[planes.num()];
    memset(idealp, 0, planes.num()*sizeof(int));
  }


  // If there are axes present we will use the first axis as reference.
  // Since the axes were sorted by axis order before our reference will
  // be the axis with the highest order (or one of them).
  // Then align any horizontal and vertical planes with that axis.
  if (axes.num()) {

    // Make horizontal refplane truly perpendicular to reference axis
    if (horizontalplane>=0) {
      //msgInfo << "Idealizing horizontal plane " << horizontalplane << ", numplanes="
      //	      << planes.num() << sendmsg;
      vec_copy(planes[horizontalplane].v, axes[0].v);
      idealp[horizontalplane] = 1;
    } 

    if (numverticalplanes) {
      // Find the first vertical plane and align it with the reference axis so it
      // becomes our vertical reference plane.
      int vref = -1;
      for (i=0; i<planes.num(); i++) {
        if (planes[i].type&VERTICALPLANE) {
          //msgInfo << "Idealizing vertical plane " << i << sendmsg;
          vref = i;
          float normal[3];
          cross_prod(normal, axes[0].v, plane(vref));
          cross_prod(planes[vref].v, axes[0].v, normal);
          vec_normalize(planes[vref].v);
          idealp[vref] = 1;
          break;
        }
      }

      // Find other vertical planes and idealize them wrt the vertical reference plane.
      // To do so, we first have to align the found vertical plane with the reference
      // axis and then idealize the angle wrt the vertical reference plane.
      for (i=vref+1; i<planes.num(); i++) {
        if (planes[i].type&VERTICALPLANE) {
          align_plane_with_axis(plane(i), axes[0].v, plane(i));
          //msgInfo << "Idealizing plane " << i << " wrt vertical plane " << vref << sendmsg;

          float idealplane[3];
          if (!idealize_angle(planes[vref].v, axes[0].v, plane(i), idealplane, axes[0].order)) {
            if (verbose>3)
              msgInfo << "Couldn't idealize vertical plane " << i << sendmsg;
            continue;
          }

          int first = plane_exists(idealplane);
          if (first>=0) {
            // Equivalent plane exists already.
            // Since idealp[i] will not be set 1, this plane will be deleted
            // at the end of this function.
            if (!idealp[first]) {
              vec_copy(planes[first].v, idealplane);
              idealp[first] = 1;
            }
          } else {
            vec_copy(planes[i].v, idealplane);
            idealp[i] = 1;
          }
        }
      }
    }
     

  } else {    // No axis present

    // If we have only one plane, it is automatically ideal.
    if (planes.num()<=1) {
      if (idealp) delete [] idealp;
      return;
    }

    // Actually, if there is more than one plane, at least one axis
    // should have been found by intersection. If we don't have it
    // here, it was probably purged in find_symmetry_elements().
    // That means that at least one of the planes is most likely bad.
    // We will keep the only best plane and mark it as ideal.

    int bestplane = 0;
    float bestscore = 0.f;
    for (i=0; i<planes.num(); i++) {
      float score = planes[i].overlap*planes[i].weight;
      if (score>bestscore) {
        bestplane = i;
        bestscore = score;
      }
      idealp[i] = 0;
    }
    idealp[bestplane] = 1;
    
    if (verbose>4) {
      msgErr << "Found planes without intersection axis!" << sendmsg;
      char buf[256];
      for (i=0; i<planes.num(); i++) {
        sprintf(buf, "plane[%i] {% .3f % .3f % .3f} overlap = %f, weight = %d", i,
                planes[i].v[0], planes[i].v[1], planes[i].v[2],
                planes[i].overlap, planes[i].weight);
        msgErr << buf << sendmsg;
      }
    }
  }

  int numidealaxes = 0;
  int *ideala = NULL;
  if (axes.num()) {
    ideala = new int[axes.num()];
    memset(ideala, 0, axes.num()*sizeof(int));
    ideala[0] = 1;
  }

  int geometry = 0;
  if      (maxaxisorder==2) geometry = 4;
  else if (maxaxisorder==3) geometry = TETRAHEDRON;
  else if (maxaxisorder==4) geometry = OCTAHEDRON;
  else if (maxaxisorder==5) geometry = DODECAHEDRON;


  for (i=1; i<numaxes(); i++) {
    if (axes[i].order<maxaxisorder) continue;
    int j, vref=-1;
    
    //msgInfo << "Idealize axis " << i << " (C" << axes[i].order << ")" << sendmsg;

    // Find plane that contains both axes[0] and axis[i]
    // and idealize axis[i] wrt reference axis[0]
    for (j=0; j<numplanes(); j++) {
      if (orthogonal(planes[j].v, axes[i].v, orthogtol) &&
          orthogonal(planes[j].v, axes[0].v, orthogtol)) {
        if (!idealp[j]) {
          // The plane couldn't be idealized in the previous step.
          // Skip it.
          continue;
        }
	
        float idealaxis[3];
        if (!idealize_angle(axes[0].v, planes[j].v, axes[i].v,
                            idealaxis, geometry)) {
          if (verbose>4) {
            msgInfo << "Couldn't idealize axis " << i
                    << " wrt reference axis in plane " << j
                    << "." << sendmsg;
          }
          continue;
        }
        vec_copy(axes[i].v, idealaxis);
        vref = j;
        ideala[i] = 1;
        break;
      }
    }

    if (vref<0) continue;

    // Find planes that are vertical to the current axis and
    // idealize them wrt plane vref.
    for (j=0; j<planes.num(); j++) {
      if (idealp[j]) continue;

      // Check if plane is vertical to axis[i]; 
      if (orthogonal(planes[j].v, axes[i].v, orthogtol)) {
        // align the current plane with the idealized axis
        align_plane_with_axis(planes[j].v, axes[i].v, planes[j].v);
        //msgInfo << "Idealizing plane " <<j<< " wrt vertical plane " <<vref<< sendmsg;

        // idealize vertical plane wrt vref
        float idealplane[3];
        if (!idealize_angle(planes[vref].v, axes[i].v, plane(j), idealplane, axes[i].order)) {
          if (verbose>4) {
            msgInfo << "Vertical plane " <<j<< " couldn't be idealized!" << sendmsg;
          }
          continue;
        }
        int first = plane_exists(idealplane);
        if (first>=0) {
          // Equivalent plane exists already.
          // Since idealp[i] will not be set 1, this plane will be deleted
          // below.
          if (!idealp[first]) {
            vec_copy(planes[first].v, idealplane);
            idealp[first] = 1;
          }
        } else {
          vec_copy(planes[j].v, idealplane);
          idealp[j] = 1;
        }
      }
    }
  }

  // Delete all planes that could not be idealized.
  prune_planes(idealp);


  // By now all planes should be idealized, so we can idealize remaining
  // axes that are intersections of planes.
  for (i=0; i<numplanes(); i++) {
    int j;
    for (j=i+1; j<numplanes(); j++) {
      // Get intersection of the two planes
      float intersect[3];
      cross_prod(intersect, plane(i), plane(j));
      vec_normalize(intersect);

      int k;
      for (k=0; k<axes.num(); k++) {
        if (ideala[k]) continue;
	
        if (collinear(intersect, axes[k].v, collintol)) {
          // Idealize axis by intersection of two planes
          vec_copy(axes[k].v, intersect);
          ideala[k] = 1; numidealaxes++;
          break;
        }
      }
    }
  }

  float halfcollintol = cosf(float(DEGTORAD(5.0f)));
  // Idealize axes that are bisectors of two planes.
  // We place this search after the one for plane intersections because 
  // it showed that otherwise we would get false positive bisectors.
  for (i=0; i<numplanes(); i++) {
    int j;
    for (j=i+1; j<numplanes(); j++) {
      // Get first axis bisecting the two planes
      float bisect1[3];
      vec_add(bisect1, planes[i].v, planes[j].v);
      vec_normalize(bisect1);

      // Get second axis bisecting the two planes
      float bisect2[3], tmp[3];
      vec_negate(tmp, planes[i].v);
      vec_add(bisect2, tmp, planes[j].v);
      vec_normalize(bisect2);
      
      int k;
      int foundbi1=0, foundbi2=0;
      for (k=0; k<axes.num(); k++) {
        if (ideala[k]) continue;
	
        if (!foundbi1 && collinear(bisect1, axes[k].v, halfcollintol)) {
          //printf("idealized bisect1 %i (C%i)\n", k, axes[k].order);
          vec_copy(axes[k].v, bisect1);
          ideala[k] = 1; numidealaxes++;
          foundbi1 = 1;
          if (foundbi2) break;
        }
        if (!foundbi2 && collinear(bisect2, axes[k].v, halfcollintol)) {
          //printf("idealized bisect2 %i (C%i)\n", k, axes[k].order);
          vec_copy(axes[k].v, bisect2);
          ideala[k] = 1; numidealaxes++;
          foundbi2 = 1;
          if (foundbi1) break;
        }
      }
    }
  }
  

  // We might still have axes that are not related to planes or we didn't
  // have planes at all (as in Dn). These axes should be orthogonal to
  // the reference.
  if (numidealaxes<axes.num()) {
    int firstorth = -1;
    for (i=0; i<numaxes(); i++) {
      if (ideala[i]) continue;

      if (!orthogonal(axes[i].v, axes[0].v, orthogtol)) continue;

      // make axis truly orthogonal to reference      
      float tmp[3];
      cross_prod(tmp, axes[0].v, axes[i].v);
      vec_normalize(tmp);
      cross_prod(axes[i].v, tmp, axes[0].v);

      if (firstorth<0) {
        firstorth = i;
        ideala[i] = 1;
        continue;
      }
      
      // Idealize angle between current and first orthogonal axis
      if (!idealize_angle(axes[firstorth].v, axes[0].v, axes[i].v, tmp, axes[0].order)) {
        if (verbose>4) {
          msgInfo << "Couldn't idealize axis "<<i<<" to first orthogonal axis "
                  <<firstorth<< "!" << sendmsg;
        }
        continue;
      }
      
      vec_copy(axes[i].v, tmp);
      ideala[i] = 1;
    }
  }

  // Delete all axes that could not be idealized
  prune_axes(ideala);

  if (ideala) delete [] ideala;
  if (idealp) delete [] idealp;

  // Idealize rotary reflections with their corresponding rotary axes.
  // If no such axis is found we delete the rotary reflection.
  if (numrotreflect()) {
    int *idealr = new int[numrotreflect()];
    memset(idealr, 0, numrotreflect()*sizeof(int));

    for (i=0; i<numrotreflect(); i++) {
      int j, success=0;
      for (j=0; j<numaxes(); j++) {
        float dot = dot_prod(rotreflections[i].v, axes[j].v);
        if (fabs(dot) > collintol) {
          float tmp[3];
          if (dot>0) vec_negate(tmp, axes[j].v);
          else       vec_copy(tmp, axes[j].v);
	  
          vec_copy(rotreflections[i].v, tmp);
          rotreflections[i].type = j;
          success = 1;
          break;
        }
      }
      
      if (success) {
        // keep this rotary reflection
        idealr[i] = 1;
      }
    }

    prune_rotreflections(idealr);
    delete [] idealr;
  }

}

// Assuming that myaxis is close to a symmetry axis the function computes
// the idealized axis, i.e. the closest symmetry axis. The reference axis
// is rotated by the ideal angle around hub and the result is returned in
// idealaxis. All multiples of 180/reforder are considered ideal angles.
// Typically hub would correspond to an already idealized axis and one
// one would just use the order of thar axis for the parameter reforder.
// For instance, if the order of the principle axis is 3 than one would
// expect that other elements like perpendicular axes or vertical planes
// to be spaced at an angle of 180/3=60 degrees. Special values for
// reforder (TETRAHEDRON, OCTAHEDRON, DODECAHEDRON) request ideal angles
// (109, 90, 117) for the accordeing geometries.
// If the measured angle between myaxis and refaxis is within a tolerance
// of an ideal angle then the ideal axis is set and 1 is returned,
// otherwise the return value is 0.
int Symmetry::idealize_angle(const float *refaxis, const float *hub,
                             const float *myaxis, float *idealaxis, int reforder) {
  float alpha = angle(refaxis, myaxis);

  const float tetrahedron = float(RADTODEG(2.0f*atanf(sqrtf(2.0f)))); // tetrahedral angle ~109 deg
  const float octahedron = 90.0f;                           // octahedral angle = 90 deg
  const float dodecahedron = float(RADTODEG(acosf(-1/sqrtf(5.0f)))); // dodecahedral angle ~116.565 deg
  const float tol = 5.0f;

  int success = 0;
  float idealangle=0.0f;

  if      (reforder==TETRAHEDRON)  idealangle = tetrahedron;
  else if (reforder==OCTAHEDRON)   idealangle = octahedron;
  else if (reforder==DODECAHEDRON) idealangle = dodecahedron;

  if (reforder<0) {
    if (fabs(idealangle-alpha)<tol) {
      alpha = idealangle;
      success = 1;

    } else if (fabs(180.0-idealangle-alpha)<tol) {
      alpha = 180-idealangle;
      success = 1;
    }
  }

  int i;
  for (i=1; i<reforder; i++) {
    idealangle = i*180.0f/reforder;
    if (fabs(alpha-idealangle)<tol) {
      alpha = idealangle;
      success = 1;
      break;
    }
  }

  // Determine the ideal angle of the axis wrt the reference axis
  if (fabs(alpha)<tol || fabs(alpha-180)<tol) {
    // same axis
    alpha = 0;
    success = 1;
  }

  if (!success) {
    //printf("alpha = %.2f, tol = %.2f deg, reforder=%i\n", alpha, tol, reforder);
    return 0;
  }
  
  float normal[3];
  cross_prod(normal, refaxis, myaxis);
  if (dot_prod(hub, normal) < 0) alpha = -alpha;

  // Idealize the axis by rotating the reference axis
  // by the ideal angle around the hub.
  Matrix4 rot;
  rot.rotate_axis(hub, float(DEGTORAD(alpha)));
  rot.multpoint3d(refaxis, idealaxis);
  
  return 1;
}


// Determine which atoms are unique for the given rotary axis.
// Result is a modified array of flags 'uniquelist' where 1 indicates
// that the corresponding atom is unique.
void Symmetry::collapse_axis(const float *axis, int order,
                             int refatom, const int *matchlist,
                             int *(&connectedunique)) {
  int i;
  float refcoor[3];
  vec_sub(refcoor, coor+3L*refatom, rcom);

  // Project reference coordinate on the plane defined by the rotary axis
  float r0[3];
  vec_scale(r0, dot_prod(refcoor, axis), axis);
  vec_sub(r0, refcoor, r0);

  for (i=0; i<sel->selected; i++) {
    if (!uniqueatoms[i] || i==matchlist[i]) continue;

    // The unique atoms we have at this point will be scattered
    // randomly over the molecule. However, we want to find a
    // set of unique coordinates that is connected and confined
    // to one segment of the rotation.
    // Here we loop over all rotary images of the current unique
    // atom (regaring the given axis) and use the image that is
    // within the samme rotary segment as the reference atom
    // as the new unique atom.
    int image, found=0;
    int k = i;
    for (image=0; image<order; image++) {
      float tmp[3];
      vec_sub(tmp, idealcoor+3L*k, rcom);
      
      // Project coordinate on the plane defined by the rotary axis
      float r[3];
      vec_scale(r, dot_prod(tmp, axis), axis);
      vec_sub(r, tmp, r);
      
      // Measure angle between projected coordinates of current and
      // reference atom. If the atom is outside the angle range we
      // swap with its image.
      if (angle(r, r0) <= 180.0/order) {
        found = 1;
        break;
      }
      k = matchlist[k];
    }
    if (found && k!=i) {
      //printf("atom[%i] --> %i image=%i\n", i, k, image);
      uniqueatoms[i] = 0;
      uniqueatoms[k] = 1;
    }
  }

  // Find a connected set of unique atoms
  wrap_unconnected_unique_atoms(refatom, matchlist, connectedunique);
}


// Find best set of unique atoms, i.e. the set with the most
// atoms connected to atom 'root'. 
// Array matchlist shall contains the closest match of the
// same atom type for each selected atom.
void Symmetry::wrap_unconnected_unique_atoms(int root,
                                             const int *matchlist,
                                             int *(&connectedunique)) {
  int i, k=0;
  int numswapped = 0;

  // The first time we call this function we need to construct
  // an initial set of unique connected atoms connected to the
  // first atom.
  if (!connectedunique) {
    connectedunique = new int[sel->selected];
  }
  find_connected_unique_atoms(connectedunique, root);
  
  // Repeatedly loop through the list of unconnected unique atoms
  // and try to swap them with images that are connected of that
  // are directly bonded to connected ones.
  do {
    numswapped = 0;
    for (i=0; i<sel->selected; i++) {
      
      if (!uniqueatoms[i] || connectedunique[i]) continue;

      int swap = 0;
      int image = 0;
      int j = matchlist[i];
      //printf("uniqueatoms[%d] = %d matching %d:\n", i, uniqueatoms[i], j);
      while (j!=i && image<sel->selected) {
        //printf("i=%d, j=%d\n", i, j);
        if (connectedunique[j]) {
          // If the image is already in the bondtree we can just swap
          swap = 1;
        } else {
          // See if one of the atoms directly bonded to the image
          // is in the bondtree. If yes, we can swap.
          int k;
          for (k=0; k<bondsperatom[j].numbonds; k++) {
            int bondto = bondsperatom[j].bondto[k];
            if (connectedunique[bondto]) swap = 1;
          }
        }
        if (swap) break;
        
        j = matchlist[j];

        image++;
      }
      
      if (swap) {
        //printf("nonbonded unique atom %i --> %i (image %i), connected=%i\n", i, j, image, connectedunique[j]);
        uniqueatoms[i] = 0;
        uniqueatoms[j] = 1;
        connectedunique[j] = 1;
        numswapped++;
      }
    }

    if (k>=sel->selected) {
      msgErr << "Stop looping unconnected unique atoms" << sendmsg;
      break;
    }
    k++;

  } while (numswapped);

}


// Find all atoms currently marked unique that are within a bond
// tree of unique atoms rooted at 'root'. As a result the array
// 'connectedunique' is populated with flags.
void Symmetry::find_connected_unique_atoms(int *(&connectedunique),
                                           int root) {
  ResizeArray<int> leaves;
  ResizeArray<int> newleaves;
  leaves.append(root);

  memset(connectedunique, 0, sel->selected*sizeof(int));
  connectedunique[root] = 1;

  int numbonded = 1;
  int i;

  do {
    newleaves.clear();

    // Loop over all endpoints of the tree
    for (i=0; i<leaves.num(); i++) {
      int j = leaves[i];
      
      int k;
      for (k=0; k<bondsperatom[j].numbonds; k++) {
        int bondto = bondsperatom[j].bondto[k];

        if (uniqueatoms[bondto]&& !connectedunique[bondto]) {
          connectedunique[bondto] = 1;
          newleaves.append(bondto);
          numbonded++;
        }
      }

    }

    leaves.clear();
    for (i=0; i<newleaves.num(); i++) {
      leaves.append(newleaves[i]);
    }

  } while (newleaves.num());

  //   for (i=0; i<sel->selected; i++) {
  //     printf("connectedunique[%i] = %i, unique = %i\n", i, connectedunique[i], uniqueatoms[i]);
  //   }

}

// Determine the unique coordinates for the whole system.
// Result is a modified array of flags 'uniquelist' where 1
// indicates that the corresponding atom is unique.
// We begin assuming all atoms are unique and go through all
// symmetry elements subsequently flagging all atoms that can
// be obtained by the according symmetry operations as 
// not unique.
void Symmetry::unique_coordinates() {
  int i;
  for(i=0; i<sel->selected; i++) {
    uniqueatoms[i] = 1;
  }

  // We use the first atom as starting point for searching 
  // connected unique atoms. In case this atom is at the
  // center of mass it will always have itself as image and
  // we better use the second atom.
  float refcoor[3];
  int refatom = 0;
  vec_sub(refcoor, coor, rcom);
  if (norm(refcoor)<0.1) {
    refatom = 1;
    vec_sub(refcoor, coor+3L*refatom, rcom);
  }

  int *connectedunique = NULL;

  // -- Inversion --
  if (inversion) {
    Matrix4 inv;
    inv.translate(rcom[0], rcom[1], rcom[2]);
    inv.scale(-1.0);
    inv.translate(-rcom[0], -rcom[1], -rcom[2]);

    // Flag equivalent atoms
    int *matchlist = unique_coordinates(&inv);

    int j;
    for(j=0; j<sel->selected; j++) {
      if (!uniqueatoms[j] || j==matchlist[j]) continue;
      uniqueatoms[j] = 0;
      uniqueatoms[matchlist[j]] = 1;
    }

    wrap_unconnected_unique_atoms(refatom, matchlist, connectedunique);

    if (matchlist) delete [] matchlist;
  }

  // -- Rotary axes --
  for (i=0; i<numaxes(); i++) {
    Matrix4 rot;
    rot.translate(rcom[0], rcom[1], rcom[2]);
    rot.rotate_axis(axes[i].v, float(VMD_TWOPI)/axes[i].order);
    rot.translate(-rcom[0], -rcom[1], -rcom[2]);
    
    // Flag equivalent atoms
    int *matchlist = unique_coordinates(&rot);
    
    // Turn the unique list into a list of unique, connected
    // atoms that are within one segment of the rotation
    // (e.g within 90 degree for a 4th order axes).
    collapse_axis(axes[i].v, axes[i].order, refatom, matchlist,
                  connectedunique);
    
    if (matchlist) delete [] matchlist;
  }

  // -- Planes --
  for (i=0; i<numplanes(); i++) {
    Matrix4 mirror;
    mirror.translate(rcom[0], rcom[1], rcom[2]);
    mirror.scale(-1.0);  // inversion
    mirror.rotate_axis(planes[i].v, float(VMD_PI));
    mirror.translate(-rcom[0], -rcom[1], -rcom[2]);

    // Flag equivalent atoms
    int *matchlist = unique_coordinates(&mirror);

    int j;
    for(j=0; j<sel->selected; j++) {
      if (!uniqueatoms[j] || j==matchlist[j]) continue;
      
      float tmp[3];
      vec_sub(tmp, coor+3L*j, rcom);
      if (behind_plane(planes[i].v, tmp)!=behind_plane(planes[i].v, refcoor)) {
        uniqueatoms[j] = 0;
        uniqueatoms[matchlist[j]] = 1;
      }
    }
    if (matchlist) delete [] matchlist;
  }

  // -- Rotary reflections --
  for (i=0; i<numrotreflect(); i++) {
    Matrix4 rotref;
    rotref.translate(rcom[0], rcom[1], rcom[2]);
    rotref.rotate_axis(rotreflections[i].v, float(VMD_TWOPI)/rotreflections[i].order);
    rotref.scale(-1.0);  // inversion
    rotref.rotate_axis(rotreflections[i].v, float(VMD_PI));
    rotref.translate(-rcom[0], -rcom[1], -rcom[2]);
    
    // Flag equivalent atoms
    int *matchlist = unique_coordinates(&rotref);
    
    collapse_axis(rotreflections[i].v, rotreflections[i].order, refatom, matchlist, connectedunique);

    if (matchlist) delete [] matchlist;
  }

  if (connectedunique) delete [] connectedunique;
}


// Goes through the list of atoms matching with original
// ones after a transformation has been applied and assigns
// a zero to the array of unique atom flags if the matching
// index is greater than the original one.
int* Symmetry::unique_coordinates(Matrix4 *trans) {
  int nmatches;
  int *matchlist = NULL;

  // For each atom find the closest transformed image atom with
  // the same chemical element. The array matchlist will contain
  // the closest match atom index (into the collapsed list of
  // selected atoms) for each selected atom.
  identify_transformed_atoms(trans, nmatches, matchlist);
  
  int j;
  for(j=0; j<sel->selected; j++) {
    if (!uniqueatoms[j]) continue;
    int m = matchlist[j];
    
    if (m<0) continue; // no match found;
    
    if (m>j) uniqueatoms[m] = 0;
  }
  
  return matchlist;
}


/// Determine the point group from symmetry elements
void Symmetry::determine_pointgroup() {
  if (linear) {
    if (inversion) pointgroup = POINTGROUP_DINFH;
    else           pointgroup = POINTGROUP_CINFV;
  
    pointgrouporder = -1;
  }

  else if (sphericaltop && maxaxisorder>=3 && axes[0].order>=2) {
    if (maxaxisorder==3) {
      if (numplanes()) {
        if (inversion) pointgroup = POINTGROUP_TH;
        else pointgroup = POINTGROUP_TD;
      }
      else pointgroup = POINTGROUP_T;
    }
    else if (maxaxisorder==4) {
      if (numplanes() || inversion) pointgroup = POINTGROUP_OH;
      else                          pointgroup = POINTGROUP_O;
    }
    else if (maxaxisorder==5) {
      if (numplanes() || inversion) pointgroup = POINTGROUP_IH;
      else                          pointgroup = POINTGROUP_I;

    }
    else pointgroup = POINTGROUP_UNKNOWN;

  }

  else if (numaxes()) {
    // If n is the higest Cn axis order, are there n C2 axes
    // perpendicular to Cn?
    int i;
    int perpC2 = 0;
    for (i=0; i<numaxes(); i++) {
      if (axes[i].order==2 && (axes[i].type & PERPENDICULAR_AXIS)) {
        perpC2++;
      }
    }

    if (perpC2==maxaxisorder) {
      if (horizontalplane>=0) pointgroup = POINTGROUP_DNH;
      else {
        // Are there n dihedral mirror planes Sd bisecting the angles
        // formed by pairs of C2 axes?
        if (numdihedralplanes==maxaxisorder) {
          pointgroup = POINTGROUP_DND;
        }
        else {
          pointgroup = POINTGROUP_DN;
        }
      }

      pointgrouporder = maxaxisorder;
    }

    else {
      if (horizontalplane>=0) pointgroup = POINTGROUP_CNH;
      else {
        // Are there n planes vertical to the highest Cn axis?
        if (numverticalplanes==maxaxisorder) {
          pointgroup = POINTGROUP_CNV;
        }
        else {
          if (numS2n()) {
            pointgroup = POINTGROUP_S2N;
          }
          else {
            pointgroup = POINTGROUP_CN;
          }
        }
      }
      pointgrouporder = maxaxisorder;

    }
  }

  else { // numaxes==0
    if (numplanes()==1) pointgroup = POINTGROUP_CS;
    else {
      if (inversion) pointgroup = POINTGROUP_CI;
      else           pointgroup = POINTGROUP_C1;
    }
  }
}


// Determine level in the pointgroup hierarchy
// As far as I understand only crystallographic point
// groups can be ranked in hierarchy, but we can use
// heuristics to rank the others (e.g. I, Ih, C5, ...)
int Symmetry::pointgroup_rank(int pg, int order) {
  if (pg==POINTGROUP_C1) return 1;
  if (pg==POINTGROUP_CS || pg==POINTGROUP_CI) return 2;
  if (pg==POINTGROUP_CN) {
    return 1+numprimefactors(order);
  }
  if (pg==POINTGROUP_S2N) {
    return 1+numprimefactors(order*2);
  }
  if (pg==POINTGROUP_DN || pg==POINTGROUP_CNV ||
      pg==POINTGROUP_CNH) {
    return 2+numprimefactors(order);
  }
  if (pg==POINTGROUP_DND || pg==POINTGROUP_DNH) {
    return 3+numprimefactors(order);
  }
  if (pg==POINTGROUP_CINFV) return 3;
  if (pg==POINTGROUP_DINFH || pg==POINTGROUP_T) return 4;
  if (pg==POINTGROUP_TD    || pg==POINTGROUP_TH ||
      pg==POINTGROUP_O) {
    return 5; 
  }
  if (pg==POINTGROUP_OH || pg==POINTGROUP_I) return 6;
  if (pg==POINTGROUP_IH) return 7;
  return 0;
}


// Generate transformation matrix that orients the molecule
// according to the GAMESS 'master frame'.
//
// From the GAMESS documentation:
// ------------------------------
// The 'master frame' is just a standard orientation for                       
// the molecule.  By default, the 'master frame' assumes that                      
//    1.   z is the principal rotation axis (if any),                             
//    2.   x is a perpendicular two-fold axis (if any),                           
//    3.  xz is the sigma-v plane (if any), and                                   
//    4.  xy is the sigma-h plane (if any).                                       
// Use the lowest number rule that applies to your molecule.
//                                                                               
//         Some examples of these rules:                                           
// Ammonia (C3v): the unique H lies in the XZ plane (R1,R3).                       
// Ethane (D3d): the unique H lies in the YZ plane (R1,R2).                        
// Methane (Td): the H lies in the XYZ direction (R2).  Since                      
//          there is more than one 3-fold, R1 does not apply.                      
// HP=O (Cs): the mirror plane is the XY plane (R4).                               
//                                                                                
// In general, it is a poor idea to try to reorient the                            
// molecule.  Certain sections of the program, such as the                         
// orbital symmetry assignment, do not know how to deal with                       
// cases where the 'master frame' has been changed.                                
//                                                                                
// Linear molecules (C4v or D4h) must lie along the z axis,                        
// so do not try to reorient linear molecules.                            

// Note:
// With perpendicular two-fold axis they seem to mean any C2 axis
// that is not collinear with the principal axis. It does not
// have to be orthogonal to the principal axis. Otherwise the methane
// example would not work.

// We must use the first or the first two rules that apply. 
// This is always sufficient for proper orientation.
// If more that two rules apply we must ignore the additional
// ones, otherwise the orientation will be messed up.

void Symmetry::orient_molecule() {
  if (pointgroup==POINTGROUP_C1) return;

  //msgInfo << "Creating standard orientation:" << sendmsg;

  // Special case: linear molecules
  if (linear) {
    // Bring X along Z
    orient.transvec(0, 0, 1);
    // Bring axis along X
    orient.transvecinv(axes[0].v[0], axes[0].v[1], axes[0].v[2]);

    orient.translate(-rcom[0], -rcom[1], -rcom[2]);
    return;
  } 

  int i;
  Matrix4 rot;

  // GAMESS rule #1:
  // z is the principal rotation axis 
  // (x is principal axis of inertia with smallest eigenvalue)

  if (!sphericaltop && numaxes()) {
    //msgInfo << "  Applying rule 1" << sendmsg;

    // Bring X along Z
    rot.transvec(0, 0, 1);
    // Bring first axis along X
    rot.transvecinv(axes[0].v[0], axes[0].v[1], axes[0].v[2]);

    // Find an axis of inertia that is orthogonal to the primary axis
    int j;
    for (j=0; j<=2; j++) {
      if (orthogonal(axes[0].v, inertiaaxes[j], orthogtol)) break;
    }
    float *ortho_inertiaaxis = inertiaaxes[j];

    // Apply same rotation to selected orthogonal axis of inerta
    // and then get transform m to bring it along X.
    Matrix4 m;
    float tmp[3];
    rot.multpoint3d(ortho_inertiaaxis, tmp);
    m.transvecinv(tmp[0], tmp[1], tmp[2]);

    // next 2 lines: postmultiply: m*rot 
    m.multmatrix(rot);
    rot.loadmatrix(m);
  }

  // GAMESS rule #2:
  // x is a 'perpendicular' (=noncollinear) two-fold axis (if any)
  int orthC2 = -1;
  for (i=1; i<numaxes(); i++) {
    if (axes[i].order==2 && 
        (orthogonal(axes[i].v, axes[0].v, orthogtol) ||
         (pointgroup>=POINTGROUP_T && pointgroup<=POINTGROUP_IH))) {
      orthC2 = i;
      break;
    }
  }
  if (orthC2>=0) {
    //msgInfo << "  Applying rule 2\n" << sendmsg;

    // Bring orth C2 axis along X
    float tmp[3];
    rot.multpoint3d(axes[orthC2].v, tmp);
    Matrix4 m;
    m.transvecinv(tmp[0], tmp[1], tmp[2]);

    // next 2 lines: postmultiply: m*rot 
    m.multmatrix(rot);
    rot.loadmatrix(m);
  }

  // GAMESS rule #3:
  // xz is the sigma-v plane (if any)
  if (numverticalplanes && orthC2<0) {
    for (i=0; i<numplanes(); i++) {
      if (planes[i].type==VERTICALPLANE) break;
    }
    //msgInfo << "Applying rule 3\n" << sendmsg;

    Matrix4 m;

    // Bring X along Y
    float Y[]={0, 1, 0};
    m.transvec(Y[0], Y[1], Y[2]);

    // Bring plane normal along X    
    float tmp[3];
    rot.multpoint3d(planes[i].v, tmp);
    m.transvecinv(tmp[0], tmp[1], tmp[2]);

    // next 2 lines: postmultiply: m*rot 
    m.multmatrix(rot);
    rot.loadmatrix(m);
  }

  // GAMESS rule #4:
  // xy is the sigma-h plane (if any)
  // If the pointgroup is Cs then treat the only plane as horizontal
  if ((horizontalplane>=0 && !numverticalplanes && orthC2<0) ||
      pointgroup==POINTGROUP_CS) {
    //msgInfo << "Applying rule 4\n" << sendmsg;
    if (pointgroup==POINTGROUP_CS) i = 0;
    else   i = horizontalplane;

    Matrix4 m;
    float Z[]={0, 0, 1};

    // Bring X along Z
    m.transvec(Z[0], Z[1], Z[2]);

    // Bring plane normal along X    
    float tmp[3];
    rot.multpoint3d(planes[i].v, tmp);
    m.transvecinv(tmp[0], tmp[1], tmp[2]);

    // next 2 lines: postmultiply: m*rot 
    m.multmatrix(rot);
    rot.loadmatrix(m);
  }

  if (pointgroup>=POINTGROUP_T && pointgroup<=POINTGROUP_IH) {
    //msgInfo << "Applying rule 5\n" << sendmsg;

    // Find a principal axis with a unique atom
    int found = 0;
    float uniqueaxis[3];
    for (i=1; i<numaxes(); i++) {
      if (axes[i].order<maxaxisorder) break;
      int j;
      for(j=0; j<sel->selected; j++) {
        if (!uniqueatoms[j]) continue;

        float tmpcoor[3];
        vec_sub(tmpcoor, coor+3L*j, rcom);
        if (norm(tmpcoor)<0.1) continue;

        if (collinear(axes[i].v, tmpcoor, collintol)) {
          if (dot_prod(axes[i].v, tmpcoor)>0)
            vec_copy(uniqueaxis, axes[i].v);
          else
            vec_negate(uniqueaxis, axes[i].v);
          found = 1;
        }
      }
    }

    if (!found) 
      msgErr << "orient_molecule(): Couldn't find axis with unique atom!" << sendmsg;

    Matrix4 m;
    float XYZ[]={1, 1, 1};

    // Bring X along XYZ
    m.transvec(XYZ[0], XYZ[1], XYZ[2]);

    // Bring unique axis along X
    float tmp[3];
    rot.multpoint3d(uniqueaxis, tmp);
    m.transvecinv(tmp[0], tmp[1], tmp[2]);

    // next 2 lines: postmultiply: m*rot 
    m.multmatrix(rot);
    rot.loadmatrix(m);
  }

  orient.multmatrix(rot);
  orient.translate(-rcom[0], -rcom[1], -rcom[2]);
}


/// Print a summary about how many symmetry elements of which type
/// were found.
void Symmetry::print_statistics() {
  int i;

#if 0
  for (i=0; i<numplanes(); i++) {
    printf("plane[%i]: weight=%3i, overlap=%.2f\n", i, planes[i].weight, planes[i].overlap);
  }

  for (i=0; i<numaxes(); i++) {
    printf("axis[%i]: order=%i, weight=%3i, overlap=%.2f {%.2f %.2f %.2f}\n", i, axes[i].order, axes[i].weight, axes[i].overlap, axes[i].v[0], axes[i].v[1], axes[i].v[2]);
  }

  for (i=0; i<numrotreflect(); i++) {
    printf("rotrefl[%i]: order=%i, weight=%3i, overlap=%.2f {%.2f %.2f %.2f}\n", i, rotreflections[i].order, rotreflections[i].weight, rotreflections[i].overlap, rotreflections[i].v[0], rotreflections[i].v[1], rotreflections[i].v[2]);
  }
#endif

  char buf [256];
  msgInfo << sendmsg;
  msgInfo << "Summary of symmetry elements:" << sendmsg;
  msgInfo << "+===============================+" << sendmsg;
  msgInfo << "| inversion center:         " << (inversion ? "yes" : "no ") 
          << " |" << sendmsg;

  if (planes.num()) {
    int havehorizplane = 0;
    msgInfo << "|                               |" << sendmsg;
    if (horizontalplane>=0) {
      msgInfo << "| horizonal planes:          1  |" << sendmsg;
      havehorizplane = 1;
    }
    if (numverticalplanes) {
      sprintf(buf, "|  vertical planes:         %2d  |", numverticalplanes);
      msgInfo << buf << sendmsg;
    }
    if (numdihedralplanes) {
      sprintf(buf, "|  (%-2d of them dihedral)        |", numdihedralplanes);
      msgInfo << buf << sendmsg;
    }
    int numregplanes = numplanes()-numverticalplanes-havehorizplane;
    if (numregplanes) {
      sprintf(buf, "|   regular planes:         %2d  |", numregplanes);
      msgInfo << buf << sendmsg;
    }
    if (numplanes()>1) {
      msgInfo << "| ----------------------------- |" << sendmsg;
      sprintf(buf, "|     total planes:         %2d  |", numplanes());
      msgInfo << buf << sendmsg;
    }
  }

  if (axes.num()) {
    msgInfo << "|                               |" << sendmsg;
    if (elementsummary.Cinf) {
      sprintf(buf, "| Cinf  rotation axes:      %2d  |", (int)elementsummary.Cinf);
      msgInfo << buf << sendmsg;
    }
    for (i=maxaxisorder; i>=1; i--) {
      if (elementsummary.C[i]) {
        sprintf(buf, "| C%-4i rotation axes:      %2d  |", i, elementsummary.C[i]);
        msgInfo << buf << sendmsg;
      }
    }
    if (numaxes()>1) {
      msgInfo << "| ----------------------------- |" << sendmsg;
      sprintf(buf, "| total rotation axes:      %2d  |", numaxes());
      msgInfo << buf << sendmsg;
    }
  }
    
  if (rotreflections.num()) {
    msgInfo << "|                               |" << sendmsg;
    for (i=maxrotreflorder; i>=3; i--) {
      if (elementsummary.S[i-3]) {
        sprintf(buf, "| S%-4i rotary reflections: %2d  |", i, elementsummary.S[i-3]);
        msgInfo << buf << sendmsg;
      }
    }
    if (rotreflections.num()>1) {
      msgInfo << "| ----------------------------- |" << sendmsg;
      sprintf(buf, "| total rotary reflections: %2ld  |", long(rotreflections.num()));
      msgInfo << buf << sendmsg;
    }
  }
  msgInfo << "+===============================+" << sendmsg;

  msgInfo << sendmsg;
  msgInfo << "Element summary: " << elementsummarystring << sendmsg;
}


// Build a summary matrix of found symmetry elements
void Symmetry::build_element_summary() {
  memset(&elementsummary, 0, sizeof(ElementSummary));

  elementsummary.inv   = inversion;
  elementsummary.sigma = planes.num();

  int i;
  for (i=0; i<numaxes(); i++) {
    if (axes[i].order==INFINITE_ORDER) {
      elementsummary.Cinf++;
    } else if (axes[i].order<=MAXORDERCN) {
      int j;
      for (j=2; j<=axes[i].order; j++) {
        if (axes[i].order % j == 0) {
          (elementsummary.C[j])++;
        }
      }
    }
  }

  for (i=0; i<numrotreflect(); i++) {
    if (rotreflections[i].order<=2*MAXORDERCN) {
      (elementsummary.S[rotreflections[i].order-3])++;
    }
  }
}


// Create human-readable string summarizing symmetry elements
// given in summary.
void Symmetry::build_element_summary_string(ElementSummary summary, char *(&sestring)) {
  int i ;
  if (sestring) delete [] sestring;
  sestring = new char[2 + 10L*(MAXORDERCN+2*MAXORDERCN+summary.Cinf
                              +(summary.sigma?1:0)+summary.inv)];
  char buf[100] ;
  sestring[0] = '\0';

  if (inversion) strcat(sestring, "(inv) ");

  if (summary.sigma==1) strcat(sestring, "(sigma) ");
  if (summary.sigma>1) {
    sprintf(buf, "%d*(sigma) ", summary.sigma);
    strcat(sestring, buf);
  }

  if (summary.Cinf==1)
    strcat(sestring, "(Cinf) ");
  else if (summary.Cinf>1) {
    sprintf(buf, "%d*(Cinf) ", summary.Cinf);
    strcat(sestring, buf);
  }
  
  for (i=MAXORDERCN; i>=2; i--) {
    if (summary.C[i]==1) {
      sprintf(buf, "(C%d) ", i);
      strcat(sestring, buf);
    }
    else if (summary.C[i]>1) {
      sprintf(buf, "%d*(C%d) ", summary.C[i], i);
      strcat(sestring, buf);
    }
  }
  
  for (i=2*MAXORDERCN; i>=3; i--) {
    if (summary.S[i-3]==1) {
      sprintf(buf, "(S%d) ", i);
      strcat(sestring, buf);
    }
    else if (summary.S[i-3]>1) {
      sprintf(buf, "%d*(S%d) ", summary.S[i-3], i);
      strcat(sestring, buf);
    }
  }
}


// For the given point group name compare the ideal numbers of 
// symmetry elements with the found ones and determine which
// elements are missing and which were found in addition to the
// ideal ones.
void Symmetry::compare_element_summary(const char *pointgroupname) {
  missingelementstring[0]    = '\0';
  additionalelementstring[0] = '\0';

  if (!strcmp(pointgroupname, "Unknown")) return;

  unsigned int i;
  for (i=0; i<NUMPOINTGROUPS; i++) {
    if (!strcmp(pointgroupname, pgdefinitions[i].name)) {

      if      (elementsummary.inv<pgdefinitions[i].summary.inv) 
        strcat(missingelementstring, "(inv) ");
      else if (elementsummary.inv>pgdefinitions[i].summary.inv) 
        strcat(additionalelementstring, "(inv) ");

      char buf[100];
      if    (elementsummary.sigma<pgdefinitions[i].summary.sigma) {
        sprintf(buf, "%i*(sigma) ",
                pgdefinitions[i].summary.sigma-elementsummary.sigma);
        strcat(missingelementstring, buf);
      }
      else if (elementsummary.sigma>pgdefinitions[i].summary.sigma) {
        sprintf(buf, "%i*(sigma) ",
                elementsummary.sigma-pgdefinitions[i].summary.sigma);
        strcat(additionalelementstring, buf);
      }

      int j;
      for (j=MAXORDERCN; j>=2; j--) {
        if (elementsummary.C[j]<pgdefinitions[i].summary.C[j]) {
          sprintf(buf, "%i*(C%i) ", pgdefinitions[i].summary.C[j]-elementsummary.C[j], j);
          strcat(missingelementstring, buf);
        }
        if (elementsummary.C[j]>pgdefinitions[i].summary.C[j]) {
          sprintf(buf, "%i*(C%i) ", elementsummary.C[j]-pgdefinitions[i].summary.C[j], j);
          strcat(additionalelementstring, buf);
        }
      }

      for (j=2*MAXORDERCN; j>=3; j--) {
        if (elementsummary.S[j-3]<pgdefinitions[i].summary.S[j-3]) {
          sprintf(buf, "%i*(S%i) ",
                  pgdefinitions[i].summary.S[j-3]-elementsummary.S[j-3], j);
          strcat(missingelementstring, buf);
        }
        if (elementsummary.S[j-3]>pgdefinitions[i].summary.S[j-3]) {
          sprintf(buf, "%i*(S%i) ",
                  elementsummary.S[j-3]-pgdefinitions[i].summary.S[j-3], j);
          strcat(additionalelementstring, buf);
        }
      }
      break;
    }
  }
}


void Symmetry::print_element_summary(const char *pointgroupname) {
  int i, found = 0;
  for (i=0; i<(int)NUMPOINTGROUPS; i++) {
    if (!strcmp(pointgroupname, pgdefinitions[i].name)) {
      found = 1;
      break;
    }
  }
  if (!found) return;

  char *idealstring=NULL;
  build_element_summary_string(pgdefinitions[i].summary, idealstring);

  // If the found elements differ from the ideal elements
  // print a comparison
  if (strcmp(idealstring, elementsummarystring)) {
    char buf[256];
    sprintf(buf, "Ideal elements (%5s): %s\n", pgdefinitions[i].name, idealstring);
    msgInfo << buf << sendmsg;
    sprintf(buf, "Found elements (%5s): %s\n", pointgroupname, elementsummarystring);
    msgInfo << buf << sendmsg;
    if (strlen(additionalelementstring))
      msgInfo << "Additional elements:    " << additionalelementstring << sendmsg;
    if (strlen(missingelementstring))
      msgInfo << "Missing elements:       " << missingelementstring    << sendmsg;
  }
  delete [] idealstring;
}


// Draws a atom-colored spheres for each atom at the transformed
// position and labels them according to the atom ID.
// Use for debuggging only!
void Symmetry::draw_transformed_mol(Matrix4 rot) {
  int i;
  Molecule *mol = mlist->mol_from_id(sel->molid());
  MoleculeGraphics *gmol = mol->moleculeGraphics();
  const float *pos = sel->coordinates(mlist);
  for (i=0; i<sel->num_atoms; i++) {
    switch (mol->atom(i)->atomicnumber) {
    case 1:
      gmol->use_color(8);
      break;
    case 6:
      gmol->use_color(10);
      break;
    case 7:
      gmol->use_color(0);
      break;
    case 8:
      gmol->use_color(1);
      break;
    default:
      gmol->use_color(2);
      break;
    }
    float p[3];
    rot.multpoint3d(pos+3L*i, p);
    gmol->add_sphere(p, 2*sigma, 16);
    char tmp[10];
    sprintf(tmp, "     %i", i);
    gmol->add_text(p, tmp, 1, 1.0f);
  }
}


/***********  HELPER FUNCTIONS  ************/

#if 0
static inline bool coplanar(const float *normal1, const float *normal2, float tol) {
  return collinear(normal1, normal2, tol);
}
#endif

static inline bool collinear(const float *axis1, const float *axis2, float tol) {
  if (fabs(dot_prod(axis1, axis2)) > tol) return 1;
  return 0;
}

static inline bool orthogonal(const float *axis1, const float *axis2, float tol) {
  if (fabs(dot_prod(axis1, axis2)) < tol) return 1;
  return 0;
}

static inline bool behind_plane(const float *normal, const float *coor) {
  return (dot_prod(normal, coor)>0.01);
}

// currently unused
static void align_plane_with_axis(const float *normal, const float *axis, float *alignedplane) {
  float inplane[3];
  cross_prod(inplane, axis, normal);
  cross_prod(alignedplane, inplane, axis);
  vec_normalize(alignedplane);
}


// Returns 1 if x is a prime number
static int isprime(int x) {
  int i;
  for (i=2; i<x; i++) {
    if (!(x%i)) return 0;
  }
  return 1;
}


// Returns the number of prime factors for x
static int numprimefactors(int x) {
  int i, numfac=0;
  for (i=2; i<=x; i++) {
    if (!isprime(i)) continue;
    if (!(x%i)) {
      x /= i;
      numfac++;
      i--;
    }
  }
  return numfac;
}


inline int Symmetry::find_collinear_axis(const float *myaxis) {
  int i;
  for (i=0; i<axes.num(); i++) {
    if (collinear(myaxis, axes[i].v, collintol)) {
      return i;
    }
  }
  return -1;
}

/// Return index of the first found plane that is coplanar to myplane
inline int Symmetry::plane_exists(const float *myplane) {
  int i;
  for (i=0; i<planes.num(); i++) {
    if (collinear(myplane, planes[i].v, collintol)) {
      return i;
    }
  }
  return -1;
}


// Checks if the molecule is planar with respect to the given plane normal.
// Rotates the molecule so that he given normal is aligned with the x-axis
// and then tests if all x coordinates are close to zero.
int Symmetry::is_planar(const float *normal) {
  Matrix4 alignx;
  alignx.transvecinv(normal[0], normal[1], normal[2]);
  int j, k;
  float xmin=0.0, xmax=0.0;
  for (j=0; j<sel->selected; j++) {
    float tmpcoor[3];
    alignx.multpoint3d(coor+3L*j, tmpcoor);
    xmin = tmpcoor[0];
    xmax = tmpcoor[0];
    break;
  }
  
  for (k=j+1; k<sel->selected; k++) {
    float tmpcoor[3];
    alignx.multpoint3d(coor+3L*k, tmpcoor);
    if      (tmpcoor[0]<xmin) xmin = tmpcoor[0];
    else if (tmpcoor[0]>xmax) xmax = tmpcoor[0];
  }
  
  if (xmax-xmin < 1.5*sigma) return 1;
  
  return 0;
}


// Assign bond topology information to each atom
void Symmetry::assign_bonds() {
  Molecule *mol = mlist->mol_from_id(sel->molid());

  // Assign an array of atom indexes associating local indexes
  // with the global ones.
  int i, j=0;
  for (i=0; i<sel->num_atoms; i++) {
    if (sel->on[i]) atomindex[j++] = i;
  }

  for (i=0; i<sel->selected; i++) {
    int j = atomindex[i];

    bondsperatom[i].numbonds = 0;
    if (bondsperatom[i].bondto) delete [] bondsperatom[i].bondto;
    if (bondsperatom[i].length) delete [] bondsperatom[i].length;
    bondsperatom[i].bondto = new int[mol->atom(j)->bonds];
    bondsperatom[i].length = new float[mol->atom(j)->bonds];

    int k;
    for (k=0; k<mol->atom(j)->bonds; k++) {
      int bondto = mol->atom(j)->bondTo[k];

      // Only consider bonds to atoms within the selection
      if (!sel->on[bondto]) continue;

      float order = mol->getbondorder(j, k);
      if (order<0.f) order = 1.f;

      if (bondto > j) {	
        // Find the local index of the bonded atom
        int m;
        int found = 0;
        for (m=i+1; m<sel->selected; m++) {
          if (atomindex[m]==bondto) { found = 1; break; }
        }

        // If the local index is not found then this means that
        // the bonded atom is outside the selection.
        if (found) {
          // Add a new bond to the list
          Bond b;
          b.atom0 = i; // index into collapsed atomlist
          b.atom1 = m; // index into collapsed atomlist
          b.order = order;
          //printf("i=%i, m=%i, j=%i, bondto=%i\n", i, m, j, bondto);
          b.length = distance(coor+3L*i, coor+3L*m);
          bonds.append(b);
        }
      }
    }
  }

  for (i=0; i<bonds.num(); i++) {
    if (verbose>3) {
      char buf[256];
      sprintf(buf, "Bond %d: %d--%d %.1f L=%.2f", i,
              atomindex[bonds[i].atom0],
              atomindex[bonds[i].atom1],
              bonds[i].order, bonds[i].length);
      msgInfo << buf << sendmsg;
    }
    int numbonds;
    numbonds = bondsperatom[bonds[i].atom0].numbonds;
    bondsperatom[bonds[i].atom0].bondto[numbonds] = bonds[i].atom1;
    bondsperatom[bonds[i].atom0].length[numbonds] = bonds[i].length;
    bondsperatom[bonds[i].atom0].numbonds++;

    numbonds = bondsperatom[bonds[i].atom1].numbonds;
    bondsperatom[bonds[i].atom1].bondto[numbonds] = bonds[i].atom0;
    bondsperatom[bonds[i].atom1].length[numbonds] = bonds[i].length;
    bondsperatom[bonds[i].atom1].numbonds++;
  }
}


// Build a list of vectors that are the sum of all bond directions
// weighted by bondorder. This serves like a checksum for bonding
// topology and orientation for each atom. Comparing the bondsum
// between atoms and its transformed images can be used to purge
// symmetry elements that would reorient the bonding pattern.
// For example, two sp2 carbons will always have the same atom type
// but the bondsum of will point in direction of the double bond, 
// thus allowing to detect if the bonding pattern has changed.
void Symmetry::assign_bondvectors() {
  Molecule *mol = mlist->mol_from_id(sel->molid());
  memset(bondsum, 0, 3L*sel->selected*sizeof(float));
  int i;
  for (i=0; i<sel->selected; i++) {
    int k;
    for (k=0; k<bondsperatom[i].numbonds; k++) {
      int bondto = bondsperatom[i].bondto[k];

      // Only consider bonds to atoms within the selection
      if (!sel->on[bondto]) continue;

      int j = atomindex[i];
      float order = mol->getbondorder(j, k);
      if (order<0.f) order = 1.f;

      float bondvec[3];
      vec_sub(bondvec, coor+3L*bondto, coor+3L*i);
      vec_normalize(bondvec);
      vec_scaled_add(bondsum+3L*i, order, bondvec);
    }
  }
}

// Copy coordinates of selected atoms into local array and assign
// and atomtypes based on chemial element and topology.
// Since we also use this in measure_trans_overlap() we don't make it
// a class member of Symmetry:: and pass parameters instead.
static void assign_atoms(AtomSel *sel, MoleculeList *mlist, float *(&mycoor), int *(&atomtype)) {
  // get atom coordinates
  const float *allcoor = sel->coordinates(mlist);

  Molecule *mol = mlist->mol_from_id(sel->molid());

  // array of strings describing the atom type
  char **typestringptr = new char*[sel->selected];
  int numtypes = 0;

  int i, j=0;
  for (i=0; i<sel->num_atoms; i++) {
    if (!sel->on[i]) continue;

    // copy coordinates of selected atoms into local array
    vec_copy(mycoor+3L*j, allcoor+3L*i);


    // Calculate lightest and heaviest element bonded to this atom
    int k;
    int minatomicnum = 999;
    int maxatomicnum = 0;
    for (k=0; k<mol->atom(i)->bonds; k++) {
      int bondto = mol->atom(i)->bondTo[k];
      int atomicnum = mol->atom(bondto)->atomicnumber;
      if (atomicnum<minatomicnum) minatomicnum = atomicnum;
      if (atomicnum>maxatomicnum) maxatomicnum = atomicnum;
    }

    // Build up a string describing the atom type. It is not meant
    // to be human-readable, so it's not nice but since the contained
    // properties are ordered it can be used to compare two atoms.
    char *typestring = new char[8L+12L*mol->atom(i)->bonds];
    typestring[0] = '\0';
    char buf[100];

    // atomic number and number of bonds
    sprintf(buf, "%i %i ", mol->atom(i)->atomicnumber, mol->atom(i)->bonds);
    strcat(typestring, buf);

    // For each chemical element get the number of bonds for each 
    // bond order. We distinguish half and integer bond orders,
    // e.g. 2 and 3 mean bond orders 1 and 1.5 respectively.
    int m;
    for (m=minatomicnum; m<=maxatomicnum; m++) {
      unsigned char bondorder[7];
      memset(bondorder, 0, 7L*sizeof(unsigned char));

      unsigned char bondedatomicnum = 0;
      for (k=0; k<mol->atom(i)->bonds; k++) {
        if (m == mol->atom(mol->atom(i)->bondTo[k])->atomicnumber) {
          bondedatomicnum++;
          float bo = mol->getbondorder(i, k);
          if (bo<0.f) bo = 1.f;
          (bondorder[(long)(2L*bo)])++;
        }
      }
      for (k=0; k<7; k++) {
        if (bondorder[k]) {
          sprintf(buf, "%i*(%i)%i ", bondorder[k], k, m);
          strcat(typestring, buf);
        }
      }
    }

    // Try to find this atom's type in the list, if it doesn't exist
    // add a new string and numerical type.     
    int found = 0;
    for (k=0; k<numtypes; k++) {
      if (!strcmp(typestringptr[k], typestring)) {
        atomtype[j] = k;
        found = 1;
        delete [] typestring;
        break;
      }
    }
    if (!found) {
      atomtype[j] = numtypes;
      typestringptr[numtypes++] = typestring;
    }

    //printf("%i: type=%i {%s}\n", j, atomtype[j], typestringptr[atomtype[j]]);
    j++;
  }

  for (i=0; i<numtypes; i++) {
    delete [] typestringptr[i];
  }
  delete [] typestringptr;
}


// Calculate the structural overlap of a set of points with a copy of
// themselves that was transformed according to the given transformation
// matrix.
// Returns the normalized sum over all gaussian function values of the
// pair distances between the atoms in the original and the transformed
// position.
// Two atoms are only considered overlapping if their atom type is
// identical. Atom types must be provided as an array with an integer
// for each atom.
inline static float trans_overlap(int *atomtype, float *(&coor), int numcoor,
                                  const Matrix4 *trans, float sigma,
                                  bool skipident, int maxnatoms) {
  float overlappermatch;
  return trans_overlap(atomtype, coor, numcoor, trans, sigma, skipident, maxnatoms, overlappermatch);
}

static float trans_overlap(int *atomtype, float *(&coor), int numcoor,
                           const Matrix4 *trans, float sigma,
                           bool skipident, int maxnatoms, float &overlappermatch) {
  // get atom coordinates
  const float *posA;
  posA = coor;

  int   *flgs = new int[numcoor];
  float *posB = new float[3L*numcoor];
  memset(flgs, 0, numcoor*sizeof(int));

  // generate transformed coordinates
  int i, ncompare=0;
  for(i=0; i<numcoor; i++) {
    trans->multpoint3d(posA+3L*i, posB+3L*i);

    // Depending on the flag skip atoms that underwent an almost
    // identical transformation.
    if (!(skipident && distance(posA+3L*i, posB+3L*i) < sigma)) {
      flgs[i] = 1;
      ncompare++;   // # of compared atoms with dist<sigma
    }
  }

  if (ncompare<0.5*numcoor) {
    // Not enough atoms to compare
    delete [] flgs;
    delete [] posB;
    return 0.0;
  }

  // If the pair distance gets too small the performance is terrible
  // even for small systems. So we limit it to at least 4.5A
  float dist;
  //if (sigma<1.5) dist = 4.5;
  dist = 3*sigma;
  float wrongelementpenalty = 100.0f/ncompare;

  float overlap = 0.0;
  float antioverlap = 0.0;
  int i1, nmatches = 0, noverlap = 0, nwrongelement = 0;
  float maxr2=powf(1.0f*dist, 2);
  float itwosig2 = 1.0f/(2.0f*sigma*sigma);

  // Now go through all atoms and find matching pairs
  for (i1=0; i1<numcoor; i1++) {
    if (!flgs[i1]) continue;

    float minr2 = maxr2+1.0f;
    float r2 = 0.0f;

    // Find the nearest atom
    int j, i2=-1;
    for (j=0; j<numcoor; j++) {
      if (!flgs[j]) continue;

      r2 = distance2(posA+3L*i1, posB+3L*j);

      if (r2<minr2) { minr2 = r2; i2 = j; }
    }

    // Compute the score for the closest atom
    if (minr2<maxr2) {
      noverlap++;
      
      // consider only pairs with identical atom types
      if (atomtype[i1]==atomtype[i2]) {
        // Gaussian function of the pair distance
        overlap += expf(-itwosig2*minr2);
        nmatches++;
      }
      else {
        // wrong element matching 
        antioverlap += wrongelementpenalty*expf(-itwosig2*r2);
        nwrongelement++;
        //printf("wrong elements %i-%i (atoms %i-%i)\n", mol->atom(i1)->atomicnumber,mol->atom(i2)->atomicnumber,i1,i2);
      }
    }
  }

  float nomatchpenalty = 0.0;
  overlappermatch = 0.0;
  int numnomatch = ncompare-nmatches;

  // We make the penalty for unmatched atoms dependent on the
  // average overlap for the overlapping atoms with correct element
  // matching. In other words, the better the structures match in
  // general the higher the penalty for a nonmatching element.
  // This helps for instance to prevent mistaking nitrogens for
  // carbons in rings.
  // On the other hand the algorithm is, at least for larger structures,
  // fairly forgiving about not matching an atom at all. This helps
  // recognizing symmetry when one or very few atoms are astray.
  if (nmatches) overlappermatch = overlap/nmatches;

  overlap -= antioverlap;

  if (!(numnomatch==0)) {
    //nomatchpenalty = powf(overlappermatch, 5)*(1-powf(0.2f+float(numnomatch),-2));
    nomatchpenalty = powf(overlappermatch, 5);//*(1-powf(0.1f,4.0f)/powf(float(numnomatch)/ncompare,4.0f));
    //overlap *= 1- nomatchpenalty;
    overlap -= 8*numnomatch*nomatchpenalty;
  }
  if (overlap<0) overlap = 0.0f;

  overlap /= ncompare;

  //printf("nsel=%i, wrongelementpen=%.2f, nmatches/noverlap/ncompare=%i/%i/%i, overlap/match=%.2f, nomatchpen=%.2f, ov=%.2f\n",
  //numcoor, wrongelementpenalty, nmatches, noverlap, ncompare, overlappermatch, nomatchpenalty, overlap);

  delete [] flgs;
  delete [] posB;

  return overlap;
}



/*******  OTHER FUNCTIONS WITH TCL INTERFACE  *********/



// Backend of the TCL interface for measure transoverlap:
// Calculate the structural overlap of a selection with a copy of itself
// that is transformed according to a given transformation matrix.
// Returns the normalized sum over all gaussian function values of the
// pair distances between atoms in the original and the transformed
// selection.
// Two atoms are only considered overlapping if their atom type is
// identical. The atom type is determined based on the chemical element
// and number and order of bonds to different elements.
int measure_trans_overlap(AtomSel *sel, MoleculeList *mlist, const Matrix4 *trans,
                          float sigma, bool skipident, int maxnatoms, float &overlap) {
  if (!sel)                     return MEASURE_ERR_NOSEL;
  if (sel->num_atoms == 0)      return MEASURE_ERR_NOATOMS;

  float *coor = new float[3L*sel->selected];

  int *atomtypes = new int[sel->selected];
  assign_atoms(sel, mlist, coor, atomtypes);

  overlap = trans_overlap(atomtypes, coor, sel->selected, trans, sigma, skipident, maxnatoms);

  return MEASURE_NOERR;
}



// Calculate the structural overlap between two point sets.
// The overlap of two atoms is defined as the Gaussian of their distance.
// If two atoms have identical positions the overlap is 1. The parameter
// sigma controls the greediness (i.e. the width) of the overlap function.
// Returns the average overlap for all atoms.
int measure_pointset_overlap(const float *posA, int natomsA, int *flagsA,
                             const float *posB, int natomsB, int *flagsB,
                             float sigma, float pairdist, float &overlap) {

  int nsmall = natomsA<natomsB ? natomsA : natomsB;
  
  int maxpairs = -1;
  GridSearchPair *pairlist, *p;
  pairlist = vmd_gridsearch3(posA, natomsA, flagsA, posB, natomsB, flagsB, pairdist,
                             1, maxpairs);

  overlap = 0.0;
  int i1, i2;
  float dx, dy, dz, r2, itwosig2 = 1.0f/(2.0f*sigma*sigma);
  for (p=pairlist; p; p=p->next) {
    i1 = p->ind1;
    i2 = p->ind2;
    dx = posA[3L*i1  ]-posB[3L*i2];
    dy = posA[3L*i1+1]-posB[3L*i2+1];
    dz = posA[3L*i1+2]-posB[3L*i2+2];
    r2 = dx*dx + dy*dy + dz*dz;
    overlap += expf(-itwosig2*r2);
  }

  overlap /= nsmall;

  return MEASURE_NOERR;
}

