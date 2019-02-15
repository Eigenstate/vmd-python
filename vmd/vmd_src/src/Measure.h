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
 *      $RCSfile: Measure.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.73 $       $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Code to measure atom distances, angles, dihedrals, etc.
 ***************************************************************************/

#include "ResizeArray.h"
#include "Molecule.h"

class AtomSel;
class Matrix4;
class MoleculeList;

#define MEASURE_NOERR                0
#define MEASURE_ERR_NOSEL           -1
#define MEASURE_ERR_NOATOMS         -2
#define MEASURE_ERR_BADWEIGHTNUM    -3
#define MEASURE_ERR_NOWEIGHT        -4
#define MEASURE_ERR_NOCOM           -4
#define MEASURE_ERR_NOMINMAXCOORDS  -4
#define MEASURE_ERR_NORGYR          -4
#define MEASURE_ERR_BADWEIGHTSUM    -5
#define MEASURE_ERR_NOMOLECULE      -6
#define MEASURE_ERR_BADWEIGHTPARM   -7
#define MEASURE_ERR_NONNUMBERPARM   -8
#define MEASURE_ERR_MISMATCHEDCNT   -9
#define MEASURE_ERR_RGYRMISMATCH    -10
#define MEASURE_ERR_NOFRAMEPOS      -11
#define MEASURE_ERR_NONZEROJACOBI   -12
#define MEASURE_ERR_GENERAL         -13
#define MEASURE_ERR_NORADII         -14
#define MEASURE_ERR_BADORDERINDEX   -15
#define MEASURE_ERR_NOTSUP          -16
#define MEASURE_ERR_BADFRAMERANGE   -17
#define MEASURE_ERR_MISMATCHEDMOLS  -18
#define MEASURE_ERR_REPEATEDATOM    -19
#define MEASURE_ERR_NOFRAMES        -20
#define MEASURE_ERR_BADATOMID       -21
#define MEASURE_ERR_BADCUTOFF       -22
#define MEASURE_ERR_ZEROGRIDSIZE    -23

#define MEASURE_BOND  2
#define MEASURE_ANGLE 3
#define MEASURE_DIHED 4
#define MEASURE_IMPRP 5
#define MEASURE_VDW   6
#define MEASURE_ELECT 7

// symbolic flags for cluster analysis
enum {MEASURE_DIST_RMSD=0,
      MEASURE_DIST_FITRMSD,
      MEASURE_DIST_RGYRD,
      MEASURE_DIST_CUSTOM,
      MEASURE_NUM_DIST};

extern const char *measure_error(int errnum);

// apply a matrix transformation to the coordinates of a selection
extern int measure_move(const AtomSel *sel, float *framepos, 
                        const Matrix4 &mat);

// Calculate average position of selected atoms over selected frames
extern int measure_avpos(const AtomSel *sel, MoleculeList *mlist, 
                         int start, int end, int step, float *avpos);

// find the center of the selected atoms in sel, using the coordinates in
// framepos and the weights in weight.  weight has sel->selected elements.
// Place the answer in com.  Return 0 on success, or negative on error.
extern int measure_center(const AtomSel *sel, const float *framepos, 
                          const float *weight, float *com);

// find the center of the selected atoms in sel, using the coordinates in
// framepos and the weights in weight.  weight has sel->selected elements.
// Place the answer in com.  Return number of residues on success, 
// or negative on error.
extern int measure_center_perresidue(MoleculeList *mlist, const AtomSel *sel, 
                                     const float *framepos,
                                     const float *weight, float *com);

// find the dipole of the selected atoms in sel, using the coordinates in
// framepos, and the atom charges.
// Place the answer in dipole.  Return 0 on success, or negative on error.
// Default units are elementary charges/Angstrom.
// Setting unitsdebye to 1 will scale the results by 4.77350732929 (default 0)
// For charged systems the point of reference for computing the dipole is
// selected with usecenter: 
//   1 = geometrical center (default),
//  -1 = center of mass,
//   0 = origin
extern int measure_dipole(const AtomSel *sel, MoleculeList *mlist,
                          float *dipole, int unitsdebye, int usecenter);

extern int measure_hbonds(Molecule *mol, AtomSel *sel1, AtomSel *sel2, 
                          double cut, double maxangle, int *donlist, 
                          int *hydlist, int *acclist, int maxsize);

// Find the transformation which aligns the atoms of sel1 and sel2 optimally,
// meaning it minimizes the RMS distance between them, weighted by weight.
// The returned matrix will have positive determinant, even if an optimal
// alignment would produce a matrix with negative determinant; the last 
// row in the matrix is flipped to change the sign of the determinant. 
// sel1->selected == sel2->selected == len(weight).
extern int measure_fit(const AtomSel *sel1, const AtomSel *sel2, 
                  const float *x, const float *y, const float *weight, 
                  const int *order, Matrix4 *mat);

// compute the axis aligned aligned bounding box for the selected atoms
// returns 0 if success
// returns <0 if not
// If the selection contains no atoms, return {0 0 0} for min and max.
extern int measure_minmax(int num, const int *on, const float *framepos,
                          const float *radii, 
                          float *min_coord, float *max_coord);

// Calculate the radius of gyration of the given selection, using the
// given weight, placing the result in rgyr.
extern int measure_rgyr(const AtomSel *sel, MoleculeList *mlist, 
                        const float *weight, float *rgyr);

// Calculate the RMS distance between the atoms in the two selections, 
// weighted by weight.  Same conditions on sel1, sel2, and weight as for
// measure_fit.  
extern int measure_rmsd(const AtomSel *sel1, const AtomSel *sel2,
                        int num, const float *f1, const float *f2,
                        float *weight, float *rmsd);

// Calculate the RMS distance between the atoms in the two selections,
// weighted by weight.  Same conditions on sel1, sel2, and weight as for
// measure_fit. Now done per residue (so the pointer is expected to be an array)
extern int measure_rmsd_perresidue(const AtomSel *sel1, const AtomSel *sel2, 
                                   MoleculeList *mlist, int num, 
                                   float *weight, float *rmsd);

// Measure RMS distance between two selections as with measure_rmsd(),
// except that it is computed with an implicit best-fit alignment 
// by virtue of the QCP algorithm.
extern int measure_rmsd_qcp(VMDApp *app,
                            const AtomSel *sel1, const AtomSel *sel2,
                            int num, const float *f1, const float *f2,
                            float *weight, float *rmsd);

// Measure matrix of RMS distance between all selected trajectory frames,
// computed with an implicit best-fit alignment by virtue of the QCP algorithm.
extern int measure_rmsdmat_qcp(VMDApp *app,
                               const AtomSel *sel, MoleculeList *mlist,
                               int num, float *weight,
                               int start, int end, int step,
                               float *rmsd);

// Given the component sums of QCP inner products, uses
// QCP algorithm to solve for best-fit RMSD and rotational alignment
extern int FastCalcRMSDAndRotation(double *rot, double *A, float *rmsd,
                                   double E0, int len, double minScore);

// Calculate RMS fluctuation of selected atoms over selected frames
extern int measure_rmsf(const AtomSel *sel, MoleculeList *mlist, 
                        int start, int end, int step, float *rmsf);

// Calculate RMSF per residue.
extern int measure_rmsf_perresidue(const AtomSel *sel, MoleculeList *mlist,
                        int start, int end, int step, float *rmsf);

extern int measure_sumweights(const AtomSel *sel, int numweights, 
                              const float *weights, float *weightsum);


// find the solvent-accessible surface area of atoms in the given selection.
// Use the assigned radii for each atom, and extend this radius by the
// parameter srad to find the points on a sphere that are exposed to solvent.
// Optional parameters (pass NULL to ignore) are:
//   pts: fills the given array with the location of the points that make
//        up the surface.
//   restrictsel: Only solvent accessible points near the given selection will 
//        be considered.
//   nsamples: number of points to use around each atom.
extern int measure_sasa(const AtomSel *sel, const float *framepos,
    const float *radius, float srad, float *sasa, ResizeArray<float> *pts,
    const AtomSel *restrictsel, const int *nsamples);

// XXX experimental version that processes a list of selections at a time
extern int measure_sasalist(MoleculeList *mlist,
                            const AtomSel **sellist, int numsels,
                            float srad, float *sasalist, const int *nsamples);


// perform cluster analysis
extern int measure_cluster(AtomSel *sel, MoleculeList *mlist, 
                           const int numcluster, const int algorithm,
                           const int likeness, const double cutoff,
                           int *clustersize, int **clusterlist,
                           int first, int last, int step, int selupdate,
                           float *weights);

// perform cluster size analysis
extern int measure_clustsize(const AtomSel *sel, MoleculeList *mlist,
                             const double cutoff, int *clustersize,
                             int *clusternum, int *clusteridx, 
                             int minsize, int numshared, int usepbc);

// calculate g(r) for two selections
extern int measure_gofr(AtomSel *sel1, AtomSel *sel2,
                        MoleculeList *mlist, 
                        const int count_h, double *gofr, double *numint, 
                        double *histog, const float delta, 
                        int first, int last, int step, int *framecntr,
                        int usepbc, int selupdate);

// calculate g(r) for two selections
extern int  measure_rdf(VMDApp *app, 
                        AtomSel *sel1, AtomSel *sel2,
                        MoleculeList *mlist, 
                        const int count_h, double *gofr, double *numint, 
                        double *histog, const float delta, 
                        int first, int last, int step, int *framecntr,
                        int usepbc, int selupdate);

int measure_geom(MoleculeList *mlist, int *molid, int *atmid, ResizeArray<float> *gValues,
		 int frame, int first, int last, int defmolid, int geomtype);

// calculate the value of this geometry, and return it
int calculate_bond(MoleculeList *mlist, int *molid, int *atmid, float *value);

// calculate the value of this geometry, and return it
int calculate_angle(MoleculeList *mlist, int *molid, int *atmid, float *value);

// calculate the value of this geometry, and return it
int calculate_dihed(MoleculeList *mlist, int *molid, int *atmid, float *value);

// check whether the given molecule & atom index is OK
// if OK, return Molecule pointer; otherwise, return NULL
int check_mol(Molecule *mol, int a);

// for the given Molecule, find the UNTRANSFORMED coords for the given atom
// return Molecule pointer if successful, NULL otherwise.
int normal_atom_coord(Molecule *m, int a, float *pos);

int measure_energy(MoleculeList *mlist, int *molid, int *atmid, int natoms, ResizeArray<float> *gValues,
		   int frame, int first, int last, int defmolid, double *params, int geomtype);
int compute_bond_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
			float k, float x0);
int compute_angle_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
			 float k, float x0, float kub, float s0);
int compute_dihed_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
			 float k, int n, float delta);
int compute_imprp_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
			 float k, float x0);
int compute_vdw_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
		       float rmin1, float eps1, float rmin2, float eps2, float cutoff, float switchdist);
int compute_elect_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
			 float q1, float q2, bool flag1, bool flag2, float cutoff);

// compute matrix that transforms coordinates from an arbitrary PBC cell 
// into an orthonormal unitcell.
int measure_pbc2onc(MoleculeList *mlist, int molid, int frame, const float *center, Matrix4 &transform);

// does the low level work for the above
void get_transform_to_orthonormal_cell(const float *cell, const float center[3], Matrix4 &transform);

// get atoms in PBC neighbor cells
int measure_pbc_neighbors(MoleculeList *mlist, AtomSel *sel, int molid,
			  int frame, const Matrix4 *alignment,
			  const float *center, const float *cutoff, const float *box,
			  ResizeArray<float> *extcoord_array,
			  ResizeArray<int> *indexmap_array);

// compute the orthogonalized bounding box for the PBC cell.
int compute_pbcminmax(MoleculeList *mlist, int molid, int frame, 
               const float *center, const Matrix4 *transform,
               float *min, float *max);


// Return the list of atoms within the specified distance of the surface
// where the surface depth is specified by sel_dist, the grid resolution is
// approximately gridsz, and atoms are assume to have size  radius
// If any of a, b, c, alpha, or gamma are zero, assume non-periodic,
// otherwise assume periodic
// returns 0 if success
// returns <0 if not
extern int measure_surface(AtomSel *sel, MoleculeList *mlist,
                           const float *framepos, 
                           const double gridsz,
                           const double radius,
                           const double sel_dist,
                           int **surface, int *n_surf);


// Calculate center of mass, principle axes and moments of inertia for
// selected atoms. The corresponding eigenvalues are also returned, 
// and might tell you if two axes are equivalent.
extern int measure_inertia(AtomSel *sel, MoleculeList *mlist, const float *coor, float rcom[3],
			   float priaxes[3][3], float itensor[4][4], float evalue[3]);

