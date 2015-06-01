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
 *	$RCSfile: MeasureSymmetry.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.31 $	$Date: 2010/12/16 04:08:23 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * Thy Symmetry class is the work horse behind the "measure symmetry"
 * command which guesses the pointgroup of a selection and returns the
 * according symmetry elements (mirror planes rotary axes, rotary reflection
 * axes). The algorithm is is fairly forgiving about molecules where atoms
 * are perturbed from the ideal position and tries its best to guess the
 * correct point group anyway. 
 * The 'forgivingness' can be controlled with the sigma parameter which is
 * the average allowed deviation from the ideal position.
 * Works nice on my 30 non-patholocical test cases. A pathological case
 * would for instance be a system with more than only a few atoms (say 15)
 * where only one atom distinguishes between two point groups.
 * If your selection contains more than a certain number of atoms then only
 * that much randomly chosen atoms are used to find planes and axes in order
 * to save time.
 *
 ***************************************************************************/

#ifndef MEASURESYMMETRY_H
#define MEASURESYMMETRY_H

#include "ResizeArray.h"

// Flags for trans_overlap() specifying wether atoms that underwent
// an identical transformation should be skipped in the evaluation.
#define SKIP_IDENTICAL   1
#define NOSKIP_IDENTICAL 0

// Rotary axis classification
#define PRINCIPAL_AXIS     1
#define PERPENDICULAR_AXIS 2

// Maximum order for Cn and S2n axes
#define MAXORDERCN 8


typedef struct plane {
  float v[3];
  float overlap;
  int weight;
  int type;
} Plane;

typedef struct axis {
  float v[3];
  float overlap;
  int order;
  int weight;
  int type;
} Axis;

// Summary of existing symmetry elements
typedef struct elementsummary {
  char inv;     // inversion center (0/1)?
  char sigma;   // number of sigma planes
  char Cinf;    // number of infinite order axes (0/1)
  // Numbers of C and S axes with order 2, 3, 4, ... repectively
  char C[MAXORDERCN+1]; // 0,1 unused; accessed by order 2, ... MAXORDERCN;
  char S[MAXORDERCN*2+1];
} ElementSummary;

typedef struct bond {
  int atom0;
  int atom1;
  float order;
  float length;
} Bond;

typedef struct bondlist {
  int numbonds;
  int *bondto;
  float *length;
} Bondlist;


class Symmetry {
private:
  AtomSel *sel;              ///< selection of atoms we are analyzing
  MoleculeList *mlist;

  float *coor;               ///< local copy of coordinates of the selected atoms
  float *bondsum;            ///< sum of all bond vectors per atom weighted by bond order
  float rcom[3];             ///< center of mass
  float inertiaeigenval[3];  ///< eigenvalues of moments of inertia tensor
  float inertiaaxes[3][3];   ///< primary axes of inertia
  int uniqueprimary[3];      ///< flags for the unique primary axes of inertia
  int numverticalplanes;     ///< # of vertical mirror planes
  int numdihedralplanes;     ///< # of dihedral mirror planes
  int horizontalplane;       ///< index of horizontal mirror plane
  int maxaxisorder;          ///< max. order of rotary axes
  int maxrotreflorder;       ///< max. order of rotary reflection axes
  int maxnatoms;             ///< max. # of atoms used for finding sym. elements
  int checkbonds;            ///< flag whether to use bond order and orientation
  int verbose;               ///< level of output verbosity
  int *atomtype;             ///< a numerical atom type for each atom
  int *atomindex;            ///< original atom index;
  int *uniqueatoms;          ///< array of flags indicating unique atoms
  float *idealcoor;          ///< idealized coordinates

  char *elementsummarystring;    ///< string summarizing symmetry elements
  char *missingelementstring;    ///< string summarizing missing elements
  char *additionalelementstring; ///< string summarizing elements found in addition to ideal ones

  Matrix4 orient;            ///< Transformation aligning mol with GAMESS standard orientation

  // Summarized numbers of elements in table form.
  ElementSummary elementsummary;

  int   maxweight;           ///> maximum weight  among all elements 
                             // XXX seems to be set but never used;
  float maxoverlap;          ///< maximum overlap among all elements
  float rmsd;                ///< RMSD between original and idealized coordinates

  int pointgroup;            ///< result of the point group guessing,
                             ///< without the order. E.g. Cnv, Dnh, Ci.
 
  int pointgrouporder;       ///< order of the guesses point group (if applicable)
                             ///< e.g. 3 for C3v
 
  ResizeArray<Plane> planes;         ///< mirror planes
  ResizeArray<Axis>  axes;           ///< rotary axes
  ResizeArray<Axis>  rotreflections; ///< rotary reflection axes
  ResizeArray<Bond>  bonds;          ///< bond list

  Bondlist *bondsperatom;  ///< Bonds in selection based indices


  bool linear;        ///< is molecule linear?
  bool planar;        ///< is molecule planar?
  bool inversion;     ///< do we have an inversion center?
  bool sphericaltop;  ///< is it a spherical top?
  bool symmetrictop;  ///< is it a symmetrical top?


  /// Self consistent iteration of symmetry element search
  int iterate_element_search();

  /// Find all symmetry elements (inversion, axes, planes, rotary reflections)
  void find_symmetry_elements();

  /// Find symmetry elements based on the primary axes of inertia.
  void find_elements_from_inertia();

  /// Find mirror planes based on atoms with the same distance from COM.
  void find_planes();

  /// Find rotary axes as intersections of mirror planes.
  void find_axes_from_plane_intersections();

  /// Find remaining Cn axes based on atoms with the same distance from COM
  void find_C2axes();
  void find_axes(int order);


  /// Check if there's an inversion center
  void check_add_inversion();   

  /// Check if the given normal defines a mirror plane and add
  /// it to the list if an equivalent one does not yet exist.
  void check_add_plane(const float *normal, int weight=1);

  /// Check if the given vector defines a C2 rotary axis and add
  /// it to the list if an equivalent one does not yet exist.
  void check_add_axis(const float *testaxis, int order);

  /// Check if the given vector feines a rotary reflection axis and
  /// add it to the list if if an equivalent one does not yet exist.
  void check_add_rotary_reflection(const float *testaxis, int maxorder);


  /// Classify vertical, dihedral and horizontal planes
  void classify_planes();

  /// Classify perpendicular axes
  void classify_axes();

  /// Remove planes with too bad overlap score from the list
  void purge_planes(float cutoff);

  /// Remove axes with too bad overlap score from the list
  void purge_axes(float cutoff);

  /// Remove rotary reflections with too bad overlap score from the list
  void purge_rotreflections(float cutoff);

  /// For each plane i prune from the list if flag keep[i]==0.
  void prune_planes(int *keep);

  /// For each axis i prune from the list if flag keep[i]==0.
  void prune_axes(int *keep);

  /// For each axis i prune from the list if flag keep[i]==0.
  void prune_rotreflections(int *keep);

  /// Assign orders to all rotary axes
  void assign_axis_orders();

  /// Assign orders to all preliminary rotary axes
  void assign_prelimaxis_orders(int order);

  /// Query the order of the given rotary axis
  int axis_order(const float *axis, float *overlap);

  /// Sort the axes according to decreasing order
  void sort_axes();

  /// Sort planes (horizontal, dihedral, vertical, rest)
  void sort_planes();

  /// Print summary information about found elements.
  void print_statistics();

  /// Build a summary matrix of found symmetry elements
  void build_element_summary();

  /// Create human-readable string summarizing symmetry elements
  void build_element_summary_string(ElementSummary summary, char *(&sestring));

  /// Compare the ideal numbers of elements with the found ones
  void compare_element_summary(const char *pointgroupname);

  /// Print the element summary strings
  void print_element_summary(const char *pointgroupname);

  /// Adjust all symmetry elements so that they have certain geometries
  /// with respect to each other.
  void idealize_elements();

  /// Idealize myaxis by rotating refaxis by the ideal angle (according
  /// to reforder) around hub and put the result in idealaxis.
  int idealize_angle(const float *refaxis, const float *hub,
		    const float *myaxis, float *idealaxis, int reforder);

  /// Generate symmetricized coordinates
  void idealize_coordinates();

  /// Averages between original and transformed coordinates.
  int average_coordinates(Matrix4 *trans);

  /// Check the bondsum for atom j and its image m generated by 
  // transformation trans.
  int check_bondsum(int j, int m, Matrix4 *trans);

  /// Match atoms of original and transformed coordinates
  void identify_transformed_atoms(Matrix4 *trans, int &nmatches, int *(&matchlist));

  /// Compute the RMSD between original and idealized coordinates
  float ideal_coordinate_rmsd ();

  /// Check the sanity of idealized coordinates
  /// by testing the distance between all atom pairs.
  int ideal_coordinate_sanity();

  /// Determine the unique coordinates for the whole system.
  void unique_coordinates();

  /// Determine the unique coordinates for the given transformation
  int* unique_coordinates(Matrix4 *trans);

  /// Collapse atoms to unique coordinates around given rotary axis
  void collapse_axis(const float *axis, int order, int refatom, const int *matchlist, int *(&connectedunique));

  /// Determine level in the pointgroup hierarchy
  int pointgroup_rank(int pg, int order);

  /// Orient the molecule according to GAMESS' rules
  void orient_molecule();

  /// Determine the point group from symmetry elements
  void determine_pointgroup();

  /// Return index of the first found axis collinear to given axis, -1 otherwise
  inline int find_collinear_axis(const float *myaxis);

  /// Return index of the first found plane that is coplanar to myplane
  inline int plane_exists(const float *myplane);

  /// Return 1 if the molecule is planar, 0 otherwise.
  int is_planar(const float *axis);

  /// Assign bond topology information to each atom
  void assign_bonds();

  /// Build a list of vectors that are the sum of all bond directions
  /// weighted by bondorder. The bondsum is useful as a checksum for
  /// bonding topology and orientation for each atom.
  void assign_bondvectors();

  /// Find set of unique atoms with the most atoms connected to atom 0.
  void wrap_unconnected_unique_atoms(int root, const int *matchlist, int *(&connectedunique));

  // Find all unique atoms that are connnected to root
  void find_connected_unique_atoms(int *(&connectedunique), int root);

  /// Draws a atom-colored spheres for each atom at the transformed
  /// position (for debuggging only).
  void draw_transformed_mol(Matrix4 rot);

public:
  Symmetry(AtomSel *sel, MoleculeList *mlist, int verbosity);
  ~Symmetry(void);

  float sigma;     // atomic overlap tolerance
  float collintol; // collinearity tolerance in radians
  float orthogtol; // coplanarity tolerance in radians

  // Check if center of mass is an inversion center.
  float score_inversion();

  /// Check if vector testaxis defines a rotary axis of the given order.
  float score_axis(const float *testaxis, int order);

  /// Check if the given normal represents a mirror plane.
  float score_plane(const float *normal);

  /// Get the score for given rotary reflection.
  float score_rotary_reflection(const float *testaxis, int order);

  /// Impose certain symmetry elements on structure
  /// by wrapping coordinates around and averaging them.
  void impose(int have_inversion, 
              int nplanes, const float *planev,
              int naxes, const float *axisv, const int *axisorder,
              int nrotrefl, const float* rotreflv,
              const int *rotreflorder);

  /// Determine the symmetry pointgroup and order
  int guess(float mysigma);

  /// Get the guessed pointgroup and order
  void get_pointgroup(char pg[8], int *order);

  /// Get RMSD between original and idealized coordinates
  float get_rmsd() { return rmsd; }

  /// Get the principal axes of inertia
  float* get_inertia_axes()      { return &(inertiaaxes[0][0]); }

  /// Get eigenvalues of the principal axes of inertia
  float* get_inertia_eigenvals() { return &(inertiaeigenval[0]); }

  /// Get a triple of flags telling if the axes of inertia are unique
  int*   get_inertia_unique()    { return &(uniqueprimary[0]); }

  /// Get order of specified axes
  int  get_axisorder(int n);

  /// Get order of specified rotary reflection
  int  get_rotreflectorder(int n);

  /// Get transformation aligning molecule with GAMESS standard orientation
  Matrix4* get_orientation() { return &orient; }

  /// Get string summarizing symmetry elements
  const char *get_element_summary()     { return elementsummarystring; }

  /// Get string summarizing missing symmetry elements
  const char *get_missing_elements()    { return missingelementstring; }

  /// Get string summarizing symmetry elements additional to ideal ones
  const char *get_additional_elements() { return additionalelementstring; }

  /// Set the max. # of atoms used for finding symmetry elements.
  void set_maxnumatoms(int n) { maxnatoms = n; }

  /// Set the max. # of atoms used for finding symmetry elements.
  void set_overlaptol(float tol) { sigma = tol; }

  /// Set flag whether to use bond order and orientation
  void set_checkbonds(int flag) { checkbonds = flag; }

  /// Get total number of mirror planes
  int numplanes()     { return planes.num(); }

  /// Get total number of rotation axes
  int numaxes()       { return axes.num(); }

  /// Get number of equivalent primary axes
  int numprimaryaxes();

  /// Get total number of rotary reflection axes
  int numrotreflect() { return rotreflections.num(); }

  /// Return the unmber of S2N rotary reflection (Sn with even n)
  int numS2n();

  /// Return 1 if inversion center is present, zero otherwise
  int has_inversion() { return inversion; }

  /// Return 1 if the molecule represents a spherical top
  int is_spherical_top() { return sphericaltop; }

  /// Return 1 if the molecule represents a symmetric top
  int is_symmetric_top() { return symmetrictop; }

  /// Return the center of mass
  float* center_of_mass() { return rcom; }

  /// Return pointer to the i-th plane
  float* plane(int i)      { return (float *) &(planes[i].v); }

  /// Return pointer to the i-th axis
  float* axis(int i)       { return (float *) &(axes[i].v); }

  /// Return pointer to the i-th rotary reflection axis
  float* rotreflect(int i) { return (float *) &(rotreflections[i].v); }

  /// Return type of the i-th plane
  int get_planetype(int i)     { return planes[i].type; }

  /// Return type of the i-th axis
  int get_axistype(int i)      { return axes[i].type; }

  /// Return type of the i-th axis
  int get_rotrefltype(int i)   { return rotreflections[i].type; }

  /// Return pointer to the i-th atoms idealized coordinates
  float *idealpos(int i)   { return idealcoor+3*i; }

  /// Return a flag telling if atom i is considered unique
  int get_unique_atom(int i)   { return uniqueatoms[i]; }
};



/// Calculate the structural overlap of a selection with a copy of itself
/// that is transformed according to a given transformation matrix.
/// Returns the normalized sum over all gaussian function values 
/// of the pair distances between atoms in the original and the transformed
/// selection.
int measure_trans_overlap(AtomSel *sel, MoleculeList *mlist, const Matrix4 *trans,
			  float sigma, bool skipident, int maxnatoms, float &overlap);


int measure_pointset_overlap(const float *posA, int natomsA, int *flagsA,
			     const float *posB, int natomsB, int *flagsB,
			     float sigma, float pairdist, float &similarity);




#endif
