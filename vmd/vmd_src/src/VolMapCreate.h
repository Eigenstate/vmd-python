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
 *      $RCSfile: VolMapCreate.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.93 $       $Date: 2019/01/23 21:33:54 $
 *
 **************************************************************************/

#include "Matrix4.h"

// enable multilevel summation by default
#define VMDUSEMSMPOT 1

class VMDApp;
class VolumetricData;
class AtomSel;
class Molecule;

/// Virtual class for dealing with the computation of VolMaps, based on
/// atomic selections and atomic data and coordinates. It provides utilities
/// for computing such maps. Various algorithms and specific calculations 
/// are done by child classes. The end result is a VolMap object, which can 
/// be passed to VMD (N/A yet) or written to a file.

class VolMapCreate {
public:
  typedef enum {COMBINE_AVG, COMBINE_MIN, COMBINE_MAX, COMBINE_STDEV, COMBINE_PMF} CombineType;
  
protected:
  VMDApp *app;
  AtomSel *sel;
  float delta;            // resolution (same along x, y and z)
  int computed_frames;    // frame counter
  int checkpoint_freq;    // write checkpoint file every xxx steps
  char *checkpoint_name;  // checkpoint file name
  bool user_minmax;       // true = user specified a minmax box, false = compute default minmax
  float min_coord[3], max_coord[3]; // used to pass user defaults, avoid using for computations!

protected:
  virtual int compute_frame(int frame, float *voldata) = 0;
  int compute_init(float padding);
  
  /// called before computing individual frames
  virtual int compute_init() {return compute_init(0.);}
  
  /// calculates minmax (into preallocated float[3] arrays) using volmapcreate's selection
  int calculate_minmax (float *min_coord, float *max_coord);
  
  /// calculates max_radius (into a provided float) using volmapcreate's selection
  int calculate_max_radius (float &radius);
  
  /// for checkpointing
  void combo_begin(CombineType method, void **customptr, void *params);
  void combo_addframe(CombineType method, float *voldata, void *customptr, float *framedata);
  void combo_export(CombineType method, float *voldata, void *customptr);
  void combo_end(CombineType method, void *customptr);


public:
  VolumetricData *volmap;
  
  VolMapCreate(VMDApp *app, AtomSel *sel, float resolution);
  virtual ~VolMapCreate();
  
  void set_minmax (float minx, float miny, float minz, float maxx, float maxy, float maxz);

  void set_checkpoint (int checkpointfreq, char *checkpointname);
  
  int compute_all(bool allframes, CombineType method, void *params);

  /// For now this will call write_dx_file, but this is going to change to using
  /// molfileplugin instead. In VolMapCreate*Energy this method is
  /// overridden by one that adds temperature and weight information to the 
  /// data set name string.
  virtual void write_map(const char *filename);

  // We temporarily need our own file writer until we use molfile plugin
  int write_dx_file (const char *filename);

};


class VolMapCreateMask: public VolMapCreate {
protected:
  int compute_init();
  int compute_frame(int frame, float *voldata);
private:
  float atomradius;

public:
  VolMapCreateMask(VMDApp *app, AtomSel *sel, float res, float the_atomradius) : VolMapCreate(app, sel, res) {
    atomradius = the_atomradius;
  }
};


class VolMapCreateDensity : public VolMapCreate {
protected:
  float *weight;
  char const *weight_string;
  int weight_mutable;
  int compute_init();
  int compute_frame(int frame, float *voldata);
  float radius_scale; // mult. factor for atomic radii
  
public:
  VolMapCreateDensity(VMDApp *app, AtomSel *sel, float res, float *the_weight, char const *the_weight_string, int the_weight_mutable, float the_radscale) : VolMapCreate(app, sel, res) {
    weight = the_weight;
    weight_string = the_weight_string;
    weight_mutable = the_weight_mutable;
    // number of random points to use for each atom's gaussian distr.
    radius_scale = the_radscale;
  }
};


class VolMapCreateInterp : public VolMapCreate {
protected:
  float *weight;
  char const *weight_string;
  int weight_mutable;
  int compute_init();
  int compute_frame(int frame, float *voldata);

public:
  VolMapCreateInterp(VMDApp *app, AtomSel *sel, float res, float *the_weight, char const *the_weight_string, int the_weight_mutable) : VolMapCreate(app, sel, res) {
    weight = the_weight;
    weight_string = the_weight_string;
    weight_mutable = the_weight_mutable;
  }
};


class VolMapCreateOccupancy : public VolMapCreate {
private:
  bool use_points;
protected:
  int compute_init();
  int compute_frame(int frame, float *voldata);  
public:
  VolMapCreateOccupancy(VMDApp *app, AtomSel *sel, float res, bool use_point_particles) : VolMapCreate(app, sel, res) {
    use_points = use_point_particles;
  }
};


class VolMapCreateDistance : public VolMapCreate {
protected:
  float max_dist;
  int compute_init();
  int compute_frame(int frame, float *voldata);  
public:
  VolMapCreateDistance(VMDApp *app, AtomSel *sel, float res, float the_max_dist) : VolMapCreate(app, sel, res) {
    max_dist = the_max_dist;
  }
};


class VolMapCreateCoulombPotential : public VolMapCreate {
protected:
  int compute_init();
  int compute_frame(int frame, float *voldata);
  
public:
  VolMapCreateCoulombPotential(VMDApp *app, AtomSel *sel, float res) : VolMapCreate(app, sel, res) {
  }
};


#if defined(VMDUSEMSMPOT)
class VolMapCreateCoulombPotentialMSM : public VolMapCreate {
protected:
  int compute_init();
  int compute_frame(int frame, float *voldata);
  
public:
  VolMapCreateCoulombPotentialMSM(VMDApp *app, AtomSel *sel, float res) : VolMapCreate(app, sel, res) {
  }
};
#endif


/// Implicit Ligand Sampling (ILS) algorithm.
/// It finds the energy of placing a monoatomic or diatomic
/// ligand at many points in the protein.
class VolMapCreateILS {
private:
  VMDApp *app;
  int molid;     // the molecule we are operating on

  int num_atoms; // # atoms in the system

  VolumetricData *volmap;  // our result: the free energy map
  VolumetricData *volmask; // mask defining valid gridpoints in volmap

  float delta;        // distance of samples for ILS computation
  int   nsubsamp;     // # samples in each dim. downsampled into
                      // each gridpoint of the final map.

  // Number of samples used during computation.
  int   nsampx, nsampy, nsampz;

  float minmax[6];     // minmax coords of bounding box
  float gridorigin[3]; // center of the first grid cell

  float cutoff;        // max interaction dist between any 2 atoms
  float extcutoff;     // cutoff corrected for the probe size
  float excl_dist;     // cutoff for the atom clash pre-scanning

  bool compute_elec;   // compute electrostatics? (currently unused)

  // Control of the angular spacing of probe orientation vectors:
  // 1 means using 1 orientation only
  // 2 corresponds to 6 orientations (vertices of octahedron)
  // 3 corresponds to 8 orientations (vertices of hexahedron)
  // 4 corresponds to 12 orientations (faces of dodecahedron)
  // 5 corresponds to 20 orientations (vertices of dodecahedron)
  // 6 corresponds to 32 orientations (faces+vert. of dodecah.)
  // 7 and above: geodesic subdivisions of icosahedral faces
  //              with frequency 1, 2, ...
  // Probes with tetrahedral symmetry: 
  // # number of rotamers for each of the 8 orientations
  // (vertices of tetrahedron and its dual tetrahedron).
  //
  // Note that the angular spacing of the rotations around
  // the orientation vectors is chosen to be about the same
  // as the angular spacing of the orientation vector
  // itself.
  int conformer_freq; 
  
  int num_conformers;   // # probe symmetry unique orientations and
                        // rotations sampled per grid point
  float *conformers;    // Stores the precomputed atom positions
                        // (relative to the center of mass)
                        // of the different probe orientations and
                        // rotations.
  int num_orientations; // # probe symmetry unique orientations
  int num_rotations;    // # symmetry unique rotations sampled
                        // per orientation

  // We store the VDW parameters once for each type:
  float *vdwparams;       // VDW well depths and radii for all types
  int   *atomtypes;       // index list for vdw parameter types

  int   num_unique_types; // # unique atom types

  float temperature;  // Temp. in Kelvin at which the MD sim. was performed

  int num_probe_atoms;  // # atoms in the probe (the ligand)
  float  probe_effsize; // effective probe radius
  float *probe_coords;  // probe coordinates

  // The two highest symmetry axes of the probe
  float probe_symmaxis1[3];
  float probe_symmaxis2[3];
  int probe_axisorder1, probe_axisorder2;
  int probe_tetrahedralsymm; // probe has tetrahedral symmetry flag

  // VDW parameters for the probe:
  // A tuple of eps and rmin is stored for each atom.
  // Actually we store beta*sqrt(eps) and rmin/2 
  // (see function set_probe()).
  float *probe_vdw; 
  float *probe_charge; // charge for each probe atom

  int first, last;        // trajectory frame range
  int computed_frames;    // # frames processed

  float max_energy;   // max energy considered in map, all higher energies
                      // will be clamped to this value.
  float min_occup;    // occupancies below this value will be treated
                      // as zero.

  bool pbc;           // If flag is set then periodic boundaries are taken
                      // into account.
  bool pbcbox;        // If flag is set then the grid dimensions will be chosen
                      // as the orthogonalized bounding box for the PBC cell.
  float pbccenter[3]; // User provided PBC cell center.

  AtomSel *alignsel;  // Selection to be used for alignment
  const float *alignrefpos; // Stores the alignment reference position

  Matrix4 transform;  // Transformation matrix that was used for the
                      // alignment of the first frame.

  int maskonly;       // If set, compute only a mask map telling for which
                      // gridpoints we expect valid energies, i.e. the points
                      // for which the maps overlap for all frames. 


  // Check if the box given by the minmax coordinates is located
  // entirely inside the PBC unit cell of the given frame and in
  // this case return 1, otherwise return 0. 
  int box_inside_pbccell(int frame, float *minmax);

  // Check if the entire volmap grid is located entirely inside
  // the PBC unit cell of the given frame (taking the alignment
  // into account) and in this case return 1, otherwise return 0.
  int grid_inside_pbccell(int frame,  float *voldata,
                          const Matrix4 &alignment);

  // Set grid dimensions to the minmax coordinates and
  // align grid with integer coordinates.
  int set_grid();

  // Initialize the ILS calculation
  int initialize();

  // ILS calculation for the given frame
  int compute_frame(int frame, float *voldata);

  // Align current frame to the reference
  void align_frame(Molecule *mol, int frame, float *coords,
		   Matrix4 &alignment);

  // Get array of coordinates of selected atoms and their
  // neighbors (within a cutoff) in the PBC images.
  int get_atom_coordinates(int frame, Matrix4 &alignment,
                           int *(&vdwtypes),
                           float *(&coords));

  // Check if probe is a linear molecule and returns
  // the Cinf axis.
  int is_probe_linear(float *axis);

  // Simple probe symmetry check
  void check_probe_symmetry();

  // Determine probe symmetry and generate probe orientations
  // and rotations.
  void initialize_probe();
  void get_eff_proberadius();


  // Generate conformers for tetrahedral symmetry
  int gen_conf_tetrahedral(float *(&conform), int freq,
                           int &numorient, int &numrot);

  // Generate conformers for all other symmetries
  int gen_conf(float *(&conform), int freq,
               int &numorient, int &numrot);

  float dimple_depth(float phi);

  // Create list of unique VDW parameters which can be accessed
  // through the atomtypes index list. 
  int create_unique_paramlist();


public:   
  VolMapCreateILS(VMDApp *_app, int molid, int firstframe,
                  int lastframe, float T, float res, 
                  int subr, float cut, int maskonly);
  ~VolMapCreateILS();

  VolumetricData* get_volmap() { return volmap; };

  // Perform ILS calculation for all specified frames.
  int compute();

  /// Add volumetric data to the molecule
  int add_map_to_molecule();

  /// Include temperature and weight information to the data set
  /// name string and write the map into a dx file.
  int write_map(const char *filename);

  // Set probe coordinates,charges and VDW parameters
  void set_probe(int num_probe_atoms, int num_conf,
                 const float *probe_coords,
                 const float *vdwrmin, const float *vdweps,
                 const float *charge);

  // Set the two highest symmetry axes for the probe and a flag
  // telling if we have a tetrahedral symmetry.
  // If the axes are not orthogonal the lower axis will be ignored.
  void set_probe_symmetry(int order1, const float *axis1,
                          int order2, const float *axis2,
                          int tetrahedral);

  // Set minmax coordinates of rectangular molecule bounding box
  void set_minmax (float minx, float miny, float minz, float maxx, float maxy, float maxz);

  // Request PBC aware computation.
  void set_pbc(float center[3], int bbox);

  // Set maximum energy considered in the calculation.
  void set_maxenergy(float maxenergy);

  // Set selection to be used for alignment.
  void set_alignsel(AtomSel *asel);

  // Set transformation matrix that was used for the
  // alignment of the first frame.
  void set_transform(const Matrix4 *mat);

  int get_conformers(float *&conform) const {
    conform = conformers;
    return num_conformers;
  }

  void get_statistics(int &numconf, int &numorient,
                      int &numrot) {
    numconf   = num_conformers;
    numorient = num_orientations;
    numrot    = num_rotations;
  }
};


// Write given map as a DX file.
int volmap_write_dx_file (VolumetricData *volmap, const char *filename);
