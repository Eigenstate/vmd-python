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
 *      $RCSfile: VolMapCreateILS.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.169 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************/

#include <math.h>
#include <stdio.h>
#include "VolMapCreate.h"
#include "MoleculeList.h"
#include "VolumetricData.h"
#include "utilities.h"
#include "WKFUtils.h"

/* avoid parameter name collisions with AIX5 "hz" macro */
#undef hz

//#define DEBUG 1
#include "VMDApp.h"
#if defined(DEBUG)
#include "MoleculeGraphics.h"  // needed only for debugging
#endif

#include "Measure.h"
#include "Inform.h"

#if defined(VMDCUDA)
#include "CUDAKernels.h"
#endif

#define TIMING

typedef union flint_t {
  float f;
  int i;
  char c[4];
} flint;

#define BIN_DEPTH     8  // number of slots per bin
#define BIN_SLOTSIZE  4  // slot permits x, y, z, vdwtype  (4 elements)
#define BIN_SIZE      (BIN_DEPTH * BIN_SLOTSIZE)  // size given in "flints"


#define BIN_DEPTH  8


struct AtomPosType {
  float x, y, z;                   // position coordinates of atom
  int vdwtype;                     // type index for van der Waals params
};


struct BinOfAtoms {
  AtomPosType atom[BIN_DEPTH];     // use fixed bin depth for CUDA
};


typedef struct ComputeOccupancyMap_t {

  // these are initialized by caller (pointers to existing memory allocations)

  float *map;                    // buffer space for occupancy map
  int mx, my, mz;                // map dimensions
  float lx, ly, lz;              // map lengths
  float x0, y0, z0;              // map origin
  float ax[3], ay[3], az[3];     // map basis vectors, aligned
  float alignmat[16];            // 4x4 matrix used for alignment
  int num_coords;                // number of atoms
  const float *coords;           // atom coords x/y/z, length 3*num_coords

  const int *vdw_type;           // type index for each atom, length num_coords

  const float *vdw_params;       // vdw parameters for system atoms, listed as
                                 // (epsilon, rmin) pairs for each type number

  const float *probe_vdw_params; // vdw parameters for probe atoms, listed as
                                 // (epsilon, rmin) pairs for each probe atom,
                                 // length 2*num_probes

  const float *conformers;       // length 3*num_probes*num_conformers,
                                 // with probe atoms listed consecutively
                                 // for each conformer

  int num_probes;                // number of probe atoms
  int num_conformers;            // number of conformers

  float cutoff;                  // cutoff distance
  float extcutoff;               // extended cutoff, includes probe radius

  float excl_dist;               // exclusion distance threshold
  float excl_energy;             // exclusion energy threshold

  int kstart, kstop;             // z-direction indexing for multi-threaded
                                 // slab decomposition of maps
                                 // kstart = starting index of slab
                                 // kstop = last index + 1

  // internal data (memory is allocated)

  float hx, hy, hz;              // derived map spacings
  float bx, by, bz;              // atom bin lengths
  float bx_1, by_1, bz_1;        // reciprocal bin lengths
  int mpblx, mpbly, mpblz;       // map points per bin side in x, y, z
  int cpu_only;                  // flag indicates that CUDA cannot be used

  BinOfAtoms *bin;               // padded bin of atoms
  BinOfAtoms *bin_zero;          // bin pointer shifted to (0,0,0)-index
  char *bincnt;                  // count occupied slots in each bin
  char *bincnt_zero;             // bincnt pointer shifted to (0,0,0)-index

  int nbx, nby, nbz;             // number of bins in x, y, z directions
  int padx, pady, padz;          // bin padding

  char *bin_offsets;             // bin neighborhood index offset
  int num_bin_offsets;           // length of tight 

  AtomPosType *extra;            // extra atoms that over fill bins
  int num_extras;                // number of extra atoms

  char *exclusions;              // same dimensions as map

  // data for CUDA goes here

} ComputeOccupancyMap;


#define DEFAULT_EXCL_DIST      1.f
#define DEFAULT_EXCL_ENERGY   87.f

#define DEFAULT_BIN_LENGTH     3.f
#define MAX_BIN_VOLUME        27.f
#define MIN_BIN_VOLUME         8.f


// Must already have ComputeOccupancyMap initialized.
// Performs geometric hashing of atoms into bins, along with all related
// memory management.  Also allocates memory for map exclusions.
static int ComputeOccupancyMap_setup(ComputeOccupancyMap *);

// Calculate occupancy for slab of map indexed from kstart through kstop.
// Starts by finding the exclusions, then continues by calculating the
// occupancy for the given probe and conformers.
static int ComputeOccupancyMap_calculate_slab(ComputeOccupancyMap *);

// Cleanup memory allocations.
static void ComputeOccupancyMap_cleanup(ComputeOccupancyMap *);


// Write bin histogram into a dx map
static void write_bin_histogram_map(
    const ComputeOccupancyMap *,
    const char *filename
    );

static void atom_bin_stats(const ComputeOccupancyMap *);


// XXX slow quadratic complexity algorithm for checking correctness
//     for every map point, for every probe atom in all conformers,
//     iterate over all atoms
static void compute_allatoms(
    float *map,                    // return calculated occupancy map
    int mx, int my, int mz,        // dimensions of map
    float lx, float ly, float lz,  // lengths of map
    const float origin[3],         // origin of map
    const float axes[9],           // basis vectors of aligned map
    const float alignmat[16],      // 4x4 alignment matrix
    int num_coords,                // number of atoms
    const float *coords,           // atom coordinates, length 3*num_coords
    const int *vdw_type,           // vdw type numbers, length num_coords
    const float *vdw_params,       // scaled vdw parameters for atoms
    float cutoff,                  // cutoff distance
    const float *probe_vdw_params, // scaled vdw parameters for probe atoms
    int num_probe_atoms,           // number of atoms in probe
    int num_conformers,            // number of conformers
    const float *conformers,       // length 3*num_probe_atoms*num_conformers
    float excl_energy              // exclusion energy threshold
    );


/////// VolMapCreateILS ///////

// Computes free energy maps from implicit ligand sampling.
// I.e. it creates a map of the estimated potential of mean force
// (in units of k_B*T at the specified temperature) of placing a 
// weakly-interacting gas monoatomic or diatomic ligand in every 
// voxel. Each voxel can be divided into a subgrid in order to obtain
// a potential value for that voxel that is averaged over its subgrid 
// positions. For diatomic ligands for each (sub-)gridpoint one can
// (and should) also average over different random rotamers.

// For the computation the trajectory frames must be aligned. One can
// either manually align them beforehand or provide a selection
// (AtomSel *alignsel) on which automatic alignment should be based.
// If such a selection was provided then it is used to align all
// trajectory frames to the first frame.
// Note that the results are slightly different from what you get when
// you align all frames manually prior to the computation.
// The differences are quite noticable when comparing the files as
// ascii texts but actually they are of numerical nature and when you
// display the maps they look the same.
// Suppose you want to align your trajectory to a reference frame from a
// different molecule then you can specify the alignment matrix (as
// returned by "measure fit") that would align the first frame to the
// reference. The corresponding member variable is Matrix4 transform.

VolMapCreateILS::VolMapCreateILS(VMDApp *_app,
                                 int mol, int firstframe, int lastframe,
                                 float T, float res, int nsub,
                                 float cut, int mask) : 
  app(_app), molid(mol), cutoff(cut),
  temperature(T), first(firstframe), last(lastframe),
  maskonly(mask)
{
  compute_elec = false;

  num_probe_atoms = 0;
  num_conformers = 0;
  conformer_freq = 1;
  conformers = NULL;
  probe_coords = NULL;
  probe_vdw    = NULL;
  probe_charge = NULL;
  probe_axisorder1 = 1;
  probe_axisorder2 = 1;
  probe_tetrahedralsymm = 0;
  probe_effsize = 0.f;
  extcutoff = cutoff;

  max_energy = DEFAULT_EXCL_ENERGY;
  min_occup = expf(-max_energy);

  nsubsamp = 1;
  if (nsub>1) nsubsamp = nsub;
  delta = res/float(nsubsamp);

  alignsel = NULL;
  pbc = false;
  pbcbox = false;
  pbccenter[0] = pbccenter[1] = pbccenter[2] = 0.f;
  
  num_unique_types = 0;
  vdwparams  = NULL;
  atomtypes  = NULL;

  minmax[0] = 0.f;
  minmax[1] = 0.f;
  minmax[2] = 0.f;
  minmax[3] = 0.f;
  minmax[4] = 0.f;
  minmax[5] = 0.f;
  
  volmap  = NULL;
  volmask = NULL;
}


VolMapCreateILS::~VolMapCreateILS() {
  if (probe_coords) delete [] probe_coords;
  if (probe_vdw)    delete [] probe_vdw;
  if (probe_charge) delete [] probe_charge;

  if (conformers) delete [] conformers;
  if (vdwparams)  delete [] vdwparams;
  if (atomtypes)  delete [] atomtypes;

  if (volmap) delete volmap;

  //  if (volmask) delete volmask;
}


/// Include temperature and weight information to the data set
/// name string and write the map into a dx file.
int VolMapCreateILS::write_map(const char *filename) {
  // Override the name string:
  // We include the number of frames used for sampling and the temperature
  // of the underlying MD simulation in the dataset name string.
  // This way we can use the VolumetricData class without adding new
  // non-general fields to it.
  char tmpstr[256];
  if (maskonly) {
    sprintf(tmpstr, "ligand pmf mask, %i frames, cutoff = %.2fA",
            computed_frames, cutoff);
  } else {
    sprintf(tmpstr, "ligand pmf [kT], %i frames, T = %.2f Kelvin, maxenergy = %g, cutoff = %.2fA",
	    computed_frames, temperature, max_energy, cutoff);
  }
  volmap->set_name(tmpstr);


  // Add volmap to a molecule so that we can use
  // the plugin interface to write the dx map.
  if (!add_map_to_molecule()) return 0;

  Molecule *mol = app->moleculeList->mol_from_id(molid);
  FileSpec spec;
  spec.nvolsets = 1;        // one volumeset to write
  spec.setids = new int[1]; // ID of the volumeset
  spec.setids[0] = mol->num_volume_data()-1;

  if (!app->molecule_savetrajectory(molid, filename, "dx", &spec)) {
    msgErr << "Couldn't write dx file!" << sendmsg;
  }

  return 1;
}


// Add volumetric data to the molecule
int VolMapCreateILS::add_map_to_molecule() {
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  float origin[3], xaxis[3], yaxis[3], zaxis[3];
  int i;
  for (i=0; i<3; i++) {
    origin[i] = (float) volmap->origin[i];
    xaxis[i] = (float) volmap->xaxis[i];
    yaxis[i] = (float) volmap->yaxis[i];
    zaxis[i] = (float) volmap->zaxis[i];
  }
  float *voldata = volmap->data;

  int err = app->molecule_add_volumetric(molid, 
              volmap->name, origin, xaxis, yaxis, zaxis,
              volmap->xsize, volmap->ysize, volmap->zsize,
              voldata);
  if (err != 1) {
    msgErr << "ERROR: Adding volmap " << mol->num_volume_data()-1
           << " to molecule " << molid << " was unsuccessful!"
           << sendmsg;
    return 0;
  }

  // Avoid data being deleted by volmap's destructor
  // since it is now owned by the molecule and will be
  // freed by its destructor .
  volmap->data = NULL;

  msgInfo << "Added volmap " << mol->num_volume_data()-1
          << " to molecule " << molid << "." << sendmsg;

  return 1;
}

// Set probe coordinates,charges and VDW parameters.
// Currently we only support up to 2 probe atoms but
// we should change that later.
void VolMapCreateILS::set_probe(int numatoms, int num_conf,
                                const float *pcoords,
                                const float *vdwrmin,
                                const float *vdweps,
                                const float *charge) {
  if (numatoms<=0) return;
  if (numatoms==1 && num_conf>0) num_conf = 0;

  conformer_freq  = num_conf;
  num_probe_atoms = numatoms;
  probe_coords = new float[3*numatoms];
  probe_vdw    = new float[2*numatoms];
  probe_charge = new float[numatoms];

  // Thermopdynamic beta = 1/kT factor (k = Boltzmann const.)
  // 1/k [kcal/mol] = 1.0/(1.38066*6.022/4184) = 503.2206
  const float beta = 503.2206f/temperature;

  // The combination rules for VDW parameters of
  // two atoms are:
  //
  // eps(i,j)  = sqrt(eps(i) * eps(j))
  // rmin(i,j) = rmin(i)/2 + rmin(j)/2
  //
  // where rmin(i,j) is the distance of the two atoms at the
  // energy minimum. The parameters are provided from the
  // user interface in form the contribution of each atom
  // rmin(i)/2 (which can be interpreted as the VDW radius).

  // We take the sqrt(eps) here already so that during
  // the computation we merely have to multiply the two
  // parameters.
  // We are computing the occupancy
  //   rho = sum_i exp(-1/kT * U[i]) 
  //     U = eps*[(rmin/r)^12 - 2*(rmin/r)^6]
  // We include beta = 1/kT into the probe eps parameter
  // for speed.
  int i;
  for (i=0; i<num_probe_atoms; i++) {
    probe_vdw[2*i  ] = beta*sqrtf(-vdweps[i]);
    probe_vdw[2*i+1] = vdwrmin[i];
  }

  if (charge) {
    memcpy(probe_charge, charge, numatoms*sizeof(float));
  } else {
    memset(probe_charge, 0, numatoms*sizeof(float));
  }

  // Get geometric center:
  float cgeom[3];
  vec_zero(cgeom);

  for (i=0; i<num_probe_atoms; i++) {
    vec_add(cgeom, cgeom, &pcoords[3*i]);
  }
  vec_scale(cgeom, 1.f/(float)num_probe_atoms, cgeom);

  // Shift geometric center to origin:
  for (i=0; i<num_probe_atoms; i++) {
    vec_sub(&probe_coords[3*i], &pcoords[3*i], cgeom);
  }
}


// Set the two highest symmetry axes for the probe and a flag
// telling if we have a tetrahedral symmetry.
void VolMapCreateILS::set_probe_symmetry(
        int order1, const float *axis1,
        int order2, const float *axis2,
        int tetrahedral) {
  probe_tetrahedralsymm = tetrahedral;

  if (axis1 && order1>1) {
    probe_axisorder1 = order1;
    vec_copy(probe_symmaxis1, axis1);
    vec_normalize(probe_symmaxis1);
  }
  if (axis2 && order2>1) {
    probe_axisorder2 = order2;
    vec_copy(probe_symmaxis2, axis2);
    vec_normalize(probe_symmaxis2);
  }
  if (!tetrahedral && probe_axisorder1>1 && probe_axisorder2>1 &&
      dot_prod(probe_symmaxis1, probe_symmaxis2) > 0.05) {
    // Axes not orthogonal, drop lower order axis
    if (probe_axisorder1<probe_axisorder2) {
      probe_axisorder1 = probe_axisorder2;
    }
    probe_axisorder2 = 1;
  }
}


// Set minmax coordinates of rectangular grid bounding box
void VolMapCreateILS::set_minmax (float minx, float miny, float minz,
                                  float maxx, float maxy, float maxz) {
  minmax[0] = minx;
  minmax[1] = miny;
  minmax[2] = minz;
  minmax[3] = maxx;
  minmax[4] = maxy;
  minmax[5] = maxz;
}


// Request PBC aware computation.
// Optionally one can set the PBC cell center
// (default is {0 0 0}).
// If bbox is TRUE then the bounding box of the grid will
// be chosen as the bounding box of the (possibly rhombic)
// PBC cell.
void VolMapCreateILS::set_pbc(float center[3], int bbox) {
  pbc    = 1;
  pbcbox = bbox;
  if (center) vec_copy(pbccenter, center);
}


// Set maximum energy considered in the calculation.
void VolMapCreateILS::set_maxenergy(float maxenergy) {
  max_energy = maxenergy;
  if (max_energy>DEFAULT_EXCL_ENERGY) {
    max_energy = DEFAULT_EXCL_ENERGY;
  }
}


// Set selection to be used for alignment.
void VolMapCreateILS::set_alignsel(AtomSel *asel) {
  if (asel) alignsel = asel;
}


// Set transformation matrix that was used for the
// alignment of the first frame.
void VolMapCreateILS::set_transform(const Matrix4 *mat) {
  transform = *mat;
}


// Set grid dimensions to the minmax coordinates and
// pad the positive side of each dimension.
//
//        lattice
//   +---+---+---+---+
//  _|___|___|___|_  |  _
// | |   |   |   | | |  |
// | o---o---o---o---+  |
// | |   |   |   | | |  |
// | |   |   |   | | |  |
// | o---o---o---o---+  |minmax
// |_|_  |   |   | | |  |
// | | | |   |   | | |  |
// | o---o---o---o---+  |
// |___|___________|    L
//  
//   |---|       
//   delta
//               
//   |-----------|
//   xaxis
//
int VolMapCreateILS::set_grid() {
  // Number of samples in the final downsampled map.
  int nx = (int) ceilf((minmax[3]-minmax[0])/(delta*nsubsamp));
  int ny = (int) ceilf((minmax[4]-minmax[1])/(delta*nsubsamp));
  int nz = (int) ceilf((minmax[5]-minmax[2])/(delta*nsubsamp));

  // Number of samples used during computation.
  nsampx = nx*nsubsamp;
  nsampy = ny*nsubsamp;
  nsampz = nz*nsubsamp;

  // For volumetric maps the origin is the center of the
  // first cell (i.e. the location of the first sample)
  // rather than the lower corner of the minmax box.

  // Origin for final downsampled map
  float origin[3];
  origin[0] = minmax[0] + 0.5f*delta*nsubsamp;
  origin[1] = minmax[1] + 0.5f*delta*nsubsamp;
  origin[2] = minmax[2] + 0.5f*delta*nsubsamp;

  // Origin for the highres map used during computation
  gridorigin[0] = minmax[0] + 0.5f*delta;
  gridorigin[1] = minmax[1] + 0.5f*delta;
  gridorigin[2] = minmax[2] + 0.5f*delta;
  

  // Cell spanning vectors,
  // (delta is the distance between two samples)
  float cellx[3], celly[3], cellz[3];
  cellx[0] = delta*nsubsamp;
  cellx[1] = 0.f;
  cellx[2] = 0.f;
  celly[0] = 0.f;
  celly[1] = delta*nsubsamp;
  celly[2] = 0.f;
  cellz[0] = 0.f;
  cellz[1] = 0.f;
  cellz[2] = delta*nsubsamp;

  // The axes that span the whole lattice i.e. the vector
  // between the first and the last sample point in each
  // dimension.
  float xaxis[3], yaxis[3], zaxis[3];
  int i;
  for (i=0; i<3; i++) {
    xaxis[i] = cellx[i]*(nx-1);
    yaxis[i] = celly[i]*(ny-1);
    zaxis[i] = cellz[i]*(nz-1);
  }

  // Initialize the final downsampled map:
  float *data = new float[nx*ny*nz];
  if (maskonly) {
    // Fill mask map with ones
    for (i=0; i<nx*ny*nz; i++) {
      data[i] = 1.f;
    }    
  } else {
    memset(data, 0, nx*ny*nz*sizeof(float));
  }

  volmap = new VolumetricData("\0", origin,
                  xaxis, yaxis, zaxis, nx, ny, nz, data);

  if (!volmap->gridsize())
    return MEASURE_ERR_ZEROGRIDSIZE;

  return 0;
}


// Initialize the ILS calculation
int VolMapCreateILS::initialize() {
  if (!app->molecule_numframes(molid))
    return MEASURE_ERR_NOFRAMES;

  msgInfo << "\n-- Implicit Ligand Sampling --\n\n"
	  << "This computes the potential of mean force (free energy) over\n"
	  << "a 3-D grid, for a small ligand.\n\n" << sendmsg;
    //	  << "If you use this method in your work, please cite:\n\n"
    //	  << "  J COHEN, A ARKHIPOV, R BRAUN and K SCHULTEN, \"Imaging the\n"
    //	  << "  migration pathways for O2, CO, NO, and Xe inside myoglobin\".\n"
    //	  << "  Biophysical Journal 91:1844-1857, 2006.\n\n" << sendmsg;
  char tmpstr[256];
  sprintf(tmpstr, "Temperature:    %g K", temperature);
  msgInfo << tmpstr << sendmsg;
  sprintf(tmpstr, "Energy cutoff:  %g kT", max_energy);
  msgInfo << tmpstr << sendmsg;

  // Compute electrostatics only if probe charges are nonzero
  if (compute_elec && probe_charge[0]==0.0 && probe_charge[1]==0.0) {
    compute_elec = 0;
  }
  msgInfo << "Electrostatics: ";
  if (compute_elec) msgInfo << "on"  << sendmsg;
  else              msgInfo << "off" << sendmsg;

  sprintf(tmpstr, "Map resolution: %g Angstrom", delta);  
  msgInfo << tmpstr << sendmsg;
  sprintf(tmpstr, "Subgrid res:    %d points", nsubsamp);
  msgInfo << tmpstr << sendmsg;
    
  // Initialize parameters
  if (num_probe_atoms==1) {
    sprintf(tmpstr, "VDW cutoff:     %6.3f Angstrom", cutoff);
    msgInfo << tmpstr << sendmsg;
  }
  else {
    // Get the max probe radius, i.e. the largest distance
    // of an atom to the center of mass

    float max_proberad = 0.f;
    int i;
    for (i=0; i<num_probe_atoms; i++) {
      float dist = norm(&probe_coords[3*i]);
      if (dist>max_proberad) max_proberad = dist;
    }

    sprintf(tmpstr, "VDW cutoff:     %6.3f Angstrom (%6.3f + probe radius)",
            cutoff+max_proberad, cutoff);
    msgInfo << tmpstr << sendmsg;

    extcutoff = cutoff + max_proberad;
  }

  if (alignsel) {
    // Get coordinates of the alignment reference frame (the current frame of alignsel)
    alignrefpos = alignsel->coordinates(app->moleculeList);

  } else if (!alignsel && last-first>1) {
    msgWarn << sendmsg;
    msgWarn << "Use of periodic boundaries requested (-pbc) but" << sendmsg;
    msgWarn << "no alignment matrix specified (you didn't use -alignsel)." << sendmsg;
    msgWarn << "Have you aligned your structure prior to this calculation?" << sendmsg;
    msgWarn << "Hopefully not, since it will have totally messed" << sendmsg;
    msgWarn << "up the definition of your PBC cells. Instead you" << sendmsg;
    msgWarn << "should use the -alignsel option and let volmap handle" << sendmsg;
    msgWarn << "the alignment." << sendmsg;
  }

  if (pbc && pbcbox) {
    // Compute minmax based on the PBC cell of the first frame:
    // Get the smallest rectangular box that encloses the
    // entire (possibly nonorthogonal) PBC cell
    int err = compute_pbcminmax(app->moleculeList, molid, first,
                   pbccenter, &transform, &minmax[0], &minmax[3]);
    if (err) return err;
  }

  sprintf(tmpstr, "{%g %g %g} {%g %g %g}\n", minmax[0], minmax[1], minmax[2], minmax[3], minmax[4], minmax[5]);
  msgInfo << "Grid minmax = " << tmpstr << sendmsg; 

  // Automatically add the force field cutoff to the system
  float ffcutoff[3];
  ffcutoff[0] = ffcutoff[1] = ffcutoff[2] = extcutoff;
  float cutminmax[6];
  vec_sub(cutminmax,   minmax,   ffcutoff);
  vec_add(cutminmax+3, minmax+3, ffcutoff);
  
  if (!box_inside_pbccell(first, minmax)) {
    if (!pbc) {
      msgWarn << sendmsg;
      msgWarn << "Selected grid exceeds periodic cell boundaries." << sendmsg;
      msgWarn << "Parts of the map will be undefined!" << sendmsg;
      msgWarn << "(Consider using -pbc flag to ensure map is valid over entire grid.)"
              << sendmsg << sendmsg;
    } else {
      msgInfo << "Selected grid exceeds periodic cell boundaries." << sendmsg;
      msgInfo << "Using PBC image atoms to ensure map is valid over entire grid."
              << sendmsg;
    }
  } else if (!box_inside_pbccell(first, cutminmax)) {
    if (!pbc) {
      msgWarn << sendmsg;
      msgWarn << "Nonbonded interaction cutoff region needed around" << sendmsg;
      msgWarn << "selected grid exceeds periodic cell boundaries." << sendmsg;
      msgWarn << "Parts of the map will be ill-defined!" << sendmsg;
      msgWarn << "(Consider using -pbc flag to ensure map is valid over entire grid.)"
              << sendmsg << sendmsg;;
    } else {
      msgInfo << "Nonbonded interaction cutoff region needed around" << sendmsg;
      msgInfo << "selected grid exceeds periodic cell boundaries." << sendmsg;
      msgInfo << "Using PBC image atoms to ensure map is valid over entire grid."
              << sendmsg;
    }
  }

  // Compute and set the grid dimensions
  set_grid();
  msgInfo << "Grid origin = {"
          << volmap->origin[0] << " "
          << volmap->origin[1] << " "
          << volmap->origin[2] << "}" << sendmsg;

  char tmp[64];
  sprintf(tmp, "Grid size   = %dx%dx%d (%.1f MB)",
          nsampx, nsampy, nsampz,
          sizeof(float)*nsampx*nsampy*nsampz/(1024.*1024.));
  msgInfo << tmp << sendmsg;

  if (nsubsamp>1) {
    sprintf(tmp, "Downsampling final map to %dx%dx%d  (%.1f MB)",
            volmap->xsize, volmap->ysize, volmap->zsize,
            sizeof(float)*volmap->gridsize()/(1024.*1024.));
    msgInfo << tmp << sendmsg;
  }

  Molecule *mol = app->moleculeList->mol_from_id(molid);
  num_atoms = mol->nAtoms;

  msgInfo << "Global transformation for all frames:" << sendmsg;
  print_Matrix4(&transform);

  if (alignsel) msgInfo << "Aligning all frames to the first one." << sendmsg;
  else          msgInfo << "Assuming all frames are aligned." << sendmsg;

  if (maskonly) {
    msgInfo << sendmsg << "Masking mode:" << sendmsg
            << "Generating a mask map containing 1 for grid points that" << sendmsg
            << "have a valid contribution from each frame and 0 otherwise."
            << sendmsg << sendmsg;
    return MEASURE_NOERR;
  }


  // Create list of unique VDW parameters which can be accessed
  // through an index list.
  create_unique_paramlist();

  // Find smallest VDW rmin parameter for all system atoms 
  float min_sysrmin = vdwparams[1];
  int i;
  for (i=1; i<num_unique_types; i++) {
    if (vdwparams[2*i+1]<min_sysrmin) min_sysrmin = vdwparams[2*i+1];
  }

  // Find largest VDW rmin parameter for all probe atoms
  float min_probermin = probe_vdw[1];
  for (i=1; i<num_probe_atoms; i++) {
    if (probe_vdw[2*i+1]<min_probermin) min_probermin = probe_vdw[2*i+1];
  }
  

  const float invbeta = temperature/503.2206f;
  msgInfo << "Probe with "<<num_probe_atoms<<" atoms:" << sendmsg;
  for (i=0; i<num_probe_atoms; i++) {
    sprintf(tmpstr, "  atom %d: epsilon = %g, rmin/2 = %g, charge = % .3f",
            i, -pow(invbeta*probe_vdw[2*i],2),
            probe_vdw[2*i+1], probe_charge[i]);
    msgInfo << tmpstr << sendmsg;
  }

  // Create conformers for multiatom probes
  if (conformer_freq && num_probe_atoms>1) {
    initialize_probe();
  } else {
    msgInfo << "Ignoring orientations for monoatomic probe." << sendmsg;
  }

  get_eff_proberadius();


  // Define a cutoff for identifying obvious atom clashes:
  sprintf(tmpstr, "Clash exclusion distance:  %.3f Angstrom", excl_dist);
  msgInfo << tmpstr << sendmsg;

//   volmask = new VolumetricData(volmap->name, volmap->origin,
//                     volmap->xaxis, volmap->yaxis, volmap->zaxis,
//                     volmap->xsize, volmap->ysize, volmap->zsize, NULL);
  //volmask = new VolumetricData(*volmap);
  //volmask->data = new float[gridsize*sizeof(float)];
  //memset(volmask->data, 0, sizeof(float)*gridsize);
  //char name[8];
  //strcpy(name, "mask");
  //volmask->set_name(name);

  return MEASURE_NOERR;
}


void VolMapCreateILS::get_eff_proberadius() {
  int numconf, numorient, numrot;
  float *conf;
  if (probe_tetrahedralsymm) {
    numconf = gen_conf_tetrahedral(conf, 6, numorient, numrot);
  } else {
    numconf = gen_conf(conf, 8, numorient, numrot);
  }

  int t, i, j, k;
  float max_proberad = 0.f;
  for (k=0; k<num_probe_atoms; k++) {
    float dist = norm(&probe_coords[3*k]);
    float rmin = probe_vdw[2*k+1];
    if (dist+rmin>max_proberad) max_proberad = dist+rmin;
  }

  float stepsize = 0.01f;
  float *effrad = new float[num_unique_types];
  excl_dist = 999999.f;

  for (t=0; t<num_unique_types; t++) {
    float vdweps  = vdwparams[2*t  ];
    float vdwrmin = vdwparams[2*t+1];
    //printf("vdweps=%f, vdwrmin=%f\n", vdweps, vdwrmin);
    float begin = max_proberad + vdwrmin;
    int maxnumstep = int(0.5+begin/stepsize);
    //printf("maxproberad=%.2f\n", max_proberad);
    float Wmin = 0.f;
//    float Ropt = 0.f;

    for (i=0; i<maxnumstep; i++) {
      float dist = begin - float(i)*stepsize;
      if (dist<=0.0f) break;

      float avgocc = 0.f;
      for (j=0; j<numconf; j++) {
        float *coor = &conf[3*num_probe_atoms*j];
        float u = 0.f;
        for (k=0; k<num_probe_atoms; k++) {
          float dx = dist-coor[3*k];
          float dy = coor[3*k+1];
          float dz = coor[3*k+2];
          float r2 = dx*dx + dy*dy + dz*dz;
          float epsilon = vdweps  * probe_vdw[2*k];
          float rmin    = vdwrmin + probe_vdw[2*k+1];
          float rm6 = rmin*rmin / r2;
          rm6 = rm6 * rm6 * rm6;
          u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
          //printf("     u[%d] = %f, r2=%f\n", k, epsilon * rm6 * (rm6 - 2.f), r2);
        }
        //printf("  u[%d] = %f\n", j, u);

        float occ = expf(-float(u));

        avgocc += occ;
      }
      avgocc /= float(numconf);
      float W = -logf(avgocc);
      //printf("dist=%.2f occ=%g; dG=%g\n", dist, avgocc, -logf(avgocc));
      if (W<Wmin) { 
        Wmin = W;
//        Ropt = dist; 
      }
      if (W>max_energy+5.f) {
        effrad[t] = dist;
        break;
      }
    }

    //printf("effrad[%d]=%.3f Ropt=%.3f, Wmin=%f\n", t, effrad[t], Ropt, Wmin);
    if (effrad[t]<excl_dist) excl_dist = effrad[t];
  }

  delete [] effrad;
  delete [] conf;
}


// Check if two vectors are collinear within a given tolerance.
// Assumes that the vectors are normalized.
static bool collinear(const float *axis1, const float *axis2, float tol) {
  if (fabs(dot_prod(axis1, axis2)) > 1.f-DEGTORAD(tol)) return 1;
  return 0;
}


// Check if probe is a linear molecule and returns
// the Cinf axis.
int VolMapCreateILS::is_probe_linear(float *axis) {
  if (num_probe_atoms==1) return 0;

  float vec0[3], vec1[3];
  vec_sub(vec0, &probe_coords[3], &probe_coords[0]);
  vec_copy(axis, vec0);
  if (num_probe_atoms==2) return 1;

  float norm0 = norm(vec0);
  int i;
  for (i=2; i<num_probe_atoms; i++) {
    vec_sub(vec1, &probe_coords[3*i], &probe_coords[0]);
    float dot = dot_prod(vec0, vec1)/(norm0*norm(vec1));
    if (fabs(dot) < 0.95f) return 0;

    if (dot>0.f) vec_add(axis, axis, vec1);
    else         vec_sub(axis, axis, vec1);
  }
  vec_normalize(axis);
  return 1;
}


// Subdivide a triangle specified by the points pole, eq1 and eq2.
// A freqency of 1 means no partitioning, 2 signifies that each edge
// is subdivided into 2 segments and so on.
// The resulting vertices are returned in v.
static int triangulate(const float *pole, const float *eq1,
                        const float *eq2, int freq, float *v) {
  if (freq==0) {
    vec_copy(v, pole);
    return 1;
  }

  float meridian[3], parallel[3];
  vec_sub(meridian, eq1, pole);
  vec_sub(parallel, eq2, eq1);
  float mlen = norm(meridian);
  float plen = norm(parallel);
  vec_normalize(meridian);
  vec_normalize(parallel);
  int i, k = 0;
  for (i=0; i<=freq; i++) {
    float latitude = float(i)/float(freq)*mlen;
    float p[3], p0[3];
    vec_copy(p0, pole);
    vec_scaled_add(p0, latitude, meridian);
    int j;
    for (j=0; j<=i; j++) {
      float longitude = float(j)/float(freq)*plen;
      vec_copy(p, p0);
      vec_scaled_add(p, longitude, parallel);
      vec_copy(&v[3*k], p);
      k++;
    }
  }
  return k;
}


// Generate the 6 vertices of an octahedron
// (or only 3 if the symmetry flag was set)
static void octahedron(float *vertices, int C2symm) {
  const float v[] = {1.f, 0.f, 0.f,
                     0.f, 1.f, 0.f,
                     0.f, 0.f, 1.f};
  memcpy(vertices, v, 9*sizeof(float));
  if (!C2symm) {
    int i;
    for (i=0; i<3; i++) {
      vec_negate(&vertices[9+3*i], &vertices[3*i]);
    }
  }
}

// Generate the 8 vertices of a hexahedron
// (or only 4 if the symmetry flag was set)
static void hexahedron(float *vertices, int C2symm) {
  const float v[] = {1.f,  1.f,  1.f,
                     1.f, -1.f,  1.f,
                     1.f,  1.f, -1.f,
                     1.f, -1.f, -1.f};
  memcpy(vertices, v, 12*sizeof(float));

  if (!C2symm) {
    int i;
    for (i=0; i<4; i++) {
      vec_negate(&vertices[12+3*i], &vertices[3*i]);
    }
  }
}

// Generate normal vectors for the 12 dodecahedral faces
// (or only 6 if the symmetry flag was set) and the 20
// vertices. The vertices are at the same time the faces
// of the icosahedron, the dual of the dodecahedron.
// XXX I know that this takes more space than just listing
//     the hardcoded points...
static void dodecahedron(float *faces, float *vertices, int C2symm) {
  // Angle between two faces
  const float dihedral = float(180.f-RADTODEG(acos(-1.f/sqrtf(5.f))));

  // Faces:
  float x[3];
  vec_zero(x); x[0] = 1.f;
  vec_copy(&faces[0], x);
  // Contruct first point in ring
  Matrix4 rot;
  rot.rot(dihedral, 'z');
  rot.multpoint3d(&faces[0], &faces[3]);
  // Get other points by rotation
  int i;
  for (i=1; i<5; i++) {
    rot.identity();
    rot.rot(float(i)*72.f, 'x');
    rot.multpoint3d(&faces[3], &faces[3+3*i]);
  }
  if (!C2symm) {
    for (i=0; i<6; i++) {
      vec_negate(&faces[18+3*i], &faces[3*i]);
    }
  }

  // Vertices
  // center and first ring
  for (i=0; i<5; i++) {
    vec_copy(&vertices[3*i], &faces[0]);
    vec_add(&vertices[3*i], &vertices[3*i], &faces[3*(i+1)]);
    vec_add(&vertices[3*i], &vertices[3*i], &faces[3*((i+1)%5+1)]);
    vec_normalize(&vertices[3*i]);
  }
  // second ring
  vec_copy(&vertices[3*5], &faces[3*1]);
  vec_add(&vertices[3*5], &vertices[3*5], &faces[3*2]);
  vec_normalize(&vertices[3*5]);
  float cross[3];
  cross_prod(cross, &vertices[3*5], &faces[0]);
  vec_normalize(cross);
  float phi = angle(&vertices[3*5], &vertices[0]);
  rot.identity();
  rot.rotate_axis(cross, float(DEGTORAD(-phi)));
  rot.multpoint3d(&vertices[3*5], &vertices[3*5]);
  for (i=1; i<5; i++) {
    rot.identity();
    rot.rot(float(i)*72.f, 'x');
    rot.multpoint3d(&vertices[3*5], &vertices[3*5+3*i]);
  }

  // opposite orientations
  if (!C2symm) {
    for (i=0; i<10; i++) {
      vec_negate(&vertices[30+3*i], &vertices[3*i]);
    }
  }
}


// Geodesic triangulation of an icosahedron.
// Parameter freq is the geodesic frequency, the 
// flag C2symm signifies if the molecule is symmetric
// and we can omit one hemisphere.
// Allocates memory for the orientation vectors
// and returns the number of generated orientations.
static int icosahedron_geodesic(float *(&orientations),
                                int C2symm, int freq) {
  int i;
  float faces[3*12];
  float junk[3*20];
  dodecahedron(faces, junk, 0);
  float meridian[3], parallel[3];

  int numvertex = 0; // number of triangle vertices
  for (i=1; i<=freq+1; i++) numvertex += i;
  int symmfac = C2symm ? 2 : 1;
  int numorient = (10*freq*freq + 2)/symmfac;
  orientations = new float[3*numorient];

  // Add pole to array of orientations
  vec_copy(&orientations[0], &faces[0]);
  int k = 1;

  for (i=0; i<5; i++) {
    // First ring
    float p0[3], p1[3], p2[3];
    vec_sub(parallel, &faces[3+3*((i+1)%5)], &faces[3*(i+1)]);
    float edgelen = norm(parallel);
    vec_normalize(parallel);
    vec_sub(meridian, &faces[3*(i+1)], &faces[0]);
    vec_normalize(meridian);
    vec_copy(p0, &faces[0]);
    vec_copy(p1, &faces[3*(i+1)]);
    vec_copy(p2, &faces[3+3*((i+1)%5)]);
    vec_scaled_add(p0,  1.f/float(freq)*edgelen, meridian);
    vec_scaled_add(p2, -1.f/float(freq)*edgelen, parallel);
    triangulate(p0, p1, p2, freq-1, &orientations[3*k]);
    k += numvertex-(freq+1);

    // Second ring
    vec_sub(meridian, &faces[3*(i+1)], &faces[21+3*((i+3)%5)]);
    vec_normalize(meridian);
    vec_copy(p0, &faces[21+3*((i+3)%5)]);
    vec_copy(p1, &faces[3*(i+1)]);
    vec_scaled_add(p0,  1.f/float(freq)*edgelen, meridian);
    vec_scaled_add(p1, -1.f/float(freq)*edgelen, meridian);
    vec_copy(p2, p1);
    vec_scaled_add(p2, float(freq-2)/float(freq)*edgelen, parallel);
    triangulate(p0, p1, p2, freq-2, &orientations[3*k]);
    k += numvertex-(freq+1)-freq;
  } 

  if (!C2symm) {
    for (i=0; i<numorient/2; i++) {
      vec_negate(&orientations[3*numorient/2+3*i], &orientations[3*i]);
    }
  }

  return numorient;
}

float VolMapCreateILS::dimple_depth(float phi) {
  int i;
  phi = 0.5f*phi;
  // Find smallest system atom
  float min_syseps  = vdwparams[0];
  float min_sysrmin = vdwparams[1];
  for (i=1; i<num_unique_types; i++) {
    if (vdwparams[2*i+1]<min_sysrmin) {
      min_syseps  = vdwparams[2*i  ];
      min_sysrmin = vdwparams[2*i+1];
    }
  }
  float maxdepth = 0.f;

  // Check all probe atoms for dimple depth
  for (i=0; i<num_probe_atoms; i++) {
    float d = norm(&probe_coords[3*i]);
    float a = d*sinf(float(DEGTORAD(phi)));
    float m = a/cosf(float(DEGTORAD(phi)));
    if (phi == 90.f) m = a;
    float c = probe_vdw[2*i+1] + min_sysrmin;
    if (m>c) {
      maxdepth = d;
      break;
    }
    float b = sqrtf(c*c-m*m);
    float depth = d + c - d*cosf(float(DEGTORAD(phi))) - b;
    //printf("d=%f, rp=%.3f, rs=%.3f, g=%f, b=%f, a=%f, c=%f\n",
    //       d, probe_vdw[2*i+1], min_sysrmin, d*cosf(DEGTORAD(phi)), b, m, c);

    float epsilon = min_syseps * probe_vdw[2*i];
    float rmin = min_sysrmin + probe_vdw[2*i+1];
    // Get energy in dimple (atoms are touching)
    float r2 = c*c;
    float rm6 = rmin*rmin / r2;
    rm6 = rm6 * rm6 * rm6;
    float u0 = epsilon * rm6 * (rm6 - 2.f);
    // Get energy for outer radius
    r2 = (c+depth)*(c+depth);
    rm6 = rmin*rmin / r2;
    rm6 = rm6 * rm6 * rm6;
    float u1 = epsilon * rm6 * (rm6 - 2.f);
    // Get energy for outer radius
    r2 = (c-depth)*(c-depth);
    rm6 = rmin*rmin / r2;
    rm6 = rm6 * rm6 * rm6;
    float u2 = epsilon * rm6 * (rm6 - 2.f);
    float du1 = u1-u0;
    float du2 = u2-u0;
    printf("phi = %.2f: %d dimple depth = %f = %5.2f%%, dU1 = %fkT = %5.2f%%; dU1 = %fkT = %5.2f%%\n",
           phi, i, depth, 100.f*depth/(d+probe_vdw[2*i]), du1, fabs(100.f*du1/u0), du2, fabs(100.f*du2/u0));

    if (depth>maxdepth) maxdepth = depth;
  }
  return maxdepth;
}


// Generate conformers for tetrahedral symmetries.
// Allocates memory for *conform array and returns the number
// of generated conformers.
// numorient:  # symmetry unique orientations
// numrot:     # rotamers per orientation
//
// Rotate around the 4 axes defined by the corners of
// the tetrahedron and its dual (also a tetrahedron).
// XXX:
// This approach exploits the probe symmetry very well
// but for higher frequencies you will start seeing 
// "holes" in the pattern. These holes are in the middle
// of the 12 corners of the cube spanned by the vertices
// of the two dual tetrahedra
// One idea how to fix this would be to generate two
// extra sets of conformations where the basic tetrahedra
// are rotated such that their vertices are in the holes.
int VolMapCreateILS::gen_conf_tetrahedral(float *(&conform),
                     int freq, int &numorient, int &numrot) {
  // Generate the 4 corners of the tetrahedron
  float tetra0[3], tetra1[3], tetra2[3], tetra3[3];
  vec_zero(tetra0);
  tetra0[0] = 1.f;
  Matrix4 rot;
  rot.rot(109.47122f, 'z');
  rot.multpoint3d(tetra0, tetra1);
  rot.identity();
  rot.rot(120.f, 'x');
  rot.multpoint3d(tetra1, tetra2);
  rot.multpoint3d(tetra2, tetra3);

#if defined(DEBUG)
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  MoleculeGraphics *gmol = mol->moleculeGraphics();
  gmol->use_color(8);
  gmol->add_line(tetra0, tetra1, 0, 1);
  gmol->add_line(tetra0, tetra2, 0, 1);
  gmol->add_line(tetra0, tetra3, 0, 1);
  gmol->add_line(tetra1, tetra2, 0, 1);
  gmol->add_line(tetra1, tetra3, 0, 1);
  gmol->add_line(tetra2, tetra3, 0, 1);
#endif

  // array of probe orientation vectors
  float *orientations;
  orientations = new float[3*8];

  vec_copy(&orientations[3*0], tetra0);
  vec_copy(&orientations[3*1], tetra1);
  vec_copy(&orientations[3*2], tetra2);
  vec_copy(&orientations[3*3], tetra3);
  float face[3];
  vec_copy(face, tetra0);
  vec_add(face, face, tetra1);
  vec_add(face, face, tetra2);
  vec_copy(&orientations[3*4], face);
  vec_copy(face, tetra0);
  vec_add(face, face, tetra1);
  vec_add(face, face, tetra3);
  vec_copy(&orientations[3*5], face);
  vec_copy(face, tetra0);
  vec_add(face, face, tetra2);
  vec_add(face, face, tetra3);
  vec_copy(&orientations[3*6], face);
  vec_copy(face, tetra1);
  vec_add(face, face, tetra2);
  vec_add(face, face, tetra3);
  vec_copy(&orientations[3*7], face);

  numrot = freq;  // # rotamers per orientation
  numorient = 8;  // the tetrahedron and its dual
  int numconf = numorient*numrot-6;
  conform = new float[3*num_probe_atoms*numconf];
  memset(conform, 0, 3*num_probe_atoms*numconf*sizeof(float));

  int conf = 0;
  int i;
  for (i=0; i<numorient; i++) {
    float *dir = &orientations[3*i];
    vec_normalize(dir);

    // Apply rotations around dir
    int j;
    for (j=0; j<numrot; j++) {
      // The non-rotated orientations are equivalent
      // for each tetrahedral dual so we skip them.
      if (i%4 && j==0) continue;

      float psi = float(j)/float(numrot)*360.f/float(probe_axisorder1);
      Matrix4 rot;
      if (i>=numorient/2) {
        // Flip orientation by 180 deg in order to get the
        // dual of the tetrahedron which is also a tetrahedron
        // with corners and faces exchanged.
        float z[3];
        vec_zero(z); z[2]=1.f;
        rot.rotate_axis(z, float(DEGTORAD(180.f)));
      }

      // Rotate around dir
      rot.rotate_axis(dir, float(DEGTORAD(psi)));
    
      // Apply rotation to all probe atoms
      int k;
      for (k=0; k<num_probe_atoms; k++) {
        rot.multpoint3d(&probe_coords[3*k],
                        &conform[3*num_probe_atoms*conf + 3*k]);
      }

      conf++;
    }
  }

  delete [] orientations;

  // Return the number of generated conformers
  return numconf;
}


// Compute angle that rotates cross(axis, v1) into cross(axis, v2)
// with respect to rotation around axis.
static float signed_angle(const float *axis,
                          const float *v1, const float *v2) {
  float normaxis[3], cross1[3], cross2[3], cross3[3];
  cross_prod(cross1, axis, v1);
  cross_prod(cross2, axis, v2);
  cross_prod(cross3, v1, v2);
  vec_normalize(cross3);
  vec_copy(normaxis, axis);
  vec_normalize(normaxis);
  float phi = angle(cross1, cross2);
  if (dot_prod(axis, cross3)<0) {
    phi = -phi;
  }
  return phi;
}


// Generate conformers for all non-tetrahedral symmetries.
// Allocates memory for *conform array and returns the number
// of generated conformers.
// numorient:  # symmetry unique orientations
// numrot:     # rotamers per orientation
int VolMapCreateILS::gen_conf(float *(&conform), int freq,
                              int &numorient, int &numrot) {
  int i;
  float *orientations = NULL; // array of probe orientation vectors
  int C2symm = (probe_axisorder2==2 ? 1 : 0);
  int symmfac = C2symm ? 2 : 1;
  float anglespacing = 360.f;

  switch (freq) {
  case 1:
    numorient = 1;
    orientations = new float[3*numorient];
    break;
  case 2:
    // 6 octahedral vertices
    numorient = 6/symmfac;
    orientations = new float[3*numorient];
    octahedron(orientations, C2symm);
    anglespacing = 90.f;
    break;
  case 3:
    // 8 hexahedral vertices
    numorient = 8/symmfac;
    orientations = new float[3*numorient];
    hexahedron(orientations, C2symm);
    anglespacing = 109.47122f;
    break;
  case 4:
    // 12 dodecahedral faces 
    numorient = 12/symmfac;
    orientations = new float[3*numorient];
    float vertices[3*20]; // junk
    dodecahedron(orientations, vertices, C2symm);
    anglespacing = float(180.f-RADTODEG(acos(-1.f/sqrtf(5.f))));
      break;
  case 5:
    // 20 dodecahedral vertices
    numorient = 20/symmfac;
    orientations = new float[3*numorient];
    float faces[3*12]; // junk
    dodecahedron(faces, orientations, C2symm);
    anglespacing = 180.f-138.189685f;
    break;
  case 6:
    // 12 faces and 20 vertices of a dodecahedron
    numorient = 32/symmfac;
    orientations = new float[3*numorient];
    dodecahedron(&orientations[0], &orientations[3*12/symmfac], C2symm);
    anglespacing = 37.377380f;
    break;
  default:
    // Triangulate icosahedral faces
    freq -= 5;
 
    numorient = icosahedron_geodesic(orientations, C2symm, freq);

    anglespacing = (180.f-138.189685f)/freq;
    break;
  }

  // Number of rotamers per orientation
  // Chosen such that the rotation stepping angle is as close
  // as possible to the angle between two adjacent orientations.
  numrot = 1;
  if (probe_axisorder1>=0) {
    numrot = int(floorf(360.f/probe_axisorder1/anglespacing + 0.5f));
  }

  int numconf = numorient*numrot;
  //printf("numorient=%d, numrot=%d, numconf=%d, num_probe_atoms=%d\n",numorient,numrot,numconf,num_probe_atoms);
  conform = new float[3*num_probe_atoms*numconf];
  memset(conform, 0, 3*num_probe_atoms*numconf*sizeof(float));

#if defined(DEBUG)
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  MoleculeGraphics *gmol = mol->moleculeGraphics();
  for (i=0; i<numorient; i++) {
    float dir[3];
    vec_scale(dir, 0.8, &orientations[3*i]);
    gmol->use_color(7);
    if (i==0) gmol->use_color(4);
    if (i==1) gmol->use_color(8);
    if (i==2) gmol->use_color(9);
    gmol->add_sphere(dir, 0.1, 8);
  }
#endif

  // Generate conformer coordinates
  int conf = 0;
  for (i=0; i<numorient; i++) {
    float *dir = &orientations[3*i];
    vec_normalize(dir);

    float cross[3], x[3], y[3], z[3];
    vec_zero(x); x[0]=1.f;
    vec_zero(y); y[1]=1.f;
    vec_zero(z); z[2]=1.f;
    float phi = 0.f;
    float theta = 0.f;

    if (!collinear(x, dir, 2.f)) {
      // Get rotation axis and angle phi to rotate x-axis into dir
      cross_prod(cross, x, dir);
      phi = angle(dir, x);
      // Get rotation around X so that Y would be in the plane
      // spanned by X and dir. (If we have a second symmetry axis
      // then this rotates axis2 into that plane because we have
      // previously aligned axis2 with Y.)
      float cross2[3];
      cross_prod(cross2, x, y);
      theta = signed_angle(x, cross2, cross);
    } else if (dot_prod(x, dir)<0.f) {
      // dir and x are antiparallel
      phi = 180.f;
    }
    //printf("dir[%d] = {%.2f %.2f %.2f},  phi=%.2f, theta=%.2f\n", i, dir[0], dir[1], dir[2], phi, theta);

    
    // Apply rotations around dir
    int j;
    for (j=0; j<numrot; j++) {
      Matrix4 m;
      float psi = float(j)/float(numrot)*360.f/float(probe_axisorder1);

      // Rotate around dir
      m.rotate_axis(dir, float(DEGTORAD(psi)));

      // Rotate around X
      m.rot(theta, 'x');
      
      // Tilt X into dir
      m.rotate_axis(z, float(DEGTORAD(phi)));

      // Apply rotation to all probe atoms
      int k;
      for (k=0; k<num_probe_atoms; k++) {
        m.multpoint3d(&probe_coords[3*k],
                      &conform[3*num_probe_atoms*conf + 3*k]);
      }

      conf++;
    }
  }

  delete [] orientations;

  // Return the number of generated conformers
  return numconf;
}


// Perform simple symmetry check on the probe molecule.
// Checks if the molecule is linear molecules and if it has an
// additional horizontal symmetry plane (Cinfv vs. Dinfh pointgroup)
// Also recognizes sp3 centers with 4 identical ligands like
// methane (Td pointgroup).
void VolMapCreateILS::check_probe_symmetry() {
  float principal[3];
  if (is_probe_linear(principal)) {
    probe_axisorder1 = -1;
    vec_copy(probe_symmaxis1, principal);

    // Check if there is an additional C2 axis, i.e.
    // is there an identical image for each atom.
    // This means the pointgroup would be Dinfv, 
    // otherwise we just have Cinfv.
    int Dinfv = 1;
    int i, j;
    for (i=0; i<num_probe_atoms; i++) {
      float image[3];
      vec_negate(image, &probe_coords[3*i]);
      int match = 1;
      for (j=i+1; j<num_probe_atoms; j++) {
        if (distance(&probe_coords[3*j], image)>0.05) continue;
        if (probe_vdw[2*i  ]!=probe_vdw[2*j  ] ||
            probe_vdw[2*i+1]!=probe_vdw[2*j+1] ||
            probe_charge[i]!=probe_charge[j]) {
          match = 0;
          break;
        }
      }
      if (!match) {
        Dinfv = 0;
        break;
      }
    }

    if (Dinfv) {
      // Construct perpendicular C2 symmetry axis
      float v[3]; // helper vector used to construct orthogonal
      vec_zero(v); v[0] = 1.f;
      if (fabs(dot_prod(probe_symmaxis1, v))>0.95) {
        // Almost parallel, choose a different vector
        v[0] = 0.f; v[1] = 1.f;
      }
      float cross[3];
      cross_prod(cross, probe_symmaxis1, v);
      cross_prod(probe_symmaxis2, cross, probe_symmaxis1);
      probe_axisorder2 = 2;
    }
  }
  else if (num_probe_atoms==5) {
    // Try a very simple check for tetrahedral symmetry:
    // It will recognize molecules with a central atom and
    // 4 equivalent atoms in the corners of a tetrahedron
    // such as methane.

    // Look for central atom
    int i, icenter = -1;
    float zero[3];
    vec_zero(zero);
    for (i=0; i<num_probe_atoms; i++) {
      if (distance(&probe_coords[3*i], zero)<0.05) {
        icenter = i;
        break;
      }
    }

    if (icenter>=0) {
      float corner[12];
      float vdweps=0.f, vdwrmin=0.f, charge=0.f, dist=0.f;
      int match = 1;
      int j = 0;

      // Check if all ligand atom have the same type and
      // build a coordinate list.
      for (i=0; i<num_probe_atoms; i++) {
        if (i==icenter) continue;
        if (j==0) {
          vdweps  = probe_vdw[2*i  ];
          vdwrmin = probe_vdw[2*i+1];
          charge  = probe_charge[i];
          dist = norm(&probe_coords[3*i]);
        }
        else if (probe_vdw[2*i  ] != vdweps  ||
                 probe_vdw[2*i+1] != vdwrmin ||
                 probe_charge[i]  != charge   ||
                 norm(&probe_coords[3*i])-dist > 0.05) {
          match = 0;
          break;
        }

        vec_copy(&corner[3*j], &probe_coords[3*i]);
        j++;
      }

      // Check the tetrahedral angles
      if (match &&
          angle(&corner[0], &corner[3])-109.47122f < 5.f &&
          angle(&corner[0], &corner[6])-109.47122f < 5.f &&
          angle(&corner[0], &corner[9])-109.47122f < 5.f &&
          angle(&corner[3], &corner[6])-109.47122f < 5.f &&
          angle(&corner[3], &corner[9])-109.47122f < 5.f &&
          angle(&corner[6], &corner[9])-109.47122f < 5.f) {
        probe_tetrahedralsymm = 1;
        probe_axisorder1 = 3;
        probe_axisorder2 = 3;
        vec_copy(probe_symmaxis1, &corner[0]);
        vec_copy(probe_symmaxis2, &corner[3]);
      }
    }
  }
}


// Generate probe conformations with uniformly distributed
// orientations and rotations around the orientation vector
// and store the resulting probe coordinates in *conformers.
void VolMapCreateILS::initialize_probe() {
  // Perform simple check on probe symmetry and determine
  // symmetry axes and order.
  check_probe_symmetry();

  // We can make use of up to two symmetry axes in two
  // independent operations: The orientation of the probe's
  // principal axis and the rotation around it.
  // If only one axis was found or specified we will use it to
  // exploit symmetry during rotation of the probe around the
  // orientation vectors.
  // In case we have an additional (orthogonal) 2-fold rotary
  // axis we can omit half of the orientations. The idea is that
  // if a 180 deg rotation turns the molecule into an identical
  // image then we don't have to generate the conformer corresponding
  // to the orientation vector pointing in the opposite direction.
  // Actually this applies only to linear symmetric molecules like
  // oxygen, but since for each orientation the molecule will also be
  // rotated around the orientation vector we can extend the concept
  // to cases where such an additional rotation turns the flipped
  // molecule into the identical image. Strictly, this rotation
  // should correspond to one of the generated rotamers but because
  // the phase for generating the rotamers was chosen arbitrarily
  // anyway we don't need this correspondence.
  //
  // probe_axisorder1: for probe rotation
  // probe_axisorder2: for probe orientation
  if (probe_axisorder1==1 || 
      (probe_axisorder2!=1 && probe_axisorder2!=2)) {
    // Swap the axes
    int tmpord = probe_axisorder1;
    probe_axisorder1 = probe_axisorder2;
    probe_axisorder2 = tmpord;

    float tmp[3];
    vec_copy(tmp, probe_symmaxis1);
    vec_copy(probe_symmaxis1, probe_symmaxis2);
    vec_copy(probe_symmaxis2, tmp);
  }


  // Rotate the probe so that symmetry axis 1 is along X
  Matrix4 rot;
  rot.transvecinv(probe_symmaxis1[0], probe_symmaxis1[1], probe_symmaxis1[2]);
  int i;
  for (i=0; i<num_probe_atoms; i++) {
    rot.multpoint3d(&probe_coords[3*i], &probe_coords[3*i]);
  }
  rot.multpoint3d(probe_symmaxis1, probe_symmaxis1);
  rot.multpoint3d(probe_symmaxis2, probe_symmaxis2);

  // Rotate the probe so that symmetry axis 2 is along Y
  if (probe_axisorder2>1) {
    float cross[3], z[3];
    vec_zero(z); z[2] = 1.f;
    cross_prod(cross, probe_symmaxis1, probe_symmaxis2);

    float phi = angle(cross, z);
    rot.identity();
    rot.rotate_axis(probe_symmaxis1, float(DEGTORAD(phi)));

    for (i=0; i<num_probe_atoms; i++) {
      rot.multpoint3d(&probe_coords[3*i], &probe_coords[3*i]);
    }
    rot.multpoint3d(probe_symmaxis1, probe_symmaxis1);
    rot.multpoint3d(probe_symmaxis2, probe_symmaxis2);
  }


  if (getenv("VMDILSNOSYMM")) {
    // Omit any symmetry in the probe conformer generation
    // (useful for benchmarking).
    probe_axisorder1 = 1;
    probe_axisorder2 = 1;
    probe_tetrahedralsymm = 0;
    msgWarn << "env(VMDILSNOSYMM) is set: Ignoring probe symmetry!" << sendmsg;
  }

  num_orientations = 0;  // # symmetry unique orientations
  num_rotations    = 1;  // # rotamers per orientation

  if (probe_tetrahedralsymm) {
    msgInfo << "Probe symmetry: tetrahedral" << sendmsg;

    num_conformers = gen_conf_tetrahedral(conformers,
                         conformer_freq, num_orientations,
                         num_rotations);
  }

  else {
    // Probe is not tetrahedral, generate geodesic orientations
    // based on platonic solids.

    if (probe_axisorder1<=1 && probe_axisorder1<=1) {
      msgInfo << "Probe symmetry: none" << sendmsg;
    }

    else if (probe_axisorder1==-1) {
      if (probe_axisorder2==2) {
        msgInfo << "Probe symmetry: Dinfh (linear, C2)" << sendmsg;
      } else {
        msgInfo << "Probe symmetry: Cinfv (linear)" << sendmsg;
      }
      msgInfo << "  Probe is linear, generating probe orientations only." << sendmsg;
    }
    else if (probe_axisorder1>1) {
      if (probe_axisorder2==2) {
        msgInfo << "Probe symmetry: C" << probe_axisorder1
                << ", C2" << sendmsg;
      } else {
        msgInfo << "Probe symmetry: C" << probe_axisorder1 << sendmsg;
      }
      msgInfo << "  Exploiting C" << probe_axisorder1
              << " rotary symmetry for rotation of the oriented probe." << sendmsg;
    }

    if (probe_axisorder2==2) {
      msgInfo << "  Exploiting C2 rotary symmetry for probe orientations" 
              << sendmsg;
    }

    // dimple_depth(180.f); // single orientation
    // dimple_depth(90.f); // hexahedron
    // dimple_depth(180.f-109.47122f); // octahedron
    // dimple_depth(180.f-116.56557f); // dodecahedron
    // dimple_depth(180.f-138.18966f); // icosahedron
    // dimple_depth(25.f); // icosahedron faces+vertices

    num_conformers = gen_conf(conformers, conformer_freq,
                              num_orientations, num_rotations);
  }

  msgInfo << "Probe orientations:       " << num_orientations
          << sendmsg;
  msgInfo << "Rotamers per orientation: " << num_rotations
          << sendmsg;
  msgInfo << "Conformers generated:     " << num_conformers
          << sendmsg << sendmsg;
}


// Check if the box given by the minmax coordinates is located
// entirely inside the PBC unit cell of the given frame and in
// this case return 1, otherwise return 0. 
int VolMapCreateILS::box_inside_pbccell(int frame, float *minmaxcoor) {
  Matrix4 mat;

  // Get the PBC --> orthonormal unitcell transformation
  measure_pbc2onc(app->moleculeList, molid,
                  frame, pbccenter, mat);

  // Get vectors describing the box edges
  float box[9];
  memset(box, 0, 9*sizeof(float));
  box[0] = minmaxcoor[3]-minmaxcoor[0];
  box[4] = minmaxcoor[4]-minmaxcoor[1];
  box[8] = minmaxcoor[5]-minmaxcoor[2];
  // printf("box = {%g %g %g}\n", box[0], box[1], box[2]);
  // printf("box = {%g %g %g}\n", box[3], box[4], box[5]);
  // printf("box = {%g %g %g}\n", box[6], box[7], box[8]);

  // Create coordinates for the 8 corners of the box
  // and transform them into the system of the orthonormal
  // PBC unit cell.
  float node[8*3];
  memset(node, 0, 8*3*sizeof(float));
  int n=0;
  int i, j, k;
  for (i=0; i<=1; i++) {
    for (j=0; j<=1; j++) {
      for (k=0; k<=1; k++) {
        vec_copy(node+3*n, &minmaxcoor[0]);
        vec_scaled_add(node+3*n, float(i), &box[0]);
        vec_scaled_add(node+3*n, float(j), &box[3]);
        vec_scaled_add(node+3*n, float(k), &box[6]);
        // Apply the PBC --> orthonormal unitcell transformation
        // to the current test point.
        mat.multpoint3d(node+3*n, node+3*n);
        n++;
      }
    }
  }

  // Check if corners lie inside the orthonormal unitcell
  for (n=0; n<8; n++) {
    //printf("node[%i] = {%g %g %g}\n", n, node[3*n], node[3*n+1], node[3*n+2]);
    if (node[3*n  ]<0.f) return 0;
    if (node[3*n+1]<0.f) return 0;
    if (node[3*n+2]<0.f) return 0;
    if (node[3*n  ]>1.f) return 0;
    if (node[3*n+1]>1.f) return 0;
    if (node[3*n+2]>1.f) return 0;
  }
  return 1;
}


// Check if the entire volmap grid is located entirely inside
// the PBC unit cell of the given frame (taking the alignment
// into account) and in this case return 1, otherwise return 0.
// Also sets the the gridpoints of volmap that are outside the
// PBC cell to zero (used in the maskonly mode).
int VolMapCreateILS::grid_inside_pbccell(int frame, float *maskvoldata, 
                                         const Matrix4 &alignment) {
  Matrix4 AA, BB, CC;

  Molecule *mol = app->moleculeList->mol_from_id(molid);
  mol->get_frame(frame)->get_transforms(AA, BB, CC);

  // Construct the cell spanning vectors
  float cella[3], cellb[3], cellc[3];
  cella[0] = AA.mat[12];
  cella[1] = AA.mat[13];
  cella[2] = AA.mat[14];
  cellb[0] = BB.mat[12];
  cellb[1] = BB.mat[13];
  cellb[2] = BB.mat[14];
  cellc[0] = CC.mat[12];
  cellc[1] = CC.mat[13];
  cellc[2] = CC.mat[14];
  // Construct the normals of the 6 cell boundary planes
  float normals[3*6];
  cross_prod(&normals[0], cella, cellb);
  cross_prod(&normals[3], cella, cellc);
  cross_prod(&normals[6], cellb, cellc);
  vec_normalize(&normals[0]);
  vec_normalize(&normals[3]);
  vec_normalize(&normals[6]);
  vec_scale(&normals[0], cutoff, &normals[0]);
  vec_scale(&normals[3], cutoff, &normals[3]);
  vec_scale(&normals[6], cutoff, &normals[6]);
  vec_negate(&normals[9],  &normals[0]);
  vec_negate(&normals[12], &normals[3]);
  vec_negate(&normals[15], &normals[6]);

  Matrix4 pbc2onc;
  int allinside = 1;

  // Get the PBC --> orthonormal unitcell transformation
  measure_pbc2onc(app->moleculeList, molid, frame, pbccenter, pbc2onc);

  // In order to transform a point P into the orthonormal cell (P') it 
  // first has to be unaligned (the inverse of the alignment):
  // P' = M_norm * (alignment^-1) * P
  Matrix4 alignmentinv(alignment);
  alignmentinv.inverse();

  Matrix4 coretransform(pbc2onc);
  coretransform.multmatrix(alignmentinv);

  float testpos[3], gridpos[3], extgridpos[3];

  int n;
  for (n=0; n<nsampx*nsampy*nsampz; n++) {
    // Position of grid cell's center
    gridpos[0] = float((n%nsampx)         *delta + gridorigin[0]);
    gridpos[1] = float(((n/nsampx)%nsampy)*delta + gridorigin[1]);
    gridpos[2] = float((n/(nsampx*nsampy))*delta + gridorigin[2]); 

    // Construct 6 test points that are at cutoff distance along
    // the 6 cell normal vectors. The closest system boundary
    // must lie along one of these normals. If all 6 points are
    // within the PBC cell then all possible interaction partner
    // will be within the cell, too.
    int i;
    for (i=0; i<6; i++) {
      vec_add(extgridpos, gridpos, &normals[3*i]);
      // Project into an orthonormal system that is convenient
      // for testing if a point is outside the cell:
      coretransform.multpoint3d(extgridpos, testpos);
      if (testpos[0]<0.f || testpos[0]>1.f ||
          testpos[1]<0.f || testpos[1]>1.f ||
          testpos[2]<0.f || testpos[2]>1.f) {
        // The test point is outside the PBC cell
        maskvoldata[n] = 0.f;
        allinside = 0;
        i = 6;
      }
    }
  }

  return allinside;
}


/// Create list of unique VDW parameters which can be accessed
/// through an index list. This is cache-friendly because the per-atom
/// indexlist and the 2 (short) parameter arrays comsume less memory
/// then twoper-atom parameter arrays.
/// This is currently based on all atoms in the system.
/// For calculations based on a small subsystem we might carry around
/// a few types that aren't actually used in the subsystem but for these
/// few it is probably not worth regenerating the list for each frame.
int VolMapCreateILS::create_unique_paramlist() {
  Molecule *mol = app->moleculeList->mol_from_id(molid);

  // typecast pointers to "flint" so that we can do int compare below
  const flint *radius = (flint *) mol->extraflt.data("radius");
  if (!radius) return MEASURE_ERR_NORADII;
  const flint *occupancy = (flint *) mol->extraflt.data("occupancy");
  if (!occupancy) return MEASURE_ERR_NORADII;

  int i, j;

#define MAX_UNIQUE_TYPES  200
  // Any sane data set should have no more than about 25 unique types.
  // We guard against malformed data by setting an upper bound on the
  // number of types, preventing our O(N) search to devolve into O(N^2).

  atomtypes = new int[mol->nAtoms];  // each atom stores its type index
  atomtypes[0] = 0;  // first atom is automatically assigned first type
  flint *unique_occ = new flint[MAX_UNIQUE_TYPES];
  flint *unique_rad = new flint[MAX_UNIQUE_TYPES];
  unique_occ[0].f = occupancy[0].f;
  unique_rad[0].f = radius[0].f;
  num_unique_types = 1;

  for (i=1; i<mol->nAtoms; i++) {
    int found = 0;
    // Compare VDW params of current atom against all
    // existing unique VDW types.
    for (j=0; j<num_unique_types; j++) {
      // perform == test on ints because it's safer
      if (occupancy[i].i==unique_occ[j].i && radius[i].i==unique_rad[j].i) {
        found = 1;
        break;
      }
    }
    if (!found) {
      if (MAX_UNIQUE_TYPES==num_unique_types) {
        msgErr << "Exceeded maximum number " << MAX_UNIQUE_TYPES
               << " of unique atom parameter types" << sendmsg;
        return -1;
      }
      // No matching VDW type found, create a new one
      unique_occ[j].f = occupancy[i].f;
      unique_rad[j].f = radius[i].f;
      num_unique_types++;
    }
    atomtypes[i] = j;
  }

  vdwparams  = new float[2*num_unique_types];
  for (j=0; j<num_unique_types; j++) {
    // check validity of VDW parameters
    if ( !(unique_occ[j].f <= 0.f && unique_rad[j].f > 0.f) ) {
      msgErr << "Found invalid VDW parameters " << j
        << ": occupancy=" << unique_occ[j].f
        << ", radius=" << unique_rad[j].f
        << sendmsg;
      return -1;
    }
    // The combination rule for VDW eps of 2 atoms is:
    // eps(i,j)  = sqrt(eps(i)) * sqrt(eps(j))
    // so we take the sqrt here already.
    vdwparams[2*j  ] = sqrtf(-unique_occ[j].f);
    vdwparams[2*j+1] = unique_rad[j].f;
  }
  delete [] unique_occ;
  delete [] unique_rad;

  msgInfo << "Number of atom types: " << num_unique_types << sendmsg;

#if 0
  float *epsilon = new float[mol->nAtoms];
  for (i=0; i<mol->nAtoms; i++) {
    // The combination rule for VDW eps of 2 atoms is:
    // eps(i,j)  = sqrt(eps(i)) * sqrt(eps(j))
    // so we take the sqrt here already.
    epsilon[i] = sqrtf(-occupancy[i]);
  }  

  atomtypes = new int[mol->nAtoms];
  atomtypes[0] = 0;
  float *unique_eps = new float[mol->nAtoms];
  float *unique_rad = new float[mol->nAtoms];
  unique_eps[0] = epsilon[0];
  unique_rad[0] = radius[0];
  num_unique_types = 1;

  for (i=1; i<mol->nAtoms; i++) {
    int found = 0;
    // Compare VDW params of current atom against all
    // existing unique VDW types.
    for (j=0; j<num_unique_types; j++) {
      if (epsilon[i]==unique_eps[j] && radius[i]==unique_rad[j]) {
        found = 1;
        break;
      }
    }
    if (!found) {
      // No matching VDW type found, create a new one
      unique_eps[j] = epsilon[i];
      unique_rad[j] = radius[i];
      num_unique_types++;
    }
    atomtypes[i] = j;
  }

  vdwparams  = new float[2*num_unique_types];
  for (j=0; j<num_unique_types; j++) {
    vdwparams[2*j  ] = unique_eps[j];
    vdwparams[2*j+1] = unique_rad[j];
    //printf("eps=%f, rmin=%f\n", unique_eps[j], unique_rad[j]);
  }
  delete [] epsilon;
  delete [] unique_eps;
  delete [] unique_rad;

  msgInfo << "Number of atom types: " << num_unique_types << sendmsg;
#endif

  return 0;
}


// Perform ILS calculation for all specified frames.
int VolMapCreateILS::compute() {
  int numframes = app->molecule_numframes(molid);
  if (first<0 || last>=numframes) return -1;

  int err = initialize();
  if (err) return err;
  
  int n, frame;
  int gridsize = nsampx*nsampy*nsampz;
  float *frame_voldata = new float[gridsize]; // individual frame voldata
  float *combo_voldata = new float[gridsize]; // combo cache voldata
  if (maskonly) {
    // Fill mask map with ones
    for (n=0; n<gridsize; n++) {
      combo_voldata[n] =  1.f;
      frame_voldata[n] =  1.f;
    }    
  } else {
    memset(combo_voldata, 0, gridsize*sizeof(float));
  }

  msgInfo << "Processing frames " << first << "-" << last
          << ", " << last-first+1 << " frames in total..." << sendmsg;

  computed_frames = 0;

  wkf_timerhandle timer = wkf_timer_create();
  wkf_timerhandle alltimer = wkf_timer_create();
  wkf_timer_start(alltimer);

  // Combine frame_voldata into combo_voldata, one frame at a time
  for (frame=first; frame<=last; frame++) { 
    msgInfo << "ILS frame " << frame-first+1 << "/" << last-first+1;

#ifdef TIMING
    msgInfo << sendmsg;
#else
    msgInfo << "   ";
#endif

    wkf_timer_start(timer);

    // Perform the actual ILS calculation for this frame
    compute_frame(frame, frame_voldata);

    msgInfo << "Total frame time = " << wkf_timer_timenow(timer) << " s" << sendmsg;

    if (maskonly) {
      for (n=0; n<gridsize; n++) {
        combo_voldata[n] *= frame_voldata[n];
      }
    } else {
      // For each cell combine occupancies of the new frame with the
      // sum of the existing ones (we will divide by the number of
      // frames later to get the average).
      int numexcl = 0;
      for (n=0; n<gridsize; n++) {
        combo_voldata[n] += frame_voldata[n];
        if (frame_voldata[n]<=min_occup) numexcl++;
      }
      //printf("numexcl = %d/%d\n", numexcl, gridsize);
    }

    computed_frames++;
  }

  double allframetime = wkf_timer_timenow(alltimer);

  // Downsampling of the final map
  if (nsubsamp>1||1) {
    int ndownsampx = volmap->xsize;
    int ndownsampy = volmap->ysize;
    int ix, iy, iz;

    if (maskonly) {
      for (iz=0; iz<nsampz; iz++) {
        int isubz = iz/nsubsamp*ndownsampy*ndownsampx;
        for (iy=0; iy<nsampy; iy++) {
          int isuby = iy/nsubsamp*ndownsampx;
          for (ix=0; ix<nsampx; ix++) {
            n = iz*nsampy*nsampx + iy*nsampx + ix;
            float val = combo_voldata[n];
            int m = isubz + isuby + ix/nsubsamp; 
            // If any of the subsamples where zero,
            // the downsampled voxel will be zero:
            volmap->data[m] *= val; 
          }
        }
      }

    } else {
      for (iz=0; iz<nsampz; iz++) {
        int isubz = iz/nsubsamp*ndownsampy*ndownsampx;
        for (iy=0; iy<nsampy; iy++) {
          int isuby = iy/nsubsamp*ndownsampx;
          for (ix=0; ix<nsampx; ix++) {
            n = iz*nsampy*nsampx + iy*nsampx + ix;
            float val = combo_voldata[n];
            int m = isubz + isuby + ix/nsubsamp; 
            //printf("%d: val[%2d,%2d,%2d]=%g -->%d\n", n, ix, iy, iz, val, m);
            volmap->data[m] += val;
          }
        }
      }

      // Finally, we have to divide by the number of frames
      // and by the number of subsamples.
      float nsamppercell = float(nsubsamp*nsubsamp*nsubsamp*computed_frames);
      for (n=0; n<volmap->gridsize(); n++) {
        volmap->data[n] = volmap->data[n]/nsamppercell;
      }
    }

    if (!maskonly) {
      // Our final maps contain the PMF W which is related to the
      // occupancy rho (the probability to find a particle at
      // that point) by
      //
      // W = -ln(rho);  [in units of kT]
      //
      // Additionally we clamp large energies to the user-provided
      // max_energy value.
      //
      for (n=0; n<volmap->gridsize(); n++) {
        float val = volmap->data[n];
        if (val<=min_occup) {
          volmap->data[n] = max_energy;
        } else {
          val = -logf(val);
          if (val > max_energy) val = max_energy;
          volmap->data[n] = val;
        }
      }
    }
  }

  delete[] frame_voldata;
  delete[] combo_voldata;

  msgInfo << "#################################################"
          << sendmsg << sendmsg;
  msgInfo << "Total time for all frames = "
          << allframetime << " s" << sendmsg;
  msgInfo << "Avg time per frame        = " 
          << allframetime/(last-first+1) << " s" << sendmsg;
  msgInfo << "Downsampling              = "
          << wkf_timer_timenow(alltimer)-allframetime << " s"
          << sendmsg << sendmsg;
  msgInfo << "#################################################"
          << sendmsg << sendmsg;

  wkf_timer_destroy(timer);
  wkf_timer_destroy(alltimer);

  return 0;
}



// Align current frame to the reference
void VolMapCreateILS::align_frame(Molecule *mol, int frame, float *coords,
                                  Matrix4 &alignment) {
  // In case alignsel is not NULL we align the current frame to the
  // first frame according to the provided selection.
  if (alignsel) {
    int i;
    int save_frame_alignsel = alignsel->which_frame;
    
    alignsel->which_frame = frame;
    alignsel->change(NULL, mol);

    float *weight = new float[alignsel->selected]; 
    for (i=0; i<alignsel->selected; i++) weight[i] = 1.0;
    
    measure_fit(alignsel, alignsel, coords, alignrefpos, weight, NULL, &alignment);
    delete[] weight;
    
    if (!getenv("VMDILSALIGNMAPS")) {
      // Do the alignment
      // (For the neighboring pbc coordinates the alignment is 
      // implicitely done below).
      for (i=0; i < mol->nAtoms; i++) { 
        alignment.multpoint3d(coords, coords);
        coords += 3;
      }
    }
    
    alignsel->which_frame = save_frame_alignsel;
  }

  // Combine frame alignment with general transform (global alignment)
  alignment.multmatrix(transform);
}


// Get array of coordinates of selected atoms.
// If the pbc flag was set we also generate coordinates
// for atoms within a cutoff in adjacent pbc image cells.
// The image coordinates can be related to the according atoms
// in the main cell through the indexmap.
int VolMapCreateILS::get_atom_coordinates(int frame, Matrix4 &alignment,
                                          int *(&vdwtypes),
                                          float *(&coords)) {
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);

  // Select all atoms within the extended cutoff of the
  // user specified grid minmax box.
  int *selon = new int[num_atoms];
  memset(selon, 0, num_atoms*sizeof(int));

  float minx = minmax[0]-extcutoff;
  float miny = minmax[1]-extcutoff;
  float minz = minmax[2]-extcutoff;
  float maxx = minmax[3]+extcutoff;
  float maxy = minmax[4]+extcutoff;
  float maxz = minmax[5]+extcutoff;

  int numselected = 0;
  int i;
  for (i=0; i<num_atoms; i++) {
    float x = coords[3*i  ];
    float y = coords[3*i+1];
    float z = coords[3*i+2];
    if (x>=minx && x<=maxx &&
        y>=miny && y<=maxy &&
        z>=minz && z<=maxz) {
      selon[i] = 1;
      numselected++;
    }
  }

  int numcoords = numselected;

  float *selcoords = NULL;

  // If pbc is set the user requests a PBC aware computation and we
  // must generate the extended coordinates, i.e. the positions of
  // the atoms in the neighboring pbc cells.
  if (pbc) {
    // Automatically add the force field cutoff to the system.
    float ffcutoff[3];
    ffcutoff[0] = ffcutoff[1] = ffcutoff[2] = cutoff;

    // Positions of the atoms in the neighboring pbc cells.
    ResizeArray<float> extcoord_array;

    // The indexmap_array contains the index of the according
    // unitcell atom for each extended atom.
    ResizeArray<int>   indexmap_array;

    // Generate coordinates for atoms in the neighboring cells.
    // The indexmap_array tells to which atom the extended coordinate
    // corresponds.
    // We have to use NULL instead of sel (2nd parameter) in order
    // to return all PBC neighbors and not only the ones within
    // ffcutoff of sel. The reason is that we have no atomselection
    // and are computing the interaction between the all system atoms
    // and the probe located at the gridpoints.
    measure_pbc_neighbors(app->moleculeList, NULL, molid, frame,
                          &alignment, pbccenter, ffcutoff, minmax, 
                          &extcoord_array, &indexmap_array);

    numcoords = numselected+indexmap_array.num();

    selcoords = new float[3*numcoords];
    vdwtypes  = new int[numcoords];

    int j = numselected;
    for (i=0; i<indexmap_array.num(); i++) {
      selcoords[3*j  ] = extcoord_array[3*i  ];
      selcoords[3*j+1] = extcoord_array[3*i+1];
      selcoords[3*j+2] = extcoord_array[3*i+2];
      vdwtypes[j] = atomtypes[indexmap_array[i]];
      j++;
    }

    //printf("volmap: considering %d PBC neighbor atoms.\n", indexmap_array.num());
  } else {
    selcoords = new float[3*numcoords];
    vdwtypes  = new int[numcoords];
  }

  // Get the core coordinates (selected atoms in the main PBC cell)
  int j=0;
  for (i=0; i<num_atoms; i++) { 
    if (!selon[i]) continue; //atom is not selected
    selcoords[3*j  ] = coords[3*i  ];
    selcoords[3*j+1] = coords[3*i+1];
    selcoords[3*j+2] = coords[3*i+2];
    vdwtypes[j] = atomtypes[i];
    j++;
  }
  //printf("volmap: considering %d core atoms.\n", j);

  coords = selcoords;

  delete [] selon;

#ifdef TIMING
  msgInfo << "Coord setup: " << wkf_timer_timenow(timer) << " s" << sendmsg;
#endif
  wkf_timer_destroy(timer);

  return numcoords;
}



/////////////////////////////////////////////////////////
//  This is the function driving ILS for each frame    //
/////////////////////////////////////////////////////////

// Computes, for each gridpoint, the VdW energy to the nearest atoms
int VolMapCreateILS::compute_frame(int frame, float *voldata) { 
  Matrix4 alignment;
  float *coords;

  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) return -1;

#ifdef TIMING
  char report[128];
  wkf_timerhandle timer = wkf_timer_create();
#endif

  // Advance to next frame
  coords = mol->get_frame(frame)->pos;

  // In case alignsel is not NULL, align the current frame to the
  // first frame according to the provided selection and get the
  // alignment transformation matrix.
  align_frame(mol, frame, coords, alignment);

  if (maskonly) {
#ifdef TIMING
    wkf_timer_start(timer);
#endif

    // We are only creating a mask map that defines the
    // gridpoints that (after alignment) still overlap in 
    // each frame with the reference grid and thus are not
    // undersampled.
    grid_inside_pbccell(frame, voldata, alignment);

#ifdef TIMING
    msgInfo << "Masking:     " << wkf_timer_timenow(timer) << " s" << sendmsg;
#endif

    return MEASURE_NOERR;
  }


  // Get array of coordinates of selected atoms and their
  // neighbors (within a cutoff) in the PBC images.
  // Memory for *vdwtypes and *coords will be allocated.
  int *vdwtypes = NULL;
  int numcoords;
  float originalign[3];
  float axesalign[9];
  float gridaxes[9];
  memset(gridaxes, 0, 9*sizeof(float));
  gridaxes[0] = gridaxes[4] = gridaxes[8] = 1.f;

  if (getenv("VMDILSALIGNMAPS")) {
    // We use all atoms unaligned, but be align the map instead
    numcoords = num_atoms;
    vdwtypes = atomtypes;
    alignment.multpoint3d(gridorigin, originalign);
    alignment.multpoint3d(&gridaxes[0], &axesalign[0]);
    alignment.multpoint3d(&gridaxes[3], &axesalign[3]);
    alignment.multpoint3d(&gridaxes[6], &axesalign[6]);
    msgInfo << "Aligning maps." << sendmsg;
  } else {
    // Get extended list of aligned atom coordinates
    numcoords = get_atom_coordinates(frame, alignment,
                                     vdwtypes, coords);
    memcpy(originalign, gridorigin, 3*sizeof(float));
    memcpy(axesalign,   gridaxes,   9*sizeof(float));
    msgInfo << "Aligning frames." << sendmsg;
  }
  
  if (getenv("VMDALLATOMILS")) {
#ifdef TIMING
    wkf_timer_start(timer);
#endif
    
    // Assuming the grid is aligned with the coordinate axes:
    float lenx = float(nsampx*delta);
    float leny = float(nsampy*delta);
    float lenz = float(nsampz*delta);
    
    compute_allatoms(voldata, nsampx, nsampy, nsampz,
                     lenx, leny, lenz, originalign, axesalign,
                     alignment.mat, numcoords, coords,
                     vdwtypes, vdwparams, cutoff, probe_vdw, num_probe_atoms,
                     num_conformers, conformers, max_energy); 
    
#ifdef TIMING
    sprintf(report, "compute_allatoms()                                     "
        "%f s\n", wkf_timer_timenow(timer));
    msgInfo << report << sendmsg;
#endif

  } else {

#ifdef TIMING
    wkf_timer_start(timer);
#endif

    // Assuming the grid is aligned with the coordinate axes:
    float lenx = float(nsampx*delta);
    float leny = float(nsampy*delta);
    float lenz = float(nsampz*delta);

    int retval;

    ComputeOccupancyMap om;

    // must be set by caller
    om.map = voldata;
    om.mx = nsampx;
    om.my = nsampy;
    om.mz = nsampz;
    om.lx = lenx;
    om.ly = leny;
    om.lz = lenz;
    om.x0 = originalign[0];
    om.y0 = originalign[1];
    om.z0 = originalign[2];
    memcpy(om.ax, &axesalign[0], 3*sizeof(float));
    memcpy(om.ay, &axesalign[3], 3*sizeof(float));
    memcpy(om.az, &axesalign[6], 3*sizeof(float));
    memcpy(om.alignmat, alignment.mat, 16*sizeof(float));
    om.num_coords = numcoords;
    om.coords = coords;
    om.vdw_type = vdwtypes;
    om.vdw_params = vdwparams;
    om.probe_vdw_params = probe_vdw;
    om.conformers = conformers;
    om.num_probes = num_probe_atoms;
    om.num_conformers = num_conformers;
    om.cutoff = cutoff;
    om.extcutoff = extcutoff;
    om.excl_dist = excl_dist; 
    om.excl_energy = max_energy;

    // single threaded version calculates one largest slab
    om.kstart = 0;
    om.kstop = om.mz;

    retval = ComputeOccupancyMap_setup(&om);

#ifdef TIMING
    sprintf(report, "ComputeOccupancyMap_setup()                            "
        "%f s\n", wkf_timer_timenow(timer));
    msgInfo << report << sendmsg;
#endif

    if (getenv("VMDILSVERBOSE")) { // XXX debugging
      atom_bin_stats(&om
          /*
          gridorigin, lenx, leny, lenz, extcutoff, cutoff,
          om.bincnt, om.nbx, om.nby, om.nbz, om.padx,
          om.num_bin_offsets, om.num_extras */);
    }

    if (retval != 0) {
      if (getenv("VMDILSVERBOSE")) { // XXX debugging
        int i, j, k;
        int total_extra_atoms = 0;
        for (k = 0;  k < om.nbz;  k++) {
          for (j = 0;  j < om.nby;  j++) {
            for (i = 0;  i < om.nbx;  i++) {
              int index = (k*om.nby + j)*om.nbx + i;
              if (om.bincnt[index] > BIN_DEPTH) {
                printf("*** bin[%d,%d,%d] tried to fill with %d atoms\n",
                    i, j, k, om.bincnt[index]);
                total_extra_atoms += om.bincnt[index] - BIN_DEPTH;
              }
            }
          }
        }
        // XXX should have total_extra_atoms > num_extra_atoms
        printf("*** can't handle total of %d extra atoms\n", total_extra_atoms);
      }
      ComputeOccupancyMap_cleanup(&om);
      return -1;
    }

#if defined(VMDCUDA)
    if (getenv("VMDILSVERBOSE")) {
      printf("*** cpu only = %d\n", om.cpu_only);
    }
    if (!getenv("VMDNOCUDA") && !(om.cpu_only) &&
        (retval=
         vmd_cuda_evaluate_occupancy_map(om.mx, om.my, om.mz, om.map,
           om.excl_energy, om.cutoff, om.hx, om.hy, om.hz,
           om.x0, om.y0, om.z0, om.bx_1, om.by_1, om.bz_1, 
           om.nbx, om.nby, om.nbz, (float *) om.bin, (float *) om.bin_zero,
           om.num_bin_offsets, om.bin_offsets,
           om.num_extras, (float *) om.extra,
           num_unique_types, om.vdw_params,
           om.num_probes, om.probe_vdw_params,
           om.num_conformers, om.conformers)) == 0) {
      // successfully ran ILS with CUDA, otherwise fall back on CPU
    } else {
      if (retval != 0) {
        msgInfo << "vmd_cuda_evaluate_occupancy_map() FAILED,"
          " using CPU for calculation\n" << sendmsg;
      }
#endif /* CUDA... */

#ifdef TIMING
      wkf_timer_start(timer);
#endif

      retval = ComputeOccupancyMap_calculate_slab(&om);

#ifdef TIMING
      sprintf(report, "ComputeOccupancyMap_calculate_slab()                   "
              "%f s\n", wkf_timer_timenow(timer));
      msgInfo << report << sendmsg;
#endif

      if (retval != 0) {
        if (getenv("VMDILSVERBOSE")) { // XXX debugging
          printf("*** ComputeOccupancyMap_calculate_slab() failed\n");
        }
        ComputeOccupancyMap_cleanup(&om);
        return -1;
      }

#if defined(VMDCUDA)
    } // end else not VMDCUDAILS
#endif

    ComputeOccupancyMap_cleanup(&om);
  }

  if (!getenv("VMDILSALIGNMAPS")) {
    delete[] coords;
    delete[] vdwtypes;
  }

#ifdef TIMING
  wkf_timer_destroy(timer);
#endif
      
  return MEASURE_NOERR; 
}


/////////////////////////////////////////////////////////
// Here follows the new implementation.                //
/////////////////////////////////////////////////////////

static int fill_atom_bins(ComputeOccupancyMap *p);
static void tighten_bin_neighborhood(ComputeOccupancyMap *p);
static void find_distance_exclusions(ComputeOccupancyMap *p);
static void find_energy_exclusions(ComputeOccupancyMap *p);
static void compute_occupancy_monoatom(ComputeOccupancyMap *p);
static void compute_occupancy_multiatom(ComputeOccupancyMap *p);


int ComputeOccupancyMap_setup(ComputeOccupancyMap *p) {

  // initialize pointer fields to zero
  p->bin = NULL;
  p->bin_zero = NULL;
  p->bincnt = NULL;
  p->bincnt_zero = NULL;
  p->bin_offsets = NULL;
  p->extra = NULL;
  p->exclusions = NULL;

  // initialize occupancy map, allocate and initialize exclusion map
  int mtotal = p->mx * p->my * p->mz;
  memset(p->map, 0, mtotal * sizeof(float));  // zero occupancy by default
  p->exclusions = new char[mtotal];
  memset(p->exclusions, 0, mtotal * sizeof(char));  // no exclusions yet

  // derive map spacing based on length and number of points
  p->hx = p->lx / p->mx;
  p->hy = p->ly / p->my;
  p->hz = p->lz / p->mz;

  p->cpu_only = 0;  // attempt to use CUDA

  // set expected map points per bin length
  // note: we want CUDA thread blocks to calculate 4^3 map points
  //       we want each thread block contained inside a single bin
  //       we want bin volume to be no more than MAX_BIN_VOLUME (27 A^3)
  //       and no smaller than MIN_BIN_VOLUME (8 A^3) due to fixed bin depth
  //       we expect map spacing to be about 0.25 A but depends on caller

  // start with trying to pack 3^3 thread blocks per atom bin
  p->mpblx = 12;
  p->mpbly = 12;
  p->mpblz = 12;

  // starting bin lengths
  p->bx = p->mpblx * p->hx;
  p->by = p->mpbly * p->hy;
  p->bz = p->mpblz * p->hz;

  // refine bin side lengths if volume of bin is too large
  while (p->bx * p->by * p->bz > MAX_BIN_VOLUME) {

    // find longest bin side and reduce its length
    if (p->bx > p->by && p->bx > p->bz) {
      p->mpblx -= 4;
      p->bx = p->mpblx * p->hx;
    }
    else if (p->by >= p->bx && p->by > p->bz) {
      p->mpbly -= 4;
      p->by = p->mpbly * p->hy;
    }
    else {
      p->mpblz -= 4;
      p->bz = p->mpblz * p->hz;
    }

  } // end refinement of bins

  if (p->bx * p->by * p->bz < MIN_BIN_VOLUME) {
    // refinement failed due to some hx, hy, hz being too large
    // now there is no known correspondence between map points and bins
    p->bx = p->by = p->bz = DEFAULT_BIN_LENGTH;
    p->mpblx = p->mpbly = p->mpblz = 0;
    p->cpu_only = 1;  // CUDA can't be used, map too coarse
  }

  p->bx_1 = 1.f / p->bx;
  p->by_1 = 1.f / p->by;
  p->bz_1 = 1.f / p->bz;

  if (fill_atom_bins(p)) {
    return -1;  // failed due to too many extra atoms for bin size
  }

  tighten_bin_neighborhood(p);

  return 0;
} // ComputeOccupancyMap_setup()


int ComputeOccupancyMap_calculate_slab(ComputeOccupancyMap *p) {
#ifdef TIMING
  char report[128];
  wkf_timerhandle timer = wkf_timer_create();
#endif

  // each of these routines operates on the slab
  // designated by kstart through kstop (z-axis indices)
  //
  // XXX we are planning CUDA kernels for each of the following routines

#if 1
#ifdef TIMING
  wkf_timer_start(timer);
#endif
  find_distance_exclusions(p);
  int i, numexcl=0;
  for (i=0; i<p->mx * p->my * p->mz; i++) {
    if (p->exclusions[i]) numexcl++;
  }
#ifdef TIMING
  sprintf(report, "ComputeOccupancyMap: find_distance_exclusions()        "
      "%f s\n", wkf_timer_timenow(timer));
  msgInfo << report << sendmsg;
#endif
#endif

  if (1 == p->num_probes) {
#ifdef TIMING
    wkf_timer_start(timer);
#endif
    compute_occupancy_monoatom(p);
#ifdef TIMING
    sprintf(report, "ComputeOccupancyMap: compute_occupancy_monoatom()      "
        "%f s\n", wkf_timer_timenow(timer));
    msgInfo << report << sendmsg;
#endif

  }
  else {

#if 1
#ifdef TIMING
    wkf_timer_start(timer);
#endif
    find_energy_exclusions(p);
    int i, numexcl=0;
    for (i=0; i<p->mx * p->my * p->mz; i++) {
      if (p->exclusions[i]) numexcl++;
    }
#ifdef TIMING
    sprintf(report, "ComputeOccupancyMap: find_energy_exclusions()          "
        "%f s  -> %d exclusions\n", wkf_timer_timenow(timer), numexcl);
    msgInfo << report << sendmsg;
#endif
#endif

#ifdef TIMING
    wkf_timer_start(timer);
#endif
    compute_occupancy_multiatom(p);
#ifdef TIMING
    sprintf(report, "ComputeOccupancyMap: compute_occupancy_multiatom()     "
        "%f s\n", wkf_timer_timenow(timer));
    msgInfo << report << sendmsg;
#endif

  }

#ifdef TIMING
  wkf_timer_destroy(timer);
#endif

  return 0;
} // ComputeOccupancyMap_calculate_slab()


void ComputeOccupancyMap_cleanup(ComputeOccupancyMap *p) {
  delete[] p->bin_offsets;
  delete[] p->extra;
  delete[] p->bincnt;
  delete[] p->bin;
  delete[] p->exclusions;
} // ComputeOccupancyMap_cleanup()


// XXX the CUDA kernels can handle up to 50 extra atoms, set this as
// the bound on "max_extra_atoms" rather than the heuristic below
#define MAX_EXTRA_ATOMS  50

int fill_atom_bins(ComputeOccupancyMap *p) {
  int too_many_extra_atoms = 0;  // be optimistic
  int max_extra_atoms = MAX_EXTRA_ATOMS;
  //int max_extra_atoms = (int) ceilf(p->num_coords / 10000.f);
      // assume no more than 1 over full bin per 10000 atoms
  int count_extras = 0;
  int n, i, j, k;

  const int *vdw_type = p->vdw_type;
  const float *coords = p->coords;
  const int num_coords = p->num_coords;
  const float lx = p->lx;
  const float ly = p->ly;
  const float lz = p->lz;
  const float bx_1 = p->bx_1;
  const float by_1 = p->by_1;
  const float bz_1 = p->bz_1;
  const float x0 = p->x0;
  const float y0 = p->y0;
  const float z0 = p->z0;
  const float extcutoff = p->extcutoff;

  // padding is based on extended cutoff distance
  p->padx = (int) ceilf(extcutoff * bx_1);
  p->pady = (int) ceilf(extcutoff * by_1);
  p->padz = (int) ceilf(extcutoff * bz_1);

  const int nbx = p->nbx = (int) ceilf(lx * bx_1) + 2*p->padx;
  const int nby = p->nby = (int) ceilf(ly * by_1) + 2*p->pady;
  p->nbz = (int) ceilf(lz * bz_1) + 2*p->padz;

  int nbins = nbx * nby * p->nbz;

  BinOfAtoms *bin = p->bin = new BinOfAtoms[nbins];
  char *bincnt = p->bincnt = new char[nbins];
  AtomPosType *extra = p->extra = new AtomPosType[max_extra_atoms];

  memset(bin, 0, nbins * sizeof(BinOfAtoms));
  memset(bincnt, 0, nbins * sizeof(char));

  // shift array pointer to the (0,0,0)-bin, which will correspond to
  // the map origin
  BinOfAtoms *bin_zero
    = p->bin_zero = bin + ((p->padz*nby + p->pady)*nbx + p->padx);
  char *bincnt_zero
    = p->bincnt_zero = bincnt + ((p->padz*nby + p->pady)*nbx + p->padx);

  for (n = 0;  n < num_coords;  n++) {
    float x = coords[3*n    ];  // atom coordinates
    float y = coords[3*n + 1];
    float z = coords[3*n + 2];

    float sx = x - x0;  // translate relative to map origin
    float sy = y - y0;
    float sz = z - z0;

    if (sx < -extcutoff || sx > lx + extcutoff ||
        sy < -extcutoff || sy > ly + extcutoff ||
        sz < -extcutoff || sz > lz + extcutoff) {
      continue;  // atom is beyond influence of lattice
    }

    i = (int) floorf(sx * bx_1);  // bin number
    j = (int) floorf(sy * by_1);
    k = (int) floorf(sz * bz_1);

    /*
    // XXX this test should never be true after passing previous test
    if (i < -p->padx || i >= p->nbx + p->padx ||
        j < -p->pady || j >= p->nby + p->pady ||
        k < -p->padz || k >= p->nbz + p->padz) {
      continue;  // atom is outside bin array
    }
    */

    int index = (k*nby + j)*nbx + i;  // flat index into bin array
    int slot = bincnt_zero[index];  // slot within bin to place atom

    if (slot < BIN_DEPTH) {
      AtomPosType *atom = bin_zero[index].atom;  // place atom in next slot
      atom[slot].x = x;
      atom[slot].y = y;
      atom[slot].z = z;
      atom[slot].vdwtype = vdw_type[n];
    }
    else if (count_extras < max_extra_atoms) {
      extra[count_extras].x = x;
      extra[count_extras].y = y;
      extra[count_extras].z = z;
      extra[count_extras].vdwtype = vdw_type[n];
      count_extras++;
    }
    else {
      // XXX debugging
      printf("*** too many extras, atom index %d\n", n);
      too_many_extra_atoms = 1;
    }

    bincnt_zero[index]++;  // increase count of atoms in bin
  }
  p->num_extras = count_extras;

  // mark unused atom slots
  // XXX set vdwtype to -1
  for (n = 0;  n < nbins;  n++) {
    for (k = bincnt[n];  k < BIN_DEPTH;  k++) {
      bin[n].atom[k].vdwtype = -1;
    }
  }

  return (too_many_extra_atoms ? -1 : 0);
} // fill_atom_bins()


// setup tightened bin index offset array of 3-tuples
void tighten_bin_neighborhood(ComputeOccupancyMap *p) {
  const int padx = p->padx;
  const int pady = p->pady;
  const int padz = p->padz;
  const float bx2 = p->bx * p->bx;
  const float by2 = p->by * p->by;
  const float bz2 = p->bz * p->bz;
  const float r = p->extcutoff + sqrtf(bx2 + by2 + bz2);  // add bin diagonal
  const float r2 = r*r;
  int n = 0, i, j, k;
  char *bin_offsets
    = p->bin_offsets = new char[3 * (2*padx+1)*(2*pady+1)*(2*padz+1)];
  for (k = -padz;  k <= padz;  k++) {
    for (j = -pady;  j <= pady;  j++) {
      for (i = -padx;  i <= padx;  i++) {
        if (i*i*bx2 + j*j*by2 + k*k*bz2 >= r2) continue;
        bin_offsets[3*n    ] = (char) i;
        bin_offsets[3*n + 1] = (char) j;
        bin_offsets[3*n + 2] = (char) k;
        n++;
      }
    }
  }
  p->num_bin_offsets = n;
} // tighten_bin_neighborhood()


// For each grid point loop over the close atoms and 
// determine if one of them is closer than excl_dist
// away. If so we assume the clash with the probe will
// result in a very high interaction energy and we can 
// exclude this point from calcultation.
void find_distance_exclusions(ComputeOccupancyMap *p) {
  const AtomPosType *extra = p->extra;
  const BinOfAtoms *bin_zero = p->bin_zero;
  char *excl = p->exclusions;

  int i, j, k, n, index;
  int ic, jc, kc;

  const int mx = p->mx;
  const int my = p->my;
  const int kstart = p->kstart;
  const int kstop = p->kstop;
  const int nbx = p->nbx;
  const int nby = p->nby;
  const float excl_dist = p->excl_dist;
  const float bx_1 = p->bx_1;
  const float by_1 = p->by_1;
  const float bz_1 = p->bz_1;
  const float hx = p->hx;
  const float hy = p->hy;
  const float hz = p->hz;
  const float x0 = p->x0;
  const float y0 = p->y0;
  const float z0 = p->z0;
  const int num_extras = p->num_extras;
  const int bdx = (int) ceilf(excl_dist * bx_1);  // width of nearby bins
  const int bdy = (int) ceilf(excl_dist * by_1);  // width of nearby bins
  const int bdz = (int) ceilf(excl_dist * bz_1);  // width of nearby bins
  const float excldist2 = excl_dist * excl_dist;

  for (k = kstart;  k < kstop;  k++) {  // k index loops over slab
    for (j = 0;  j < my;  j++) {
      for (i = 0;  i < mx;  i++) {  // loop over map points

        float px = i*hx;
        float py = j*hy;
        float pz = k*hz;  // translated coordinates of map point

        int ib = (int) floorf(px * bx_1);
        int jb = (int) floorf(py * by_1);
        int kb = (int) floorf(pz * bz_1);  // zero-based bin index

        px += x0;
        py += y0;
        pz += z0;  // absolute position

        for (kc = kb - bdz;  kc <= kb + bdz;  kc++) {
          for (jc = jb - bdy;  jc <= jb + bdy;  jc++) {
            for (ic = ib - bdx;  ic <= ib + bdx;  ic++) {

              const AtomPosType *atom
                = bin_zero[(kc*nby + jc)*nbx + ic].atom;

              for (n = 0;  n < BIN_DEPTH;  n++) {  // atoms in bin
                if (-1 == atom[n].vdwtype) break;  // finished atoms in bin
                float dx = px - atom[n].x;
                float dy = py - atom[n].y;
                float dz = pz - atom[n].z;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 <= excldist2) {
                  index = (k*my + j)*mx + i;
                  excl[index] = 1;
                  goto NEXT_MAP_POINT;  // don't have to look at more atoms
                }
              } // end loop over atoms in bin

            }
          }
        } // end loop over nearby bins

        for (n = 0;  n < num_extras;  n++) {  // extra atoms
          float dx = px - extra[n].x;
          float dy = py - extra[n].y;
          float dz = pz - extra[n].z;
          float r2 = dx*dx + dy*dy + dz*dz;
          if (r2 <= excldist2) {
            index = (k*my + j)*mx + i;
            excl[index] = 1;
            goto NEXT_MAP_POINT;  // don't have to look at more atoms
          }
        } // end loop over extra atoms

NEXT_MAP_POINT:
        ; // continue loop over lattice points

      }
    }
  } // end loop over lattice points

} // find_distance_exclusions()



// For each grid point sum up the energetic contribution
// of all close atoms. If that interaction energy is above
// the excl_energy cutoff value we don't have to consider
// this grid point in the subsequent calculation.
// Hence, we save the computation of the different probe
// orientation for multiatom probes.
void find_energy_exclusions(ComputeOccupancyMap *p) {
  const char *bin_offsets = p->bin_offsets;
  const float *vdw_params = p->vdw_params;
  const AtomPosType *extra = p->extra;
  const BinOfAtoms *bin_zero = p->bin_zero;
  char *excl = p->exclusions;

  const float probe_vdweps = p->probe_vdw_params[0];   // use first probe param
  const float probe_vdwrmin = p->probe_vdw_params[1];  // for epsilon and rmin

  const int mx = p->mx;
  const int my = p->my;
  const int kstart = p->kstart;
  const int kstop = p->kstop;
  const int nbx = p->nbx;
  const int nby = p->nby;
  const float hx = p->hx;
  const float hy = p->hy;
  const float hz = p->hz;
  const float bx_1 = p->bx_1;
  const float by_1 = p->by_1;
  const float bz_1 = p->bz_1;
  const float x0 = p->x0;
  const float y0 = p->y0;
  const float z0 = p->z0;
  const int num_bin_offsets = p->num_bin_offsets;
  const int num_extras = p->num_extras;
  const float excl_energy = p->excl_energy;

  int i, j, k, n, index;
  const float cutoff2 = p->cutoff * p->cutoff;

  for (k = kstart;  k < kstop;  k++) {  // k index loops over slab
    for (j = 0;  j < my;  j++) {
      for (i = 0;  i < mx;  i++) {  // loop over map points

        int lindex = (k*my + j)*mx + i;  // map index
        if (excl[lindex]) continue;  // already excluded based on distance

        float px = i*hx;
        float py = j*hy;
        float pz = k*hz;  // translated coordinates of map point

        int ib = (int) floorf(px * bx_1);
        int jb = (int) floorf(py * by_1);
        int kb = (int) floorf(pz * bz_1);  // zero-based bin index

        px += x0;
        py += y0;
        pz += z0;  // absolute position

        float u = 0.f;

        for (index = 0;  index < num_bin_offsets;  index++) { // neighborhood
          int ic = ib + (int) bin_offsets[3*index    ];
          int jc = jb + (int) bin_offsets[3*index + 1];
          int kc = kb + (int) bin_offsets[3*index + 2];

          const AtomPosType *atom
            = bin_zero[(kc*nby + jc)*nbx + ic].atom;

          for (n = 0;  n < BIN_DEPTH;  n++) {  // atoms in bin
            if (-1 == atom[n].vdwtype) break;  // finished atoms in bin
            float dx = px - atom[n].x;
            float dy = py - atom[n].y;
            float dz = pz - atom[n].z;
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= cutoff2) continue;
            int pindex = 2 * atom[n].vdwtype;
            float epsilon = vdw_params[pindex] * probe_vdweps;
            float rmin = vdw_params[pindex + 1] + probe_vdwrmin;
            float rm6 = rmin*rmin / r2;
            rm6 = rm6 * rm6 * rm6;
            u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
          } // end loop atoms in bin

        } // end loop bin neighborhood

        for (n = 0;  n < num_extras;  n++) {  // extra atoms
          float dx = px - extra[n].x;
          float dy = py - extra[n].y;
          float dz = pz - extra[n].z;
          float r2 = dx*dx + dy*dy + dz*dz;
          if (r2 >= cutoff2) continue;
          int pindex = 2 * extra[n].vdwtype;
          float epsilon = vdw_params[pindex] * probe_vdweps;
          float rmin = vdw_params[pindex + 1] + probe_vdwrmin;
          float rm6 = rmin*rmin / r2;
          rm6 = rm6 * rm6 * rm6;
          u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
        } // end loop over extra atoms

        if (u >= excl_energy) excl[lindex] = 1;

      }
    }
  } // end loop over lattice

} // find_energy_exclusions()


// For a monoatomic probe compute the occupancy rho
// (probability of finding the probe)
//
// For each map point the occupancy is computed as
//
//   rho = exp(-U)
//
// where U is the interaction energy of the probe with the system
// due to the VDW force field.
//
void compute_occupancy_monoatom(ComputeOccupancyMap *p) {
  const char *bin_offsets = p->bin_offsets;
  const float *vdw_params = p->vdw_params;
  const AtomPosType *extra = p->extra;
  const BinOfAtoms *bin_zero = p->bin_zero;
  const char *excl = p->exclusions;
  float *map = p->map;

  const int mx = p->mx;
  const int my = p->my;
  const int kstart = p->kstart;
  const int kstop = p->kstop;
  const int nbx = p->nbx;
  const int nby = p->nby;
  const float hx = p->hx;
  const float hy = p->hy;
  const float hz = p->hz;
  const float bx_1 = p->bx_1;
  const float by_1 = p->by_1;
  const float bz_1 = p->bz_1;
  const float x0 = p->x0;
  const float y0 = p->y0;
  const float z0 = p->z0;
  const int num_bin_offsets = p->num_bin_offsets;
  const int num_extras = p->num_extras;

  float probe_vdweps = p->probe_vdw_params[0];   // use first probe param
  float probe_vdwrmin = p->probe_vdw_params[1];  //   for epsilon and rmin

  int i, j, k, n, index;
  float max_energy = p->excl_energy;
  float cutoff2 = p->cutoff * p->cutoff;

  for (k = kstart;  k < kstop;  k++) {  // k index loops over slab
    for (j = 0;  j < my;  j++) {
      for (i = 0;  i < mx;  i++) {  // loop over lattice points

        int lindex = (k*my + j)*mx + i;  // map index
#if 1
        if (excl[lindex]) {  // is map point excluded?
          map[lindex] = 0.f;  // clamp occupancy to zero
          continue;
        }
#endif

        float px = i*hx;
        float py = j*hy;
        float pz = k*hz;  // translated coordinates of lattice point

        int ib = (int) floorf(px * bx_1);
        int jb = (int) floorf(py * by_1);
        int kb = (int) floorf(pz * bz_1);  // zero-based bin index

        px += x0;
        py += y0;
        pz += z0;  // absolute position

        float u = 0.f;

        for (index = 0;  index < num_bin_offsets;  index++) { // neighborhood
          int ic = ib + (int) bin_offsets[3*index    ];
          int jc = jb + (int) bin_offsets[3*index + 1];
          int kc = kb + (int) bin_offsets[3*index + 2];

          const AtomPosType *atom
            = bin_zero[(kc*nby + jc)*nbx + ic].atom;

          for (n = 0;  n < BIN_DEPTH;  n++) {  // atoms in bin
            if (-1 == atom[n].vdwtype) break;  // finished atoms in bin
            float dx = px - atom[n].x;
            float dy = py - atom[n].y;
            float dz = pz - atom[n].z;
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= cutoff2) continue;
            int pindex = 2 * atom[n].vdwtype;
            float epsilon = vdw_params[pindex] * probe_vdweps;
            float rmin = vdw_params[pindex + 1] + probe_vdwrmin;
            float rm6 = rmin*rmin / r2;
            rm6 = rm6 * rm6 * rm6;
            u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
          } // end loop atoms in bin

        } // end loop bin neighborhood

        for (n = 0;  n < num_extras;  n++) {  // extra atoms
          float dx = px - extra[n].x;
          float dy = py - extra[n].y;
          float dz = pz - extra[n].z;
          float r2 = dx*dx + dy*dy + dz*dz;
          if (r2 >= cutoff2) continue;
          int pindex = 2 * extra[n].vdwtype;
          float epsilon = vdw_params[pindex] * probe_vdweps;
          float rmin = vdw_params[pindex + 1] + probe_vdwrmin;
          float rm6 = rmin*rmin / r2;
          rm6 = rm6 * rm6 * rm6;
          u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
        } // end loop over extra atoms

        float occ = 0.f;
        if (u < max_energy) {
          occ = expf(-u);
        }
        map[lindex] = occ;  // the occupancy

      }
    }
  } // end loop over lattice

} // compute_occupancy_monoatom()


// For a multiatom probe compute the occupancy rho
// (probability of finding the probe)
//
// Calculate occupancy rho at each map point,
// where rho = (1/m) sum_m ( -exp(u[i]) ) over m conformers,
// u[i] is the potential energy of the i-th conformer.
//
void compute_occupancy_multiatom(ComputeOccupancyMap *p) {
  const char *bin_offsets = p->bin_offsets;
  const float *vdw_params = p->vdw_params;
  const float *probe_vdw_params = p->probe_vdw_params;
  const float *conformers = p->conformers;
  const AtomPosType *extra = p->extra;
  const float hx = p->hx;
  const float hy = p->hy;
  const float hz = p->hz;
  const float x0 = p->x0;
  const float y0 = p->y0;
  const float z0 = p->z0;
  const float bx_1 = p->bx_1;
  const float by_1 = p->by_1;
  const float bz_1 = p->bz_1;
  const float inv_num_conformers = 1.f / (float) p->num_conformers;
  const int num_bin_offsets = p->num_bin_offsets;
  const int num_extras = p->num_extras;
  const int num_probes = p->num_probes;
  const int num_conformers = p->num_conformers;
  const int nbx = p->nbx;
  const int nby = p->nby;
  const int mx = p->mx;
  const int my = p->my;
  const int kstart = p->kstart;
  const int kstop = p->kstop;

  const BinOfAtoms *bin_zero = p->bin_zero;
  const char *excl = p->exclusions;
  float *map = p->map;

  int i, j, k, n, nb;

  const float minocc = expf(-p->excl_energy);
  const float cutoff2 = p->cutoff * p->cutoff;

  float *u = new float[p->num_conformers];  // cal potential for each conformer

  for (k = kstart;  k < kstop;  k++) {  // k index loops over slab
    for (j = 0;  j < my;  j++) {
      for (i = 0;  i < mx;  i++) {  // loop over lattice points

        int lindex = (k*my + j)*mx + i;  // map index
        if (excl[lindex]) {  // is map point excluded?
          map[lindex] = 0.f;  // clamp occupancy to zero
          continue;
        }

        float px = i*hx;
        float py = j*hy;
        float pz = k*hz;  // translated coordinates of lattice point

        int ib = (int) floorf(px * bx_1);
        int jb = (int) floorf(py * by_1);
        int kb = (int) floorf(pz * bz_1);  // zero-based bin index

        int m, ma;

        px += x0;
        py += y0;
        pz += z0;  // absolute position

        memset(u, 0, num_conformers * sizeof(float));

        for (nb = 0;  nb < num_bin_offsets;  nb++) { // bin neighborhood
          int ic = ib + (int) bin_offsets[3*nb    ];
          int jc = jb + (int) bin_offsets[3*nb + 1];
          int kc = kb + (int) bin_offsets[3*nb + 2];

          const AtomPosType *atom = bin_zero[(kc*nby + jc)*nbx + ic].atom;

          for (n = 0;  n < BIN_DEPTH;  n++) {  // atoms in bin
            if (-1 == atom[n].vdwtype) break;  // finished atoms in bin

            for (m = 0;  m < num_conformers;  m++) {  // conformers
              float v = 0.f;
              for (ma = 0;  ma < num_probes;  ma++) {  // probe
                int index = m*num_probes + ma;
                float dx = conformers[3*index    ] + px - atom[n].x;
                float dy = conformers[3*index + 1] + py - atom[n].y;
                float dz = conformers[3*index + 2] + pz - atom[n].z;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 >= cutoff2) continue;
                int pindex = 2 * atom[n].vdwtype;
                float epsilon = vdw_params[pindex] * probe_vdw_params[2*ma];
                float rmin = vdw_params[pindex + 1] + probe_vdw_params[2*ma + 1];
                float rm6 = rmin*rmin / r2;
                rm6 = rm6 * rm6 * rm6;
                v += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
              } // end loop probe

              u[m] += v;  // contribution of one system atom to conformer

            } // end loop conformers

          } // end loop atoms in bin

        } // end loop bin neighborhood

        for (n = 0;  n < num_extras;  n++) {  // extra atoms
          for (m = 0;  m < num_conformers;  m++) {  // conformers
            float v = 0.f;
            for (ma = 0;  ma < num_probes;  ma++) {  // probe
              int index = m*num_probes + ma;
              float dx = conformers[3*index    ] + px - extra[n].x;
              float dy = conformers[3*index + 1] + py - extra[n].y;
              float dz = conformers[3*index + 2] + pz - extra[n].z;
              float r2 = dx*dx + dy*dy + dz*dz;
              if (r2 >= cutoff2) continue;
              int pindex = 2 *extra[n].vdwtype;
              float epsilon = vdw_params[pindex] * probe_vdw_params[2*ma];
              float rmin = vdw_params[pindex + 1] + probe_vdw_params[2*ma + 1];
              float rm6 = rmin*rmin / r2;
              rm6 = rm6 * rm6 * rm6;
              v += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
            } // end loop probe

            u[m] += v;  // contribution of one system atom to conformer

          } // end loop conformers
        } // end loop over extra atoms

        // now we have energies of all conformers u[i], i=0..m-1

        float z = 0.f;
        for (m = 0;  m < num_conformers;  m++) {  // average over conformers
          z += expf(-u[m]);
        }

        float occ = z * inv_num_conformers;  // the occupency
        map[lindex] = (occ > minocc ? occ : 0.f);
      }
    }
  } // end loop over lattice

  delete[] u;  // free extra memory
} // compute_occupancy_multiatom()


// Write bin histogram into a dx map
static void write_bin_histogram_map(
    const ComputeOccupancyMap *p,
    const char *filename
    ) {
  float xaxis[3], yaxis[3], zaxis[3];
  memset(xaxis, 0, 3*sizeof(float));
  memset(yaxis, 0, 3*sizeof(float));
  memset(zaxis, 0, 3*sizeof(float));
  xaxis[0] = p->nbx * p->bx;
  yaxis[1] = p->nby * p->by;
  zaxis[2] = p->nbz * p->bz;

  int gridsize = p->nbx * p->nby * p->nbz;
  float *data = new float[gridsize];

  int i;
  for (i=0; i<gridsize; i++) {
    data[i] = (float) p->bincnt[i];
  }

  float ori[3];
  ori[0] = p->x0 - p->padx * p->bx;
  ori[1] = p->y0 - p->pady * p->by;
  ori[2] = p->z0 - p->padz * p->bz;
 
  VolumetricData *volhist;
  volhist = new VolumetricData("atom binning histogram",
                               ori, xaxis, yaxis, zaxis,
                               p->nbx, p->nby, p->nbz, data);

  // Call the file writer in the VolMapCreate.C:
  volmap_write_dx_file(volhist, filename);

  delete volhist;  // XXX does data get deleted as part of volhist?
}


// XXX print out histogram of atom bins
void atom_bin_stats(const ComputeOccupancyMap *p) {
  int histogram[10] = { 0 };
  int i, j, k;

  printf("*** origin = %g %g %g\n", p->x0, p->y0, p->z0);
  printf("*** lenx = %g  leny = %g  lenz = %g\n", p->lx, p->ly, p->lz);
  printf("*** bin lengths = %g %g %g\n", p->bx, p->by, p->bz);
  printf("*** inverse bin lengths = %g %g %g\n", p->bx_1, p->by_1, p->bz_1);
  printf("*** bin array:  %d X %d X %d\n", p->nbx, p->nby, p->nbz);
  printf("*** bin padding:  %d %d %d\n", p->padx, p->pady, p->padz);
  printf("*** cutoff = %g\n", p->cutoff);
  printf("*** extcutoff = %g\n", p->extcutoff);
  printf("*** size of tight neighborhood = %d\n", p->num_bin_offsets);

  for (k = 0;  k < p->nbz;  k++) {
    for (j = 0;  j < p->nby;  j++) {
      for (i = 0;  i < p->nbx;  i++) {
        int index = (k*p->nby + j)*p->nbx + i;
        int count = p->bincnt[index];
        histogram[(count <= 9 ? count : 9)]++;
      }
    }
  }

  printf("*** histogram of bin fill:\n");
  for (i = 0;  i < (int) (sizeof(histogram) / sizeof(int));  i++) {
    if (i < 9) {
      printf("***     atom count %d     number of bins %d\n",
          i, histogram[i]);
    }
    else {
      printf("***     atom count > 8   number of bins %d\n", histogram[i]);
    }
  }
  printf("*** number of extra atoms = %d\n", p->num_extras);

  write_bin_histogram_map(p, "binning_histogram.dx");
}


// XXX slow quadratic complexity algorithm for checking correctness
//     for every map point, for every probe atom in all conformers,
//     iterate over all atoms
void compute_allatoms(
    float *map,                    // return calculated occupancy map
    int mx, int my, int mz,        // dimensions of map
    float lx, float ly, float lz,  // lengths of map
    const float origin[3],         // origin of map
    const float axes[9],           // basis vectors of aligned map
    const float alignmat[16],      // 4x4 alignment matrix
    int num_coords,                // number of atoms
    const float *coords,           // atom coordinates, length 3*num_coords
    const int *vdw_type,           // vdw type numbers, length num_coords
    const float *vdw_params,       // scaled vdw parameters for atoms
    float cutoff,                  // cutoff distance
    const float *probe_vdw_params, // scaled vdw parameters for probe atoms
    int num_probe_atoms,           // number of atoms in probe
    int num_conformers,            // number of conformers
    const float *conformers,       // length 3*num_probe_atoms*num_conformers
    float excl_energy              // exclusion energy threshold
    ) {
  const float theTrivialConformer[] = { 0.f, 0.f, 0.f };

  if (0 == num_conformers) {  // fix to handle trivial case consistently
    num_conformers = 1;
    conformers = theTrivialConformer;
  }

  float hx = lx / mx;  // lattice spacing
  float hy = ly / my;
  float hz = lz / mz;

  int i, j, k, n, m, ma;

  float cutoff2 = cutoff * cutoff;
  float minocc = expf(-excl_energy);  // minimum nonzero occupancy

  float *u = new float[num_conformers];  // calc potential for each conformer

#if 1
  printf("*** All atoms calculation (quadratic complexity)\n");
  printf("***   number of map points = %d\n", mx*my*mz);
  printf("***   number of atoms = %d\n", num_coords);
  printf("***   number of conformers = %d\n", num_conformers);
  printf("***   number of probe atoms = %d\n", num_probe_atoms);
#endif

  for (k = 0;  k < mz;  k++) {
    for (j = 0;  j < my;  j++) {
      for (i = 0;  i < mx;  i++) {  // loop over lattice points

        int mindex = (k*my + j)*mx + i;  // map flat index

        float px = i*hx + origin[0];
        float py = j*hy + origin[1];
        float pz = k*hz + origin[2];  // coordinates of lattice point

        memset(u, 0, num_conformers * sizeof(float));

        for (n = 0;  n < num_coords;  n++) {  // all atoms

          for (m = 0;  m < num_conformers;  m++) {  // conformers
            float v = 0.f;
            for (ma = 0;  ma < num_probe_atoms;  ma++) {  // probe atoms
              int index = m*num_probe_atoms + ma;
              float dx = conformers[3*index    ] + px - coords[3*n    ];
              float dy = conformers[3*index + 1] + py - coords[3*n + 1];
              float dz = conformers[3*index + 2] + pz - coords[3*n + 2];
              float r2 = dx*dx + dy*dy + dz*dz;
              if (r2 >= cutoff2) continue;
              int pindex = 2 * vdw_type[n];
              float epsilon = vdw_params[pindex] * probe_vdw_params[2*ma];
              float rmin = vdw_params[pindex + 1] + probe_vdw_params[2*ma + 1];
              float rm6 = rmin*rmin / r2;
              rm6 = rm6 * rm6 * rm6;
              v += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
            } // end loop probe atoms

            u[m] += v;  // contribution of one system atom to conformer

          } // end loop conformers

        } // end loop over all atoms

        float z = 0.f;
        for (m = 0;  m < num_conformers;  m++) {  // average conformer energies
          z += expf(-u[m]);
        }

        map[mindex] = z / num_conformers;  // store average occupancy
        if (map[mindex] < minocc) map[mindex] = 0.f;
      }
    }
  } // end loop over map

  delete[] u;
}

