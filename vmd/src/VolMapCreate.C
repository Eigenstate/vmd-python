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
 *      $RCSfile: VolMapCreate.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.120 $      $Date: 2012/07/24 15:20:30 $
 *
 ***************************************************************************/

/* Functions for creating useful volumetric maps based on the 3-D molecular structure */

// Todo List: 
// - Document!
// - Don't just output to a DX file... give user more control (use plugins)
// - Allow a framerange param, don't just use all available frames (and get rid
// of the VMDApp dependency). Also, VMD needs a general FrameRange object...

// Note:
// All functions in here loop over x fastest and z slowest and also create
// maps with that order of data. This matches the order used in
// VolumetricData which is ultimately dictated by the way graphics hardware 
// works.

// XXX VolMapCreate provides its own dx file writer.
// Actualy the maps should be generated in memory only and only dumped
// to files using the molfile_plugin interface. This is currently
// not yet possible but will hopefully be enabled soon.

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h> // only for write_dx_file()

#include "VMDApp.h"
#include "MoleculeList.h"
#include "Molecule.h"
#include "Timestep.h"
#include "Measure.h"
#include "SpatialSearch.h"
#include "VolCPotential.h"
#include "VolumetricData.h"
#include "VolMapCreate.h"
#include "utilities.h"
#include "ResizeArray.h"
#include "Inform.h"
#include "WKFUtils.h"

#if defined(VMDUSEMSMPOT)
#include "msmpot.h"
#endif

// Conversion factor between raw units (e^2 / A) and kT/e
#define POT_CONV 560.47254

#define MIN(X,Y) (((X)<(Y))? (X) : (Y))
#define MAX(X,Y) (((X)>(Y))? (X) : (Y))

/// maximum energy (above which all energies are considered infinite, for
/// performance purposes of the Energy/PMF map type)
static const float MAX_ENERGY = 150.f; 



////////////// VolMapCreate //////////////

VolMapCreate::VolMapCreate(VMDApp *the_app, AtomSel *the_sel, float resolution) {
  volmap = NULL;
  app = the_app;
  sel = the_sel;
  delta = resolution;
  computed_frames = 0;
  checkpoint_freq = 0;
  checkpoint_name = NULL;

  char dataname[1];
  strcpy(dataname, ""); // null-terminated empty string
  float zerovec[3];
  memset(zerovec, 0, 3*sizeof(float));
  volmap = new VolumetricData(dataname, zerovec,
			      zerovec, zerovec, zerovec, 0, 0, 0, NULL);
  user_minmax = false;
}


VolMapCreate::~VolMapCreate() {
  if (volmap) delete volmap;
  if (checkpoint_name) delete[] checkpoint_name;
}


void VolMapCreate::set_minmax (float minx, float miny, float minz, float maxx, float maxy, float maxz) {
  user_minmax = true;
  min_coord[0] = minx;
  min_coord[1] = miny;
  min_coord[2] = minz;
  max_coord[0] = maxx;
  max_coord[1] = maxy;
  max_coord[2] = maxz;
}


void VolMapCreate::set_checkpoint (int checkpointfreq_tmp, char *checkpointname_tmp) {
  if (checkpointfreq_tmp > -1) checkpoint_freq = checkpointfreq_tmp;
  if (!checkpointname_tmp) return;
  
  if (checkpoint_name) delete[] checkpoint_name;
  checkpoint_name = new char[strlen(checkpointname_tmp)+1];
  strcpy(checkpoint_name, checkpointname_tmp);
}


/// Computes the minmax values over all frames and fills them into
/// two pre-allocated float[3] arrays.
int VolMapCreate::calculate_minmax (float *my_min_coord, float *my_max_coord) {
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  int numframes = app->molecule_numframes(sel->molid()); // XXX need a frame selection object
  
  float frame_min_coord[3], frame_max_coord[3], *coords;
  
  msgInfo << "volmap: Computing bounding box coordinates" << sendmsg;

  int save_frame = sel->which_frame;
  int frame;
  for (frame=0; frame<numframes; frame++) {
    sel->which_frame=frame;
    sel->change(NULL,mol);
    coords = sel->coordinates(app->moleculeList);
    if (!coords) continue;

    int err = measure_minmax(sel->num_atoms, sel->on, coords, NULL, 
                             frame_min_coord, frame_max_coord);
    if (err != MEASURE_NOERR) {
      sel->which_frame = save_frame;
      return err;
    }
    
    int i;
    for (i=0; i<3; i++) {
      if (!frame || frame_min_coord[i] < my_min_coord[i]) my_min_coord[i] = frame_min_coord[i];
      if (!frame || frame_max_coord[i] > my_max_coord[i]) my_max_coord[i] = frame_max_coord[i];
    }
  }
  sel->which_frame = save_frame;
  
  return 0;
}


/// Calculates the maximum vdW radius of all atoms in the selection
int VolMapCreate::calculate_max_radius (float &max_rad) {
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  if (!mol) return -1;
  const float *radius = mol->extraflt.data("radius");
  if (!radius) return MEASURE_ERR_NORADII;
  
  max_rad = 0.f;
  int i;
  for (i=sel->firstsel; i<=sel->lastsel; i++) 
    if (sel->on[i] && radius[i] > max_rad) max_rad = radius[i];
  
  return 0;
}

// Combo routines are used to combine the different frame maps together into a
// final entity

// Initialize the frame combination buffer
void VolMapCreate::combo_begin(CombineType method, void **customptr, void *params) {
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;
  
  *customptr = NULL;
  memset(volmap->data, 0, sizeof(float)*gridsize);
  computed_frames = 0;
  
  // these combine types need additional storage
  if (method == COMBINE_STDEV) {
    float *voldata2 = new float[gridsize];
    memset(voldata2, 0, gridsize*sizeof(float));
    *customptr = (void*) voldata2;
  }
}

// Add a frame to the combination buffer
void VolMapCreate::combo_addframe(CombineType method, float *voldata, void *customptr, float *frame_voldata) {
  float *voldata2 = (float*) customptr;
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;
  int n;
  
  computed_frames++;
     
  if (computed_frames == 1) { // FIRST FRAME
    switch (method) {
    case COMBINE_AVG:
    case COMBINE_MAX:    
    case COMBINE_MIN:    
      memcpy(voldata, frame_voldata, gridsize*sizeof(float));
      break;
    case COMBINE_PMF:
      memcpy(voldata, frame_voldata, gridsize*sizeof(float));
      break;
    case COMBINE_STDEV:    
      memcpy(voldata, frame_voldata, gridsize*sizeof(float));
      for (n=0; n<gridsize; n++) voldata2[n] = frame_voldata[n]*frame_voldata[n];
      break;
    }
    
    return;
  }

  // THE FOLLOWING ONLY APPLIES TO OTHER FRAMES THAN FIRST
  switch (method) {
    case COMBINE_AVG:
      for (n=0; n<gridsize; n++) voldata[n] += frame_voldata[n];
      break;
    case COMBINE_PMF:
      for (n=0; n<gridsize; n++) voldata[n] = (float) -log(exp(-voldata[n]) + exp(-frame_voldata[n]));
      break;
    case COMBINE_MAX:    
      for (n=0; n<gridsize; n++) voldata[n] = MAX(voldata[n], frame_voldata[n]);
      break;
    case COMBINE_MIN:    
      for (n=0; n<gridsize; n++) voldata[n] = MIN(voldata[n], frame_voldata[n]);
      break;
    case COMBINE_STDEV:    
      for (n=0; n<gridsize; n++) voldata[n] += frame_voldata[n];
      for (n=0; n<gridsize; n++) voldata2[n] += frame_voldata[n]*frame_voldata[n];
      break;
  }
}


/// Output a copy of the combination buffer, to which a final transform
/// appropriate for the combination type has been applied to. Frames can still
/// be appended to the original combo buffer. This procedure should be used to
/// create the final map as well as each checkpoint map.
void VolMapCreate::combo_export(CombineType method, float *voldata, void *customptr) {
  float *voldata2 = (float*) customptr;
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;
  int n;
  
  switch (method) {
  case COMBINE_AVG:
    for (n=0; n<gridsize; n++)
      volmap->data[n] = voldata[n]/computed_frames;
    break;
  case COMBINE_PMF:
    for (n=0; n<gridsize; n++) {
      float val = voldata[n] + logf((float) computed_frames);
      if (val > MAX_ENERGY) val = MAX_ENERGY;  // weed out outlying data
      volmap->data[n] = val;
    }
    break;
  case COMBINE_MAX:
  case COMBINE_MIN:    
    memcpy(volmap->data, voldata, gridsize*sizeof(float));    
    break;
  case COMBINE_STDEV:    
    for (n=0; n<gridsize; n++) {
      volmap->data[n] = voldata[n]/computed_frames;
      volmap->data[n] = sqrtf(voldata2[n]/computed_frames - volmap->data[n]*volmap->data[n]); 
    }
    break;
  }
}


/// Do some cleaning up of the combination buffer
void VolMapCreate::combo_end(CombineType method, void *customptr) {
  if (method == COMBINE_STDEV) {
    float *voldata2 = (float*) customptr;
    delete[] voldata2;
  }
}


/// computes volmap for all (selected) frames and takes the average, min or max
/// depending on what was specified by "method", for most cases, 
/// params is unused (set to NULL)
/// XXX the "allframes" param should be a FrameRange object, but for now its a boolean
int VolMapCreate::compute_all (bool allframes, CombineType method, void *params) {
  int err = this->compute_init();
  if (err) return err;
  
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;

  // Special case: if only have one frame do it here the fast way
  if (!allframes) {
    if (volmap->data) delete[] volmap->data;
    volmap->data = new float[gridsize];         // final exported voldata
  
    msgInfo << "volmap: grid size = " << volmap->xsize
            <<"x"<< volmap->ysize <<"x"<< volmap->zsize;
    char tmp[64];
    sprintf(tmp, " (%.1f MB)", sizeof(float)*gridsize/(1024.*1024.));
    msgInfo << tmp << sendmsg;

    // only compute the current frame of the given selection
    this->compute_frame(sel->which_frame, volmap->data); 
    
    //err = this->compute_end();
    //if (err) return err;
    return 0; // no error
  }
  
  
  msgInfo << "volmap: grid size    = " << volmap->xsize
          <<"x"<< volmap->ysize <<"x"<< volmap->zsize;
  char tmp[64];
  sprintf(tmp, " (%.1f MB)", sizeof(float)*gridsize/(1024.*1024.));
  msgInfo << tmp << sendmsg;

  int numframes = app->molecule_numframes(sel->molid());
  msgInfo << "volmap: Computing " << numframes << " frames in total..." << sendmsg;

  if (volmap->data) delete[] volmap->data;
  volmap->data = new float[gridsize];         // final exported voldata
  float *frame_voldata = new float[gridsize]; // individual frame voldata
  float *voldata = new float[gridsize];       // combo cache voldata
  
  void *customptr = NULL;
  combo_begin(method, &customptr, params);
  wkf_timerhandle timer = wkf_timer_create();

  // Combine frame_voldata into voldata, one frame at a time, starting with 1st frame
  int frame;
  for (frame=0; frame<numframes; frame++) { 
    // XXX to-do, only take frames from a frame selection
    msgInfo << "volmap: frame " << frame << "/" << numframes;
#ifdef TIMING
    msgInfo << sendmsg;
#else
    msgInfo << "   ";
#endif

    wkf_timer_start(timer);

    this->compute_frame(frame, frame_voldata);
    wkf_timer_stop(timer);
    msgInfo << "Total time = " << wkf_timer_time(timer) << " s" << sendmsg;

    combo_addframe(method, voldata, customptr, frame_voldata);
    if (checkpoint_freq && computed_frames && !(computed_frames%checkpoint_freq)) {
      combo_export(method, voldata, customptr);
      const char *filename;
      if (checkpoint_name) filename=checkpoint_name;
      else filename = "checkpoint.dx";
      write_map(filename);
    }
  }
    
  wkf_timer_destroy(timer);

  delete[] frame_voldata;
  
  // All frames have been combined, perform finishing steps here
  combo_export(method, voldata, customptr);
  combo_end (method, customptr);
  delete[] voldata;
       
  return 0; // no error
}



// compute_init() sets up the grid coordinate system and dimensions
// If the user did not specify the grid's minmax boundary, it is
// defaulted to the trajectory's minmax coordinates, to which "padding"
// is added in all dimensions. 
int VolMapCreate::compute_init (float padding) {
  if (!sel) return MEASURE_ERR_NOSEL;
  if (sel->num_atoms == 0) return MEASURE_ERR_NOATOMS;
  
  int err, i;
  
  if (!volmap) return -1;
  
  if (user_minmax)
    padding = 0.;  // don't want to pad user's defaults
  else {
    // The minmax coordinates over all frames
    err = calculate_minmax(min_coord, max_coord);
    if (err) return err;
  }
  
  // Depending on the volmap type we are computing we add some padding
  // to the dimensions of the grid. This function is called by the 
  // compute_init() methods of the VolMapCreate<type> subclasses. 
  // The caller provides a reasonable map type specific padding value.
  // The padding can for example be the radius of the largest atom or
  // an interaction distance cutoff.
  // After padding we align the grid with integer coordinates.
  for (i=0; i<3; i++) {
    //adjust padding and ensure that different maps are properly aligned
    min_coord[i] = (float) floor((min_coord[i] - padding)/delta)*delta;    
    max_coord[i] = (float)  ceil((max_coord[i] + padding)/delta)*delta;
  }
  
  volmap->xsize = MAX((int)((max_coord[0] - min_coord[0])/delta), 0);
  volmap->ysize = MAX((int)((max_coord[1] - min_coord[1])/delta), 0);
  volmap->zsize = MAX((int)((max_coord[2] - min_coord[2])/delta), 0);

  char tmpstr[256];
  sprintf(tmpstr, "{%g %g %g} {%g %g %g}\n", min_coord[0], min_coord[1], min_coord[2], max_coord[0], max_coord[1], max_coord[2]);
  msgInfo << "volmap: grid minmax = " << tmpstr << sendmsg; 
  
  float cellx[3], celly[3], cellz[3];
  cellx[0] = delta;
  cellx[1] = 0.f;
  cellx[2] = 0.f;
  celly[0] = 0.f;
  celly[1] = delta;
  celly[2] = 0.f;
  cellz[0] = 0.f;
  cellz[1] = 0.f;
  cellz[2] = delta;

  // define origin by shifting to middle of each cell,
  // compute_frame() needs to take this into account
  for (i=0; i<3; i++) 
    volmap->origin[i] = min_coord[i] + \
                 0.5f*(cellx[i] + celly[i] + cellz[i]);
  int d;
  for (d=0; d<3; d++) {
    volmap->xaxis[d] = cellx[d]*(volmap->xsize-1);
    volmap->yaxis[d] = celly[d]*(volmap->ysize-1);
    volmap->zaxis[d] = cellz[d]*(volmap->zsize-1);
  }
  
  if (!volmap->xsize*volmap->ysize*volmap->zsize)
    return MEASURE_ERR_ZEROGRIDSIZE;

  return 0; // no error
}



////////////// VolMapCreateMask //////////////

/// This creates a "mask", i.e. a map that is either 1 or 0 and can be
/// multiplied to another map in order to isolate a particular region
/// in space. Currently, the mask is created by painting a sphere of
/// radius 5 around each selected atom.

int VolMapCreateMask::compute_init() {
  char tmpstr[255];
  sprintf(tmpstr, "mask (%s.200)", sel->cmdStr);
  volmap->set_name(tmpstr);
    
  return VolMapCreate::compute_init(atomradius+0.5f);
}


int VolMapCreateMask::compute_frame (int frame, float *voldata) {
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  if (!mol) return -1;
  
  int i;
  int GRIDSIZEX = volmap->xsize;
  int GRIDSIZEY = volmap->ysize;
  int GRIDSIZEZ = volmap->zsize;
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;
  
  //create volumetric mask grid
  memset(voldata, 0, gridsize*sizeof(float));
  int save_frame = sel->which_frame;
  sel->which_frame=frame;
  sel->change(NULL,mol);
  
  const float *coords = sel->coordinates(app->moleculeList);
  if (!coords) {
    sel->which_frame = save_frame;
    return -1;
  }
  
  float cellx[3], celly[3], cellz[3];
  volmap->cell_axes(cellx, celly, cellz);

  float min_coords[3];
  for (i=0; i<3; i++)
    min_coords[i] = float(volmap->origin[i] - 0.5f*(cellx[i] + celly[i] + cellz[i]));
  
  // paint atomic spheres on map
  int gx, gy, gz;
  for (i=sel->firstsel; i<=sel->lastsel; i++) { 
    if (!sel->on[i]) continue; //atom is not selected

    gx = (int) ((coords[3*i  ] - min_coords[0])/delta);
    gy = (int) ((coords[3*i+1] - min_coords[1])/delta);
    gz = (int) ((coords[3*i+2] - min_coords[2])/delta);
      
    int steps = (int)(atomradius/delta)+1;
    int iz, iy, ix;
    for (iz=MAX(gz-steps,0); iz<=MIN(gz+steps,GRIDSIZEZ-1); iz++)
    for (iy=MAX(gy-steps,0); iy<=MIN(gy+steps,GRIDSIZEY-1); iy++)
    for (ix=MAX(gx-steps,0); ix<=MIN(gx+steps,GRIDSIZEX-1); ix++) {
      int n = ix + iy*GRIDSIZEX + iz*GRIDSIZEY*GRIDSIZEX;
      float dx = float(coords[3*i  ] - volmap->origin[0] - ix*delta);
      float dy = float(coords[3*i+1] - volmap->origin[1] - iy*delta);
      float dz = float(coords[3*i+2] - volmap->origin[2] - iz*delta);
      float dist2 = dx*dx+dy*dy+dz*dz;
      if (dist2 <= atomradius*atomradius) voldata[n] = 1.f;
    }
  }
  
  sel->which_frame = save_frame;
  
  return 0;
}  



/////// VolMapCreateDensity ///////

/// Creates a map of the weighted atomic density at each gridpoint.
/// This is done by replacing each atom in the selection with a
/// normalized gaussian distribution of width (standard deviation)
/// equal to its atomic radius. The gaussian distribution for each
/// atom is then weighted using an optional weight, and defaults
/// to a weight of one (i.e, the number density). The various
/// gaussians are then additively distributed on a grid.

int VolMapCreateDensity::compute_init () {
  char tmpstr[255];
  sprintf(tmpstr, "density (%.200s) [A^-3]", sel->cmdStr);
  volmap->set_name(tmpstr);
  
  float max_rad=0.f;
  calculate_max_radius(max_rad);
    
  return VolMapCreate::compute_init(MAX(3.f*radius_scale*max_rad,10.f));
}


int VolMapCreateDensity::compute_frame (int frame, float *voldata) {
  if (!weight) return MEASURE_ERR_NOWEIGHT;
    
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  if (!mol) return -1;
    int i;
    
  const float *radius = mol->extraflt.data("radius");
  if (!radius) return MEASURE_ERR_NORADII;
  
  int GRIDSIZEX = volmap->xsize;
  int GRIDSIZEY = volmap->ysize;
  int GRIDSIZEZ = volmap->zsize;
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;

  //create volumetric density grid
  memset(voldata, 0, gridsize*sizeof(float));
  int save_frame = sel->which_frame;
  sel->which_frame = frame;
  sel->change(NULL,mol);
  const float *coords = sel->coordinates(app->moleculeList);
  if (!coords) {
    sel->which_frame = save_frame;
    return -1;
  }
  
  float cellx[3], celly[3], cellz[3];
  volmap->cell_axes(cellx, celly, cellz);

  float min_coords[3];
  for (i=0; i<3; i++) 
    min_coords[i] = float(volmap->origin[i] - 0.5f*(cellx[i] + celly[i] + cellz[i]));
  
  int w_index=0;
  int gx, gy, gz;   // grid coord indices
  for (i=sel->firstsel; i<=sel->lastsel; i++) { 
    if (!sel->on[i]) continue; //atom is not selected

    gx = (int) ((coords[3*i  ] - min_coords[0])/delta);
    gy = (int) ((coords[3*i+1] - min_coords[1])/delta);
    gz = (int) ((coords[3*i+2] - min_coords[2])/delta);
      
    float scaled_radius = 0.5f*radius_scale*radius[i];
    float exp_factor = 1.0f/(2.0f*scaled_radius*scaled_radius);
    float norm = weight[w_index++]/(sqrtf((float) (8.0f*VMD_PI*VMD_PI*VMD_PI))*scaled_radius*scaled_radius*scaled_radius);
                  
    int steps = (int)(4.1f*scaled_radius/delta);
    int iz, iy, ix;
    for (iz=MAX(gz-steps,0); iz<=MIN(gz+steps,GRIDSIZEZ-1); iz++)
    for (iy=MAX(gy-steps,0); iy<=MIN(gy+steps,GRIDSIZEY-1); iy++)
    for (ix=MAX(gx-steps,0); ix<=MIN(gx+steps,GRIDSIZEX-1); ix++) {
      int n = ix + iy*GRIDSIZEX + iz*GRIDSIZEY*GRIDSIZEX;
      float dx = float(coords[3*i  ] - volmap->origin[0] - ix*delta);
      float dy = float(coords[3*i+1] - volmap->origin[1] - iy*delta);
      float dz = float(coords[3*i+2] - volmap->origin[2] - iz*delta);
      float dist2 = dx*dx+dy*dy+dz*dz;
      voldata[n] += norm * expf(-dist2*exp_factor);
      // Uncomment the following line for a much faster implementation
      // This is useful is all you care about is the smooth visual appearance
      // voldata[n] += exp_factor/(dist2+10.f);
    }
  }

  sel->which_frame = save_frame;
    
  return 0;
}  




/////// VolMapCreateInterp ///////
  
/// Creates a map with the atomic weights interpolated onto a grid.
/// For each atom, its weight is distributed to the 8 nearest voxels via
/// a trilinear interpolation.
  
int VolMapCreateInterp::compute_init () {
  char tmpstr[255];
  sprintf(tmpstr, "interp (%.200s) [A^-3]", sel->cmdStr);
  volmap->set_name(tmpstr);
  
  return VolMapCreate::compute_init(delta+0.5f);
}


int VolMapCreateInterp::compute_frame (int frame, float *voldata) {
  if (!weight) return MEASURE_ERR_NOWEIGHT;
    
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  if (!mol) return -1;
    int i;
    
  int GRIDSIZEX = volmap->xsize;
  int GRIDSIZEY = volmap->ysize;
  int GRIDSIZEXY = GRIDSIZEX * GRIDSIZEY;
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;

  // create volumetric density grid
  memset(voldata, 0, gridsize*sizeof(float));
  int save_frame = sel->which_frame;
  sel->which_frame = frame;
  sel->change(NULL,mol);
  const float *coords = sel->coordinates(app->moleculeList);
  if (!coords) {
    sel->which_frame = save_frame;
    return -1;
  }
  
  int w_index=0;
  int gx, gy, gz;      // grid coord indices
  float fgx, fgy, fgz; // fractional grid coord indices
  float dx, dy, dz;    // to measure distances
  for (i=sel->firstsel; i<=sel->lastsel; i++) { 
    if (!sel->on[i]) continue; //atom is not selected

    // Find position of the atom within the map ("fractional indices")
    fgx = float(coords[3*i  ] - volmap->origin[0])/delta;
    fgy = float(coords[3*i+1] - volmap->origin[1])/delta;
    fgz = float(coords[3*i+2] - volmap->origin[2])/delta;

    // Find nearest voxel with lowest indices
    gx = (int) fgx;
    gy = (int) fgy;
    gz = (int) fgz;

    // Calculate distance between atom and each voxel
    dx = fgx - gx;
    dy = fgy - gy;
    dz = fgz - gz;

    // Perform trilinear interpolation

    voldata[ gx + gy*GRIDSIZEX + gz*GRIDSIZEXY ] \
      += (1.0f - dx) * (1.0f - dy) * (1.0f - dz) * weight[w_index];

    voldata[ (gx+1) + (gy+1)*GRIDSIZEX + (gz+1)*GRIDSIZEXY ] \
      += dx * dy * dz * weight[w_index];

    voldata[ (gx+1) + (gy+1)*GRIDSIZEX + gz*GRIDSIZEXY ] \
      += dx * dy * (1.0f - dz) * weight[w_index];

    voldata[ gx + gy*GRIDSIZEX + (gz+1)*GRIDSIZEXY ] \
      += (1.0f - dx) * (1.0f - dy) * dz * weight[w_index];

    voldata[ (gx+1) + gy*GRIDSIZEX + gz*GRIDSIZEXY ] \
      += dx * (1.0f - dy) * (1.0f - dz) * weight[w_index];

    voldata[ gx + (gy+1)*GRIDSIZEX + (gz+1)*GRIDSIZEXY ] \
      += (1.0f - dx) * dy * dz * weight[w_index];

    voldata[ gx + (gy+1)*GRIDSIZEX + gz*GRIDSIZEXY ] \
      += (1.0f - dx) * dy * (1.0f - dz) * weight[w_index];

    voldata[ (gx+1) + gy*GRIDSIZEX + (gz+1)*GRIDSIZEXY ] \
      += dx * (1.0f - dy) + dz * weight[w_index++];
  }

  sel->which_frame = save_frame;
      
  return 0;
}  




  
/////// VolMapCreateOccupancy ///////

/// This creates a map that is valued at 100 for gridpoints inside
/// atoms, and 0 for gridpoints outside. If averaged over many frames,
/// it will produce the % chance of that gridpoint being occupied.
/// These maps can either be created using point particles or by
/// painting spheres using the VDW radius.

int VolMapCreateOccupancy::compute_init () {
  char tmpstr[255];
  sprintf(tmpstr, "occupancy (%.200s)", sel->cmdStr);
  volmap->set_name(tmpstr);
  
  float max_rad=0.f;  
  if (use_points)
    max_rad = 1.f;
  else
    calculate_max_radius(max_rad);
  
  return VolMapCreate::compute_init(max_rad);
}


int VolMapCreateOccupancy::compute_frame(int frame, float *voldata) { 
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  if (!mol) return -1;
  
  int GRIDSIZEX = volmap->xsize;
  int GRIDSIZEY = volmap->ysize;
  int GRIDSIZEZ = volmap->zsize;
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;
  int i;
  
  //create volumetric density grid
  memset(voldata, 0, gridsize*sizeof(float));
  int save_frame = sel->which_frame;
  sel->which_frame=frame;
  sel->change(NULL,mol);
  const float *coords = sel->coordinates(app->moleculeList);

  if (!coords) {
    sel->which_frame = save_frame;
    return -1;
  }

  float cellx[3], celly[3], cellz[3];
  volmap->cell_axes(cellx, celly, cellz);

  float min_coords[3];
  for (i=0; i<3; i++) 
    min_coords[i] = float(volmap->origin[i] - 0.5f*(cellx[i] + celly[i] + cellz[i]));

  int gx, gy, gz;
  
  if (use_points) { // draw single points
    for (i=sel->firstsel; i<=sel->lastsel; i++) { 
      if (!sel->on[i]) continue; //atom is not selected

      gx = (int) ((coords[3*i  ] - min_coords[0])/delta);
      if (gx<0 || gx>=GRIDSIZEX) continue;
      gy = (int) ((coords[3*i+1] - min_coords[1])/delta);
      if (gy<0 || gy>=GRIDSIZEY) continue;
      gz = (int) ((coords[3*i+2] - min_coords[2])/delta);
      if (gz<0 || gz>=GRIDSIZEZ) continue;

      voldata[gx+GRIDSIZEX*gy+GRIDSIZEX*GRIDSIZEY*gz] = 1.f; 
    }
  }
  else { // paint atomic spheres on map
    const float *radius = mol->extraflt.data("radius");
    if (!radius) {
      sel->which_frame = save_frame;
      return MEASURE_ERR_NORADII;
    }
  
    for (i=sel->firstsel; i<=sel->lastsel; i++) { 
      if (!sel->on[i]) continue; //atom is not selected

      gx = (int) ((coords[3*i  ] - min_coords[0])/delta);
      gy = (int) ((coords[3*i+1] - min_coords[1])/delta);
      gz = (int) ((coords[3*i+2] - min_coords[2])/delta);
      
      int steps = (int)(radius[i]/delta)+1;
      int iz, iy, ix;
      for (iz=MAX(gz-steps,0); iz<=MIN(gz+steps,GRIDSIZEZ-1); iz++)
      for (iy=MAX(gy-steps,0); iy<=MIN(gy+steps,GRIDSIZEY-1); iy++)
      for (ix=MAX(gx-steps,0); ix<=MIN(gx+steps,GRIDSIZEX-1); ix++) {
        int n = ix + iy*GRIDSIZEX + iz*GRIDSIZEY*GRIDSIZEX;
        float dx = float(coords[3*i  ] - volmap->origin[0] - ix*delta);
        float dy = float(coords[3*i+1] - volmap->origin[1] - iy*delta);
        float dz = float(coords[3*i+2] - volmap->origin[2] - iz*delta);
        float dist2 = dx*dx+dy*dy+dz*dz;
        if (dist2 <= radius[i]*radius[i]) voldata[n] = 1.f;
      }
    }
  }
  
  sel->which_frame = save_frame;
    
  return 0;
}



/////// VolMapCreateDistance ///////

/// Creates a map for which each gridpoint contains the distance between
/// that point and the edge of the nearest atom. In other words, each
/// gridpoint specifies the maximum radius of a sphere centered at that
/// point which does not intersect with the spheres of any other atoms.
/// All atoms are treated as spheres using the atoms' VMD radii.

int VolMapCreateDistance::compute_init () {
  char tmpstr[255];
  sprintf(tmpstr, "distance (%.200s) [A]", sel->cmdStr);
  volmap->set_name(tmpstr);
  
  float max_rad=0.f;
  calculate_max_radius(max_rad);
  
  return VolMapCreate::compute_init(max_rad+max_dist);
}
 

/// Computes, for each gridpoint, the distance to the nearest atom
/// boundary, as defined by the VMD's atomic VDW radii.
int VolMapCreateDistance::compute_frame(int frame, float *voldata) { 
  int i, n;  
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  if (!mol) return -1;
  const float *radius = mol->extraflt.data("radius");
  if (!radius) return MEASURE_ERR_NORADII;
  
  int GRIDSIZEX = volmap->xsize;
  int GRIDSIZEY = volmap->ysize;
  int gridsize = volmap->xsize*volmap->ysize*volmap->zsize;

  float dx, dy, dz;
  float dist, mindist, r;
  
  float max_rad=0.f;
  calculate_max_radius(max_rad);
  
  // 1. Create a fake "molecule" containing all of the grid points
  //    this is quite memory intensive but _MUCH_ faster doing it point-by point!
  
  float *gridpos = new float[3*gridsize]; 
  int *gridon = new int[gridsize]; 
  for (n=0; n<gridsize; n++) {
    gridpos[3*n  ] = float((n%GRIDSIZEX)*delta + volmap->origin[0]); //position of grid cell's center
    gridpos[3*n+1] = float(((n/GRIDSIZEX)%GRIDSIZEY)*delta + volmap->origin[1]);
    gridpos[3*n+2] = float((n/(GRIDSIZEX*GRIDSIZEY))*delta + volmap->origin[2]); 
    gridon[n] = 1;
  }

  GridSearchPair *pairlist, *p;

  int save_frame = sel->which_frame;
  sel->which_frame = frame;
  sel->change(NULL,mol);
  const float *coords = sel->coordinates(app->moleculeList);
  if (!coords) {
    sel->which_frame = save_frame;
    return -1;
  }
  
  // initialize all grid points to be the maximal allowed distance = cutoff
  for (n=0; n<gridsize; n++) voldata[n] = max_dist;
  
  // 2. Create a list of all bonds between the grid and the real molecule
  //    which are within the user-set cutoff distance 
  //    (the use of a cutoff is purely to speed this up tremendously)
  
  pairlist = vmd_gridsearch3(gridpos, gridsize, gridon, coords,
                             sel->num_atoms, sel->on, max_dist+max_rad, true, -1);
  for (p=pairlist; p; p=p->next) {
    n = p->ind1;
    // if a grid point is already known to be inside an atom, skip it and save some time
    if ((mindist = voldata[n]) == 0.f) continue;
    i = p->ind2;
    r = radius[i];
    dx = gridpos[3*n  ] - coords[3*i];
    dy = gridpos[3*n+1] - coords[3*i+1];
    dz = gridpos[3*n+2] - coords[3*i+2];
    
    // 3. At each grid point, store the _smallest_ recorded distance
    //    to a nearby atomic surface
      
    dist = sqrtf(dx*dx+dy*dy+dz*dz) - r;
    if (dist < 0) dist = 0.f;
    if (dist < mindist) voldata[n] = dist;
  }
  
  // delete pairlist
  for (p=pairlist; p;) {
    GridSearchPair *tmp = p;
    p = p->next;
    free(tmp);
  }  

  delete [] gridpos; 
  delete [] gridon; 

  sel->which_frame = save_frame;

  return MEASURE_NOERR; 
}




/////// VolMapCreateCoulombPotential ///////
  
int VolMapCreateCoulombPotential::compute_init () {
  char tmpstr[255];
  sprintf(tmpstr, "Potential (kT/e at 298.15K) (%.200s)", sel->cmdStr);
  volmap->set_name(tmpstr);
  
  float max_rad;
  calculate_max_radius(max_rad);
    
  // init object, no extra padding by default  
  return VolMapCreate::compute_init(0.f);
}


int VolMapCreateCoulombPotential::compute_frame(int frame, float *voldata) {
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  if (!mol) return -1;
    int i;
    
  const float *charge = mol->extraflt.data("charge");
  if (!charge) return MEASURE_ERR_NORADII; // XXX fix this later

  int gridsize=volmap->xsize*volmap->ysize*volmap->zsize;

  // create volumetric density grid
  memset(voldata, 0, gridsize*sizeof(float));
  int save_frame = sel->which_frame;
  sel->which_frame=frame;
  sel->change(NULL,mol);
  const float *coords = sel->coordinates(app->moleculeList);
  if (!coords) {
    sel->which_frame = save_frame;
    return -1;
  }
 
  float cellx[3], celly[3], cellz[3];
  volmap->cell_axes(cellx, celly, cellz);

  float min_coords[3];
  for (i=0; i<3; i++) 
    min_coords[i] = float(volmap->origin[i] - 0.5f*(cellx[i] + celly[i] + cellz[i]));

  // copy selected atom coordinates and charges to a contiguous memory
  // buffer and translate them to the starting corner of the map.
  float *xyzq = (float *) malloc(sel->selected * 4 * sizeof(float));
  float *curatom = xyzq;
  for (i=sel->firstsel; i<=sel->lastsel; i++) { 
    if (sel->on[i]) {
      curatom[0] = coords[3*i  ] - min_coords[0];
      curatom[1] = coords[3*i+1] - min_coords[1];
      curatom[2] = coords[3*i+2] - min_coords[2];
      curatom[3] = charge[i] * float(POT_CONV);
      curatom += 4;
    }
  }

  vol_cpotential(sel->selected, xyzq, voldata, 
                 volmap->zsize, volmap->ysize, volmap->xsize, delta);

  free(xyzq);

  sel->which_frame = save_frame;
 
  return 0;
}  

/////// VolMapCreateCoulombPotentialMSM ///////

#if defined(VMDUSEMSMPOT)
int VolMapCreateCoulombPotentialMSM::compute_init () {
  char tmpstr[255];
  sprintf(tmpstr, "Potential (kT/e at 298.15K) (%.200s)", sel->cmdStr);
  volmap->set_name(tmpstr);
  
  float max_rad;
  calculate_max_radius(max_rad);
  
  // init object, no extra padding by default
  // Note: padding would create serious problems for the periodic case 
  return VolMapCreate::compute_init(0.f);
}


int VolMapCreateCoulombPotentialMSM::compute_frame(int frame, float *voldata) {
  DrawMolecule *mol = app->moleculeList->mol_from_id(sel->molid());
  if (!mol) return -1;
    int i;

  int usepbc = 0;
    
  const float *charge = mol->extraflt.data("charge");
  if (!charge) return MEASURE_ERR_NORADII; // XXX fix this later

  int gridsize=volmap->xsize*volmap->ysize*volmap->zsize;

  // create volumetric density grid
  memset(voldata, 0, gridsize*sizeof(float));
  int save_frame = sel->which_frame;
  sel->which_frame=frame;
  sel->change(NULL,mol);
  const float *coords = sel->coordinates(app->moleculeList);
  const Timestep *ts = sel->timestep(app->moleculeList); 
  if (!coords) {
    sel->which_frame = save_frame;
    return -1;
  }
  if (!ts) {
    return -1;
  }
 
  float cellx[3], celly[3], cellz[3];
  volmap->cell_axes(cellx, celly, cellz);

  float min_coords[3];
  for (i=0; i<3; i++) 
    min_coords[i] = float(volmap->origin[i] - 0.5f*(cellx[i] + celly[i] + cellz[i]));

  // copy selected atom coordinates and charges to a contiguous memory
  // buffer and translate them to the starting corner of the map.
  float *xyzq = (float *) malloc(sel->selected * 4 * sizeof(float));
  float *curatom = xyzq;
  for (i=sel->firstsel; i<=sel->lastsel; i++) { 
    if (sel->on[i]) {
      curatom[0] = coords[3*i  ] - min_coords[0];
      curatom[1] = coords[3*i+1] - min_coords[1];
      curatom[2] = coords[3*i+2] - min_coords[2];
      curatom[3] = charge[i] * float(POT_CONV);
      curatom += 4;
    }
  }

  Msmpot *msm = Msmpot_create(); // create a multilevel summation object
#if 0
  int msmrc;
  int mx = volmap->xsize;  /* map lattice dimensions */
  int my = volmap->ysize;
  int mz = volmap->zsize;
  float lx = delta*mx;     /* map lattice lengths */
  float ly = delta*my;
  float lz = delta*mz;
  float x0=0, y0=0, z0=0;  /* map origin */
  float vx=0, vy=0, vz=0;  /* periodic domain lengths (0 for nonperiodic) */

  if (getenv("MSMPOT_NOCUDA")) {
    /* turn off use of CUDA (with 0 in last parameter) */
    Msmpot_configure(msm, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  }

  if (getenv("MSMPOT_PBCON")) {
    vx = lx, vy = ly, vz = lz;  /* use periodic boundary conditions */
  }

  if (getenv("MSMPOT_EXACT")) { 
    msmrc = Msmpot_compute_exact(msm, voldata, 
        mx, my, mz, lx, ly, lz, x0, y0, z0, vx, vy, vz,
        xyzq, sel->selected);
  }
  else {
    msmrc = Msmpot_compute(msm, voldata, 
        mx, my, mz, lx, ly, lz, x0, y0, z0, vx, vy, vz,
        xyzq, sel->selected);
  }
  if (msmrc != MSMPOT_SUCCESS) {
    printf("MSM return code: %d\n", msmrc);
    printf("MSM error string: '%s'\n", Msmpot_error_string(msmrc));
  } 
#endif
#if 1
  // New MSM API: both non-periodic and periodic MSM calcs
  int msmrc;

  // XXX hack for ease of initial testing
  if (getenv("VMDMSMUSEPBC"))
    usepbc = 1;

  if (usepbc) {
    // get periodic cell information for current frame
    float a, b, c, alpha, beta, gamma;
    a = ts->a_length;
    b = ts->b_length;
    c = ts->c_length;
    alpha = ts->alpha;
    beta = ts->beta;
    gamma = ts->gamma;

    // check validity of PBC cell side lengths
    if (fabsf(a*b*c) < 0.0001) {
      msgErr << "volmap coulombmsm: unit cell volume is zero." << sendmsg;
      return -1;
    }

    // check PBC unit cell shape to select proper low level algorithm.
    if ((alpha != 90.0) || (beta != 90.0) || (gamma != 90.0)) {
      msgErr << "volmap coulombmsm: unit cell is non-orthogonal." << sendmsg;
      return -1;
    }

#ifdef MSMPOT_COMPUTE_EXACT
  if (getenv("MSMPOT_EXACT")) {
    // XXX the current PBC code will currently use the initially specified 
    //     map dimensions and coordinates for all frames in a time-averaged
    //     calculation.  In the case that one would prefer the map to cover
    //     a fixed region of the unit cell in reciprocal space, we will need
    //     to change this code to update the effective map geometry on-the-fly.
    msgInfo << "Running EXACT periodic MSM calculation..." << sendmsg;
    msmrc = Msmpot_compute_exact(msm, voldata, 
                           volmap->xsize, volmap->ysize, volmap->zsize,
                           volmap->xsize * delta, 
                           volmap->ysize * delta, 
                           volmap->zsize * delta, 
                           0, 0, 0, // origin, already translated to min 
                           a, b, c, // pbc cell length 0 == nonperiodic calc
                           xyzq, sel->selected);
  } else {
#endif
    // XXX the current PBC code will currently use the initially specified 
    //     map dimensions and coordinates for all frames in a time-averaged
    //     calculation.  In the case that one would prefer the map to cover
    //     a fixed region of the unit cell in reciprocal space, we will need
    //     to change this code to update the effective map geometry on-the-fly.
    msgInfo << "Running periodic MSM calculation..." << sendmsg;
    msmrc = Msmpot_compute(msm, voldata, 
                           volmap->xsize, volmap->ysize, volmap->zsize,
                           volmap->xsize * delta, 
                           volmap->ysize * delta, 
                           volmap->zsize * delta, 
                           0, 0, 0, // origin, already translated to min 
                           a, b, c, // pbc cell length 0 == nonperiodic calc
                           xyzq, sel->selected);
#ifdef MSMPOT_COMPUTE_EXACT
  }
#endif

  } else {

#ifdef MSMPOT_COMPUTE_EXACT
  if (getenv("MSMPOT_EXACT")) {
    msgInfo << "Running EXACT non-periodic MSM calculation..." << sendmsg;
    msmrc = Msmpot_compute_exact(msm, voldata, 
                           volmap->xsize, volmap->ysize, volmap->zsize,
                           volmap->xsize * delta, 
                           volmap->ysize * delta, 
                           volmap->zsize * delta, 
                           0, 0, 0, // origin, already translated to min 
                           0, 0, 0, // pbc cell length 0 == nonperiodic calc
                           xyzq, sel->selected);
  } else {
#endif
    msgInfo << "Running non-periodic MSM calculation..." << sendmsg;
    msmrc = Msmpot_compute(msm, voldata, 
                           volmap->xsize, volmap->ysize, volmap->zsize,
                           volmap->xsize * delta, 
                           volmap->ysize * delta, 
                           volmap->zsize * delta, 
                           0, 0, 0, // origin, already translated to min 
                           0, 0, 0, // pbc cell length 0 == nonperiodic calc
                           xyzq, sel->selected);
#ifdef MSMPOT_COMPUTE_EXACT
  }
#endif

  }

  if (msmrc != MSMPOT_SUCCESS) {
    printf("MSM return code: %d\n", msmrc);
    printf("MSM error string: '%s'\n", Msmpot_error_string(msmrc));
  } 
#else
  // old MSM API: non-periodic MSM calcs only
  int msmrc = Msmpot_compute(msm, voldata, 
                             volmap->xsize, volmap->ysize, volmap->zsize,
                             delta, delta, delta, 
                             0, 0, 0,
                             0, 0, 0, // origin, already translated to min 
                             xyzq, sel->selected);
  if (msmrc != MSMPOT_ERROR_NONE) {
    printf("MSM return code: %d\n", msmrc);
    printf("MSM error string: '%s'\n", Msmpot_error_string(msmrc));
  } 
#endif
  Msmpot_destroy(msm);

  free(xyzq);

  sel->which_frame = save_frame;
 
  return 0;
}  

#endif





// Write the map as a DX file.
// This is the default base class function which can be
// overridden by the derived classes. 
// E.g. VolMapCreateFastEnergy defines its own write_map().
void VolMapCreate::write_map(const char *filename) {
  volmap_write_dx_file(volmap, filename);
}


int volmap_write_dx_file (VolumetricData *volmap, const char *filename) {
  if (!volmap->data) return -1; // XXX is this a good random error code?
  int i;
  int xsize = volmap->xsize;
  int ysize = volmap->ysize;
  int zsize = volmap->zsize;
  int gridsize = xsize*ysize*zsize;

  float cellx[3], celly[3], cellz[3];
  volmap->cell_axes(cellx, celly, cellz);

  
  msgInfo << "volmap: writing file \"" << filename << "\"." << sendmsg;
  
  FILE *fout = fopen(filename, "w");
  if (!fout) {
    msgErr << "volmap: Cannot open file \"" << filename
           << "\" for writing." << sendmsg;
    return errno;
  };
    
  fprintf(fout, "# Data calculated by the VMD volmap function\n");

  // Since the data origin and the grid origin are aligned we have
  // grid centered data, even though we were thinking in terms of
  // voxels centered in datapoints. VMD treats all dx file maps as
  // grid centered data, so this is right.
  fprintf(fout, "object 1 class gridpositions counts %d %d %d\n", xsize, ysize, zsize);
  fprintf(fout, "origin %g %g %g\n", volmap->origin[0], volmap->origin[1], volmap->origin[2]);
  fprintf(fout, "delta %g %g %g\n", cellx[0], cellx[1], cellx[2]);
  fprintf(fout, "delta %g %g %g\n", celly[0], celly[1], celly[2]);
  fprintf(fout, "delta %g %g %g\n", cellz[0], cellz[1], cellz[2]);
  fprintf(fout, "object 2 class gridconnections counts %d %d %d\n", xsize, ysize, zsize);
  fprintf(fout, "object 3 class array type double rank 0 items %d data follows\n", gridsize);
  
  // This reverses the ordering from x fastest to z fastest changing variable
  float val1,val2,val3;
  int gx=0, gy=0, gz=-1;
  for (i=0; i < (gridsize/3)*3; i+=3)  {
    if (++gz >= zsize) {
      gz=0;
      if (++gy >= ysize) {gy=0; gx++;}
    }
    val1 = volmap->voxel_value(gx,gy,gz);
    if (++gz >= zsize) {
      gz=0;
      if (++gy >= ysize) {gy=0; gx++;}
    }
    val2 = volmap->voxel_value(gx,gy,gz);
    if (++gz >= zsize) {
      gz=0;
      if (++gy >= ysize) {gy=0; gx++;}
    }
    val3 = volmap->voxel_value(gx,gy,gz);    
    fprintf(fout, "%g %g %g\n", val1, val2, val3);
  }
  for (i=(gridsize/3)*3; i < gridsize; i++) {
    if (++gz >= zsize) {
      gz=0;
      if (++gy >= ysize) {gy=0; gx++;}
    }
    fprintf(fout, "%g ", volmap->voxel_value(gx,gy,gz));
  }
  if (gridsize%3) fprintf(fout, "\n");
  fprintf(fout, "\n");
  
  // Replace any double quotes (") by single quotes (') in the 
  // dataname string to make sure that we don't prematurely
  // terminate the string in the dx file.
  char *squotes = new char[strlen(volmap->name)+1];
  strcpy(squotes, volmap->name);
  char *s = squotes;
  while((s=strchr(s, '"'))) *s = '\'';

  if (volmap->name) {
    fprintf(fout, "object \"%s\" class field\n", squotes);
  } else {
    char dataname[10];
    strcpy(dataname, "(no name)");
    fprintf(fout, "object \"%s\" class field\n", dataname);
  }

  delete [] squotes;

  fclose(fout);
  return 0;
}
