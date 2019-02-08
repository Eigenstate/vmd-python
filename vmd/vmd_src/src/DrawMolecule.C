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
 *	$RCSfile: DrawMolecule.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.145 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Displayable version of a DrawMolecule, derived from BaseMolecule and
 * Displayable.  This contains all the info for rendering
 * the molecule.
 *
 ***************************************************************************/

#include "DrawMolecule.h"
#include "AtomColor.h"
#include "AtomRep.h"
#include "AtomSel.h"
#include "utilities.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "CommandQueue.h"
#include "CmdAnimate.h"
#include "Stride.h"
#include "PickList.h"
#include "MaterialList.h"
#include "Inform.h"
#include "TextEvent.h"
#include "DisplayDevice.h"
#include "MoleculeGraphics.h"
#include "BondSearch.h"
#include "DrawForce.h"
#include "VolumetricData.h"
#include "CUDAAccel.h"

///////////////////////  constructor and destructor

DrawMolecule::DrawMolecule(VMDApp *vmdapp, Displayable *par)
	: BaseMolecule(vmdapp->next_molid()), 
	  Displayable(par), app(vmdapp), repList(8) {
  repcounter = 0;
  curframe = -1;
  active = TRUE;
  did_secondary_structure = 0;
  molgraphics = new MoleculeGraphics(this);
  vmdapp->pickList->add_pickable(molgraphics);
  drawForce = new DrawForce(this);

  invalidate_cov_scale();
  center[0] = center[1] = center[2] = 0.0f;

  need_find_bonds = 0;
}
  
// destructor ... free up any extra allocated space (the child Displayables
// will be deleted by the Displayable destructor)
DrawMolecule::~DrawMolecule() {
  int i;

  // delete all molecule representations
  for(i=0; i < components(); i++) {
    app->pickList->remove_pickable(component(i));
    delete component(i);  
  }

  app->pickList->remove_pickable(molgraphics);

  // delete all timesteps
  for (i=timesteps.num()-1; i>=0; i--) {
    delete timesteps[i];
    timesteps.remove(i);
  }

  delete molgraphics;
}

///////////////////////  public routines

// return Nth component ... change to proper return type
DrawMolItem *DrawMolecule::component(int n) {
  if(n >= 0 && n < components())
    return repList[n];
  else
    return NULL;
}


// return the component corresponding to the pickable
DrawMolItem *DrawMolecule::component_from_pickable(const Pickable *p) {
  for (int i=0; i<components(); i++) 
    if (repList[i] == p) return repList[i];
  return NULL; // no matching component
}

// return the available CPU thread pool to the caller
wkf_threadpool_t * DrawMolecule::cpu_threadpool(void) {
  return (wkf_threadpool_t *) app->thrpool;
}

// return the available CUDA device pool to the caller
wkf_threadpool_t * DrawMolecule::cuda_devpool(void) {
  return (app->cuda != NULL) ? app->cuda->get_cuda_devpool() : NULL;
}


// Return true if ANY representation is displaying atom n
int DrawMolecule::atom_displayed(int n) {
  if (displayed() && n >= 0 && n < nAtoms) {
    for (int i=(components() - 1); i >= 0; i--) {
      if ((repList[i])->atom_displayed(n))
        return TRUE; // atom is shown
    }
  }
  return FALSE; // atom is not shown
}


// delete the Nth representation ... return success
int DrawMolecule::del_rep(int n) {
  DrawMolItem *rep = component(n);
  if (rep) {
    app->pickList->remove_pickable(rep);
    delete rep;		// delete the object
    repList.remove(n);	// and it's slot in the representation list
    invalidate_cov_scale();
  }

  return (rep != NULL);
}


void DrawMolecule::add_rep(AtomColor *ac, AtomRep *ar, AtomSel *as, 
                          const Material *am) {
  // Rep has unique name (unique within the molecule)
  char buf[50];
  sprintf(buf, "rep%d", repcounter++);
  DrawMolItem *rep = new DrawMolItem(buf, this, ac, ar, as);
  app->pickList->add_pickable(rep);
  rep->change_material(am);
  repList.append(rep);
  invalidate_cov_scale();
}

int DrawMolecule::show_rep(int repid, int onoff) {
  DrawMolItem *rep = component(repid);
  if (rep) {
    if (onoff) rep->on();
    else rep->off();
    invalidate_cov_scale();
    return TRUE;
  }
  return FALSE;
}

// change the Nth representation ... return success.
// if any object is NULL, that characteristic is not changed.
int DrawMolecule::change_rep(int n, AtomColor *ac, AtomRep *ar, const char *sel) { 
  DrawMolItem *rep = component(n);
  if (rep) {
    rep->change_color(ac);
    rep->change_rep(ar);
    rep->change_sel(sel);  // returns TRUE if there was no problem, or if
                           // sel was NULL meaning no action is to be taken
    invalidate_cov_scale();
    return TRUE;
  }

  return FALSE;
}


// redraw all the representations
void DrawMolecule::force_recalc(int reason) {
  int numcomp = components();
  for (int i=0; i<numcomp; i++) {
    component(i)->force_recalc(reason);
  }
  // The preceding loop updates all the DrawMolItem reps, but other children
  // of DrawMolecule (i.e. DrawForce) need to know about the update as well.
  // Calling need_matrix_recalc sets the _needUpdate flag for this purpose.
  need_matrix_recalc();
  app->commandQueue->runcommand(new CmdAnimNewFrame);

  // MOL_REGEN or SEL_REGEN implies our scale factor may have changed.  
  if (reason & (DrawMolItem::MOL_REGEN | DrawMolItem::SEL_REGEN)) 
    invalidate_cov_scale();
}


// tell the rep to update its PBC transformation matrices next prepare cycle
void DrawMolecule::change_pbc() {
  int numcomp = components();
  for (int i=0; i<numcomp; i++) 
    component(i)->change_pbc();
  // labels can be made between periodic images, so warn them that the
  // distance between images has changed.  Would be better to set a flag
  // so that notify can only get called once, inside of prepare().
  notify();
}


// tell the rep to update its timestep
void DrawMolecule::change_ts() {
  int numcomp = components();
  for (int i=0; i<numcomp; i++) 
    component(i)->change_ts();

  molgraphics->prepare();
  drawForce->prepare();

  notify();

  // now that internal state has been updated, notify scripts
  app->commandQueue->runcommand(new FrameEvent(id(), curframe));
}


// query whether this molecule contains a highlighted rep
int DrawMolecule::highlighted_rep() const {
  if (app->highlighted_molid != id()) 
    return -1;
  return app->highlighted_rep;
}


// get the component by its string name
int DrawMolecule::get_component_by_name(const char *nm) {
  // XXX linear search for the name is slow
  int numreps = repList.num();
  for (int i=0; i<numreps; i++) {
    if (!strcmp(repList[i]->name, nm))
      return i; // return component
  }
  return -1; // failed to find a component with that name
}


// get the name of the specified component
const char *DrawMolecule::get_component_name(int ind) {
  DrawMolItem *rep = component(ind);
  if (!rep) 
    return FALSE;
  return rep->name;
}

void DrawMolecule::prepare() {
  if (needUpdate()) {
    notify(); // notify monitors
  }
}

void DrawMolecule::override_current_frame(int n) {
    if (n == curframe) return;
    int num = timesteps.num();
    if ( num==0 ) return;
    if ( n<0 ) curframe = 0;
    else if ( n>=num ) curframe = num-1;
    else curframe = n;
    invalidate_cov_scale();
}

// notify monitors of an update
void DrawMolecule::notify() {
  int monnum = monitorlist.num();
  int nid = id();
  for (int i=0; i<monnum; i++) 
    monitorlist[i]->notify(nid);
}


// add a new frame
void DrawMolecule::append_frame(Timestep *ts) {
  timesteps.append(ts);  // add the timestep to the animation

  // To ensure compatibility with legacy behavior, always advance to the
  // newly added frame.  
  override_current_frame(timesteps.num() - 1);

  // Notify that curframe changed.  This appears to entail no significant
  // overhead: reps update lazily, molgraphics only regenerates if it's been
  // modified since the last update, and DrawForce seems to be innocuous as
  // well.
  change_ts();

  // recenter the molecule when the first coordinate frame is loaded
  if (timesteps.num() == 1) {    
#if 0
    // XXX this is a nice hack to allow easy benchmarking of real VMD
    //     trajectory I/O rates without having to first load some coords
    if (getenv("VMDNOCOVCALC") == NULL)
#endif
      app->scene_resetview_newmoldata();
  }

  // update bonds if needed, when any subsequent frame is loaded
  if (timesteps.num() >= 1) {    
    // find bonds if necessary
    if (need_find_bonds == 1) {     
      need_find_bonds = 0;
      vmd_bond_search(this, ts, -1, 0); // just add bonds, no dup checking
    } else if (need_find_bonds == 2) {
      need_find_bonds = 0;
      vmd_bond_search(this, ts, -1, 1); // add bonds checking for dups
    }
  }

  addremove_ts();              // tell all reps to update themselves
  app->commandQueue->runcommand(new CmdAnimNewNumFrames); // update frame count
}


// duplicate an existing frame
void DrawMolecule::duplicate_frame(const Timestep *ts) {
  Timestep *newts;
  if (ts == NULL) { // append a 'null' frame
    newts = new Timestep(nAtoms);
    newts->zero_values();
  } else {
    newts = new Timestep(*ts);
  }
  append_frame(newts);
}


// delete a frame
void DrawMolecule::delete_frame(int n) {
    if (n<0 || n>=timesteps.num()) return;
    delete timesteps[n];
    timesteps.remove(n);

    // notifications
    addremove_ts();
    app->commandQueue->runcommand(new CmdAnimNewNumFrames);

    // adjust current frame if necessary
    if (curframe >= timesteps.num()) {
        curframe = timesteps.num()-1;
        change_ts();
    }
}


// add or remove a timestep
void DrawMolecule::addremove_ts() {
  int numcomp = components();
  for (int i=0; i<numcomp; i++) 
    component(i)->change_traj();
}


// return the norm, double-precision arguments
static float dnorm(const double *v) {
  return (float)sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

void DrawMolecule::invalidate_cov_scale() {
  scalefactor = -1;
}

// scaling factor required to make the molecule fit within (-1 ... 1)
float DrawMolecule::scale_factor() {
  if (scalefactor < 0) update_cov_scale();
  if (scalefactor > 0) {
    return scalefactor;

  } else if (molgraphics->num_elements() > 0) {
    return molgraphics->scale_factor();

  } else if (volumeList.num() > 0) {
    // scale factor is 1.5/(maxrange), where maxrange is the largest range
    // of the data along any cardinal axis.  That's how Timestep does it, 
    // anyway.  The volumetric axes aren't necessarily orthogonal so I'll
    // just go with the largest value.
    const VolumetricData *data = volumeList[0];
    float x=dnorm(data->xaxis), y=dnorm(data->yaxis), z=dnorm(data->zaxis);
    float scale_factor = x > y ? x : y;
    scale_factor = scale_factor > z ? scale_factor : z;
    if (scale_factor > 0) return 1.5f/scale_factor;
  }
  return 1.0f;
}


// center of volume of this molecule
int DrawMolecule::cov(float& x, float& y, float& z) {
  if (scalefactor < 0) update_cov_scale();

  if (scalefactor > 0) {
    // have valid coordinate data
    x = center[0]; y = center[1]; z = center[2];

  } else if (molgraphics->num_elements() > 0) {
    // use user-defined graphics to center 
    molgraphics->cov(x, y, z);

  } else if (volumeList.num() > 0) {
    // use first volumetric data set
    const VolumetricData *data = volumeList[0];
    x = (float) (data->origin[0] + 
        0.5*(data->xaxis[0] + data->yaxis[0] + data->zaxis[0]));
    y = (float) (data->origin[1] + 
        0.5*(data->xaxis[1] + data->yaxis[1] + data->zaxis[1]));
    z = (float) (data->origin[2] + 
        0.5*(data->xaxis[2] + data->yaxis[2] + data->zaxis[2]));
  } else {
    return FALSE;
  }
  return TRUE;
}


/// recompute molecule's bonds via distance bond search from current timestep
int DrawMolecule::recalc_bonds() {
  Timestep *ts = current();

  if (ts) {
    clear_bonds();                     // clear the existing bond list
    vmd_bond_search(this, ts, -1, 0);  // just add bonds, no dup checking
    msgInfo << "Bond count: " << count_bonds() << sendmsg;
    return 0;
  } 

  msgInfo << "No coordinates" << sendmsg;
  return -1;
}

int DrawMolecule::need_secondary_structure(int calc_if_not_yet_done) {
  if (did_secondary_structure) return TRUE; 

  if (calc_if_not_yet_done) {
    if (!current()) return FALSE; // fails if there's no frame
    did_secondary_structure = TRUE;
    app->show_stride_message();
    if (ss_from_stride(this)) {
      msgErr << "Call to Stride program failed." << sendmsg;
      return FALSE;
    }
    return TRUE;
  }
  // just indicate that we don't need to do the calculation anymore
  did_secondary_structure = TRUE;
  return TRUE;
}

void DrawMolecule::invalidate_ss() {
  did_secondary_structure = 0;
}

int DrawMolecule::recalc_ss() {
  did_secondary_structure = 0;
  int success = need_secondary_structure(1);
  did_secondary_structure = 1;

  if (success) for (int i=0; i<components(); i++) component(i)->change_ss();
  return success;
}

void DrawMolecule::register_monitor(DrawMoleculeMonitor *mon) {
  monitorlist.append(mon);
}
void DrawMolecule::unregister_monitor(DrawMoleculeMonitor *mon) {
  monitorlist.remove(monitorlist.find(mon));
}

void DrawMolecule::update_cov_scale() {
  const Timestep *ts = current();
  if (!ts) return;
  int i, n = ts->num;
  // only do this if there are atoms
  if (!n) return;

  float covx, covy, covz;
  float minposx, minposy, minposz;
  float maxposx, maxposy, maxposz;

  // flags for selected atoms in displayed reps
  ResizeArray<int> tmp_(n);  // so I free automatically on return
  int *on = &tmp_[0];
  for (i=0; i<n; i++) on[i] = 0;

  int istart=n; // first on atom, init to beyond end of list
  
  // find the first selected atom, and merge selection flags 
  // from all of the active reps to get the complete set
  // of visible atoms when computing the cov
  for (int j=0; j<repList.num(); j++) {
    const DrawMolItem *rep = repList[j];
    if (!rep->displayed())
      continue; // don't process hidden/non-visible reps

    if (rep->atomSel->selected > 0) { 
      const int first = rep->atomSel->firstsel;
      const int last = rep->atomSel->lastsel;
      if (first < istart)
        istart=first;

      const int *flgs = rep->atomSel->on;
      for (i=first; i<=last; i++)
        on[i] |= flgs[i];
    }
  }

  // if there are no selected atoms, use all atom coordinates
  if (istart < 0 || istart >= n) {
    istart = 0;
    for (i=0; i<n; i++) on[i] = 1;
  }

  // initialize min/max positions with values from the first on atom
  const float *mpos = ts->pos + 3L*istart;
  minposx = maxposx = mpos[0];
  minposy = maxposy = mpos[1];
  minposz = maxposz = mpos[2];
  covx = covy = covz = 0.0;

  int icount = 0;
  for (i=istart; i<n; ++i, mpos += 3) {
    if (!on[i]) continue;
    ++icount;
  
    const float xpos = mpos[0];
    const float ypos = mpos[1]; 
    const float zpos = mpos[2]; 

    covx += xpos;
    covy += ypos;
    covz += zpos;

    if (xpos < minposx) minposx = xpos;
    if (xpos > maxposx) maxposx = xpos;

    if (ypos < minposy) minposy = ypos;
    if (ypos > maxposy) maxposy = ypos; 

    if (zpos < minposz) minposz = zpos;
    if (zpos > maxposz) maxposz = zpos;
  }

  // set the center of volume variable now
  center[0] = covx; 
  center[1] = covy; 
  center[2] = covz; 
  vec_scale(center, 1.0f / icount, center);
 
  // calculate center-of-volume and scale factor
  scalefactor = maxposx - minposx;

  // prevent getting a zero-scaled scene when loading a single atom.
  if (scalefactor == 0.0) {
    scalefactor = 3.0;
  }

  if ((maxposx - minposx) > scalefactor)
    scalefactor = maxposx - minposx;
  if ((maxposy - minposy) > scalefactor)
    scalefactor = maxposy - minposy;
  if ((maxposz - minposz) > scalefactor)
    scalefactor = maxposz - minposz;

  scalefactor = 1.5f / scalefactor;
}

