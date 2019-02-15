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
 *	$RCSfile: DrawMolecule.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.87 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Displayable version of a molecule, derived from BaseMolecule and
 * Displayable.  This contains all the info for rendering the molecule.
 *
 ***************************************************************************/
#ifndef DRAWMOLECULE_H
#define DRAWMOLECULE_H

#include "BaseMolecule.h"
#include "Displayable.h"
#include "DrawMolItem.h"
#include "ResizeArray.h"
#include "WKFThreads.h"
#include "QuickSurf.h"

class AtomColor;
class AtomRep;
class AtomSel;
class VMDApp;
class MoleculeGraphics;
class DrawForce;

// XXX this macro enables code to allow the molecular orbital
// representations within the same molecule to reuse any existing
// rep's molecular orbital grid if the orbital ID and various 
// grid-specific parameters are all compatible.  This optimization
// short-circuits the need for a rep to compute its own grid if
// any other rep already has what it needs.  For large QM/MM scenes,
// this optimization can be worth as much as a 2X speedup when
// orbital computation dominates animation performance.
#define VMDENABLEORBITALGRIDBACKDOOR 1

/// A monitor class that acts as a proxy for things like labels that
/// have to be notified when molecules change their state.  
class DrawMoleculeMonitor {
public:
  // called with id of molecule
  virtual void notify(int) = 0;
  DrawMoleculeMonitor() {}
  virtual ~DrawMoleculeMonitor() {}
};


/// Subclass of BaseMolecule and Displayable for drawing a molecule
class DrawMolecule : public BaseMolecule, public Displayable {
public:
  int active;      ///< is this molecule active?  Used by MoleculeList.

  VMDApp *app;     ///< Needed by DrawMolItem, so that reps get can access 
                   ///< to GPU global memory management routines,
                   ///< shared QuickSurf objects, and other such routines


private:
  int repcounter;  ///< counter for giving unique names to reps.

  /// a MoleculeGraphics instance, which handles custom graphics primitives
  /// added to this molecule.
  MoleculeGraphics *molgraphics;

  /// a DrawForce instance for drawing force arrows
  DrawForce *drawForce;

#if defined(VMDENABLEORBITALGRIDBACKDOOR)
/// XXX hack to let orbital rep search for existing grids among 
///     all of the reps in this molecule...
public:
#endif
  /// all representations in this molecule
  ResizeArray<DrawMolItem *> repList;

#if defined(VMDENABLEORBITALGRIDBACKDOOR)
/// XXX hack to let orbital rep search for existing grids among 
///     all of the reps in this molecule...
private:
#endif
  /// timesteps owned by the molecule
  ResizeArray<Timestep *> timesteps;

  /// current frame
  int curframe;
 
  /// calculation of the secondary structure is done on the fly, but
  /// only if I need it do I run STRIDE
  int did_secondary_structure;

  /// list of registered monitors
  ResizeArray<DrawMoleculeMonitor *> monitorlist;

  /// Tell reps that we added or removed a timestep.  
  void addremove_ts();

  float center[3];          ///< center of volume position
  float scalefactor;        ///< cached scale factor.  Initially -1 = invalid
  
  /// recompute center and scalefactor from current coordinates.
  /// Include only selected atoms in displayed reps.
  void update_cov_scale();

  /// Force recalculation of center and scale on next access
  void invalidate_cov_scale();

public:
  /// constructor ... pass in a VMDApp * and a parent displayable. 
  DrawMolecule(VMDApp *, Displayable *);
  virtual ~DrawMolecule();

  
  //
  // public utility routines
  //
  wkf_threadpool_t * cpu_threadpool(void);

  /// Query CUDA device pool pointer
  wkf_threadpool_t * cuda_devpool(void);
 
  /// return whether the Nth atom is displayed.  This is true if ANY
  /// representation is displaying the given atom
  int atom_displayed(int);

  //
  // access routines for the drawn molecule components (which are the children)
  //
  
  /// total number of components
  int components(void) { return repList.num(); }
  
  /// return Nth component ... change to proper return type
  DrawMolItem *component(int);

  /// return the component that matches the given Pickable, or NULL if no match
  DrawMolItem *component_from_pickable(const Pickable *);

  /// retrieve the index of the component with the given name.  Returns -1 on 
  /// failure.
  int get_component_by_name(const char *);
 
  /// Get the name of the given component.  Names are initially "repN", where
  /// N starts at 0 and increases each time a rep is created.  Return NULL
  /// if the index is invalid.
  const char *get_component_name(int);
  

  /// delete the Nth representation ... return success
  int del_rep(int);

  /// Add a new representation (component).  This always succeeds,
  /// since all parameters must have already been verified as valid.
  /// The rep takes over ownership of the parameter objects (except Material).
  void add_rep(AtomColor *, AtomRep *, AtomSel *, const Material *);

  /// change the Nth representation ... return success.
  /// if any object is NULL, that characteristic is not changed.
  int change_rep(int, AtomColor *, AtomRep *, const char *sel);

  /// turn the Nth representation on or off.   Return success.
  int show_rep(int repid, int onoff);

  /// force a recalc of all representations
  /// For MOL_REGEN, this also invalidates the value of cov and scale_factor, 
  //causing them to be recomputed on the next access.
  void force_recalc(int);

  /// Tell reps that the periodic image parameters have been changed
  void change_pbc();

  /// Tell reps that the currently displayed timestep has changed
  void change_ts();

  /// Return the highlighted rep for this molecule.  Returns -1 if there is
  /// no such rep.
  int highlighted_rep() const;

  //
  // methods for dealing with frames
  //

  /// number of frames in the files associatd with the molecule
  int numframes() const { return timesteps.num(); }

  /// index of current frame
  int frame() const { return curframe; }

  /// change current frame without firing callbacks
  void override_current_frame(int frame);

  /// get the current frame
  Timestep *current() { 
      if (curframe >= 0 && curframe < timesteps.num())
          return timesteps[curframe];
      return NULL;
  }

  /// get the specifed frame
  Timestep *get_frame(int n) {
      if ( n>= 0 && n<timesteps.num() ) {
          return timesteps[n];
      }
      return NULL;
  }

  /// get the last frame
  Timestep *get_last_frame() {
      return get_frame(timesteps.num()-1);
  }

  /// delete the nth frame
  void delete_frame(int n);

  /// append the given frame
  void append_frame(Timestep *);

  /// duplicate the given frame
  /// passing NULL adds a 'null' frame (i.e. all zeros)
  void duplicate_frame(const Timestep *);

  /// scaling factor required to make the molecule fit within (-1 ... 1)
  float scale_factor();

  /// center of volume of this molecule.  Return success.  Fails if there 
  /// aren't any coordinates, graphics, or volumetric data sets to compute
  /// the cov from.
  int cov(float &, float &, float &);

  /// recalculate bonds via distance bond search based on current timestep
  int recalc_bonds(void);

  /// request ss calculation. Return success.
  int need_secondary_structure(int);

  /// invalidate current secondary structure when structure is changed
  void invalidate_ss();

  /// recalculate the secondary structure using current coordinates
  /// Return success.
  int recalc_ss();

  // return pointer to molgraphics so that MoleculeList::check_pickable
  // can test for it.
  MoleculeGraphics *moleculeGraphics() const { return molgraphics; }

  // prepare molecule for redraw
  virtual void prepare();

  /// register monitors
  void register_monitor(DrawMoleculeMonitor *);

  /// unregister monitors
  void unregister_monitor(DrawMoleculeMonitor *);

  void notify();
};

#endif

