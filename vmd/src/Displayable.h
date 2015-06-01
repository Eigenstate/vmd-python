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
 *	$RCSfile: Displayable.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.86 $	$Date: 2010/12/16 04:08:12 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Base class for all objects which are drawn in the DisplayDevice.
 * Each Displayable is also a linked list of other Displayables, which act
 * as 'children' to the parent, i.e. they get the parent's tranformation as
 * well as their own.
 *
 ***************************************************************************/
#ifndef DISPLAYABLE_H
#define DISPLAYABLE_H

#include "Matrix4.h"
#include "ResizeArray.h"
#include "Pickable.h"
#include "utilities.h"
#include "VMDDisplayList.h"
#include "MaterialList.h"

class DisplayDevice;
class Scene;

/// Base class for all objects which are drawn by a DisplayDevice
class Displayable : public Pickable {
public:
  /// Everything in this class can be accessed by the draw function,
  /// so all data must be allocated in Scene address space.  These
  /// allocators handled specially in the CAVE and FreeVR subclasses
  void *operator new(size_t);
  void operator delete(void *, size_t);
 
  /// components of the transformation ... centering, rotating, global trans
  Matrix4 rotm;    ///< Rotation matrix       (R)
  float globt[3];  ///< Global Translation    (GT)
  float centt[3];  ///< Centering translation (CT)
  float scale;     ///< Scaling               (S)
  Matrix4 tm;      ///< concatenated transformation matrix (GT * S * R * CT)
  
  /// are all child Displayables current?
  int needUpdate(void) { return _needUpdate; } 

private:
  /// flag set to TRUE when updates are needed and FALSE when render() is called
  int _needUpdate;

  /// do we need to recalculate the trans matrix in the next call to 'prepare'?
  int needMatrixRecalc;

  /// does creation work common to parent and child displayables.
  void do_create();

protected:
  /// The list of display commands
  VMDDisplayList *cmdList;

  /// The Scene object where color data can be accessed.
  Scene *scene;

  /// append a DispCmd code with no data to the cmdList.
  void append(int d) { cmdList->append(d, 0); }

  /// which of the following operations will this object listen to
  int doCent, doRot, doGlob, doScale;

  /// is the object free to be affected by rotations/translations/etc?
  int isFixed;
  
  /// is the object to be drawn or not?
  int isOn;

  /// recalculate the transformation matrix, and replace matrix in cmdList
  void recalc_mat(void);

  /// our parent Displayable; if NULL, this is a parent Displayable
  Displayable *parent;

  /// list of all children Displayable's
  Displayable **children;
  int num_children;
  int max_children;

  virtual void do_color_changed(int cat) {}
  virtual void do_color_rgb_changed(int color) {}
  virtual void do_color_scale_changed() {}

public:
  /// constructor: specify the parent Displayable, which may NOT be NULL
  Displayable(Displayable *);

  /// alternative constructor for root Displayable; no parent
  Displayable(Scene *);

  /// destructor: delete all children as well 
  virtual ~Displayable(void);

  /// reset the display command list; remove all current commands
  void reset_disp_list(void);

  /// signal that a reset of the trans matrix is needed next 'prepare' cycle.
  void need_matrix_recalc(void) { needMatrixRecalc = TRUE; }

  //
  // routines for working as a Pickable
  //
  
  /// return our list of draw commands with picking draw commands in them
  virtual VMDDisplayList *pick_cmd_list(void);

  /// return whether the pickable object is being displayed
  virtual int pickable_on(void);
      
  /// Recompute color indices if you're using the given color category
  void color_changed(int cat);

  /// A color's rgb value has been redefined
  void color_rgb_changed(int color);

  /// The color scale has been redefined
  void color_scale_changed();

  //
  // deal with child Displayable's
  //

  /// return the Nth child displayable pointer
  Displayable *child(int N) const { return children[N]; }
  
  /// return the index of the given child displayable pointer, or (-1) if none.
  int child_index(Displayable *d) { 
    for (int i=0; i<num_children; i++) 
      if (d == child(i)) return i;
    return -1; 
  }

  /// add the given Displayable as a child (assuming it is one)
  void add_child(Displayable *);

  /// remove specified child displayable, does not delete it.  return success.
  int remove_child(Displayable *);

  /// remove specified child displayable, does not delete it.  return success.
  int remove_child(int N) { return remove_child(child(N)); }

  int displayed(void) const {  ///< is a displayable displayed
    return isOn;
  }
  void off(void);            ///< turn displayable off
  void on(void);             ///< turn displayable on

  // make the object fixed (not responsive to scale/rot/trans commands
  int fixed(void) const { return isFixed; }
  void fix(void)   { isFixed = TRUE;  }
  void unfix(void) { isFixed = FALSE; }

  //
  // preparation/update routines
  //
  
  /// update geometry before drawing, called by Scene to prepare all objects
  // Return 0 if we do not require a redraw, or 1 if we do. 
  int draw_prepare();

  /// specific preparations, called by draw_prepare, supplied by derived class
  virtual void prepare();

  /// call DisplayDevice::render() on the list, then draw children recursively
  void draw(DisplayDevice *) const;

  //
  // command to set whether to be affected by particular trans routines
  //
  void scale_on(void) { doScale = TRUE; }
  void scale_off(void) { doScale = FALSE; }
  int scaling(void) const { return doScale && !fixed(); }
  
  void rot_on(void) { doRot = TRUE; }
  void rot_off(void) { doRot = FALSE; }
  int rotating(void) const { return doRot && !fixed(); }

  void cent_trans_on(void) { doCent = TRUE; }
  void cent_trans_off(void) {  doCent = FALSE; }
  int cent_translating(void) const { return doCent; }

  void glob_trans_on(void) { doGlob = TRUE; }
  void glob_trans_off(void) { doGlob = FALSE; }
  int glob_translating(void) const { return doGlob && !fixed(); }

  //
  // command to change transformation
  //

  /// reset to identity matrix, virtual so resets affect other factors as well
  virtual void reset_transformation(void);

  void set_scale(float s);           ///< set the scale factor
  void mult_scale(float s);          ///< multiply the existing scale factor
  
  void add_rot(float x, char axis);  ///< add a rotation to the specified axis
  void add_rot(const Matrix4 &);     ///< concatenate in a new rotation
  void set_rot(float x, char axis);  ///< set the rotation on a given axis
  void set_rot(const Matrix4 &);     ///< set the rotation matrix

  void set_glob_trans(float, float, float); ///< set the global translation
  void add_glob_trans(float, float, float); ///< add to the global translation

  void set_cent_trans(float, float, float); ///< set the centering transform
  void add_cent_trans(float, float, float); ///< add to the centering transform

  /// change centt and globt so (x,y,z) is in the center and
  /// tm(old) * (x,y,z) = tm(new) * (x,y,z);
  void change_center(float x, float y, float z);

  void cacheskip(int onoff);         ///< whether to skip display list caching

  // 
  // Material functions 
  // 
  void change_material(const Material *);
  int curr_material() const;
  void update_material(const Material *mat);
  void delete_material(int n, const MaterialList *);

  //
  // Clipping plane functions; these are just wrappers for the VMDDisplayList
  // methods
  //
  const VMDClipPlane *clipplane(int i) {
    return cmdList->clipplane(i);
  }
  int set_clip_center(int i, const float *center) {
    int rc = cmdList->set_clip_center(i, center);
    if (rc) _needUpdate = 1;
    return rc;
  }
  int set_clip_normal(int i, const float *normal) {
    int rc = cmdList->set_clip_normal(i, normal);
    if (rc) _needUpdate = 1;
    return rc;
  }
  int set_clip_color(int i, const float *color) {
    int rc = cmdList->set_clip_color(i, color);
    if (rc) _needUpdate = 1;
    return rc;
  }
  int set_clip_status(int i, int mode) {
    int rc = cmdList->set_clip_status(i, mode);
    if (rc) _needUpdate = 1;
    return rc;
  } 
};

#endif

