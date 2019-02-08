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
 *	$RCSfile: Displayable.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.117 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Base class for all objects which are drawn in the DisplayDevice.
 *
 ***************************************************************************/

#include <string.h>
#include "Displayable.h"
#include "DispCmds.h"
#include "PickList.h"
#include "utilities.h"
#include "VMDApp.h"
#include "DisplayDevice.h"

/*
 Notes about how displayables work
 ---------------------------------
 1. Each Displayable contains an array of drawing/display commands.
    This array actually has some character-sized flags at the beginning, and
    then a sequential list of commands.  The format for the cmdList is:
	Integer: On (T) or Off (F)
	Integer: Fixed (T) or Free (F)
	Push Matrix command
	Multiply Matrix (with this objs trans matrix) command
	... all commands for this object
	Pop Matrix command

  2. Each command has the form (as set up in DispCmds.h):
	command code (integer)
	data for command (0 or more bytes)

  3. A Displayable 'registers' with one or more Scene objects; this
     means it gives the display command list pointer to the Scene, which then
     uses that to draw the object.  Before drawing, however, the Scene calls
     the 'prepare' routine for each Displayable.  When deleted, the Displayable
     'unregisters' with all Scene objs it has previously registered with.
     If a Scene is deleted, it unregisters the Displayables in it's possession.

  4. A Displayable is either a 'parent' one or a 'child' one.  The
     difference is that parent Displayables register with the scene, and have
     only one transformation; children do NOT register with the scene, and have
     not only their main transformation (which is the same as the parent) but
     also a second one which multiplies the first.  Note that children can have
     children of their own, but are not 'parents', i.e. child Displayables do 
     not register with a scene even if they have children of their own.
  5. Children do register their DISPLAY LISTS, just not themselves
     (thus, they are not prepared, etc.; just rendered.)
  6. Children are created as normal, but with the parent Displayable
     specified instead of the Scene.  As part of the creation, the child will 
     be added to the parent's list via the 'add child' routine.
*/


void *Displayable::operator new(size_t n) {
  return vmd_alloc(n);
}

void Displayable::operator delete(void *p, size_t) {
  vmd_dealloc(p);
}

Displayable::Displayable(Scene *s) : scene(s) {
  // Initialize scaling and other transformations, since we're a parent 
  parent = NULL;
  scale = 1;
  globt[0] = globt[1] = globt[2] = 0;
  centt[0] = centt[1] = centt[2] = 0;
  
  // get values for this items as default values
  isOn = TRUE;
  doCent = doRot = doGlob = doScale = TRUE;
  do_create();
}

Displayable::Displayable(Displayable *pops)  : scene(pops->scene) {

  _needUpdate = 1; // default to needing an update
  parent = pops;

  // get copies of all of parents tranformation matrices from parent
  vec_copy(centt, parent->centt);
  rotm = parent->rotm;
  vec_copy(globt, parent->globt);
  scale = parent->scale;
  tm = parent->tm;

  isOn = parent->displayed();
  doCent = parent->doCent;
  doRot = parent->doRot;
  doGlob = parent->doGlob;
  doScale = parent->doScale;

  // do common creation action
  do_create();
  
  // take initial material settings from parent
  cmdList->ambient = parent->cmdList->ambient;
  cmdList->specular = parent->cmdList->specular;
  cmdList->diffuse = parent->cmdList->diffuse;
  cmdList->shininess = parent->cmdList->shininess;
  cmdList->mirror = parent->cmdList->mirror;
  cmdList->opacity = parent->cmdList->opacity;
  cmdList->outline = parent->cmdList->outline;
  cmdList->outlinewidth = parent->cmdList->outlinewidth;
  cmdList->transmode = parent->cmdList->transmode;
  cmdList->materialtag = parent->cmdList->materialtag;

  // inherit cacheability from parent
  cmdList->cacheskip = parent->cmdList->cacheskip;

  // finally, add this Displayable as a child to the parent
  parent->add_child(this);
}


// does all the creation work after variables have been initialized
void Displayable::do_create() {

  children = (Displayable **)vmd_alloc(16L*sizeof(Displayable*));
  num_children = 0;
  max_children = 16;

  // initialize the display command list 
  cmdList = new VMDDisplayList;

  // initialize flags and scalar settings
  needMatrixRecalc = TRUE;
  isFixed = FALSE;

  // initialize display list
  cmdList->mat = tm;

}

// reset the display command list; remove all current commands
void Displayable::reset_disp_list(void) {
  _needUpdate = 1;

  // Must use a unique rep serial number, so that display list caching
  // works correctly in the rendering code
  cmdList->reset_and_free(VMDApp::get_repserialnum()); 
}

// destructor; free up allocated space
Displayable::~Displayable(void) {
  // delete all children still around; also unregistered them
  while(num_children > 0)
    // delete first child object, the child destructor then removes the
    // child from this parent's list, until there are no more
    delete child(0);

  cmdList->reset_and_free(0); // free space allocated for disp storage
  delete cmdList;             // free the cmdList itself back to scene memory

  // if this is a child, remove it from it's parent's list of children
  if (parent)
    parent->remove_child(this);

  vmd_dealloc(children);
}


///////////////////////////  protected routines 

// recalculate the transformation matrix, and replace matrix in cmdList
// This is composed of these operations (applied as R to L):
//	TM =  GlobalTrans * Scale * Rotation * CenterTrans
void Displayable::recalc_mat(void) {
  if (needMatrixRecalc) {
    _needUpdate = 1;
    tm.identity();
    tm.translate(globt);
    tm.multmatrix(rotm);
    tm.scale(scale);
    tm.translate(centt);
    // reload this matrix in the display command list
    cmdList->mat = tm;

    needMatrixRecalc = FALSE;
  }

  // recalc matrix for all children
  for (int i=0; i < num_children; i++)
    child(i)->recalc_mat();
}

///////////////////////////  public routines 

// turn this object on or off
void Displayable::off(void) { 
  isOn = FALSE;
  _needUpdate = 1;
}

void Displayable::on(void) { 
  isOn = TRUE;
  _needUpdate = 1;
}
  
// add the given Displayable as a child (assuming it is one)
void Displayable::add_child(Displayable *d) {
    
  // append child to list of children
  children[num_children++] = d;
  if (num_children == max_children) {
    void *tmp = vmd_alloc(max_children*2L*sizeof(Displayable*));
    memcpy(tmp,children,max_children*sizeof(Displayable*));
    vmd_dealloc(children);
    children = (Displayable **)tmp;
    max_children *= 2;
  } 
}


// remove the given Displayable as a child. return success.
int Displayable::remove_child(Displayable *d) {
  // remove first child that matches the pointer, if available.
  int n = child_index(d);
  if (n >= 0) {
    // copy the entries from children+n+1  
    for (int i=n; i<num_children-1; i++) {
      children[i] = children[i+1];
    }
    num_children--;
    _needUpdate = 1;
    return TRUE;
  }
  return FALSE;
}

//
// prepare/update routines
//

// prepare for drawing, called by draw_prepare, supplied by derived class.
void Displayable::prepare() { }
  
// prepare to draw; possibly recalc the trans matrix, and do particular preps
int Displayable::draw_prepare() {
  int needupdate;

  if (parent == NULL)
    recalc_mat();    // update matrix if this is a parent Displayable
    
  prepare();         // do derived class preparations for this object

  needupdate = _needUpdate; // cache update state before we clear it

  // prepare child displayables; done after the parent has been prepared.
  for (int i=0; i < num_children; i++)
    needupdate |= child(i)->draw_prepare();

  // set the _needUpdate flag to zero _after_ all children have been updated
  // so that they can check (through needUpdate() if their parent has been
  // updated.  DrawForce currently uses this to determine whether to redraw. 
  _needUpdate = 0;          // once we've been prepared, we're ready to draw
  return needupdate; // return whether this or child displayables need updating
}

// do the actual drawing
void Displayable::draw(DisplayDevice *d) const {
  // only render myself and my children if parent is turned on
  if (isOn) {
    d->render(cmdList);
    for (int i=0; i<num_children; i++)
      child(i)->draw(d);
  }
}


//
// commands to change transformation
//

// reset the transformation to the identity matrix
void Displayable::reset_transformation(void) {
  // only reset if we're not fixed and given operations are allowed
  if (scaling())           scale=1;
  if (rotating())          { rotm.identity(); }
  if (glob_translating())  globt[0] = globt[1] = globt[2] = 0;
  if (cent_translating())  centt[0] = centt[1] = centt[2] = 0;	
  need_matrix_recalc();

  // do reset for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->reset_transformation();
}

void Displayable::set_scale(float s) {
  if (fixed())  return;		// only transform unfixed objects

  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->set_scale(s);

  if (!scaling())  return;
  scale = s;
  need_matrix_recalc();
}

void Displayable::mult_scale(float s) {
  if (fixed())  return;		// only transform unfixed objects

  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->mult_scale(s);

  if (!scaling())  return;
  scale *= s;
  need_matrix_recalc();
}

void Displayable::add_rot(float x, char axis) {
  if (fixed())  return;		// only transform unfixed objects

  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->add_rot(x, axis);

  if (!rotating())  return;

  // Need to apply the new rotation first
  Matrix4 mat;
  mat.rot(x, axis);
  mat.multmatrix(rotm);
  rotm = mat;
  need_matrix_recalc();
}

void Displayable::set_rot(float x, char axis) {
  if (fixed())  return;		// only transform unfixed objects

  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->set_rot(x, axis);

  if (!rotating())  return;
  // apply rotation to identity, and then multiply this by old rot matrix
  rotm.identity();
  rotm.rot(x,axis);
  need_matrix_recalc();
}

void Displayable::add_rot(const Matrix4 &m) {
  if (fixed())  return;		// only transform unfixed objects

  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->add_rot(m);

  if (!rotating())  return;
  Matrix4 mat(m);
  mat.multmatrix(rotm);
  rotm = mat;
  need_matrix_recalc();
}

void Displayable::set_rot(const Matrix4 &m) {
  if (fixed())  return;		// only transform unfixed objects

  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->set_rot(m);

  if (!rotating())  return;
  rotm = m;
  need_matrix_recalc();
}

void Displayable::set_glob_trans(float x, float y, float z) {
  if (fixed())  return;		// only transform unfixed objects

  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->set_glob_trans(x, y, z);

  if (!glob_translating())  return;
  globt[0] = x;
  globt[1] = y;
  globt[2] = z;
  need_matrix_recalc();
}

void Displayable::add_glob_trans(float x, float y, float z) {
  if (fixed())  return;		// only transform unfixed objects

  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->add_glob_trans(x, y, z);

  if (!glob_translating())  return;
  globt[0] += x;
  globt[1] += y;
  globt[2] += z;
  need_matrix_recalc();
}

void Displayable::set_cent_trans(float x, float y, float z) {
  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->set_cent_trans(x, y, z);

  if (!cent_translating())  return;
  centt[0] = x;
  centt[1] = y;
  centt[2] = z;
  need_matrix_recalc();
}

void Displayable::add_cent_trans(float x, float y, float z) {
  // do trans for all children as well
  for (int i=0; i < num_children; i++)
    child(i)->add_cent_trans(x, y, z);

  if (!cent_translating())  return;
  centt[0] += x;
  centt[1] += y;
  centt[2] += z;
  recalc_mat();
}

void Displayable::change_center(float x, float y, float z)
{
    // Here's the math:
    //  T = global translation (offset) matrix
    //  M = scaling*rotation matrix
    //  C = centering (offset) matrix
    //  p = picked point
    //  x = any point
    // and let G = T*M*C, the global transformation

    // the current transformation is: T*M*C * p
    // I want a new T', C' such that
    //   C' * p = {0 0 0 1}
    // and
    //   T'*M*C' * x = T M C x

    // JRG: Here's my new math:
    // C' * p = {0 0 0 1} so T' M C' p = G p = T' M {0 0 0 1}
    // Hence T' = translate( G*p - M*{0 0 0 1} )
    // and we don't need any inverses

  float p[4], g[4], ident[4], m[4];
  ident[0]=0.0; ident[1] = 0.0; ident[2]=0.0; ident[3]=1.0;
  p[0]=x; p[1]=y; p[2]=z; p[3]=1.0;

  // Set g = G * p
  tm.multpoint4d(p,g);

  // Set m = M * {0 0 0 1}
  Matrix4 M(rotm);
  M.scale(scale);
  M.multpoint4d(ident, m);

    // and apply the result
  set_cent_trans(-x, -y, -z);

  // Set Tprime = translate(g - m)
  set_glob_trans(g[0]-m[0], g[1]-m[1], g[2]-m[2]);
}

void Displayable::change_material(const Material *mat) {
  _needUpdate = 1;
  cmdList->ambient = mat->ambient;
  cmdList->specular = mat->specular;
  cmdList->diffuse = mat->diffuse;
  cmdList->shininess = mat->shininess;
  cmdList->mirror = mat->mirror;
  cmdList->opacity = mat->opacity;
  cmdList->outline = mat->outline;
  cmdList->outlinewidth = mat->outlinewidth;
  cmdList->transmode = mat->transmode;
  cmdList->materialtag = mat->ind;
}

void Displayable::cacheskip(int onoff) {
  cmdList->cacheskip = onoff;
}

int Displayable::curr_material() const {
  return cmdList->materialtag;
}

void Displayable::update_material(const Material *mat) {
  if (mat->ind == curr_material()) change_material(mat);
  for (int i=0; i<num_children; i++) children[i]->update_material(mat);
}

void Displayable::delete_material(int n, const MaterialList *mlist) {
  if (n == curr_material()) {
    change_material(mlist->material(0)); // 0th material can't be deleted 
  }
  for (int i=0; i<num_children; i++) children[i]->delete_material(n, mlist);
}

//
// routines for working as a Pickable
//
  
// return our list of draw commands with picking draw commands in them
VMDDisplayList *Displayable::pick_cmd_list(void) {
  return cmdList;
}

// return whether the pickable object is being displayed
int Displayable::pickable_on(void) {
  return displayed();
}

void Displayable::color_changed(int cat) {
  do_color_changed(cat);
  for (int i=0; i<num_children; i++) child(i)->color_changed(cat);
}

void Displayable::color_rgb_changed(int color) {
  do_color_rgb_changed(color);
  for (int i=0; i<num_children; i++) child(i)->color_rgb_changed(color);
}

void Displayable::color_scale_changed() {
  do_color_scale_changed();
  for (int i=0; i<num_children; i++) child(i)->color_scale_changed();
}

