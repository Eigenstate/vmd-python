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
 *      $RCSfile: MaterialList.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.36 $      $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Material properties list structure, master list of all materials
 ***************************************************************************/

#include "MaterialList.h"
#include "Displayable.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static const Material defOpaque = {
  0.00f, // ambient
  0.50f, // specular
  0.65f, // diffuse
 40.00f, // shininess 
  0.00f, // mirror    
  1.00f, // opacity
  0.00f, // outline
  0.00f, // outlinewidth
  0.00f, // transmode
  0      // index
};

static const Material defTransp = {
  0.00f, // ambient
  0.50f, // specular
  0.65f, // diffuse
 40.00f, // shininess 
  0.00f, // mirror    
  0.30f, // opacity
  0.00f, // outline
  0.00f, // outlinewidth
  0.00f, // transmode
  1      // index
};


// raw shininess values will run from 1 to 1000, but be scaled from 0 to 1
// in the user interface.  If y is the internal value and x is the 
// external (user interface value), then y = A exp(k * (x-0.5)), where...
static const double SHINY_A = sqrt(1000.0);  // ...and...
static const double SHINY_k = log(1000.0);

MaterialList::MaterialList(Displayable *rootd) : root(rootd) {
  mlist = new NameList<Material *>;
  Material *opaque, *transp;
  opaque = new Material;
  transp = new Material;
  memcpy(opaque, &defOpaque, sizeof(Material));
  memcpy(transp, &defTransp, sizeof(Material));
  
  // add to list 
  mlist->add_name("Opaque",opaque);
  mlist->add_name("Transparent",transp);

  // the update_material makes all child displayables update as well; this
  // is needed in the case of Scene::root to make sure that Scene::background
  // is properly initialized.
  root->update_material(material(0));
  matcounter = 2;
}

void MaterialList::notify(int ind) {
  const Material *mat = mlist->data(ind);
  root->update_material(mat);
}

MaterialList::~MaterialList() {
  for (int i=0; i<mlist->num(); i++) delete mlist->data(i);
  delete mlist;
}

void MaterialList::set_name(int i, const char *nm) {
  mlist->set_name(i, nm);
}

void MaterialList::set_ambient(int i, float f) {
  if (i<0 || i >= num()) return;
  mlist->data(i)->ambient = f;
  notify(i);
}
 
void MaterialList::set_specular(int i, float f) {
  if (i<0 || i >= num()) return;
  mlist->data(i)->specular = f;
  notify(i);
}
 
void MaterialList::set_diffuse(int i, float f) {
  if (i<0 || i >= num()) return;
  mlist->data(i)->diffuse = f;
  notify(i);
}
 
void MaterialList::set_shininess(int i, float f) {
  if (i<0 || i >= num()) return;
  double val = SHINY_A * exp(SHINY_k * (f-0.5));
  mlist->data(i)->shininess = (float) val;
  notify(i);
}

void MaterialList::set_mirror(int i, float f) {
  if (i<0 || i >= num()) return;
  mlist->data(i)->mirror = f;
  notify(i);
}
 
void MaterialList::set_opacity(int i, float f) {
  if (i<0 || i >= num()) return;
  mlist->data(i)->opacity = f;
  notify(i);
}

void MaterialList::set_outline(int i, float f) {
  if (i<0 || i >= num()) return;
  mlist->data(i)->outline = f;
  notify(i);
}

void MaterialList::set_outlinewidth(int i, float f) {
  if (i<0 || i >= num()) return;
  mlist->data(i)->outlinewidth = f;
  notify(i);
}

void MaterialList::set_transmode(int i, float f) {
  if (i<0 || i >= num()) return;
  mlist->data(i)->transmode = f;
  notify(i);
}
 
const char *MaterialList::add_material(const char *newname, 
                                       const char *copyfrom) {
  int copyind = 0;
  if (copyfrom) {
    copyind = material_index(copyfrom);
    if (copyind < 0) return NULL;  // material to copy from does not exist
  }
  char buf[20];
  const char *name = newname;
  if (!name) {
    // find a new unique name.
    int id = mlist->num();
    do {
      sprintf(buf, "Material%d", id++);
      name = buf;
    } while (mlist->typecode(name) >= 0);
  } else {
    if (mlist->typecode(name) >= 0)
      return NULL; // name already present
  }
  Material *newmat = new Material;
  memcpy(newmat, mlist->data(copyind), sizeof(Material));
  newmat->ind = matcounter++;
  int ind = mlist->add_name(name, newmat);
  return mlist->name(ind);
} 

float MaterialList::get_ambient(int i) {
  return mlist->data(i)->ambient;
}

float MaterialList::get_specular(int i) {
  return mlist->data(i)->specular;
}

float MaterialList::get_diffuse(int i) {
  return mlist->data(i)->diffuse;
}

float MaterialList::get_shininess(int i) {
  double val = mlist->data(i)->shininess;
  return ((float) (log(val/SHINY_A)/SHINY_k + 0.5f)); 
}

float MaterialList::get_mirror(int i) {
  return mlist->data(i)->mirror;
}

float MaterialList::get_opacity(int i) {
  return mlist->data(i)->opacity;
}

float MaterialList::get_outline(int i) {
  return mlist->data(i)->outline;
}

float MaterialList::get_outlinewidth(int i) {
  return mlist->data(i)->outlinewidth;
}

float MaterialList::get_transmode(int i) {
  return mlist->data(i)->transmode;
}

int MaterialList::delete_material(int n) {
  int i, num = mlist->num();
  if (n < 2 || n >= num) 
    return FALSE;

  NameList<Material *> *newlist = new NameList<Material *>;
  for (i=0; i<num; i++) {
    Material *mat = mlist->data(i);
    if (i == n) {
      delete mat;
    } else {
      mat->ind = newlist->num();   
      newlist->add_name(mlist->name(i), mat); 
    }
  }
  delete mlist;
  mlist = newlist;

  // notify displayables that material n has been deleted.  Pass reference
  // to self so that the displayable can get a new material if it's old
  // one was deleted.
  root->delete_material(n, this);
  return TRUE;
}

int MaterialList::restore_default(int ind) {
  if (ind == 0 || ind == 1) {
    Material *mat = mlist->data(ind);
    const Material *from = (ind == 0) ? &defOpaque : &defTransp;
    memcpy(mat, from, sizeof(Material));
    notify(ind);
    return TRUE;
  }
  return FALSE;
}

