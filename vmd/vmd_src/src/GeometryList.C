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
 *      $RCSfile: GeometryList.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.66 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * This maintains a set of lists of geometry monitors, and draws them to
 * the scene.  This is a Displayable which keeps the graphical representations
 * of the geometry monitors; this is also a Pickable, and creates and controls
 * the PickMode objects which create new geometry monitors via the pointers.
 * This object keeps a set of ResizeArray, each of which is a 'category' for
 * geometry monitors (i.e Atoms, Bonds, etc.) which contain GeometryMol objects. 
 *
 ***************************************************************************/

#include "GeometryList.h"
#include "GeometryAtom.h"
#include "GeometryBond.h"
#include "GeometryAngle.h"
#include "GeometryDihedral.h"
#include "GeometrySpring.h"
#include "DisplayDevice.h"
#include "DispCmds.h"
#include "MoleculeList.h"
#include "utilities.h"
#include "VMDApp.h"
#include "Scene.h"
#include "Inform.h"

// default colors to use
#define ATOMGEOMCOL  REGGREEN
#define BONDGEOMCOL     REGWHITE
#define ANGLEGEOMCOL  REGYELLOW
#define DIHEGEOMCOL  REGCYAN
#define TUGGEOMCOL  REGWHITE
#define SPRINGGEOMCOL  REGORANGE


////////////////////////  constructor  /////////////////////////
GeometryList::GeometryList(VMDApp *vmdapp, Displayable *disp)
  : Displayable(disp) {

  // indicate we don't yet have a color object to use
  colorCat = (-1);

  // save the list of molecules to use for data
  app = vmdapp;

  // default size of text labels is 1.0 (no scaling)
  labelsize = 1.0f;
  labelthickness = 1.0f;

  // create default lists for atom, bond, angle, and dihedral measurements
  add_geom_list("Atoms", ATOMGEOMCOL);
  add_geom_list("Bonds", BONDGEOMCOL);
  add_geom_list("Angles", ANGLEGEOMCOL);
  add_geom_list("Dihedrals", DIHEGEOMCOL);
  add_geom_list("Springs", SPRINGGEOMCOL);

  colorCat = scene->add_color_category("Labels");
  for (int i=0; i<num_lists(); i++) {
    scene->add_color_item(colorCat, geomLists.name(i), 
        geomLists.data(i)->defaultColor);
  }

  // Displayable characteristics
  rot_off();
  scale_off();
  glob_trans_off();
  cent_trans_off();
}


////////////////////////  destructor  ////////////////////////////
GeometryList::~GeometryList(void) {

  // for all the lists, delete all geometry monitors
  for(int i=(num_lists() - 1); i >= 0; i--)
    del_geom_list(i);
}


///////////////////////////  protected virtual routines  ///////////////////

void GeometryList::do_color_changed(int clr) {
  if (clr == colorCat) {
    for (int i=0; i<num_lists(); i++) {
      GeomListPtr glist = geom_list(i);
      GeomListStruct *s = geomLists.data(i);
      int ind = scene->category_item_index(colorCat, geom_list_name(i));
      int c = scene->category_item_value(colorCat, ind);
      s->curColor = c;
      for (int j=0; j<glist->num(); j++)
        (*glist)[j]->set_color(c);
    }
  }
}

///////////////////////////  public routines  ////////////////////////////

// add a new category: specify the name, and default color.  Return index of
// new list.
int GeometryList::add_geom_list(const char *nm, int clr) {

  // make sure we do not already have a category with this name
  int oldlist = geom_list_index(nm);
  if(oldlist >= 0)
    return oldlist;
    
  // create a new struct
  GeomListStruct *newlist = new GeomListStruct;
  newlist->geomList = new ResizeArray<GeometryMol *>(8);
  newlist->defaultColor = clr;
  newlist->curColor = clr;
  
  // add the new category to the big list
  return geomLists.add_name(nm, newlist);
}


// delete the Nth category.  Return success.
int GeometryList::del_geom_list(int n) {
  GeomListStruct *oldlist = NULL;

  // make sure we do have a category with this name
  if(n >= 0 && n < num_lists()) {
    // get data for Nth list
    oldlist = geomLists.data(n);
    GeomListPtr glist = oldlist->geomList;
    
    // go through the list and delete all current GeometryMol objects
    for(int i=(glist->num() - 1); i >= 0; i--) {
      delete (*glist)[i];
    }

    // delete the old list storage and structure
    delete glist;
    delete oldlist;
    geomLists.set_data(n, (GeomListStruct *) NULL);
  }

  // return whether we were successful
  return (oldlist != NULL);
}

int GeometryList::add_geometry(const char *geomcatname, const int *molids,
    const int *atomids, const int *cells, float k, int toggle) {

  // check that geometry category name is valid
  int geomcat = geom_list_index(geomcatname);
  GeometryMol *g = NULL;
  MoleculeList *mlist = app->moleculeList;
  if (geomcat == geom_list_index("Atoms")) 
    g = new GeometryAtom(*molids, *atomids, cells, mlist, app->commandQueue, this);
  else if (geomcat == geom_list_index("Bonds"))
    g = new GeometryBond((int *)molids, (int *)atomids, cells, mlist, app->commandQueue, this);
  else if (geomcat == geom_list_index("Angles"))
    g = new GeometryAngle((int *)molids, (int *)atomids, cells, mlist, app->commandQueue, this);
  else if (geomcat == geom_list_index("Dihedrals"))
    g = new GeometryDihedral((int *)molids, (int *)atomids, cells, mlist, app->commandQueue, this);
  else if (geomcat == geom_list_index("Springs"))
    g = new GeometrySpring((int *)molids, (int *)atomids, mlist, app->commandQueue, k, this);
  else {
    msgErr << "Unknown geometry category '" << geomcatname << "'." << sendmsg;
    return -1;
  }
  if(g && g->ok()) {
    GeomListPtr glist = geom_list(geomcat);

    // if there is already an identical label in the list,
    // do not add this one, instead toggle the displayed
    // status of the old one and return the index.
    for(int i=(glist->num() - 1); i >= 0; i--) {
      GeometryMol *g2 = (*glist)[i];
      if(!strcmp(g2->unique_name(), g->unique_name())) {
        // name matches
        if (toggle) {
          if (g2->displayed())
            g2->off();
          else
            g2->on();
        }
      
        delete g;
        return i;
      }
    }

    // spam the console
    msgInfo << "Added new " << geomcatname << " label " << g->name();
    if (g->has_value())
      msgInfo << " = " << g->calculate();
    msgInfo << sendmsg;

    // add the geometry object
    glist->append(g);
    
    // calculate the value for the first time
    g->calculate();

    // set the color, which also causes the display list to be generated
    g->set_color(geomLists.data(geomcat)->curColor);
    g->set_text_size(labelsize);
    g->set_text_thickness(labelthickness);

    // set the pick variables
    g->set_pick();

    // indicate we were successful; return index of this object
    return glist->num() - 1;
  }
  
  // if here, something did not work
  return -1;
}


// del a new geometry object from the list with the given index. Return success.
// if n < 0, delete all markers
int GeometryList::del_geometry(int glindex, int n) {

  // make sure the given GeometryMol object is OK
  if(glindex >= 0 && glindex < num_lists()) {

    // get the list of geometry objects for the given index
    GeomListPtr glist = geom_list(glindex);

    // make sure the geometry index is ok
    if(n >= 0 && n < glist->num())  {
      // delete and remove the geometry object
      delete (*glist)[n];
      glist->remove(n);
      
      // indicate we were successful
      return TRUE;
    } else if(n < 0) {
      // delete and remove all geometry objects
      for(int j=(glist->num() - 1); j >= 0; j--) {
        delete (*glist)[j];
        glist->remove(j);
      }
      
      // indicate we were successful
      return TRUE;
    }
  }
  
  // if here, something did not work
  return FALSE;
}

// toggle whether to show or hide a geometry monitor.  If the monitor 
// specified is < 0, does so for all monitors in the given category
int GeometryList::show_geometry(int glindex, int n, int s) {

  // make sure the given GeometryMol object is OK
  if(glindex >= 0 && glindex < num_lists()) {
    
    // get the list of geometry objects for the given index
    GeomListPtr glist = geom_list(glindex);

    // make sure the geometry index is ok
    if(n >= 0 && n < glist->num())  {
      // hide or show the specified object
      if (s)
        (*glist)[n] -> on();
      else 
        (*glist)[n] -> off();

      
      // indicate we were successful
      return TRUE;
    } else if(n < 0) {
      // delete and remove all geometry objects
      for(int j=(glist->num() - 1); j >= 0; j--) {
        if (s)
          (*glist)[j] -> on();
        else
          (*glist)[j] -> off();
      }
     
      
      // indicate we were successful
      return TRUE;
    }
  }
  
  // if here, something did not work
  return FALSE;
}

int GeometryList::setTextSize(float newsize) {
  if (newsize <= 0)
    return FALSE;

  if (newsize == labelsize) return TRUE;

  labelsize = newsize;
  for(int i=(num_lists() - 1); i >= 0; i--) {
    GeomListPtr glist = geom_list(i);
    for(int j=(glist->num() - 1); j >= 0; j--) {
      GeometryMol *g = (*glist)[j];
      g->set_text_size(newsize);
    }
  }

  return TRUE;
}

int GeometryList::setTextThickness(float newthick) {
  if (newthick <= 0)
    return FALSE;

  if (newthick == labelthickness) return TRUE;

  labelthickness = newthick;
  for(int i=(num_lists() - 1); i >= 0; i--) {
    GeomListPtr glist = geom_list(i);
    for(int j=(glist->num() - 1); j >= 0; j--) {
      GeometryMol *g = (*glist)[j];
      g->set_text_thickness(newthick);
    }
  }

  return TRUE;
}

const float *GeometryList::getTextOffset(const char *nm, int n) {
  GeomListPtr glist = geom_list(nm);
  if (!glist) return NULL;
  if (n < 0 || n >= glist->num()) return NULL;
  GeometryMol *g = (*glist)[n];
  return g->text_offset();
}

int GeometryList::setTextOffset(const char *nm, int n, const float delta[2]) {

  GeomListPtr glist = geom_list(nm);
  if (!glist) return FALSE;
  if (n < 0 || n >= glist->num()) return FALSE;
  GeometryMol *g = (*glist)[n];
  g->set_text_offset(delta);
  return TRUE;
}

const char *GeometryList::getTextFormat(const char *nm, int n) {
  // XXX  get/set Text Offset/Format duplicate their first four lines...
  GeomListPtr glist = geom_list(nm);
  if (!glist) return NULL;
  if (n < 0 || n >= glist->num()) return NULL;
  GeometryMol *g = (*glist)[n];
  return g->text_format();
}

int GeometryList::setTextFormat(const char *nm, int n, const char *format) {
  GeomListPtr glist = geom_list(nm);
  if (!glist) return FALSE;
  if (n < 0 || n >= glist->num()) return FALSE;
  GeometryMol *g = (*glist)[n];
  g->set_text_format(format);
  return TRUE;
}

///////////////////////  public virtual routines  ////////////////////////

// prepare for drawing ... do any updates needed right before draw.
// For now, this always recreates the display list
void GeometryList::prepare() {

  // go through all the geometry objects, recalculate, and find out if
  // something is no longer 'ok'.  If not, delete it.
  for(int i=(num_lists() - 1); i >= 0; i--) {
    GeomListPtr glist = geom_list(i);
    for(int j=(glist->num() - 1); j >= 0; j--) {
      GeometryMol *g = (*glist)[j];
      if(!g->ok()) {
        del_geometry(i, j);
      }
    }
  }
}

