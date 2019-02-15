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
 *      $RCSfile: GeometryList.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.43 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * This maintains a set of lists of geometry monitors, and draws them to
 * the scene.  This is a Displayable which keeps the graphical representations
 * of the geometry monitors; this is also a Pickable, and creates and controls
 * the PickMode objects which create new geometry monitors via the pointers.
 * This object keeps a set of ResizeArray, each of which is a 'category' for
 * geometry monitors (i.e Atoms, Bonds, etc.) containing GeometryMol objects.
 *
 ***************************************************************************/
#ifndef GEOMETRYLIST_H
#define GEOMETRYLIST_H

#include "Displayable.h"
#include "NameList.h"
#include "ResizeArray.h"
#include "GeometryMol.h"
#include "DispCmds.h"
class VMDApp;


/// structure used to store data about each category
typedef ResizeArray<GeometryMol *>  *GeomListPtr;

/// structure used to store data about each category
typedef struct {
  GeomListPtr geomList;
  int defaultColor;
  int curColor;
} GeomListStruct;


/// Displayable subclass to maintain geometry monitors, and draw them to a scene
class GeometryList : public Displayable {
private:
  /// list of ResizeArray's which hold the geometry objects
  NameList<GeomListStruct *> geomLists;
  
  /// color category index with the colors to use.  If < 0, use default colors
  int colorCat;

  VMDApp *app;

  /// add a new category: specify the name, and default color.  Return index of
  /// new list.
  int add_geom_list(const char *, int);
  
  /// delete the Nth category.  Return success.
  int del_geom_list(int);

  /// relative text size for labels
  float labelsize;
  /// relative thickness for labels
  float labelthickness;

protected:
  /// changes the color of all geometries
  virtual void do_color_changed(int);

public:
  GeometryList(VMDApp *, Displayable *);
  
  /// destructor: must clean up all lists
  virtual ~GeometryList(void);
  
  //
  // return information about this class
  //
  
  /// number of geometry lists
  int num_lists(void) { return geomLists.num(); }
  
  /// access the pointer to the Nth geometry list
  GeomListPtr geom_list(int n) { return (geomLists.data(n))->geomList; }
  
  /// return pointer to the geometry list with the given name, NULL if not found
  GeomListPtr geom_list(const char *nm) {
    int glistindex = geom_list_index(nm);
    return (glistindex >= 0 ? geom_list(glistindex) : (GeomListPtr) NULL);
  }

  /// return the name of the Nth geometry list
  const char *geom_list_name(int n) { return geomLists.name(n); }
  
  /// return the index of the geom list with the given name, -1 if not found
  int geom_list_index(const char *nm) { return geomLists.typecode(nm); }
  
  //
  // routines to add/delete/modify geometry objects
  // NOTE: after GeometryMol objects have been added, they remain in the
  //	lists until either explicitely deleted, or their 'ok' routine
  //	no longer returns 'TRUE'.  If 'ok' returns FALSE, they are
  //	deleted.
  //

  /// add a new geometry object to the list with the given name.  Return
  /// the index of the geometry object on success (>= 0), or -1 on 
  /// failure. 
  int add_geometry(const char *geomcat, const int *molids, const int *atomids,
      const int *cells, float k, int toggle);

  /// delete the Nth geometry object with the given index, return success.
  int del_geometry(int, int);
  
  /// delete the Nth geometry object with the given name, return success.
  int del_geometry(const char *nm, int n) {
    return del_geometry(geom_list_index(nm), n);
  }
  
  /// show/hide the Nth geometry monitor in the given category.  If
  /// N < 0, hide ALL monitors in that category.  Return success.
  /// args: category, geometry monitor index, show (T) or hide (F)
  int show_geometry(int, int, int);

  /// same as above, but giving a name for the category instead of index
  int show_geometry(const char *nm, int n, int s) {
    return show_geometry(geom_list_index(nm), n, s);
  }

  /// Get/set text size.  Affects all labels.
  float getTextSize() const { return labelsize; }
  int setTextSize(float);

  /// Get/set text thickness.  Affects all labels.
  float getTextThickness() const { return labelthickness; }
  int setTextThickness(float);

  const float *getTextOffset(const char *nm, int n);

  /// set text offset for specfied label
  int setTextOffset(const char *nm, int n, const float delta[2]);

  const char *getTextFormat(const char *nm, int n);
  int setTextFormat(const char *nm, int n, const char *format);

  // 
  // public virtual drawing routines
  // 

  /// prepare for drawing ... do any updates needed right before draw.
  virtual void prepare();
};

#endif
  
