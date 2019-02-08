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
 *      $RCSfile: MaterialList.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.28 $      $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Material properties list structure, master list of all materials
 ***************************************************************************/

#ifndef MATERIAL_LIST_H__
#define MATERIAL_LIST_H__

#include "NameList.h"
#include "ResizeArray.h"

class Displayable;

/// structure containing material properties used to shade a Displayable
struct Material {
  float ambient;
  float specular;
  float diffuse;
  float shininess;
  float mirror;
  float opacity;
  float outline;
  float outlinewidth;
  float transmode;
  int ind;
};


/// enumeration of all material properties
enum MaterialProperty { 
  MAT_AMBIENT, MAT_SPECULAR, MAT_DIFFUSE, MAT_SHININESS, MAT_MIRROR,
  MAT_OPACITY, MAT_OUTLINE, MAT_OUTLINEWIDTH, MAT_TRANSMODE
};
                        

/// manages a list of materials that can be applied to Displayable objects
class MaterialList {
protected:
  // list of materials 
  NameList<Material *> *mlist;
 
  // root displayable, used to propogate changes in material settings down
  // to all the displayables.
  Displayable *root;

  // tell users that settings have changed 
  void notify(int);

  // counter used to create unique material names
  int matcounter;

public:
  // constructor: root of displayable tree
  MaterialList(Displayable *);
  ~MaterialList();

  // query raw material properties
  // The renderers would access the material properties from the
  // index in the DispCmd.
  int num() const { return mlist->num(); }
  const char *material_name(int i) const { return mlist->name(i); }
  const Material *material(int i) const  { return mlist->data(i); }

  // Displayables get the index for a material here.  If the name is no longer
  // valid, index 0 (opaque) is returned. 
  int material_index(const char *nm) const { return mlist->typecode(nm); }

  // modify material properties - the raw values may be scaled internally
  void set_name(int, const char *);
  void set_ambient(int, float); 
  void set_specular(int, float); 
  void set_diffuse(int, float); 
  void set_shininess(int, float); 
  void set_mirror(int, float); 
  void set_opacity(int, float); 
  void set_outline(int, float); 
  void set_outlinewidth(int, float); 
  void set_transmode(int, float); 

  // query material properties, with values scaled from 0 to 1
  float get_ambient(int); 
  float get_specular(int); 
  float get_diffuse(int); 
  float get_shininess(int); 
  float get_mirror(int); 
  float get_opacity(int); 
  float get_outline(int); 
  float get_outlinewidth(int); 
  float get_transmode(int); 
  
  // Add material with given name, or use new unique name if NULL.  
  // Copy settings from material with given name, or use material 0 
  // if copyfrom is NULL.  Return name of new material, or NULL on
  // error.
  const char *add_material(const char *name, const char *copyfrom); 

  // delete material; return success.
  int delete_material(int);

  // restore the default value of the given material.  Return success.
  int restore_default(int);
};

#endif

