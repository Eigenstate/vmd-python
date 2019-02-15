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
 *      $RCSfile: GeometryMol.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.34 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Base class for all Geometry objects which measure information about
 * atoms in a molecule.  A molecule Geometry monitor is assumed to operate
 * on N atoms, and be able to calculate a single floating-point value for
 * those atoms.  (i.e. the angle formed by three atoms in space)
 *
 ***************************************************************************/
#ifndef GEOMETRYMOL_H
#define GEOMETRYMOL_H

class MoleculeList;
class Molecule;
class VMDDisplayList;
class CommandQueue;

#include "Displayable.h"
#include "ResizeArray.h"
#include "utilities.h"
#include "JString.h"

class GeometryMonitor;

/// Base class for objects that make atomic measurements in a molecule
class GeometryMol : public Displayable {
public:
  /// number of items in this object
  int items(void) { return numItems; }

  /// direct access to the component ids
  int com_index(int i) {
    return comIndex[i];
  }

  /// direct access to the object ids
  int obj_index(int i) {
    return objIndex[i];
  }
  
  /// most recent value
  float value(void) { return geomValue; }

  /// do we have a useful value to print?
  int has_value(void) { return hasValue; }

  /// recalculate the value of this geometry, and return it
  virtual float calculate(void) { return 0.0; }
  
  /// if so, call this command to set the variable(s)
  virtual void set_pick(void) { }

protected:
  MoleculeList *molList;
  CommandQueue *cmdqueue;

  int numItems;       ///< number of components used to calculate geometry value
  int *objIndex;      ///< indices of objects containing the components
  int *comIndex;      ///< indices of the components
  int *cellIndex;     ///< unit cell IDs for the components
  float geomValue;    ///< most recent value
  int hasValue;       ///< do we have a useful value to print?
  float valuePos[3];  ///< location where to print value in the scene
  char *gmName;       ///< name of the molecule geometry monitor
  char *uniquegmName; ///< unique name of the geometry monitor

  //
  // protected functions
  //

  /// set the name of this item
  void geom_set_name(void);

  /// sort the elements in the list, so that the lowest atom index is first
  /// (but preserve the relative order, i.e. a-b-c or c-b-a)
  void sort_items(void);

  /// check whether the given molecule m & atom index a is OK
  /// if OK, return Molecule pointer; otherwise, return NULL
  Molecule *check_mol(int m, int a);

  /// for the given pick point, find the TRANSFORMED coords for the given atom.
  /// return Molecule pointer if successful, NULL otherwise.
  Molecule *transformed_atom_coord(int ind, float *);
  
  /// for the given Molecule, find the UNTRANSFORMED coords for the given atom.
  /// return Molecule pointer if successful, NULL otherwise.
  /// Also applies current periodic image for this coordinate.
  Molecule *normal_atom_coord(int ind, float *);

  /// draws a line between the two given points
  void display_line(float *, float *, VMDDisplayList *);

  /// print given text at current valuePos position
  void display_string(const char *, VMDDisplayList *);

  void atom_full_name(char *buf, Molecule *mol, int ind) ;
  void atom_short_name(char *buf, Molecule *mol, int ind) ;
  void atom_formatted_name(JString &str, Molecule *mol, int ind);

  /// methods for setting Tcl variables
  void set_pick_selection(int, int, int*);
  void set_pick_selection();
  void set_pick_value(double);

public:
  /// constructor: # items, molecule list to use, command queue for events
  GeometryMol(int, int *, int *, const int *cells, MoleculeList *mlist, 
      CommandQueue *, Displayable *);

  virtual ~GeometryMol(void);
  
  /// return the name of this geometry marker; by default, just blank
  const char *name(void);

  /// return 'unique' name of the marker, which should be different than
  /// other names for different markers of this same type
  const char *unique_name(void);

  /// check whether the geometry value can still be calculated
  int ok(void);

  /// calculate a list of items, if this object can do so.  Return success.
  int calculate_all(ResizeArray<float> &);

  /// redraw the label due to molecule rotations or changes in coordinates
  void update() { create_cmd_list(); }

  /// set the color used to draw the label
  void set_color(int c) { my_color = c; update(); }

  /// set the size used for text labels
  void set_text_size(float size) { my_text_size = size; update(); }

  /// set the line thickness used for text labels
  void set_text_thickness(float thick) { my_text_thickness = thick; update(); }

  const float *text_offset() const { return my_text_offset; }

  void set_text_offset(const float offset[2]) {
    my_text_offset[0] = offset[0];
    my_text_offset[1] = offset[1];
    update();
  }

  /// text format: the string to be printed for atom labels, with the
  /// following substitutions:
  ///   %R = resname
  ///   %r = resname in "camel case" (i.e. only first letter capitalized) 
  ///   %d = resid
  ///   %a = atom name
  ///   %q = atom charge
  ///   %i = zero-based atom index

  const char *text_format() const { return (const char *)my_text_format; }
  void set_text_format(const char *aFormat) {
    my_text_format = aFormat;
    update();
  }
  
protected:
  virtual void create_cmd_list() = 0;
  ResizeArray<GeometryMonitor *> monitors;
  int my_color;

private:
  float my_text_size;
  float my_text_thickness;
  float my_text_offset[2];
  JString my_text_format;
};

#endif

