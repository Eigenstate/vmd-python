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
 *      $RCSfile: AtomColor.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.46 $      $Date: 2011/03/05 05:13:32 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Parse and maintain the data for how a molecule should be colored.
 *
 ***************************************************************************/
#ifndef ATOMCOLOR_H
#define ATOMCOLOR_H

#include "utilities.h"
class MoleculeList;
class DrawMolecule;
class Scene;

// default atom coloring method, and max cmd string size
#define DEFAULT_ATOMCOLOR	AtomColor::NAME
#define MAX_ATOMCOLOR_CMD	255

/// Parse and maintain the data for how a molecule should be colored.
class AtomColor {
public:
  /// different methods for coloring atoms
  /// XXX the order of these enums matters, as the AtomColor class uses
  /// the ordering to eliminate some complexity in testing.
  enum ColorMethod { NAME, TYPE, ELEMENT, RESNAME, RESTYPE, RESID, CHAIN, 
                     SEGNAME, CONFORMATION, MOLECULE, STRUCTURE, COLORID, 
                     BETA, OCCUP, MASS, CHARGE, 
                     POS, POSX, POSY, POSZ,
                     USER, USER2, USER3, USER4, FRAGMENT,
                     INDEX, BACKBONE, THROB, PHYSICALTIME, TIMESTEP, VELOCITY, 
                     VOLUME, TOTAL };
	
  char cmdStr[MAX_ATOMCOLOR_CMD + 1]; ///< initial string with rep command
  int *color;               ///< color determined for each atom
  int nAtoms;               ///< number of atoms these colors are for

private:
  ColorMethod colorMethod;  ///<  how to represent atoms, and some parameters
  int colIndex;	            ///< index, if method = COLORID
  int volIndex;             ///< index, if method = VOLUME
  const Scene *scene;       ///< Scene object this object should use

  /// min and max range of data.  When using color scale methods like mass,
  /// data that falls outside this range will be colored by the min or max
  /// color.  minRange and maxRange are calculated only once, at the first
  /// opportunity after the coloring method is chosen (the molecule must have
  /// a timestep), or when rescale_colorscale_minmax() is called.
  float minRange, maxRange;

  int need_recalc_minmax;   ///< flag for whether to recalc minmax or not
  MoleculeList *molList;    ///< list of molecules to use, may be NULL
  DrawMolecule *mol;        ///< molecule used to base the selection on
  int parse_cmd(const char *); ///< parse command, store results, return success

public:
  AtomColor(MoleculeList *, const Scene *);
  AtomColor(AtomColor &);
  ~AtomColor(void);
  
  /// equal operator, to change the current settings.
  AtomColor& operator=(const AtomColor &);

  /// return whether the color category index is the
  /// setting for the current coloring method
  int current_color_use(int);

  /// find the color index for the atoms of the given molecule. Return success.
  int find(DrawMolecule *);

  /// provide new settings; does a 'find' at the end if a mol has
  /// been previously provided.
  int change(const char *newcmd) {
    int retval = parse_cmd(newcmd);
    if(retval && mol)
      retval = find(mol);
    return retval;
  }

  //
  // info about current settings
  //

  /// return representation method
  int method(void) { return colorMethod; }
  
  /// return color index (may not be applicable)
  int color_index(void) { return colIndex; }

  /// return volume index (may not be applicable)
  int volume_index(void) { return volIndex; }

  /// get current minRange and maxRange for color scale
  void get_colorscale_minmax(float *min, float *max) const {
    *min = minRange;
    *max = maxRange;
  }

  /// Set minRange and maxRange.  Return success
  int set_colorscale_minmax(float min, float max) {
    if (min > max) return FALSE;
    minRange = min;
    maxRange = max;
    need_recalc_minmax = FALSE;  // don't override these values
    return TRUE;
  }

  /// Rescale the minRange and maxRange on the next update.
  void rescale_colorscale_minmax() {
    need_recalc_minmax = TRUE;
  }

  /// returns true if the current coloring method uses the color scale
  int uses_colorscale() const;

  /// flag for whether to re-find atom colors every time the frame changes
  int do_update;
};

/// XXX a global string array with text descriptions of representation methods
extern const char *AtomColorName[AtomColor::TOTAL];
extern const char *AtomColorMenuName[AtomColor::TOTAL];

#endif

