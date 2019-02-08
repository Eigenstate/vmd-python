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
 *      $RCSfile: MaterialFltkMenu.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $      $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Material properties GUI form.
 ***************************************************************************/
#ifndef MATERIAL_FLTK_MENU_H__
#define MATERIAL_FLTK_MENU_H__

#include "VMDFltkMenu.h"

class Fl_Value_Slider;
class Fl_Hold_Browser;
class Fl_Button;
class Fl_Input;

/// VMDFltkMenu subclass implementing a GUI for creating
/// and configuring material properties
class MaterialFltkMenu: public VMDFltkMenu {
private:
  int curmat;                     ///< current material

  void fill_material_browser();
  void set_sliders();

  void init(void);                ///< initialize the user interface

  Fl_Value_Slider *ambient;       ///< ambient lighting coefficient
  Fl_Value_Slider *specular;      ///< specular reflection coefficient
  Fl_Value_Slider *diffuse;       ///< diffuse reflection coefficient
  Fl_Value_Slider *shininess;     ///< Phong shininess exponent control
  Fl_Value_Slider *mirror;        ///< mirror reflection coefficient
  Fl_Value_Slider *opacity;       ///< surface opacity
  Fl_Value_Slider *outline;       ///< edge cueing amplitude
  Fl_Value_Slider *outlinewidth;  ///< edge cueing exponent
  Fl_Hold_Browser *browser;
  Fl_Check_Button *transmode;     ///< enable/disable transparency modulation
  Fl_Input *nameinput;
  Fl_Button *deletebutton;
  Fl_Button *defaultbutton;

private:
  static void slider_cb(Fl_Widget *w, void *v);
  static void createnew_cb(Fl_Widget *w, void *v);
  static void delete_cb(Fl_Widget *w, void *v);
  static void browser_cb(Fl_Widget *w, void *v);
  static void name_cb(Fl_Widget *w, void *v);
  static void default_cb(Fl_Widget *w, void *v);

protected:
  int act_on_command(int, Command *);

public:
  MaterialFltkMenu(VMDApp *);
};

#endif
