/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

#ifndef COLOR_FLTK_MENU_H__
#define COLOR_FLTK_MENU_H__

#include "VMDFltkMenu.h"

class VMDApp;
class Fl_Hold_Browser;
class Fl_Value_Slider;
class Fl_Button;
class Fl_Choice;

/// class to maintain a GUI-usable image of a ColorScale
class ColorscaleImage;

/// VMDFltkMenu subclass providing a GUI to change color settings 
class ColorFltkMenu : public VMDFltkMenu {
public:
  ColorFltkMenu(VMDApp *);
  void update_scaleimage();

private:
  /// Put all color categories in the browser and clear any selected category.
  void reset_color_categories();

  /// Put all color names into the browser and clear any selected items.
  void reset_color_names();

  /// Make color scale controls reflect reality
  void reset_color_scale();

  /// Select the colors in the color browsers according to what's chosen
  /// in the category and item browsers.  Also update the color definition.
  /// If no color is chosen, deselect the color choosers.
  void update_chosen_color();

  /// Update the RGB values for the currently selected color
  void update_color_definition();

  void make_window();
  Fl_Hold_Browser *categorybrowser;
  Fl_Hold_Browser *itembrowser;
  Fl_Hold_Browser *colorbrowser;
  Fl_Hold_Browser *colordefbrowser;
  Fl_Value_Slider *redscale;
  Fl_Value_Slider *greenscale;
  Fl_Value_Slider *bluescale;
  Fl_Button *grayscalebutton;
  Fl_Button *defaultbutton;
  Fl_Choice *scalemethod;
  Fl_Value_Slider *offsetvalue;
  Fl_Value_Slider *midpointvalue;
  ColorscaleImage *image;

  static void category_cb(Fl_Widget *, void *);
  static void item_cb(Fl_Widget *, void *);
  static void color_cb(Fl_Widget *, void *);
  static void colordef_cb(Fl_Widget *, void *);
  static void rgb_cb(Fl_Widget *, void *);
  static void default_cb(Fl_Widget *, void *v);
  static void scalemethod_cb(Fl_Widget *, void *);
  static void scalesettings_cb(Fl_Widget *, void *);

protected:
  int act_on_command(int, Command *);
};
#endif
