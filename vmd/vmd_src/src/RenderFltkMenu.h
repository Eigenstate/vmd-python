/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/
#ifndef RENDER_FLTK_MENU_H__
#define RENDER_FLTK_MENU_H__

#include "VMDFltkMenu.h"

class Fl_Choice;
class Fl_Button;
class Fl_Input;

/// VMDFltkMenu subclass implementing a GUI for exporting and rendering
/// VMD scenes via the FileRenderer classes in FileRenderList.
class RenderFltkMenu : public VMDFltkMenu {

public:
  RenderFltkMenu(VMDApp *);

protected:
  int act_on_command(int, Command *);

private:
  void make_window();
  void fill_render_choices();

  Fl_Choice *formatchoice;
  Fl_Input *filenameinput;
  Fl_Input *commandinput;

  /// Puts the current filename and render option into their respective inputs
  static void formatchoice_cb(Fl_Widget *, void *);

  /// Updates the saved render command each time it's changed.
  static void command_cb(Fl_Widget *, void *);

  /// Restores the saved render command to its default value
  static void default_cb(Fl_Widget *, void *);

  /// Browse for a filename to save the render outut.
  static void browse_cb(Fl_Widget *, void *);

  /// Render and report errors.
  static void render_cb(Fl_Widget *, void *);
};
#endif
