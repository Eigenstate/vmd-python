/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/


#ifndef GEOMETRYFLTKMENU_H 
#define GEOMETRYFLTKMENU_H 

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Multi_Browser.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Value_Output.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Float_Input.H>
#include <FL/Fl_Positioner.H>
#include "VMDFltkMenu.h"

#define VMD_FLCHART_WORKAROUND 1

#if defined(VMD_FLCHART_WORKAROUND)
class myFl_Chart; // XXX workaround bug in Fl_Chart
#else
class Fl_Chart;
#endif
class Fl_Input;
class GeometryList;
class Molecule;

/// VMDFltkMenu subclass to manage geometry data, labels, and picked atoms
class GeometryFltkMenu : public VMDFltkMenu {
public:
  GeometryFltkMenu(VMDApp *);
  void apply_offset_to_selected_labels(float x, float y);
  void apply_format_to_selected_labels(const char *format);

protected:
  int act_on_command(int, Command *);

private:
  /// cache GeometryList until we get an API
  GeometryList *glist;

  int user_is_typing_in_format_input;

  void make_window();
  void update_geometry_types();
  void update_labelprops();
  void fill_label_browser();
  void handle_pick(Molecule *, int, float);

  static void typechooser_cb(Fl_Widget *, void *);
  static void graphinwindow_cb(Fl_Widget *, void *);
  static void show_cb(Fl_Widget *, void *);
  static void hide_cb(Fl_Widget *, void *);
  static void delete_cb(Fl_Widget *, void *);
  static void labelbrowser_cb(Fl_Widget *, void *);
  static void exportgraph_cb(Fl_Widget *, void *);
  static void savetofile_cb(Fl_Widget *, void *);
  static void close_cb(Fl_Widget *, void *);

  Fl_Browser *labelbrowser;
  Fl_Choice *labeltypechooser;
  Fl_Group *pickinggroup;
  Fl_Output *pickedmolecule;
  Fl_Output *pickedresname;
  Fl_Output *pickedresid;
  Fl_Output *pickedname;
  Fl_Output *pickedtype;
  Fl_Output *pickedindex;
  Fl_Output *pickedchain;
  Fl_Output *pickedsegname;
  Fl_Output *pickedpos;
  Fl_Output *pickedvalue;
  Fl_Group *geometrygroup;
  Fl_Button *savetofilebutton;
  Fl_Button *previewcheckbutton;
  Fl_Button *exportgraphbutton;
#if defined(VMD_FLCHART_WORKAROUND)
  myFl_Chart *chart; // XXX workaround bug in Fl_Chart
#else
  Fl_Chart *chart;
#endif
  Fl_Button *showbutton;
  Fl_Button *hidebutton;
  Fl_Button *deletebutton;
  Fl_Group *propertiesgroup;
  Fl_Positioner *textoffsetpositioner;
  Fl_Button *offsetresetbutton;
  Fl_Input *textformatinput;
  Fl_Group *globalpropsgroup;
  Fl_Slider *textsizeslider;
  Fl_Float_Input *textsizeinput;
  Fl_Slider *textthicknessslider;
  Fl_Float_Input *textthicknessinput;
};
#endif
