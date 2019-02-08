/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/


#ifndef MAINFLTKMENU
#define MAINFLTKMENU

#include "FL/Fl_Menu_Item.H"
#include "VMDFltkMenu.h"
#include "ResizeArray.h"

class MolBrowser;
class Fl_Menu_Bar;
class Fl_Slider;
class Fl_Int_Input;
class Fl_Check_Button;
class Fl_Counter;
class Fl_Button;

#define VMDLEANGUI 1

typedef enum {MENU_ALWAYS_ON=0, MENU_NEED_SEL=1, MENU_NEED_UNIQUE_SEL=3} MenuBehavior;

  /// VMDFltkMenu subclass implementing the main molecule browser GUI, 
/// with pulldown menus to change mouse state and bring up other menus.
class MainFltkMenu: public VMDFltkMenu {
  friend class MolBrowser;
  
private:
  Fl_Menu_Bar *menubar;
  MolBrowser *browser;
 
  enum {UNDEFINED, NO_SELECTED_MOL, ONE_SELECTED_MOL, MANY_SELECTED_MOL} guistate;

  Fl_Menu_Item *file_menuitems;
  Fl_Menu_Item *molecule_menuitems;
  Fl_Menu_Item *display_menuitems;
  Fl_Menu_Item *axes_menuitems;
  Fl_Menu_Item *backgroundmode_menuitems;
  Fl_Menu_Item *stage_menuitems;
  Fl_Menu_Item *stereo_menuitems;
  Fl_Menu_Item *stereoswap_menuitems;
#if !defined(VMDLEANGUI)
  Fl_Menu_Item *cachemode_menuitems;
#endif
  Fl_Menu_Item *rendermode_menuitems;
  Fl_Menu_Item *mouse_menuitems;
  Fl_Menu_Item *browserpopup_menuitems;
    
  // these are defined to overcome an Fltk limitation
  Fl_Menu_Item *axes_menuitems_storage;
  Fl_Menu_Item *backgroundmode_menuitems_storage;
  Fl_Menu_Item *stage_menuitems_storage;
  Fl_Menu_Item *stereo_menuitems_storage;
  Fl_Menu_Item *stereoswap_menuitems_storage;
  Fl_Menu_Item *rendermode_menuitems_storage;
#if !defined(VMDLEANGUI)
  Fl_Menu_Item *cachemode_menuitems_storage;
#endif
  Fl_Menu_Item *mouse_menuitems_storage;
        
  Fl_Slider *frameslider;
  Fl_Slider *speed;
  Fl_Int_Input *curframe;
  Fl_Check_Button *zoom;
  Fl_Choice *style;

  Fl_Counter *step;
  Fl_Button *forward, *reverse;

  void update_mousemode(Command *);
  void update_dispmode();

  /// special callback to override default FLTK/VMD window close behavior
  static void vmd_main_window_cb(Fl_Widget *, void *);

  static void loadfile_cb(Fl_Widget *, void *);
  static void savefile_cb(Fl_Widget *, void *);
  static void frameslider_cb(Fl_Widget *, void *);
  static void zoom_cb(Fl_Widget *w, void *v);

  /// Gets the number of the user-selected molecule in the GUI.
  /// Returns the number of the user-selected molecule (in the GUI) or of 
  /// the first selected molecule if there are more than one. 
  /// Note: This is *not* the molecule ID (use VMDApp::molecule_id(num) for 
  /// that).  Returns -1 if there is no such molecule.
  int get_selected_molecule();

  /// Check to see if the guistate (i.e. whether a molecule is selected in 
  /// the browser or not) has changed and updates the gui accordingly 
  void update_menu_state(Fl_Menu_Item* mymenuitems, const MenuBehavior* mymenu_behavior);
  void update_gui_state();
  
protected:
  int act_on_command(int, Command *);

  /// XXX hack around a resize bug in Fltk on the Mac by overriding the width 
  /// of the menu just before drawing.
  virtual void draw();  // override Fl_Window::draw()

public:
  MainFltkMenu(VMDApp *);
  ~MainFltkMenu();
};

#endif

