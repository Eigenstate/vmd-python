/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/
#include "Inform.h"
#include "MainFltkMenu.h"
#include "FL/Fl_Menu_Bar.H"
#include "FL/Fl_Menu_Button.H"
#include "FL/Fl_Menu_Item.H"
#include "MolBrowser.h"
#include "frame_selector.h"
#include "FL/Fl_Radio_Button.H"
#include "FL/Fl_Value_Slider.H"
#include "FL/Fl_Int_Input.H"
#include "TextEvent.h"

#if FL_MAJOR_VERSION <= 1
#if FL_MINOR_VERSION < 1
#include "FL/fl_file_chooser.H"
#endif
#endif


#include "FL/forms.H"
#include "VMDApp.h"
#include "VMDMenu.h"
#include "CommandQueue.h"
#include "CmdMenu.h"
#include "CmdAnimate.h"
#include "Mouse.h"
#include "TextEvent.h"
#include "FPS.h"
#include "Stage.h"
#include "Axes.h"
#include "Scene.h"
#include "Animation.h"
#include "DisplayDevice.h"
#include "PickModeList.h"

#define EXT_MENU_NAME "Extensions"


// Special main window callback to prevent ESC from closing the 
// main window, and tie window closure via mouse to quitting VMD.
void MainFltkMenu::vmd_main_window_cb(Fl_Widget * w, void *) {
  MainFltkMenu *m = (MainFltkMenu *)w;

  if (Fl::event_key() == FL_Escape) return; // ignore Escape key

  if (fl_show_question("Really Quit?", 0))
    m->app->VMDexit("",0,0);

  // Normal code executed by all other windows is:
  // m->app->menu_show(m->get_name(), 0);
}

// callback for all pulldown menu items that just raise a form
static void menu_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  const char *name = (const char *)v;
  app->menu_show(name, 0);
  app->menu_show(name, 1);
}

static void loadnew_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->menu_select_mol("files", -1);
  app->menu_show("files", 0);
  app->menu_show("files", 1);
}

void MainFltkMenu::loadfile_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  int selmol = ((MainFltkMenu *) v)->get_selected_molecule();
  app->menu_select_mol("files", selmol);
  app->menu_show("files", 0);
  app->menu_show("files", 1);
}

void MainFltkMenu::savefile_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  int selmol = ((MainFltkMenu *) v)->get_selected_molecule();
  app->menu_select_mol("save", selmol);
  app->menu_show("save", 0);
  app->menu_show("save", 1);
}

static void render_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->menu_show("render", 0);
  app->menu_show("render", 1);
}

static void savestate_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  if (!app->save_state()) {
    fl_alert("Save State failed.");
  }
}

static void logfile_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  char *file = app->vmd_choose_file(
    "Enter filename for VMD session log:", // Title
    "*.vmd",                               // extension
    "VMD files",                           // label
    1                                      // do_save
  );
  if (!file) return;
  char *buf = new char[strlen(file)+13];
  sprintf(buf, "logfile {%s}", file);
  app->commandQueue->runcommand(new TclEvalEvent(buf));
  delete [] buf;
  delete [] file;
}

static void logconsole_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  const char *buf = "logfile console";
  app->commandQueue->runcommand(new TclEvalEvent(buf));
}

static void logoff_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  const char *buf = "logfile off";
  app->commandQueue->runcommand(new TclEvalEvent(buf));
}

static void quit_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  if (fl_show_question("Really Quit?", 0))
    app->VMDexit("",0,0);
}
  
static void aa_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->display_set_aa(
    ((Fl_Menu_ *)w)->mvalue()->value());
}
 
static void depthcue_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->display_set_depthcue(
    ((Fl_Menu_ *)w)->mvalue()->value());
}

#if !defined(VMDLEANGUI)
static void culling_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->display_set_culling(
    ((Fl_Menu_ *)w)->mvalue()->value());
}
#endif

static void fps_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->display_set_fps(
  ((Fl_Menu_ *)w)->mvalue()->value()); 
}

static void light_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  int *whichlight = (int *)v;
  int turnon = ((Fl_Menu_ *)w)->mvalue()->value();
  app->light_on(*whichlight, turnon);
}

static void stage_cb(Fl_Widget *w, void *v) {
  Fl_Menu_ *m = (Fl_Menu_ *)w;
  VMDApp *app = (VMDApp *)v;
  app->stage_set_location(m->text());
}

static void axes_cb(Fl_Widget *w, void *v) {
  Fl_Menu_ *m = (Fl_Menu_ *)w;
  VMDApp *app = (VMDApp *)v;
  app->axes_set_location(m->text());
}

static void backgroundmode_cb(Fl_Widget *w, void *v) {
  Fl_Menu_ *m = (Fl_Menu_ *)w;
  VMDApp *app = (VMDApp *)(w->user_data());
  if (!strcmp("Gradient", m->text())) {
    app->display_set_background_mode(1);
  } else {
    app->display_set_background_mode(0);
  }
}

static void stereo_cb(Fl_Widget *w, void *v) {
  Fl_Menu_ *m = (Fl_Menu_ *)w;
  VMDApp *app = (VMDApp *)v;
  app->display_set_stereo(m->text());
}

static void stereoswap_cb(Fl_Widget *w, void *v) {
  Fl_Menu_ *m = (Fl_Menu_ *)w;
  VMDApp *app = (VMDApp *)v;
  if (!strcmp("On", m->text())) {
    app->display_set_stereo_swap(1);
  } else {
    app->display_set_stereo_swap(0);
  }
}

#if !defined(VMDLEANGUI)
static void cachemode_cb(Fl_Widget *w, void *v) {
  Fl_Menu_ *m = (Fl_Menu_ *)w;
  VMDApp *app = (VMDApp *)v;
  app->display_set_cachemode(m->text());
}
#endif

static void rendermode_cb(Fl_Widget *w, void *v) {
  Fl_Menu_ *m = (Fl_Menu_ *)w;
  VMDApp *app = (VMDApp *)v;
  app->display_set_rendermode(m->text());
}

static void resetview_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->scene_stoprotation();
  app->scene_resetview();
}

static void stoprotation_cb(Fl_Widget *w, void *) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->scene_stoprotation();
}
 
static void proj_cb(Fl_Widget *w, void *) {
  Fl_Menu_ *m = (Fl_Menu_ *)w;
  VMDApp *app = (VMDApp *)(w->user_data());
  app->display_set_projection(m->text());
}

static void mouse_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->mouse_set_mode(*((int *)v), -1);
}
 
static void move_light_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->mouse_set_mode(Mouse::LIGHT, *((int *)v) );
}

static void help_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)(w->user_data());
  app->commandQueue->runcommand(new HelpEvent((const char*)v));
}

// edit menu callbacks
static void mol_top_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  MolBrowser *browser = (MolBrowser *)v;
  for (int i=0; i<browser->size(); i++) {
    if (browser->selected(i+1)) {
      app->molecule_make_top(app->molecule_id(i));
      break;
    }
  }
}

static void mol_active_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  MolBrowser *browser = (MolBrowser *)v;
  for (int i=0; i<browser->size(); i++) {
    if (browser->selected(i+1)) {
      int molid = app->molecule_id(i);
      app->molecule_activate(molid, !app->molecule_is_active(molid));
    }
  }
}

static void mol_displayed_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  MolBrowser *browser = (MolBrowser *)v;
  for (int i=0; i<browser->size(); i++) {
    if (browser->selected(i+1)) {
      int molid = app->molecule_id(i);
      app->molecule_display(molid, !app->molecule_is_displayed(molid));
    }
  }
}
  
static void mol_fixed_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  MolBrowser *browser = (MolBrowser *)v;
  for (int i=0; i<browser->size(); i++) {
    if (browser->selected(i+1)) {
      int molid = app->molecule_id(i);
      app->molecule_fix(molid, !app->molecule_is_fixed(molid));
    }
  }
}


static void mol_rename_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  MolBrowser *browser = (MolBrowser *)v;
  int molid=-1;
  for (int i=0; i<browser->size(); i++)
    if (browser->selected(i+1)) {
      molid = app->molecule_id(i);
      break;
    }
  if (molid < 0) return;
  
  // this code snippet is replicated in MolBrowser.C:
  const char *oldname = app->molecule_name(molid);
  const char *newname = fl_input("Enter a new name for molecule %d:", 
      oldname, molid);
  if (newname) app->molecule_rename(molid, newname);
}
  
  
static void mol_cancel_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  MolBrowser *browser = (MolBrowser *)v;
  for (int i=0; i<browser->size(); i++) {
    if (browser->selected(i+1)) {
      int molid = app->molecule_id(i);
      app->molecule_cancel_io(molid);
    }
  }
}

static void mol_delete_ts_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  MolBrowser *browser = (MolBrowser *)v;
  int molid=-1;
  for (int i=0; i<browser->size(); i++)
    if (browser->selected(i+1)) {
      molid = app->molecule_id(i);
      break;
    }
  if (molid < 0) return;
  
  // this code snippet is replicated in MolBrowser.C:
  int numframes = app->molecule_numframes(molid);
  if (!numframes) {
    fl_alert("Molecule %d has no frames to delete!", molid);
  } else {
    const char *molname = app->molecule_name(molid);
    int first=0, last=numframes-1, stride=0;
    int ok = frame_delete_selector(molname, last, &first, &last, &stride);
    if (ok) app->molecule_deleteframes(molid, first, last, stride);
  }
}

static void mol_delete_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  MolBrowser *browser = (MolBrowser *)v;
  ResizeArray<int> idlist;
  for (int i=0; i<browser->size(); i++) {
    if (browser->selected(i+1)) {
      idlist.append(app->molecule_id(i));
    }
  }
  for (int j=0; j<idlist.num(); j++)
    app->molecule_delete(idlist[j]);
}

static void loadstate_cb(Fl_Widget *w, void *v) {
  VMDApp *app = (VMDApp *)w->user_data();
  char *file = app->vmd_choose_file(
    "Enter filename containing VMD saved state:", // Title
    "*.vmd",                                      // extension
    "VMD files",                                  // label
    0                                             // do_save
  );
  if (!file) return;
  char *buf = new char[strlen(file)+10];
  sprintf(buf, "play {%s}", file);
  app->commandQueue->runcommand(new TclEvalEvent(buf));
  delete [] buf;
  delete [] file;
}
  

// the menu behavior describes whether or not the menu item require a 
// molecule(s) to exist or be selected in the main browser or not, in order 
// to be active. The fields need to be in the same order as they appear in 
// the menu description.

static const MenuBehavior file_menu_behavior[] = {
  MENU_ALWAYS_ON,           // new
  MENU_NEED_UNIQUE_SEL,     // load file
  MENU_NEED_UNIQUE_SEL,     // save file
  MENU_ALWAYS_ON,           // load state
  MENU_ALWAYS_ON,           // save state
  MENU_ALWAYS_ON,           // log tcl commands to console
  MENU_ALWAYS_ON,           // log tcl commands to file
  MENU_ALWAYS_ON,           // logging off
  MENU_ALWAYS_ON,           // render
  MENU_ALWAYS_ON            // quit
};

// Note: the user_data (i.e. callback argument) for all file_menu items 
// will be reset to the "this" MainFltkMenu object instance.
static const Fl_Menu_Item init_file_menuitems[] = {
  {"New Molecule...",             0, loadnew_cb},
  {"Load Data Into Molecule...",  0, NULL /*set later*/},
  {"Save Coordinates...",         0, NULL /*set later*/, NULL, FL_MENU_DIVIDER},
  {"Load Visualization State...", 0, loadstate_cb},
  {"Save Visualization State...", 0, savestate_cb,  NULL, FL_MENU_DIVIDER},
  {"Log Tcl Commands to Console", 0, logconsole_cb, NULL},
  {"Log Tcl Commands to File...", 0, logfile_cb,    NULL},
  {"Turn Off Logging",            0, logoff_cb,     NULL, FL_MENU_DIVIDER},
  {"Render...",                   0, render_cb,     NULL, FL_MENU_DIVIDER},
  {"Quit",                        0, quit_cb},
  {NULL}
};

static const MenuBehavior molecule_menu_behavior[] = {
  MENU_NEED_UNIQUE_SEL,     // top
  MENU_NEED_SEL,            // active
  MENU_NEED_SEL,            // displayed
  MENU_NEED_SEL,            // fixed
  MENU_NEED_UNIQUE_SEL,     // rename
  MENU_NEED_UNIQUE_SEL,     // delete ts
  MENU_NEED_SEL,            // cancel file i/o
  MENU_NEED_SEL             // delete mol
};

// Note: the user_data (i.e. callback argument) for all molecule_menu items 
// will be reset to this->browser.
static const Fl_Menu_Item init_molecule_menuitems[] = {
  {"Make Top",          0, mol_top_cb,       }, 
  {"Toggle Active",     0, mol_active_cb,    },
  {"Toggle Displayed",  0, mol_displayed_cb, },
  {"Toggle Fixed",      0, mol_fixed_cb,     NULL,   FL_MENU_DIVIDER},
  {"Rename...",         0, mol_rename_cb     },
  {"Delete Frames...",  0, mol_delete_ts_cb  },
  {"Abort File I/O",    0, mol_cancel_cb,    },
  {"Delete Molecule",   0, mol_delete_cb     },
  {NULL}
};


static const MenuBehavior browserpopup_menu_behavior[] = {
  MENU_ALWAYS_ON,           // new
  MENU_NEED_UNIQUE_SEL,     // load file
  MENU_NEED_UNIQUE_SEL,     // save file
  MENU_NEED_UNIQUE_SEL,     // rename
  MENU_NEED_UNIQUE_SEL,     // delete ts
  MENU_NEED_SEL,            // cancel file i/o
  MENU_NEED_SEL             // delete mol
};

// Note: the user_data (i.e. callback argument) for all molecule_menu items 
// will be reset to this->browser.
static const Fl_Menu_Item init_browserpopup_menuitems[] = {
  // Here: user_data will be set to (MainFltkMenu*) this.
  {"New Molecule...",            0, loadnew_cb    },
  {"Load Data Into Molecule...", 0, NULL /* set later */},
  {"Save Coordinates...",        0, NULL /* set later */,  NULL, FL_MENU_DIVIDER},
  // Here: user_data will be set to this->browser
  {"Rename...",         0, mol_rename_cb     },
  {"Delete Frames...",  0, mol_delete_ts_cb  },
  {"Abort File I/O",    0, mol_cancel_cb,    },
  {"Delete Molecule",   0, mol_delete_cb     },
  {NULL}
};

static const Fl_Menu_Item graphics_menuitems[] = {
  {"Representations...", 0, menu_cb, (void *)"graphics"},
  {"Colors...", 0, menu_cb, (void *)"color"},
  {"Materials...", 0, menu_cb, (void *)"material"},
  {"Labels...", 0, menu_cb, (void *)"labels", FL_MENU_DIVIDER},
  {"Tools...", 0, menu_cb, (void *)"tool"}, 
  {0}
};

static int cbdata[] = {
  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
};


enum DispMenu {
   DM_RESETVIEW=0,
   DM_STOPROTATION,
   DM_PERSPECTIVE,   
   DM_ORTHOGRAPHIC,
   DM_ANTIALIASING,
   DM_DEPTHCUEING,
#if !defined(VMDLEANGUI)
   DM_CULLING,
#endif
   DM_FPS,
   DM_LIGHT0,
   DM_LIGHT1,
   DM_LIGHT2,
   DM_LIGHT3,
   DM_AXES,
   DM_BACKGROUND,
   DM_STAGE,
   DM_STEREO,
   DM_STEREOEYESWAP,
#if !defined(VMDLEANGUI)
   DM_CACHEMODE,
#endif
   DM_RENDERMODE,
   DM_DISPSETTINGS,
   DM_LASTMENUITEM
};

static const Fl_Menu_Item init_display_menuitems[] = {
  {"Reset View", '=', resetview_cb},
  {"Stop Rotation", 0, stoprotation_cb, NULL, FL_MENU_DIVIDER},
  {"Perspective", 0, proj_cb, NULL, FL_MENU_RADIO },
  {"Orthographic", 0, proj_cb, NULL, FL_MENU_RADIO | FL_MENU_DIVIDER},
  {"Antialiasing", 0, aa_cb, NULL, FL_MENU_TOGGLE  | FL_MENU_INACTIVE},
  {"Depth Cueing", 0, depthcue_cb, NULL, FL_MENU_TOGGLE | FL_MENU_INACTIVE},
#if !defined(VMDLEANGUI)
  {"Culling", 0, culling_cb, NULL, FL_MENU_TOGGLE | FL_MENU_INACTIVE},
#endif
  {"FPS Indicator", 0, fps_cb, NULL, FL_MENU_TOGGLE | FL_MENU_DIVIDER},
  {"Light 0", 0, light_cb, cbdata+0, FL_MENU_TOGGLE},
  {"Light 1", 0, light_cb, cbdata+1, FL_MENU_TOGGLE},
  {"Light 2", 0, light_cb, cbdata+2, FL_MENU_TOGGLE},
  {"Light 3", 0, light_cb, cbdata+3, FL_MENU_TOGGLE | FL_MENU_DIVIDER},
  {"Axes",   0, NULL, NULL, FL_SUBMENU_POINTER},
  {"Background", 0, backgroundmode_cb, NULL, FL_SUBMENU_POINTER},
  {"Stage",   0, NULL, NULL, FL_SUBMENU_POINTER | FL_MENU_DIVIDER},
  {"Stereo",  0, NULL, NULL, FL_SUBMENU_POINTER},
  {"Stereo Eye Swap",  0, NULL, NULL, FL_SUBMENU_POINTER | FL_MENU_DIVIDER},
#if !defined(VMDLEANGUI)
  {"Cachemode", 0, NULL, NULL, FL_SUBMENU_POINTER},
#endif
  {"Rendermode", 0, NULL, NULL, FL_SUBMENU_POINTER | FL_MENU_DIVIDER},
  {"Display Settings...", 0, menu_cb, (void *)"display"},
  {0}
};

// forward declaration
static void cb_cb(Fl_Widget *w, void *v); 

// These are the items that appear in the mouse submenu.  
// If items are added or removed from this menu, the update_mousemode method
// must be updated to reflect the new positions if the items.
static const Fl_Menu_Item init_mouse_menuitems[] = {
  {"Rotate Mode", 'r', mouse_cb, cbdata+Mouse::ROTATION,FL_MENU_RADIO|FL_MENU_VALUE},
  {"Translate Mode",'t',mouse_cb,cbdata+Mouse::TRANSLATION,FL_MENU_RADIO},
  {"Scale Mode", 's', mouse_cb,  cbdata+Mouse::SCALING, FL_MENU_RADIO | FL_MENU_DIVIDER},
  {"Center", 'c', mouse_cb,      cbdata+Mouse::CENTER, FL_MENU_RADIO},
  {"Query", '0', mouse_cb,       cbdata+Mouse::QUERY,  FL_MENU_RADIO},
      
  {"Label",0,cb_cb,0, FL_SUBMENU | FL_MENU_TOGGLE },
    {"Atoms", '1', mouse_cb,     cbdata+Mouse::LABELATOM, FL_MENU_RADIO},
    {"Bonds", '2', mouse_cb,     cbdata+Mouse::LABELBOND, FL_MENU_RADIO},
    {"Angles", '3', mouse_cb,    cbdata+Mouse::LABELANGLE, FL_MENU_RADIO},
    {"Dihedrals", '4', mouse_cb, cbdata+Mouse::LABELDIHEDRAL, FL_MENU_RADIO},
    {0},
  {"Move",0,cb_cb,0, FL_SUBMENU | FL_MENU_TOGGLE},
    {"Atom", '5', mouse_cb,      cbdata+Mouse::MOVEATOM, FL_MENU_RADIO},
    {"Residue", '6', mouse_cb,   cbdata+Mouse::MOVERES, FL_MENU_RADIO},
    {"Fragment", '7', mouse_cb,  cbdata+Mouse::MOVEFRAG, FL_MENU_RADIO},
    {"Molecule", '8', mouse_cb,  cbdata+Mouse::MOVEMOL, FL_MENU_RADIO},
    {"Rep", '9', mouse_cb,       cbdata+Mouse::MOVEREP, FL_MENU_RADIO},
    {0},
  {"Force",0,cb_cb,0, FL_SUBMENU | FL_MENU_TOGGLE},
    {"Atom", '%', mouse_cb,      cbdata+Mouse::FORCEATOM, FL_MENU_RADIO},
    {"Residue", '^', mouse_cb,   cbdata+Mouse::FORCERES, FL_MENU_RADIO},
    {"Fragment", '&', mouse_cb,  cbdata+Mouse::FORCEFRAG, FL_MENU_RADIO},
    {0},
  {"Move Light", 0,cb_cb,0, FL_SUBMENU | FL_MENU_TOGGLE},
    {"0", 0, move_light_cb, cbdata+0, FL_MENU_RADIO},
    {"1", 0, move_light_cb, cbdata+1, FL_MENU_RADIO},
    {"2", 0, move_light_cb, cbdata+2, FL_MENU_RADIO},
    {"3", 0, move_light_cb, cbdata+3, FL_MENU_RADIO},
    {0},
  {"Add/Remove Bonds", 0, mouse_cb, cbdata+Mouse::ADDBOND, FL_MENU_RADIO},
  {"Pick", 'p', mouse_cb,        cbdata+Mouse::PICK,    FL_MENU_RADIO},
  {0}
};

static const Fl_Menu_Item init_help_menuitems[] = {
  {"Quick Help",        0, help_cb, (void*) "quickhelp"},
  {"User's Guide",        0, help_cb, (void*) "userguide"},
  {"Tutorial",        0, help_cb, (void*) "tutorial", FL_MENU_DIVIDER},
  {"Homepage",          0, help_cb, (void*) "homepage"},
  {"FAQ",               0, help_cb, (void*) "faq"},
  {"Mailing List",      0, help_cb, (void*) "maillist"},
  {"Script Library",    0, help_cb, (void*) "scripts"},
  {"Plugin Library",    0, help_cb, (void*) "plugins", FL_MENU_DIVIDER},
  {"3D Renderers",      0, 0, 0, FL_SUBMENU},
    {"POV-Ray",         0, help_cb, (void*) "povray"},
    {"Radiance",        0, help_cb, (void*) "radiance"},
    {"Raster3D",        0, help_cb, (void*) "raster3D"},
    {"Rayshade",        0, help_cb, (void*) "rayshade"},
    {"Tachyon",         0, help_cb, (void*) "tachyon"},
    {"VRML",            0, help_cb, (void*) "vrml"},
    {0},
  {"Auxiliary Programs", 0, 0, 0, FL_SUBMENU},
    {"BioCoRE",           0, help_cb, (void*) "biocore"},
    {"MSMS",              0, help_cb, (void*) "msms"},
    {"NanoShaper",        0, help_cb, (void*) "nanoshaper"},
    {"NAMD",              0, help_cb, (void*) "namd"},
    {"Tcl/Tk",            0, help_cb, (void*) "tcl"},
    {"Python",            0, help_cb, (void*) "python"},
    {0},
  {0}
};

// turn the item on if any of its children are on; otherwise restore it
// to its off state.
static void cb_cb(Fl_Widget *w, void *v) {
  Fl_Menu_Item *titleitem = (Fl_Menu_Item*) ((Fl_Menu_ *)w)->mvalue();
  const Fl_Menu_Item *item;
  for (item = titleitem+1; item->label(); item++)
    if (item->value()) {
      titleitem->set();
      return;
    }
  titleitem->clear();
}
    
void MainFltkMenu::frameslider_cb(Fl_Widget *w, void *v) {
  Fl_Valuator *val = (Fl_Valuator *)w;
  MainFltkMenu *self = (MainFltkMenu *)v;
  // If the right mouse button is active, update frame only on release...
  //if (Fl::event_button() == FL_RIGHT_MOUSE) {  XXX wrong way to do it
  if (Fl::event_state(FL_BUTTON3)) {
    if (!Fl::event_state()) {
      self->app->animation_set_frame((int)val->value());
    } else {
      // but still update the value displayed in the current frame.
      char buf[10];
      sprintf(buf, "%d", (int)val->value());
      self->curframe->value(buf);
    }
  } else {
    self->app->animation_set_frame((int)val->value());
  }
}

static void curframe_cb(Fl_Widget *w, void *v) {
  Fl_Input *inp = (Fl_Input *)w;
  VMDApp *app = (VMDApp *)v;
  int val = atoi(inp->value());
  int max = app->molecule_numframes(app->molecule_top());
  if (val < 0) val = 0;
  if (val >= max) val = max-1;
  app->animation_set_frame(val);
}

static void start_cb(Fl_Widget *, void *v) {
  VMDApp *app = (VMDApp *)v;
  app->animation_set_frame(-1);
}

static void stop_cb(Fl_Widget *, void *v) {
  VMDApp *app = (VMDApp *)v;
  app->animation_set_frame(-2);
}

static void prev_cb(Fl_Widget *, void *v) {
  VMDApp *app = (VMDApp *)v;
  app->animation_set_dir(Animation::ANIM_REVERSE1);
}

static void next_cb(Fl_Widget *, void *v) {
  VMDApp *app = (VMDApp *)v;
  app->animation_set_dir(Animation::ANIM_FORWARD1);
}

static void forward_cb(Fl_Widget *w, void *v) {
  Fl_Button *button = (Fl_Button *)w;
  VMDApp *app = (VMDApp *)v;
  if (button->value())
    app->animation_set_dir(Animation::ANIM_FORWARD);
  else
    app->animation_set_dir(Animation::ANIM_PAUSE);
}

static void reverse_cb(Fl_Widget *w, void *v) {
  Fl_Button *button = (Fl_Button *)w;
  VMDApp *app = (VMDApp *)v;
  if (button->value())
    app->animation_set_dir(Animation::ANIM_REVERSE);
  else
    app->animation_set_dir(Animation::ANIM_PAUSE);
}

static void style_cb(Fl_Widget *w, void *v) {
  Fl_Choice *choice = (Fl_Choice *)w;
  VMDApp *app = (VMDApp *)v;
  app->animation_set_style(choice->value());
}

static void step_cb(Fl_Widget *w, void *v) {
  Fl_Counter *counter = (Fl_Counter *)w;
  VMDApp *app = (VMDApp *)v;
  app->animation_set_stride((int)counter->value());
}

static void speed_cb(Fl_Widget *w, void *v) {
  Fl_Slider *slider = (Fl_Slider *)w;
  VMDApp *app = (VMDApp *)v;
  app->animation_set_speed((float) slider->value());
}

void MainFltkMenu::zoom_cb(Fl_Widget *w, void *v) {
  Fl_Button *b = (Fl_Button *)w;
  MainFltkMenu *self = (MainFltkMenu *)v;
  int numframes = self->app->molecule_numframes(self->app->molecule_top());
  if (numframes < 1) return;
  double full_range = (double)numframes;
  if (b->value()) {
    // turn on zoom: recenter the range around the current value of the slider
    double pixel_range = 100;
    if (full_range > pixel_range) {
      double curval = self->frameslider->value();
      double curfrac = curval/full_range;
      self->frameslider->range(curval - pixel_range*curfrac,
                               curval + pixel_range*(1.0-curfrac));
      self->frameslider->color(VMDMENU_SLIDER_BG, VMDMENU_SLIDER_FG);
      self->frameslider->redraw();
    }
  } else {
    // turn off zoom; make the range equal to the number of frames
    self->frameslider->range(0, full_range-1);
    self->frameslider->color(VMDMENU_SLIDER_BG, VMDMENU_SLIDER_FG);
    self->frameslider->redraw();
  }
}

void MainFltkMenu::update_mousemode(Command *cmd) {
  int mode = ((CmdMouseMode *)cmd)->mouseMode;
  int setting = ((CmdMouseMode *)cmd)->mouseSetting;
  
  Fl_Menu_Item *items = mouse_menuitems;
  int menulen = sizeof(init_mouse_menuitems)/sizeof(Fl_Menu_Item);
  for (int j=0; j<menulen; j++) // replaced hard-coded <=29 with <menulen
    items[j].clear();

  switch(mode) {
    case Mouse::ROTATION:      items[ 0].setonly(); break;
    case Mouse::TRANSLATION:   items[ 1].setonly(); break;
    case Mouse::SCALING:       items[ 2].setonly(); break;
    case Mouse::QUERY:         items[ 4].setonly(); break;
    case Mouse::CENTER:        items[ 3].setonly(); break;
    case Mouse::LABELATOM:     items[ 6].setonly(); break;
    case Mouse::LABELBOND:     items[ 7].setonly(); break;
    case Mouse::LABELANGLE:    items[ 8].setonly(); break;   
    case Mouse::LABELDIHEDRAL: items[ 9].setonly(); break;  
    case Mouse::MOVEATOM:      items[12].setonly(); break; 
    case Mouse::MOVERES:       items[13].setonly(); break; 
    case Mouse::MOVEFRAG:      items[14].setonly(); break;  
    case Mouse::MOVEMOL:       items[15].setonly(); break; 
    case Mouse::MOVEREP:       items[16].setonly(); break; 
    case Mouse::FORCEATOM:     items[19].setonly(); break; 
    case Mouse::FORCERES:      items[20].setonly(); break; 
    case Mouse::FORCEFRAG:     items[21].setonly(); break; 
    case Mouse::ADDBOND:       items[29].setonly(); break; 
    case Mouse::PICK:          items[30].setonly(); break;
    case Mouse::LIGHT:
      switch (setting) {
        case 0: items[24].setonly(); break;
        case 1: items[25].setonly(); break;
        case 2: items[26].setonly(); break;
        case 3: items[27].setonly(); break;
      }
  }
  if (mode >= Mouse::PICK) {
    items[0].setonly();  // check "rotate" mouse mode
    if (mode == Mouse::LABELATOM  || mode == Mouse::LABELBOND || \
        mode == Mouse::LABELANGLE || mode == Mouse::LABELDIHEDRAL)
      items[5].set();
    else if (mode == Mouse::MOVEATOM || mode == Mouse::MOVERES || \
             mode == Mouse::MOVEMOL  || mode == Mouse::MOVEREP)
      items[11].set();
    else if (mode == Mouse::FORCEATOM || mode == Mouse::FORCERES || mode == Mouse::FORCEFRAG)
      items[18].set();
  } else if (mode == Mouse::LIGHT) {
    if (setting >= 0 && setting <= 3) items[23].set();   
  }
}

void MainFltkMenu::update_dispmode() {
  // XXX the implementation here is ugly because older FLTK APIs
  // lack the value(int) methods, and we can only call set/clear().
  // With FLTK 1.3.x we could instead do things somewhat more cleanly,
  // display_menuitems[DM_ANTIALIASING].value((app->display->aa_enabled()!=0));

  // match the active projection string and set radio button state
  const char *projname = app->display->get_projection();
  for (int ii=DM_PERSPECTIVE; ii<=DM_ORTHOGRAPHIC; ii++) {
    if (!strupcmp(projname, display_menuitems[ii].label())) {
      display_menuitems[ii].setonly();
      break;
    }
  }

  // update antialiasing on/off state
  if (app->display->aa_enabled()) 
    display_menuitems[DM_ANTIALIASING].set();
  else
    display_menuitems[DM_ANTIALIASING].clear();

  // update depth cueing on/off state
  if (app->display->cueing_enabled()) 
    display_menuitems[DM_DEPTHCUEING].set();
  else
    display_menuitems[DM_DEPTHCUEING].clear();

#if !defined(VMDLEANGUI)
  // update backface culling on/off state
  if (app->display->culling_enabled()) 
    display_menuitems[DM_CULLING].set();
  else
    display_menuitems[DM_CULLING].clear();
#endif

  // update display FPS on/off state
  if (app->fps->displayed()) 
    display_menuitems[DM_FPS].set();
  else
    display_menuitems[DM_FPS].clear();

  // update light 0,1,2,3 on/off states
  for (int j=0; j<4; j++)
    if (app->scene->light_active(j))
      display_menuitems[DM_LIGHT0+j].set();
    else
      display_menuitems[DM_LIGHT0+j].clear();

  // set active submenu states for axes, background, stage,
  // stereo mode, stereo eye swap, display list caching, and rendering mode
  axes_menuitems[app->axes->location()].setonly();
  backgroundmode_menuitems[app->scene->background_mode()].setonly();
  stage_menuitems[app->stage->location()].setonly();
  stereo_menuitems[app->display->stereo_mode()].setonly();
  stereoswap_menuitems[app->display->stereo_swap()].setonly();
#if !defined(VMDLEANGUI)
  cachemode_menuitems[app->display->cache_mode()].setonly();
#endif
  rendermode_menuitems[app->display->render_mode()].setonly();
} 
    
    
// Add some extra space at the bottom of the menu for the OSX resizing tab;
// otherwise it obscures buttons on the menu.
#if defined(__APPLE__)
#define MAINFLTKMENUHEIGHT 205
#else
#define MAINFLTKMENUHEIGHT 190
#endif

#if 0
// original main window width used for fixed-width non-antialiased fonts
#define MAINFLTKMENUWIDTH 450
#else
// main window width needed for antialiased fonts via Xft-enabled FLTK builds
#define MAINFLTKMENUWIDTH 470
#endif


MainFltkMenu::MainFltkMenu(VMDApp *vmdapp)
: VMDFltkMenu("main", "VMD Main", vmdapp) {
  // set initial window size
  size(MAINFLTKMENUWIDTH, MAINFLTKMENUHEIGHT);
 
  // set resizable in y but not in x...
  size_range(MAINFLTKMENUWIDTH, MAINFLTKMENUHEIGHT, MAINFLTKMENUWIDTH, 0);

  command_wanted(Command::MOL_NEW);
  command_wanted(Command::MOL_DEL);
  command_wanted(Command::MOL_ACTIVE);
  command_wanted(Command::MOL_ON);
  command_wanted(Command::MOL_RENAME);
  command_wanted(Command::MOL_FIX);
  command_wanted(Command::MOL_TOP);
  command_wanted(Command::MOL_VOLUME);
  command_wanted(Command::ANIM_JUMP);
  command_wanted(Command::ANIM_NEW_FRAME);
  command_wanted(Command::ANIM_NEW_NUM_FRAMES);
  command_wanted(Command::MOUSE_MODE);
  command_wanted(Command::MENU_TK_ADD);
  command_wanted(Command::MENU_TK_REMOVE);
  command_wanted(Command::ANIM_STYLE);
  command_wanted(Command::ANIM_SKIP);
  command_wanted(Command::ANIM_SPEED);
  command_wanted(Command::ANIM_DIRECTION);
  command_wanted(Command::ANIM_JUMP);

  command_wanted(Command::DISP_DEPTHCUE);
  command_wanted(Command::DISP_CULLING);
  command_wanted(Command::DISP_ANTIALIAS);
  command_wanted(Command::DISP_FPS);
  command_wanted(Command::DISP_LIGHT_ON);
  command_wanted(Command::CMD_STAGE);
  command_wanted(Command::CMD_AXES);
  command_wanted(Command::DISP_BACKGROUNDGRADIENT);
  command_wanted(Command::DISP_PROJ);
  command_wanted(Command::DISP_STEREO);
  command_wanted(Command::DISP_STEREOSWAP);
  command_wanted(Command::DISP_CACHEMODE);
  command_wanted(Command::DISP_RENDERMODE);

  browser = new MolBrowser(vmdapp, this, 0, 60, MAINFLTKMENUWIDTH, 90);

  // ******** CREATE MENUS *********
  // We make copies of the static data because we will be changing the state
  // and contents of some menus and menu items.
  
  int menulen;
  Fl_Menu_Item nullitem = {NULL};
     
  // create menu instances and fill in user_data fields for menu callback use.
  menulen = sizeof(init_file_menuitems)/sizeof(Fl_Menu_Item);
  file_menuitems = new Fl_Menu_Item[menulen];
  int j;
  for (j=0; j<menulen; j++) {
    file_menuitems[j] = init_file_menuitems[j];
    file_menuitems[j].user_data(this);
  }
  // these are set here because the are private functions
  file_menuitems[1].callback(loadfile_cb);
  file_menuitems[2].callback(savefile_cb);
        
  menulen = sizeof(init_molecule_menuitems)/sizeof(Fl_Menu_Item);
  molecule_menuitems = new Fl_Menu_Item[menulen];
  for (j=0; j<menulen; j++) {
    molecule_menuitems[j] = init_molecule_menuitems[j];
    molecule_menuitems[j].user_data(browser);
  }
  
  
  // This is the popup menu in the molbrowser window (mix of file and molecule menus)
  menulen = sizeof(init_browserpopup_menuitems)/sizeof(Fl_Menu_Item);
  browserpopup_menuitems = new Fl_Menu_Item[menulen];
  for (j=0; j<3; j++) {
    browserpopup_menuitems[j] = init_browserpopup_menuitems[j];
    browserpopup_menuitems[j].user_data(this);
  }
  for (j=3; j<menulen; j++) {
    browserpopup_menuitems[j] = init_browserpopup_menuitems[j];
    browserpopup_menuitems[j].user_data(browser);
  }
  // these are set here because the are private functions
  browserpopup_menuitems[1].callback(loadfile_cb);
  browserpopup_menuitems[2].callback(savefile_cb);

  
  menulen = sizeof(init_display_menuitems)/sizeof(Fl_Menu_Item);
  display_menuitems = new Fl_Menu_Item[menulen];
  for (j=0; j<menulen; j++)
    display_menuitems[j] = init_display_menuitems[j];
  if (app->display->aa_available()) display_menuitems[DM_ANTIALIASING].activate();
  if (app->display->cueing_available()) display_menuitems[DM_DEPTHCUEING].activate();
#if !defined(VMDLEANGUI)
  if (app->display->culling_available()) display_menuitems[DM_CULLING].activate();
#endif
  
  menulen = app->axes->locations();
  axes_menuitems_storage = new Fl_Menu_Item[menulen+2]; 
  axes_menuitems_storage[0] = nullitem;   // pad the beginning of the array 
                                          // to prevent an Fltk crash
  axes_menuitems = axes_menuitems_storage+1; 
  for (j=0; j<menulen; j++) {
    Fl_Menu_Item item = {app->axes->loc_description(j), 0, axes_cb, app, FL_MENU_RADIO};
    axes_menuitems[j] = item;
  }
  axes_menuitems[menulen] = nullitem;
  display_menuitems[DM_AXES].user_data(axes_menuitems);

  menulen = 2;
  backgroundmode_menuitems_storage = new  Fl_Menu_Item[menulen+2];
  backgroundmode_menuitems_storage[0] = nullitem;
  backgroundmode_menuitems = backgroundmode_menuitems_storage+1;
  {
    Fl_Menu_Item item = { "Solid Color", 0, backgroundmode_cb, app, FL_MENU_RADIO};
    backgroundmode_menuitems[0] = item;
  }
  {
    Fl_Menu_Item item = { "Gradient", 0, backgroundmode_cb, app, FL_MENU_RADIO};
    backgroundmode_menuitems[1] = item;
  }
  backgroundmode_menuitems[menulen] = nullitem;
  display_menuitems[DM_BACKGROUND].user_data(backgroundmode_menuitems);
 
  menulen = app->stage->locations();
  stage_menuitems_storage = new Fl_Menu_Item[menulen+2]; 
  stage_menuitems_storage[0] = nullitem;
  stage_menuitems = stage_menuitems_storage+1;  
  for (j=0; j<menulen; j++) {
    Fl_Menu_Item item = {app->stage->loc_description(j), 0, stage_cb, app, FL_MENU_RADIO};  
    stage_menuitems[j] = item;
  }
  stage_menuitems[menulen] = nullitem;
  display_menuitems[DM_STAGE].user_data(stage_menuitems);
  
  menulen = app->display->num_stereo_modes();
  stereo_menuitems_storage = new Fl_Menu_Item[menulen+2]; 
  stereo_menuitems_storage[0] = nullitem;
  stereo_menuitems = stereo_menuitems_storage+1; 
  for (j=0; j<menulen; j++) {
    Fl_Menu_Item item = {app->display->stereo_name(j), 0, stereo_cb, vmdapp, FL_MENU_RADIO}; 
    stereo_menuitems[j] = item;
  }
  stereo_menuitems[menulen] = nullitem;
  display_menuitems[DM_STEREO].user_data(stereo_menuitems);

  menulen = 2;
  stereoswap_menuitems_storage = new Fl_Menu_Item[menulen+2]; 
  stereoswap_menuitems_storage[0] = nullitem;
  stereoswap_menuitems = stereoswap_menuitems_storage+1; 
  for (j=0; j<menulen; j++) {
    const char * StereoSwap[] = { "Off", "On" };
    Fl_Menu_Item item = {StereoSwap[j], 0, stereoswap_cb, vmdapp, FL_MENU_RADIO}; 
    stereoswap_menuitems[j] = item;
  }
  stereoswap_menuitems[menulen] = nullitem;
  display_menuitems[DM_STEREOEYESWAP].user_data(stereoswap_menuitems);

#if !defined(VMDLEANGUI)
  menulen = app->display->num_cache_modes();
  cachemode_menuitems_storage = new Fl_Menu_Item[menulen+2]; 
  cachemode_menuitems_storage[0] = nullitem;
  cachemode_menuitems = cachemode_menuitems_storage+1; 
  for (j=0; j<menulen; j++) {
    Fl_Menu_Item item = {app->display->cache_name(j), 0, cachemode_cb, vmdapp, FL_MENU_RADIO}; 
    cachemode_menuitems[j] = item;
  }
  cachemode_menuitems[menulen] = nullitem;
  display_menuitems[DM_CACHEMODE].user_data(cachemode_menuitems);
#endif
  
  menulen = app->display->num_render_modes();
  rendermode_menuitems_storage = new Fl_Menu_Item[menulen+2]; 
  rendermode_menuitems_storage[0] = nullitem;
  rendermode_menuitems = rendermode_menuitems_storage+1; 
  for (j=0; j<menulen; j++) {
    Fl_Menu_Item item = {app->display->render_name(j), 0, rendermode_cb, vmdapp, FL_MENU_RADIO}; 
    rendermode_menuitems[j] = item;
  }
  rendermode_menuitems[menulen] = nullitem;
  display_menuitems[DM_RENDERMODE].user_data(rendermode_menuitems);

  update_dispmode();

  menulen = sizeof(init_mouse_menuitems)/sizeof(Fl_Menu_Item);
  mouse_menuitems_storage = new Fl_Menu_Item[menulen+2]; 
  mouse_menuitems_storage[0] = nullitem;
  mouse_menuitems = mouse_menuitems_storage+1; 
  for (j=0; j<menulen; j++)
    mouse_menuitems[j] = init_mouse_menuitems[j];

  
  // ******** CREATE MENU BAR *********
  menubar = new Fl_Menu_Bar(0, 0, MAINFLTKMENUWIDTH, 30);
#if defined(VMDMENU_WINDOW)
  menubar->color(VMDMENU_WINDOW);
#endif
  menubar->add("File",0,0,(void *)file_menuitems,FL_SUBMENU_POINTER);
  menubar->add("Molecule",0,0,(void *)molecule_menuitems,FL_SUBMENU_POINTER);
  menubar->add("Graphics",0,0,(void *)graphics_menuitems, FL_SUBMENU_POINTER);
  menubar->add("Display",0,0,(void*)display_menuitems, FL_SUBMENU_POINTER);
  menubar->add("Mouse",0,0,(void *)mouse_menuitems, FL_SUBMENU_POINTER);
  menubar->add(EXT_MENU_NAME,0,0, NULL, FL_SUBMENU);
  menubar->add("Help",0,0,(void *)init_help_menuitems, FL_SUBMENU_POINTER);
  menubar->user_data(vmdapp);
  menubar->selection_color(VMDMENU_MENU_SEL);

  // ******** CREATE CONTROLS *********
  Fl_Group::current()->resizable(browser);

  Fl_Button *b;
  int bwidth = 20, bheight = 20;
  b = new Fl_Button(0, 150, bwidth, bheight, "@4->|");
  VMDFLTKTOOLTIP(b, "Jump to beginning")
  b->labeltype(FL_SYMBOL_LABEL);
  b->callback(start_cb, app);

  reverse = new Fl_Button(0, 150+bheight, bwidth, bheight, "@<");
  VMDFLTKTOOLTIP(reverse, "Play in reverse")
  reverse->labeltype(FL_SYMBOL_LABEL);
  reverse->type(FL_TOGGLE_BUTTON);
  reverse->callback(reverse_cb, app);

  b = new Fl_Button(bwidth, 150+bheight, bwidth, bheight, "@<|");
  VMDFLTKTOOLTIP(b, "Step in reverse")
  b->labeltype(FL_SYMBOL_LABEL);
  b->callback(prev_cb, app);

  b = new Fl_Button(MAINFLTKMENUWIDTH-bwidth, 150, bwidth, bheight, "@->|");
  VMDFLTKTOOLTIP(b, "Jump to end")
  b->labeltype(FL_SYMBOL_LABEL);
  b->callback(stop_cb, app);

  forward = new Fl_Button(MAINFLTKMENUWIDTH-bwidth, 150+bheight,
                          bwidth, bheight, "@>");
  VMDFLTKTOOLTIP(forward, "Play forward")
  forward->labeltype(FL_SYMBOL_LABEL);
  forward->type(FL_TOGGLE_BUTTON);
  forward->callback(forward_cb, app);

  b = new Fl_Button(MAINFLTKMENUWIDTH-2*bwidth, 150+bheight,
                    bwidth, bheight, "@|>");
  VMDFLTKTOOLTIP(b, "Step forward")
  b->labeltype(FL_SYMBOL_LABEL);
  b->callback(next_cb, app);
  
  curframe = new Fl_Int_Input(bwidth, 150, 2*bwidth, bheight);
  VMDFLTKTOOLTIP(curframe, "Set current frame")
  curframe->textsize(12);
  curframe->callback(curframe_cb, app);
  curframe->when(FL_WHEN_ENTER_KEY);
  curframe->selection_color(VMDMENU_VALUE_SEL2);

  frameslider = new Fl_Slider(3*bwidth, 150,
                              MAINFLTKMENUWIDTH-4*bwidth, bheight);
  VMDFLTKTOOLTIP(frameslider, "Drag to set current frame")
  frameslider->type(FL_HOR_NICE_SLIDER);
  frameslider->step(1,1);
  frameslider->callback(frameslider_cb, this);
  frameslider->color(VMDMENU_SLIDER_BG, VMDMENU_SLIDER_FG);
  frameslider->when(FL_WHEN_CHANGED | FL_WHEN_RELEASE);

  step = new Fl_Counter(220,150+bheight, 45,bheight, "step");
  VMDFLTKTOOLTIP(step, "Animation step size")
  step->labelsize(12);
  step->type(FL_SIMPLE_COUNTER);
  step->step(1,1);
  step->minimum(1);
  step->value(1);
  step->callback(step_cb, app);
  step->align(FL_ALIGN_LEFT);

  style = new Fl_Choice(120, 150+bheight, 65, bheight);
  VMDFLTKTOOLTIP(style, "Set animation looping mode")
  style->textsize(12);
  style->selection_color(VMDMENU_MENU_SEL);
  style->box(FL_THIN_UP_BOX);
  for (int s=0; s<Animation::ANIM_TOTAL_STYLES; s++)
    style->add(animationStyleName[s]);

  // XXX The Animation class starts with ANIM_LOOP as its style, so that's
  // what we do, too.
  style->value(1);
  style->callback(style_cb, app);

  zoom = new Fl_Check_Button(80, 150+bheight-2, bwidth+5, bheight+5, "zoom");
  VMDFLTKTOOLTIP(zoom, "Zoom in slider onto 100-frame subrange centered on current frame")
  zoom->labelsize(12);
  zoom->align(FL_ALIGN_LEFT);
  zoom->value(0);
  //zoom->selection_color(FL_RED);
  zoom->color(VMDMENU_CHECKBOX_BG, VMDMENU_CHECKBOX_FG);
  zoom->callback(zoom_cb, this);

  speed = new Fl_Slider(315, 150+bheight, 90, bheight, "speed");
  VMDFLTKTOOLTIP(speed, "Drag slider to change animation speed")
  speed->labelsize(12);
  speed->type(FL_HORIZONTAL);
  speed->color(VMDMENU_SLIDER_BG, VMDMENU_SLIDER_FG);
  speed->value(1.0);
  speed->callback(speed_cb, app);
  speed->align(FL_ALIGN_LEFT);

  guistate = UNDEFINED;
  update_gui_state();

  callback(vmd_main_window_cb); // override default FLTK/VMD global handlers
          
  Fl_Window::end();
}
 
int MainFltkMenu::act_on_command(int type, Command *cmd) {
  if (type == Command::MOL_NEW) {
    // XXX force set of anim style to the current GUI setting
    // when new molecules are loaded, since they get the default otherwise
    app->animation_set_style(style->value());
  } 

  if (type == Command::MOL_ACTIVE || 
      type == Command::MOL_ON ||
      type == Command::MOL_FIX  || 
      type == Command::MOL_NEW ||
      type == Command::MOL_RENAME ||
      type == Command::MOL_VOLUME ||
      type == Command::ANIM_NEW_NUM_FRAMES ||
      type == Command::MOL_DEL ||
      type == Command::MOL_TOP
     ) {
    browser->update();
  }

  if (type == Command::MOL_TOP || 
      type == Command::MOL_DEL || // XXX ought to emit a MOL_TOP too, IMHO
      type == Command::MOL_NEW ||
      type == Command::MOL_VOLUME ||
      type == Command::ANIM_JUMP ||
      type == Command::ANIM_NEW_NUM_FRAMES ||
      type == Command::ANIM_NEW_FRAME) {
    int id = app->molecule_top();
    int frame = app->molecule_frame(id);
    if (type != Command::ANIM_NEW_FRAME) {
      int max = app->molecule_numframes(id);
      frameslider->range(0, max-1);  
    } 
    frameslider->value(frame);
    char buf[20];
    sprintf(buf, "%d", frame);
    curframe->value(buf);
    if (type == Command::ANIM_JUMP) {
      forward->value(0);
      reverse->value(0);
    }
  } else if (type == Command::MOUSE_MODE) {
    update_mousemode(cmd);
  } else if (type == Command::DISP_DEPTHCUE  || type == Command::DISP_CULLING
          || type == Command::DISP_ANTIALIAS || type == Command::DISP_FPS
          || type == Command::DISP_LIGHT_ON  || type == Command::CMD_STAGE
          || type == Command::CMD_AXES       || type == Command::DISP_PROJ
          || type == Command::DISP_BACKGROUNDGRADIENT
          || type == Command::DISP_STEREO    || type == Command::DISP_STEREOSWAP
          || type == Command::DISP_CACHEMODE 
          || type == Command::DISP_RENDERMODE) {
    update_dispmode();
  } else if (type == Command::MENU_TK_ADD) {
    char *shortpath = ((CmdMenuExtensionAdd *)cmd)->menupath;
    char *longpath = new char[strlen(EXT_MENU_NAME)+strlen(shortpath)+2];
    sprintf(longpath, "%s/%s",EXT_MENU_NAME,((CmdMenuExtensionAdd *)cmd)->menupath);
    char *menuname = stringdup(((CmdMenuExtensionAdd *)cmd)->name);
    menubar->add(longpath, 0, menu_cb, menuname);
    delete[] longpath;
  } else if (type == Command::MENU_TK_REMOVE) {
    const Fl_Menu_Item *menubase = menubar->menu();
    int remove_menu_index = 0;
    int m;

    for (m=0; m<menubase->size(); m++) 
      if (!strcmp(menubase[m].label(), EXT_MENU_NAME)) break;
    const Fl_Menu_Item *extmenu = menubase+m;
    for (m=1; m<extmenu[1].size(); m++) 
      if (extmenu[m].user_data() && !strcmp((char*)extmenu[m].user_data(), ((CmdMenuExtensionRemove*)cmd)->name)) {
        remove_menu_index = extmenu-menubase+m;
        break;
      }
    if (remove_menu_index) menubar->remove(remove_menu_index);
  } else if (type == Command::ANIM_STYLE) {
    style->value((int)((CmdAnimStyle *)cmd)->newStyle);
  } else if (type == Command::ANIM_SKIP) {
    step->value(((CmdAnimSkip *)cmd)->newSkip);
  } else if (type == Command::ANIM_SPEED) {
    // XXX should put some kind of scaling in here to improve the dynamic
    // range of the slider.  Also put the inverse scaling in speed_cb.
    double val = ((CmdAnimSpeed *)cmd)->newSpeed;
    speed->value(val);
  } else if (type == Command::ANIM_DIRECTION) {
    Animation::AnimDir newDir = ((CmdAnimDir *)cmd)->newDir;
    forward->value(newDir == Animation::ANIM_FORWARD);
    reverse->value(newDir == Animation::ANIM_REVERSE);
  } else {
    return TRUE;
  }

  return FALSE;
}
 
MainFltkMenu::~MainFltkMenu() {
  delete[] file_menuitems;
  delete[] molecule_menuitems;
  delete[] display_menuitems;
  delete[] axes_menuitems_storage;
  delete[] backgroundmode_menuitems_storage;
  delete[] stage_menuitems_storage;
  delete[] stereo_menuitems_storage;
  delete[] stereoswap_menuitems_storage;
#if !defined(VMDLEANGUI)
  delete[] cachemode_menuitems_storage;
#endif
  delete[] rendermode_menuitems_storage;
  delete[] mouse_menuitems_storage;
  delete[] browserpopup_menuitems;
}

int MainFltkMenu::get_selected_molecule() {
  for (int j=0; j<browser->size(); j++)
    if (browser->selected(j+1)) 
      return j;

  return -1;
}

/// Updates the menu items dimming state to reflect the guistate (i.e. # of selected molecules)
void MainFltkMenu::update_menu_state(Fl_Menu_Item* mymenuitems, const MenuBehavior* mymenu_behavior) {
  int j;
  
  switch (guistate) {
    case MANY_SELECTED_MOL:
      for (j=0; mymenuitems[j].label(); j++) {
        if (mymenu_behavior[j] == MENU_NEED_UNIQUE_SEL) mymenuitems[j].deactivate();
        else mymenuitems[j].activate();
      }
      break;
    case ONE_SELECTED_MOL:
      for (j=0; mymenuitems[j].label(); j++)
        mymenuitems[j].activate();
      break;
    case NO_SELECTED_MOL:
      for (j=0; mymenuitems[j].label(); j++) {
        if (mymenu_behavior[j] & MENU_NEED_SEL) mymenuitems[j].deactivate();
        else mymenuitems[j].activate();
      }
      break;
    case UNDEFINED: //gets rid of g++ compiler warning 
     break;
  } 
}

    
void MainFltkMenu::update_gui_state() {
  char has_selected_mol = 0;
  int old_guistate = guistate;
  
  for (int item=1; item<=browser->size(); item++) {
    if (browser->selected(item)) { 
      has_selected_mol++; 
      if (has_selected_mol >= 2) break;
    }
  }

  if (has_selected_mol == 2) guistate = MANY_SELECTED_MOL;
  else if (has_selected_mol == 1) guistate = ONE_SELECTED_MOL;
  else if (!has_selected_mol) guistate = NO_SELECTED_MOL;

  // (de)activate the Molecule menu items 
  if (old_guistate != guistate) {
    update_menu_state(file_menuitems, file_menu_behavior);
    update_menu_state(molecule_menuitems, molecule_menu_behavior);
    update_menu_state(browserpopup_menuitems, browserpopup_menu_behavior);
  }
    
}



void MainFltkMenu::draw() {
#if defined(ARCH_MACOSX) || defined(ARCH_MACOSXX86) || defined(ARCH_MACOSXX86_64)
  size(MAINFLTKMENUWIDTH, h());
#endif
  Fl_Window::draw();
}

