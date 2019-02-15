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
 *      $RCSfile: SaveTrajectoryFltkMenu.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Window to allow the user to save trajectory frames etc.
 ***************************************************************************/
#ifndef SAVE_TRAJECTORY_FLTK_MENU_H 
#define SAVE_TRAJECTORY_FLTK_MENU_H 

#include "VMDFltkMenu.h"

class Fl_Browser;
class Fl_Button;
class Fl_Return_Button;
class Fl_Choice;
class Fl_Widget;
class Fl_Input;

/// VMDFltkMenu subclass implementing a GUI for saving trajectory 
/// and coordinate frames to files
class SaveTrajectoryFltkMenu : public VMDFltkMenu {
public:
  SaveTrajectoryFltkMenu(VMDApp *);
  int selectmol(int molindex);
    
  void do_save();

  /// "Activates" the currently selected molchooser molecule, must be called
  /// after all calls to molchooser->value(xxx) 
  void molchooser_activate_selection();

  // set the selected atoms text
  void select_atoms(const char *);

protected:
  int act_on_command(int, Command *);
  int selected_molid;
         
private:
  Fl_Choice *filetypechooser;
  Fl_Return_Button *savebutton;
  Fl_Button *closebutton;
  Fl_Choice *molchooser;

  Fl_Input *selectinput;
  Fl_Choice *repchooser;

  /// GUI elements for selecting frames, copied from FileChooser
  Fl_Group *timestepgroup;
  Fl_Button *allatoncebutton;
  Fl_Button *saveinbackgroundbutton;
  Fl_Input *firstinput;
  Fl_Input *lastinput;
  Fl_Input *strideinput;

  Fl_Browser *datasetbrowser;
  
  /// Rebuilds the list of existing molecule names and tries to keep the same 
  /// molecule selected as before (if it still exists)
  void update_molchooser(int type);
  
};
#endif
