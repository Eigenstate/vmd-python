/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

#ifndef MOLBROWSER_H
#define MOLBROWSER_H

#include "FL/Fl_Multi_Browser.H"
#include "FL/Fl_Menu_Button.H"

#include "MainFltkMenu.h"

class VMDApp;

/// Fl_Multi_Browser subclass that keeps track of the main VMD menu and
/// a VMDApp context.
class MolBrowser : public Fl_Multi_Browser {
private:
  VMDApp *app;
  MainFltkMenu *mainmenu;
  int dragpending; // flag indicating a pending drag-and-drop paste action

public:
  /// Pass the parent MainFltkMenu to the MolBrowser init'er so that it
  /// can communicate molecule selection events back to the parent; pass NULL
  /// to disable such communication.
  MolBrowser(VMDApp *, MainFltkMenu *, int x, int y, int xw, int yw);

  void update();
  int handle(int);
};

#endif
