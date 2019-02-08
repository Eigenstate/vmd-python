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
 *      $RCSfile: VMDFltkMenu.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.29 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Class to manage FLTK menus within VMD.
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <FL/Fl.H>
#include <FL/Fl_Choice.H>
#include "VMDFltkMenu.h"
#include "VMDApp.h"
#include "utilities.h"

void VMDFltkMenu::window_cb(Fl_Widget *w, void *) {
  VMDFltkMenu *m = (VMDFltkMenu *)w;
  m->app->menu_show(m->get_name(), 0);
}


VMDFltkMenu::VMDFltkMenu(const char *menuname,const char *title,VMDApp *vmdapp) 
: VMDMenu(menuname, vmdapp), Fl_Window(0,0,NULL) {
  _title=stringdup(title);
  Fl_Window::label(_title);
#if defined(VMDMENU_WINDOW)
  Fl_Window::color(VMDMENU_WINDOW);
#endif
  callback(window_cb);
}


VMDFltkMenu::~VMDFltkMenu() {
  delete [] _title;
}


void VMDFltkMenu::do_on() {
  Fl_Window::show();
}


void VMDFltkMenu::do_off() {
  Fl_Window::hide();
}


void VMDFltkMenu::move(int x, int y) {
  Fl_Widget::position(x,y);
}


void VMDFltkMenu::where(int &x, int &y) {
  x = Fl_Widget::x();
  y = Fl_Widget::y();
}


void fill_fltk_molchooser(Fl_Choice *choice, VMDApp *app, 
                          const char *extramenu) {
  int nummols = app->num_molecules();
  int has_extra = (extramenu == NULL) ? 0 : 1;
  int need_full_regen = 0;
  char buf[1024];

  // compute number of items in the menu, not counting the ending NULL marker.
  int menusz = choice->size() - 1;

#if 1
  // Optimize the cases where the chooser must be regenerated:
  // If the chooser is empty, we need to regenerate.
  // If the chooser is larger than needed, we have to 
  // re-create it from scratch since some molecule was 
  // potentially deleted from the middle.
  // If the size remains fixed, then we need to regenerate
  // because we are getting called to rename an existing molecule or
  // to change its state.
  // If the size grows by exactly one, then we can just add the new
  // molecule, and we don't need to regen, otherwise we have to regen.
  if ((menusz <= has_extra) ||
      (menusz >= (nummols + has_extra)) || 
      ((nummols + has_extra) - menusz) > 1) {
    need_full_regen = 1;
// printf("full regen: msz: %d sz: %d  N: %d  extra: %d\n", 
//        choice->size(), menusz, nummols, has_extra);
  } else {
// printf("add-newmol: msz: %d sz: %d  N: %d  extra: %d\n", 
//        choice->size(), menusz, nummols, has_extra);
  }
#else
  need_full_regen = 1;
#endif

  // either update (add an item) or completely regenerate the chooser contents
  if (need_full_regen) {
    choice->clear();

    if (has_extra)
      choice->add(extramenu);

    for (int i=0; i<nummols; i++) {
      int id = app->molecule_id(i);
      const char *s = app->molecule_name(id); 

      // Truncate molecule name to first 25 chars...
      // We must ensure we never find menu items by name, or that we 
      // truncate the search string identically to the way we do it here.
      sprintf(buf, "%d: %-.25s%s", id, s, 
              app->molecule_is_displayed(id) ? "" : " (off)");

      // Fltk doesn't allow adding a menu item with the same name as
      // an existing item, so we use replace, which also avoids 
      // problems with the escape characters interpreted by add()
      int ind = choice->add("foobar");
      choice->replace(ind, buf);
    }
  } else {
    // we just need to add one new molecule...
    int i = nummols - 1;
    int id = app->molecule_id(i);
    const char *s = app->molecule_name(id); 

    // Truncate molecule name to first 25 chars...
    // We must ensure we never find menu items by name, or that we 
    // truncate the search string identically to the way we do it here.
    sprintf(buf, "%d: %-.25s%s", id, s, 
            app->molecule_is_displayed(id) ? "" : " (off)");

    // Fltk doesn't allow adding a menu item with the same name as
    // an existing item, so we use replace, which also avoids 
    // problems with the escape characters interpreted by add()
    int ind = choice->add("foobar");
    choice->replace(ind, buf);
  }
}


char * escape_fltk_menustring(const char * menustring) {
  char * newstr;
  int len = strlen(menustring);
  int i, j;

  // don't bother being precise, these are just menu strings, and they're
  // going to be freed immediately, so allocate largest possible memory block
  // we'll ever need (every char being escape) and avoid running through the
  // string twice to accurately count the number of escaped characters.
  newstr = (char *) malloc(((len * 2) + 1) * sizeof(char)); 
  if (newstr == NULL) 
    return NULL;

  i=0;
  j=0;
  while (menustring[i] != '\0') {
    // insert an escape character if necessary
    if (menustring[i] == '/' ||
        menustring[i] == '\\' ||
        menustring[i] == '_') {
      newstr[j] = '\\'; 
      j++;
    } else if (menustring[i] == '&') {
      // FLTK won't escape '&' characters for some reason, so I skip 'em
      i++;
      continue;
    }

    newstr[j] = menustring[i];
    i++;
    j++;
  }  
  newstr[j] = '\0'; // null terminate the string

  return newstr;
}


// Find the menu index with a name that matches the string.
// Only checks leaf node menu names, not full pathnames currently
static int find_menu_from_string(const char *namestr, const Fl_Menu *m) {
  // don't crash and burn on a NULL name or menu object
  if (namestr == NULL || m == NULL)
    return -1;

  // FLTK 1.1.7 XXX should do the same use the built-in
  // find_item() or item_pathname() routines to do the same work we do below

  // FLTK 1.1.4 -- only checks leaf node menu names, ignores full pathname
  // find leaf menu name from full menu path
  const char *nstr;
  if ((nstr = strrchr(namestr, '/')) == NULL)
    nstr = namestr;
  else
    nstr++;

  int i, val;
  for (val=-1, i=0; i<m->size()-1; i++) {
    const char *mstr = m[i].text;
    // compare leaf submenu item name against left color mode name
    if ((mstr != NULL) && (!strcmp(nstr, mstr))) {
      val=i;
      break;
    }
  }

  return val;
}


// Set the chooser to the menu name matching the given string.
// Only checks the leaf node menu names, not full pathnames currently
void set_chooser_from_string(const char *namestr, class Fl_Choice *chooser) {
  int m = find_menu_from_string(namestr, chooser->menu());

  // don't set the menu if we can't find the string
  if (m >= 0)
    chooser->value(m);
}



