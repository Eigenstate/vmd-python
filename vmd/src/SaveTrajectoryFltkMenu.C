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
 *      $RCSfile: SaveTrajectoryFltkMenu.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.38 $       $Date: 2011/10/07 01:31:04 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Window to allow the user to save sets of trajectory frames etc.
 ***************************************************************************/

#include "SaveTrajectoryFltkMenu.h"
#include "Command.h"
#include "VMDApp.h"
#include "MolFilePlugin.h"
#include "frame_selector.h"
#include "CmdAnimate.h"
#include "AtomSel.h"
#include "MoleculeList.h"
#include "VolumetricData.h"

#include <stdio.h>
#include <FL/Fl.H>
#include <FL/forms.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Hold_Browser.H>
#include <FL/Fl_Return_Button.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Multi_Browser.H>
#include "utilities.h"

static void save_cb(Fl_Widget *, void *v) {
  ((SaveTrajectoryFltkMenu *)v)->do_save();
}

void SaveTrajectoryFltkMenu::do_save() {
  int ind = molchooser->value()-1;
  if (ind < 0) {
    fl_alert("Please select a molecule first.");
    return;
  }
  int molid = app->molecule_id(ind);
  if (molid < 0) return;
  // Make sure the user selected a file type
  if (filetypechooser->value() < 0) {
    fl_alert("Please select a file type first.");
    return;
  }

  const char *filetype = filetypechooser->text();
  char mask[20];
  sprintf(mask, "*.%s", (const char *)filetypechooser->mvalue()->user_data());

  // read and check selected frames
  const char *firststr = firstinput->value();
  const char *laststr = lastinput->value();
  const char *stridestr = strideinput->value();
  int max = app->molecule_numframes(molid)-1;
  int first = strlen(firststr) ? atoi(firststr) : 0;
  int last = strlen(laststr) ? atoi(laststr) : max;
  int stride = strlen(stridestr) ? atoi(stridestr) : 1;
  int waitfor = allatoncebutton->value() ? -1 : 0;

  if (first < 0) first=0;
  if (last > max) last=max;
  if (stride < 1) stride=1;

  if ((last-first)/stride < 0) {
    fl_alert("No timesteps selected; trajectory file will not be written.");
    return;
  }

  // check that selected atoms is valid
  const char *seltext = selectinput->value();
  AtomSel *atomsel = NULL;
  if (strlen(seltext)) {
    atomsel = new AtomSel(app->atomSelParser, molid);
    if (atomsel->change(seltext, app->moleculeList->mol_from_id(molid)) != AtomSel::PARSE_SUCCESS) {
      delete atomsel;
      fl_alert("Invalid atom selection: %s", seltext);
      return;
    }
  }

  char *fname = app->vmd_choose_file(
    "Choose filename to save trajectory", mask, filetype, 1);
  if (fname) {
    FileSpec spec;
    spec.first = first;
    spec.last = last;
    spec.stride = stride;
    spec.waitfor = waitfor;
    if (atomsel) {
      spec.selection = atomsel->on;
    }
    if (app->molecule_savetrajectory(molid, fname, filetype, &spec) < 0) {
      /// XXX would be nice if we had some idea of what went wrong...
      fl_alert("Error writing trajectory file.");
    }
    delete [] fname;
  }
  delete atomsel;
}

static void molchooser_cb(Fl_Widget *, void *v) {
  ((SaveTrajectoryFltkMenu *)v)->molchooser_activate_selection();
}

void SaveTrajectoryFltkMenu::molchooser_activate_selection() {
  int m;
  selected_molid = app->molecule_id(molchooser->value()-1);
  if (selected_molid < 0) return;
  
  // update frames
  int numframes = app->molecule_numframes(selected_molid); 
  firstinput->value("0");
  {
    char buf[32];
    sprintf(buf, "%d", numframes-1);
    lastinput->value(buf);
  }
  strideinput->value("1");

  // update volumetric datasets
  Molecule *mol = app->moleculeList->mol_from_id(selected_molid);
  datasetbrowser->clear();
  for (m=0; m<mol->num_volume_data(); m++) {
    datasetbrowser->add(mol->get_volume_data(m)->name);
  }

  // update reps in repchooser
  repchooser->clear();
  repchooser->add("Current selections:");
  for (m=0; m<app->num_molreps(selected_molid); m++) {
    repchooser->add(app->molrep_get_selection(selected_molid, m));
  }
  repchooser->value(0);
}

static void repchooser_cb(Fl_Widget *w, void *v) {
  Fl_Choice *c = (Fl_Choice *)w;
  if (c->value()) 
    ((SaveTrajectoryFltkMenu *)v)->select_atoms(c->text());
}

void SaveTrajectoryFltkMenu::select_atoms(const char *sel) {
  selectinput->value(sel);
}

SaveTrajectoryFltkMenu::SaveTrajectoryFltkMenu(VMDApp *vmdapp)
: VMDFltkMenu("save", "Save Trajectory", vmdapp) {
  
  size(450, 250);
    { Fl_Choice* o = molchooser = new Fl_Choice(120, 10, 320, 25, "Save data from: ");
      o->box(FL_THIN_UP_BOX);
      o->down_box(FL_BORDER_BOX);
      o->color(VMDMENU_CHOOSER_BG);
      o->selection_color(VMDMENU_CHOOSER_SEL);
      o->callback(molchooser_cb, this);
    }
    { Fl_Input *o = selectinput = new Fl_Input(120, 45, 295, 25, "Selected atoms:");
      o->selection_color(VMDMENU_VALUE_SEL);
    }
    { Fl_Choice* o = repchooser = new Fl_Choice(415, 45, 25, 25);
      o->down_box(FL_BORDER_BOX);
      o->align(FL_ALIGN_TOP_LEFT);
      o->color(VMDMENU_CHOOSER_BG, VMDMENU_CHOOSER_SEL);
      o->callback(repchooser_cb, this);
    }
    { Fl_Choice* o = filetypechooser = new Fl_Choice(20, 90, 115, 25, "File type:");
      o->down_box(FL_BORDER_BOX);
      o->align(FL_ALIGN_TOP_LEFT);
      o->color(VMDMENU_CHOOSER_BG, VMDMENU_CHOOSER_SEL);
    }
    savebutton = new Fl_Return_Button(345, 90, 95, 25, "Save...");
    savebutton->callback(save_cb, this);
    { Fl_Group* o = timestepgroup = new Fl_Group(20, 145, 165, 95, "Frames: ");
      o->box(FL_ENGRAVED_FRAME);
      o->align(FL_ALIGN_TOP_LEFT);
      { Fl_Button* o = saveinbackgroundbutton = new Fl_Round_Button(30, 215, 150, 20, "Save in background");
        o->down_box(FL_ROUND_DOWN_BOX);
        o->type(FL_RADIO_BUTTON);
      }
      { Fl_Button* o = allatoncebutton = new Fl_Round_Button(30, 195, 150, 20, "Save all at once");
        o->down_box(FL_ROUND_DOWN_BOX);
        o->type(FL_RADIO_BUTTON);
      }
      { Fl_Input* o = firstinput = new Fl_Int_Input(25, 170, 45, 20, "First:");
        o->align(FL_ALIGN_TOP);
        o->selection_color(VMDMENU_VALUE_SEL);
      }
      { Fl_Input* o = lastinput = new Fl_Int_Input(80, 170, 45, 20, "Last:");
        o->align(FL_ALIGN_TOP);
        o->selection_color(VMDMENU_VALUE_SEL);
      }
      { Fl_Input* o = strideinput = new Fl_Int_Input(135, 170, 45, 20, "Stride:");
        o->align(FL_ALIGN_TOP);
        o->selection_color(VMDMENU_VALUE_SEL);
      }
      o->end();
      datasetbrowser = new Fl_Multi_Browser(195, 145, 240, 95, "Volumetric Datasets");
      datasetbrowser->align(5);
      datasetbrowser->color(VMDMENU_BROWSER_BG, VMDMENU_BROWSER_SEL);
    } 
    end();

  allatoncebutton->value(1);
  selected_molid = -1;
  datasetbrowser->deactivate();
  
  command_wanted(Command::PLUGIN_UPDATE);
  command_wanted(Command::MOL_NEW);
  command_wanted(Command::MOL_RENAME);
  command_wanted(Command::MOL_DEL);
  command_wanted(Command::MOL_ADDREP);
  command_wanted(Command::MOL_DELREP);
  command_wanted(Command::MOL_MODREP);
  command_wanted(Command::MOL_MODREPITEM);
  command_wanted(Command::ANIM_DELETE);
}

// This rebuilds the molecule chooser menu and selects the item 
// corresponding to the previouly selected molid
void SaveTrajectoryFltkMenu::update_molchooser(int type) {
  if (type == Command::MOL_NEW ||
      type == Command::MOL_DEL ||  
      type == Command::MOL_RENAME) {
    // don't regen the molecule list for rep changes
    fill_fltk_molchooser(molchooser, app, "No Molecule Selected");
  }
 
  molchooser->value(0);
#if 1
  int m = app->molecule_index_from_id(selected_molid);
  if (m >= 0) {
    molchooser->value(m);
    molchooser_activate_selection();
  }
#else
  for (int m=1; m<molchooser->size()-1; m++) { 
    int tmpid = app->molecule_id(m-1);
    if (tmpid == selected_molid) {
      molchooser->value(m);
      molchooser_activate_selection();
      break;
    }
  }
#endif
}


// XXX just like FilesFltkMenu - maybe we should be subclassing or something
int SaveTrajectoryFltkMenu::act_on_command(int type, Command *command) {
  switch(type) {
    case Command::PLUGIN_UPDATE: 
      {
        filetypechooser->clear();
        PluginList plugins;
        int n = app->list_plugins(plugins, "mol file reader");
        for (int p=0; p<n; p++) {
          MolFilePlugin m(plugins[p]);
          if (m.can_write_timesteps())
            filetypechooser->add(m.name(), 0, 0, (void *)m.extension());
        }

        // set the chooser to the first item initially, update from there
        filetypechooser->value(0);

        // Make PDB the default filetype, if we can find it in the menu
        // XXX Current code is listing them using extensions rather than the
        //     pretty names.
        // XXX This should be updated to use the built-in FLTK routines
        //     for FLTK versions 1.1.7 and newer.
#if 1
        set_chooser_from_string("pdb", filetypechooser);
#else
        Fl_Menu_Item *pdbitem = NULL;
        pdbitem = filetypechooser->find_item("pdb");
        if (pdbitem)
          filetypechooser->value(pdbitem); // set "pdb" as default
#endif
      }
      break;

    case Command::MOL_NEW:
    case Command::MOL_DEL:
    case Command::MOL_RENAME:
    case Command::MOL_ADDREP:
    case Command::MOL_DELREP:
    case Command::MOL_MODREP:
    case Command::MOL_MODREPITEM:
      update_molchooser(type);
      break;

    case Command::ANIM_DELETE:
      if (selected_molid == ((CmdAnimDelete*)command)->whichMol)
        molchooser_activate_selection(); //to update the number of frames
      break;

    default:
      return 0;
  }

  return 1;
}


int SaveTrajectoryFltkMenu::selectmol(int molindex) {
  if (molindex < 0 || molindex >= app->num_molecules())
      molchooser->value(0);
  else {
    molchooser->value(molindex+1);
    molchooser_activate_selection();
  }
  return TRUE;
}


