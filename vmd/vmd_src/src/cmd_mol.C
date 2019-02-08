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
 *      $RCSfile: cmd_mol.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.133 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for molecule control
 ***************************************************************************/

#include <ctype.h>
#include <stdlib.h>
#include <tcl.h>

#include "config.h"
#include "utilities.h"
#include "VMDApp.h"
#include "TclCommands.h"
#include "VMDDisplayList.h"
#include "MoleculeList.h"

// the following uses the Cmdtypes MOL_NEW, MOL_DEL, MOL_ACTIVE,
// MOL_FIX, MOL_ON, MOL_TOP, MOL_SELECT, MOL_REP, MOL_COLOR, MOL_ADDREP,
// MOL_MODREP, MOL_DELREP, MOL_MODREPITEM


/*
 * NOTES:
 *
 * 1) When referring to a molecule in a command by a number, the
 * unique molecule ID is used, NOT the relative index of the molecule into
 * the molecule list.  This is because the relative index changes as molecules
 * are added/removed, but the unique ID stays the same until the molecule is
 * deleted.
 *
 */


/// calculate which molecules to operate upon from the given molecule name;
/// put the data in idList, and return the pointer to the integer list.
/// If NULL is returned, there was an error or no molecules were specified.
/// If an error occurs, this prints out the error message as well.
///
/// Molecule names are of the form
///      <n1>[|<n2>[.... [|<nN>]...]]
/// where <ni> is either "all", "top", "active", "inactive", "displayed",
/// "on", "off", "fixed", "free", "none", or an ID.
/// There should be no spaces in the name.
class IdList {
private:
  int *idList;
  int selectedMols;

  void molname_error(Tcl_Interp *interp, const char *fullname, const char *errpart) {
    char *errmsg = new char[200 + strlen(fullname) + strlen(errpart)];
    sprintf(errmsg, 
      "Illegal molecule specification '%s': Could not\nfind molecule '%s'. ",
      fullname, errpart);
    Tcl_SetResult(interp, errmsg, TCL_VOLATILE);
    delete [] errmsg;
  }

public:
  IdList() {
    idList = 0;
    selectedMols = -1;
  }
  int find(Tcl_Interp *, VMDApp *app, const char *txt, int *allmolsflag = NULL); // return num()
  ~IdList() {
    delete [] idList;
  }
  int num() const { return selectedMols; }
  int operator[](int i) const { return idList[i]; }
};

int IdList::find(Tcl_Interp *interp, VMDApp *app, const char *givenName, int *allmolsflag) {
  char name[512];      
  int i, numMols;

  if (interp == NULL || app == NULL || givenName == NULL) {
    Tcl_SetResult(interp, (char *) "ERROR) IdList::find() bad parameters", TCL_STATIC);
    return -1;
  }

  numMols = app->num_molecules();
  if (!numMols) {
    Tcl_SetResult(interp, (char *) "ERROR) No molecules loaded.", TCL_STATIC);
    return -1;
  }

  idList = new int[numMols];
  for (i=0; i<numMols; i++) idList[i] = 0;

  // start to tokenize the string, then
  // scan the string and look for molecule specifiers
  strcpy(name, givenName);
  char *tok = strtok(name, "|/:");
  while(tok) {
    // look for what kind of name this is
    if(isdigit(*tok)) {     // check if it is a molid 
      int n = atoi(tok);
      if (!app->molecule_valid_id(n)) {
        molname_error(interp, givenName, tok);
        return -1;
      }
      for (i=0; i<numMols; i++) {
        if (app->molecule_id(i) == n) {
          idList[i] = TRUE;
        }
      }
    } else if(!strupncmp(tok, "all", CMDLEN)) {     // check if "all"
      // tell caller that _all_ molecules were selected, to allow more
      // efficient processing where possible.
      if (allmolsflag != NULL) {
        *allmolsflag = 1; 
      }
      for(i=0; i < numMols; i++)
        idList[i] = TRUE;
    } else if(!strupncmp(tok, "none", CMDLEN)) {    // "none"
      for(i=0; i < numMols; i++)
        idList[i] = FALSE;
    } else if(!strupncmp(tok, "top", CMDLEN)) {     // "top"
      int top = app->molecule_top();
      for(i=0; i < numMols; i++)
        if (app->molecule_id(i) == top)
          idList[i] = 1;
    } else if(!strupncmp(tok, "active", CMDLEN)) {  // check if "active"
      for(i=0; i < numMols; i++)
        if (app->molecule_is_active(app->molecule_id(i))) 
          idList[i] = 1;
    } else if(!strupncmp(tok, "inactive", CMDLEN)) {    // check if "inactive"
      for(i=0; i < numMols; i++)
        if (!app->molecule_is_active(app->molecule_id(i)))
          idList[i] = 1;
    } else if(!strupncmp(tok, "displayed", CMDLEN) ||
    !strupncmp(tok, "on", CMDLEN)) {        // "displayed" or "on"
      for(i=0; i < numMols; i++)
        if (app->molecule_is_displayed(app->molecule_id(i)))
          idList[i] = 1;
    } else if(!strupncmp(tok, "off", CMDLEN)) { // check if "off"
      for(i=0; i < numMols; i++)
        if (!app->molecule_is_displayed(app->molecule_id(i)))
          idList[i] = 1;
    } else if(!strupncmp(tok, "fixed", CMDLEN)) {   // check if "fixed"
      for(i=0; i < numMols; i++)
        if (app->molecule_is_fixed(app->molecule_id(i)))
          idList[i] = 1;
    } else if(!strupncmp(tok, "free", CMDLEN) ||
    !strupncmp(tok, "unfix", CMDLEN)) {         // "free" or "unfix"
      for(i=0; i < numMols; i++)
        if (!app->molecule_is_fixed(app->molecule_id(i)))
          idList[i] = 1;
    } else {
      // bad molecule name; print error and return
      molname_error(interp, givenName, tok);
      return -1;
    }
    tok = strtok(NULL,"|/:");
  }

  // found the names; now convert the flag array to a list of id's.  
  selectedMols = 0;
  for(i=0; i < numMols; i++) {
    if(idList[i])
      idList[selectedMols++] = app->molecule_id(i);
  }
  return selectedMols;
}

static void print_mol_summary(Tcl_Interp *interp, VMDApp *app, int molid) {
  if (!app->molecule_valid_id(molid)) return;
  // everything except molecule name is bounded in size
  char *buf = new char[strlen(app->molecule_name(molid))+50];
  sprintf(buf, "%s  Atoms:%d  Frames (C): %d(%d)  Status:%c%c%c%c\n",
          app->molecule_name(molid), app->molecule_numatoms(molid),
          app->molecule_numframes(molid), app->molecule_frame(molid),
          (app->molecule_is_active(molid) ? 'A' : 'a'),
          (app->molecule_is_displayed(molid) ? 'D' : 'd'),
          (app->molecule_is_fixed(molid) ? 'F' : 'f'),
          (molid == app->molecule_top() ? 'T' : 't'));
  Tcl_AppendResult(interp, buf, NULL);
  delete [] buf;
}

static void print_arep_summary(Tcl_Interp *interp, VMDApp *app, int molid, 
                               int i) {
  if (!app->molecule_valid_id(molid)) return;
  if (i < 0 || i >= app->num_molreps(molid)) return;
  char buf[100];
  sprintf(buf, "%d: %s, %d atoms selected.\n", 
          i, (app->molecule_is_displayed(molid) ? " on" : "off"), 
          app->molrep_numselected(molid, i));
  Tcl_AppendResult(interp, buf, NULL);
  Tcl_AppendResult(interp, "  Coloring method: ", 
                   app->molrep_get_color(molid, i), "\n", NULL);
  Tcl_AppendResult(interp, "   Representation: ", 
                   app->molrep_get_style(molid, i), "\n", NULL);
  Tcl_AppendResult(interp, "        Selection: ", 
                   app->molrep_get_selection(molid, i), "\n", NULL);
}

static void cmd_mol_list(Tcl_Interp *interp, VMDApp *app, const char *moltxt) {
  IdList idList;
  if (idList.find(interp, app, moltxt) > 1) {
    Tcl_AppendResult(interp, "Molecule Status Overview:\n", NULL);
    Tcl_AppendResult(interp, "-------------------------\n", NULL);
    for (int i=0; i < idList.num(); i++)
      print_mol_summary(interp, app, idList[i]);
  } else if (idList.num() == 1) {
    Tcl_AppendResult(interp, "Status of molecule ", 
                     app->molecule_name(idList[0]), ":\n", NULL);
    print_mol_summary(interp, app, idList[0]);
    char buf[50];
    sprintf(buf, "Atom representations: %d\n", app->num_molreps(idList[0]));
    Tcl_AppendResult(interp, buf, NULL); 
    for (int i=0; i < app->num_molreps(idList[0]); i++)
      print_arep_summary(interp, app, idList[0], i);
  }
}

static void cmd_mol_usage(Tcl_Interp *interp) {
    Tcl_AppendResult(interp, "usage: mol <command> [args...]\n",
    "\nMolecules and Data:\n",
    "  new [file name] [options...]       -- load file into a new molecule\n",
    "  new atoms <natoms>                 -- generate a new molecule with 'empty' atoms\n",
    "  addfile <file name> [options...]   -- load files into existing molecule\n",
    "    options: type, first, last, step, waitfor, volsets, filebonds, autobonds, \n",
    "             <molid> (addfile only)\n",
    "  load <file type> <file name>       -- load molecule (obsolescent)\n" ,
    "  urlload <file type> <URL>          -- load molecule from URL\n" ,
    "  pdbload <four letter accession id> -- download molecule from the PDB\n",
    "  cancel <molid>                     -- cancel load/save of trajectory\n",  
    "  delete <molid>                     -- delete given molecule\n" , 
    "  rename <molid> <name>              -- Rename the specified molecule\n",     
    "  dataflag <molid> [set | unset] <flagname> -- Set/unset data output flags\n",
    "  list [all|<molid>]                 -- displays info about molecules\n",
    "\nMolecule GUI Properties:\n",
    "  top <molid>                        -- make that molecule 'top'\n",
    "  on <molid>                         -- make molecule visible\n" ,
    "  off <molid>                        -- make molecule invisible\n" ,
    "  fix <molid>                        -- don't apply mouse motions\n" ,
    "  free <molid>                       -- let mouse move molecules\n" ,
    "  active <molid>                     -- make molecule active\n" ,
    "  inactive <molid>                   -- make molecule inactive\n" ,
    "\nGraphical Representations:\n",
    "  addrep <molid>                     -- add a new representation\n" ,
    "  delrep <repid> <molid>             -- delete rep\n" ,
    "  default [color | style | selection | material] <value>\n",
    "  representation|rep <style>         -- set the drawing style for new reps\n" ,
    "  selection <selection>              -- set the selection for new reps\n" ,
    "  color <color>                      -- set the color for new reps\n" ,
    "  material <material>                -- set the material for new reps\n" ,
    "  modstyle <repid> <molid> <style>   -- change the drawing style for a rep\n" ,
    "  modselect <repid> <molid> <selection>  -- change the selection for a rep\n" ,
    "  modcolor <repid> <molid> <color>   -- change the color for a rep\n" ,
    "  modmaterial <repid> <molid> <material> -- change the material for a rep\n" ,
    "  repname <molid> <repid>            -- Get the name of a rep\n",
    "  repindex <molid> <repname>         -- Get the repid of a rep from its name\n",
    "  reanalyze <molid>                  -- Re-analyze structure after changes\n",
    "  bondsrecalc <molid>                -- Recalculate bonds, current timestep\n",
    "  ssrecalc <molid>                   -- Recalculate secondary structure (Cartoon)\n",
    "  selupdate <repid> <molid> [on|off] -- Get/Set auto recalc of rep selection\n",
    "  colupdate <repid> <molid> [on|off] -- Get/Set auto recalc of rep color\n",
    "  scaleminmax <molid> <repid> [<min> <max>|auto] -- Get/set colorscale minmax\n",
    "  drawframes <molid> <repid> [<framespec>|now] -- Get/Set drawn frame range\n", 
    "  smoothrep <molid> <repid> [smooth] -- Get or set trajectory smoothing value\n",
    "  showperiodic <molid> <repid> [flags] -- Get or set periodic image display\n",
    "  numperiodic <molid> <repid> <n>    -- Get or set number of periodic images\n",
    "  showrep <molid> <repid> [on|off]   -- Turn selected rep on or off\n",
    "  voldelete <molid> <volID> -- delete volumetric data\n",
    "  volmove <molid> <matrix> <volID>   -- transform volumetric data\n",
    "\nClipping Planes:\n",
    "  clipplane center <clipid> <repid> <molid> [<vector>]\n",
    "  clipplane color  <clipid> <repid> <molid> [<vector>]\n",
    "  clipplane normal <clipid> <repid> <molid> [<vector>]\n",
    "  clipplane status <clipid> <repid> <molid> [<mode>]\n",
    "  clipplane num\n",
    "\n",
    "See also the molinfo command\n",
    NULL);
}


int text_cmd_mol(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if (argc == 1) {
    cmd_mol_usage(interp);
    return TCL_ERROR;
  }
  if ((argc == 4 || argc == 6) && !strupncmp(argv[1], "load", CMDLEN)) {
    // load a new molecule
  
    // Special-case graphics molecules to load as "blank" molecules
    if (argc == 4 && !strupncmp(argv[2], "graphics", CMDLEN)) {
      int rc = app->molecule_new(argv[3], 0);
      Tcl_SetObjResult(interp, Tcl_NewIntObj(rc));
      return TCL_OK;
    }
    FileSpec spec;
    spec.waitfor = FileSpec::WAIT_ALL;
    int newmolid = app->molecule_load(-1, argv[3], argv[2], &spec);
    if (newmolid < 0) {
      Tcl_AppendResult(interp, "Unable to load structure file ",argv[3], NULL);
      return TCL_ERROR;
    }
    if (argc == 6) {
      if (app->molecule_load(newmolid, argv[5], argv[4], &spec) < 0) {
        Tcl_AppendResult(interp, "Unable to load coordinate file ", argv[5], 
                         NULL);
        return TCL_ERROR;
      }
    }
    Tcl_SetObjResult(interp, Tcl_NewIntObj(newmolid));

  } else if(argc < 4 && !strupncmp(argv[1], "list", CMDLEN)) {
    if(argc == 2)
      cmd_mol_list(interp, app, "all");
    else
      cmd_mol_list(interp, app, argv[2]);
  
  } else if (argc < 4 && !strupncmp(argv[1], "cancel", CMDLEN)) {
    IdList idList;
    if (argc == 2) {
      idList.find(interp, app, "all");
    } else {
      idList.find(interp, app, argv[2]);
    }
    for (int i=0; i<idList.num(); i++) app->molecule_cancel_io(idList[i]);
  
  } else if(!strupncmp(argv[1],"selection",CMDLEN)) {
    char *molstr = combine_arguments(argc, argv, 2);
    if (molstr) {
      app->molecule_set_selection(molstr);
      delete [] molstr;
    }
  } else if(!strupncmp(argv[1],"representation",CMDLEN) ||
            !strupncmp(argv[1],"rep",CMDLEN) ) {
    char *molstr = combine_arguments(argc, argv, 2);
    if (molstr) {
      app->molecule_set_style(molstr);
      delete [] molstr;
    }

  } else if(!strupncmp(argv[1],"color",CMDLEN)) {
    char *molstr = combine_arguments(argc, argv, 2);
    if (molstr) {
      app->molecule_set_color(molstr);
      delete [] molstr;
    }

  } else if(!strupncmp(argv[1],"material",CMDLEN)) {
    char *molstr = combine_arguments(argc, argv, 2);
    if (molstr) {
      app->molecule_set_material(molstr);
      delete [] molstr;
    }

  } else if(argc > 4 && (!strupncmp(argv[1],"modcolor",CMDLEN) ||
            !strupncmp(argv[1],"modstyle",CMDLEN) ||
            !strupncmp(argv[1],"modselect",CMDLEN) ||  
            !strupncmp(argv[1],"modmaterial",CMDLEN))) {
    IdList idList;
    if (idList.find(interp, app, argv[3]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0]; 
    char *molstr = combine_arguments(argc, argv, 4);
    if(molstr) {
      if(!strupncmp(argv[1],"modcolor",CMDLEN))
        app->molrep_set_color(molid, atoi(argv[2]), molstr);
      else if(!strupncmp(argv[1],"modstyle",CMDLEN))
        app->molrep_set_style(molid, atoi(argv[2]), molstr);
      else if(!strupncmp(argv[1],"modselect",CMDLEN))
        app->molrep_set_selection(molid, atoi(argv[2]), molstr);
      else if(!strupncmp(argv[1],"modmaterial",CMDLEN))
        app->molrep_set_material(molid, atoi(argv[2]), molstr);
      delete [] molstr;
    }
  
  } else if(argc == 3 && !strupncmp(argv[1],"addrep",CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) > 1) {
      
      Tcl_AppendResult(interp, "mol addrep operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    if (idList.num() < 1) return TCL_ERROR;
    if (!app->molecule_addrep(idList[0])) {
      Tcl_AppendResult(interp, "addrep: Unable to add rep to molecule ", 
                       argv[2], NULL);
      return TCL_ERROR;
    }
  } else if(argc == 4 && !strupncmp(argv[1],"delrep",CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[3]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    app->molrep_delete(idList[0], atoi(argv[2]));
  } else if(argc == 4 && !strupncmp(argv[1],"modrep",CMDLEN)) {
    // XXX This is freakin' lame - deprecate this, please!
    IdList idList;
    if (idList.find(interp, app, argv[3]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
    int repid = atoi(argv[2]);
    app->molrep_set_style(molid, repid, app->molecule_get_style());
    app->molrep_set_color(molid, repid, app->molecule_get_color());
    app->molrep_set_selection(molid, repid, app->molecule_get_selection());
    app->molrep_set_material(molid, repid, app->molecule_get_material());
  } else if(argc == 3 && !strupncmp(argv[1],"delete",CMDLEN)) {
    IdList idList;
    int allmolsflag=0;
    idList.find(interp, app, argv[2], &allmolsflag);

    if (allmolsflag) {
      // use most efficient molecule deletion code path
      app->molecule_delete_all();
    } else {
      // delete molecules piecemeal
      for (int i=0; i<idList.num(); i++) {
        app->molecule_delete(idList[i]);
      }
    }
  } else if(argc == 3 && (!strupncmp(argv[1],"active",CMDLEN) ||
            !strupncmp(argv[1],"inactive",CMDLEN))) {
    IdList idList;
    idList.find(interp, app, argv[2]);
    for (int i=0; i<idList.num(); i++) {
      app->molecule_activate(idList[i], !strupncmp(argv[1],"active",CMDLEN));
    }
  } else if(argc == 3 && (!strupncmp(argv[1],"on",CMDLEN) ||
            !strupncmp(argv[1],"off",CMDLEN))) {
    IdList idList;
    idList.find(interp, app, argv[2]);
    for (int i=0; i<idList.num(); i++) {
      app->molecule_display(idList[i], !strupncmp(argv[1],"on",CMDLEN));
    }
  } else if(argc == 3 && (!strupncmp(argv[1],"fix",CMDLEN) ||
            !strupncmp(argv[1],"free",CMDLEN))) {
    IdList idList;
    idList.find(interp, app, argv[2]);
    for (int i=0; i<idList.num(); i++) {
      app->molecule_fix(idList[i], !strupncmp(argv[1],"fix",CMDLEN));
    }

  } else if(argc == 3 && !strupncmp(argv[1],"top",CMDLEN)) {
    IdList idList;
    if (idList.find(interp,app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    app->molecule_make_top(idList[0]);
  } else if (argc == 4 && !strupncmp(argv[1], "urlload", CMDLEN)) {
    // load a file from a URL
    //   "mol urlload xyz http://www.umn.edu/test/me/out.xyz
    // vmd_mol_urlload url localfile 
    char *localfile = vmd_tempfile("urlload");
    char *buf = new char[strlen(localfile)+strlen(argv[3])+100];
    sprintf(buf, "vmd_mol_urlload %s %s", argv[3], localfile);
    int rc = Tcl_Eval(interp, buf);
    delete [] buf;
    if (rc != TCL_OK) {
      Tcl_AppendResult(interp, "vmd_mol_urllload failed.", NULL);
      delete [] localfile;
      return TCL_ERROR;
    }
    FileSpec spec;
    spec.waitfor = FileSpec::WAIT_ALL;
    int molid = app->molecule_load(-1, localfile, argv[2], &spec);
    delete [] localfile;
    if (molid < 0) {
      Tcl_AppendResult(interp, "urlload failed: unable to load downloaded file '", localfile, "' as file type ", argv[2], NULL);
      return TCL_ERROR;
    }
    Tcl_SetObjResult(interp, Tcl_NewIntObj(molid));
  } else if (argc == 3 && !strupncmp(argv[1], "pdbload", CMDLEN)) {
    // alias to "mol load webpdb ..."
    FileSpec spec;
    spec.waitfor = FileSpec::WAIT_ALL; 
    int rc = app->molecule_load(-1, argv[2], "webpdb", &spec);
    if (rc < 0) {
      Tcl_AppendResult(interp, "pdbload of '", argv[2], "' failed.", NULL);
      return TCL_ERROR;
    }
    Tcl_SetObjResult(interp, Tcl_NewIntObj(rc));
  } else if (argc == 5 && !strupncmp(argv[1], "dataflag", CMDLEN)) {
    IdList idList;
    idList.find(interp, app, argv[2]);
    int setval;

    if (!strcmp("set", argv[3])) {
      setval = 1;
    } else if  (!strcmp("unset", argv[3])) {
      setval = 0;
    } else {
      Tcl_AppendResult(interp, argv[1], " subcommand unrecognized", NULL);
      return TCL_ERROR;
    }

    for (int i=0; i<idList.num(); i++) {
      if (!app->molecule_set_dataset_flag(idList[i], argv[4], setval)) {
        Tcl_AppendResult(interp, " error setting dataset flag ", argv[4], NULL);
        return TCL_ERROR;
      }
    }
  } else if (argc == 3 && !strupncmp(argv[1], "reanalyze", CMDLEN)) {
    IdList idList;
    idList.find(interp, app, argv[2]);
    for (int i=0; i<idList.num(); i++) {
      app->molecule_reanalyze(idList[i]);
    }

  } else if (argc == 3 && !strupncmp(argv[1], "bondsrecalc", CMDLEN)) {
    IdList idList;
    idList.find(interp, app, argv[2]);
    for (int i=0; i<idList.num(); i++) {
      app->molecule_bondsrecalc(idList[i]);
    }

  } else if (argc == 3 && !strupncmp(argv[1], "ssrecalc", CMDLEN)) {
    IdList idList;
    idList.find(interp, app, argv[2]);
    for (int i=0; i<idList.num(); i++) {
      app->molecule_ssrecalc(idList[i]);
    }

  } else if (argc == 12 && !strupncmp(argv[1], "volume", CMDLEN)) {
    float origin[3], xaxis[3], yaxis[3], zaxis[3];
    int xsize, ysize, zsize;
    float *data;  // new'ed here; passed to molecule

    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
   
    if (tcl_get_vector(argv[4], origin, interp) != TCL_OK ||
        tcl_get_vector(argv[5], xaxis, interp) != TCL_OK ||
        tcl_get_vector(argv[6], yaxis, interp) != TCL_OK ||
        tcl_get_vector(argv[7], zaxis, interp) != TCL_OK) {
      return TCL_ERROR;
    }
    if (Tcl_GetInt(interp, argv[8], &xsize) != TCL_OK ||
        Tcl_GetInt(interp, argv[9], &ysize) != TCL_OK ||
        Tcl_GetInt(interp, argv[10], &zsize) != TCL_OK) {
      return TCL_ERROR;
    }
    long size = xsize*ysize*zsize;
    data = new float[size];
   
    int ndata;
    const char **dataAsString;
    if (Tcl_SplitList(interp, argv[11], &ndata, &dataAsString) != TCL_OK) {
      Tcl_SetResult(interp, (char *) "mol volume: invalid data block", TCL_STATIC);
      delete [] data;
      return TCL_ERROR;
    }
    if (ndata != size) {
      Tcl_SetResult(interp, (char *) "mol volume: size of data does not match specified sizes", TCL_STATIC);
      Tcl_Free((char *)dataAsString);
      delete [] data;
      return TCL_ERROR;
    }
    for (int i=0; i<ndata; i++) {
      double tmp;
      if (Tcl_GetDouble(interp, dataAsString[i], &tmp) != TCL_OK) {
        Tcl_SetResult(interp, (char *) "graphics: volume: non-numeric found in data block", TCL_STATIC);
        Tcl_Free((char *)dataAsString);
        delete [] data;
        return TCL_ERROR;
      }
      data[i] = (float)tmp;
    }
    app->molecule_add_volumetric(molid, argv[3], origin, xaxis, yaxis, zaxis,
                                 xsize, ysize, zsize, data);
    Tcl_Free((char *)dataAsString);

  } else if ((argc == 4 || argc == 5) && !strupncmp(argv[1], "selupdate", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[3]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
    int repid = atoi(argv[2]);
    if (argc == 4) {
      char tmpstring[64];
      sprintf(tmpstring, "%d", app->molrep_get_selupdate(molid, repid));
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    } else {
      int onoff;
      if (Tcl_GetBoolean(interp, argv[4], &onoff) != TCL_OK)
        return TCL_ERROR; // GetBoolean sets error message.
      if (!app->molrep_set_selupdate(molid, repid, onoff)) {
        Tcl_AppendResult(interp, "Could not set auto update", NULL);
        return TCL_ERROR;
      }
    }
  } else if ((argc == 4 || argc == 5) && !strupncmp(argv[1], "colupdate", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[3]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
    int repid = atoi(argv[2]);
    if (argc == 4) {
      char tmpstring[64];
      sprintf(tmpstring, "%d", app->molrep_get_colorupdate(molid, repid));
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    } else {
      int onoff;
      if (Tcl_GetBoolean(interp, argv[4], &onoff) != TCL_OK)
        return TCL_ERROR; // GetBoolean sets error message.
      if (!app->molrep_set_colorupdate(molid, repid, onoff)) {
        Tcl_AppendResult(interp, "Could not set auto color update", NULL);
        return TCL_ERROR;
      }
    }

  // Clipping plane functions:
  // mol clipplane center <clipid> <repid> <molid> [<vector>]
  //   set/get the clipplane center
  // mol clipplane color <clipid> <repid> <molid> [<vector>]
  //   set/get the clipplane color 
  // mol clipplane normal <clipid> <repid> <molid> [<vector>]
  //   set/get the clipplane normal 
  // mol clipplane status <clipid> <repid> <molid> [<mode>]
  // mol clipplane num
  } else if (argc >= 2 && !strupncmp(argv[1], "clipplane", CMDLEN)) {
    if (argc == 3 && !strupncmp(argv[1], "clipplane", CMDLEN) &&
        !strupncmp(argv[2], "num", CMDLEN)) {
      Tcl_SetObjResult(interp, Tcl_NewIntObj(app->num_clipplanes()));
      return TCL_OK;
    } else if (argc >= 6 && !strupncmp(argv[1], "clipplane", CMDLEN)) {
      int clipid = atoi(argv[3]);
      int repid = atoi(argv[4]);

      // int molid = atoi(argv[5]);  // XXX should use IDList
      IdList idList;
      if (idList.find(interp, app, argv[5]) != 1) {
        Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
        return TCL_ERROR;
      }
      int molid = idList[0];

      if (clipid < 0 || clipid >= app->num_clipplanes()) {
        Tcl_AppendResult(interp, "Invalid clip plane id specified: ", 
                         argv[3], NULL);
        return TCL_ERROR;
      }
      if (!app->molecule_valid_id(molid)) {
        Tcl_AppendResult(interp, "Invalid molid specified:", argv[5], NULL);
        return TCL_ERROR;
      }
      if (repid < 0 || repid >= app->num_molreps(molid)) {
        Tcl_AppendResult(interp, "Invalid repid specified:", argv[4], NULL);
        return TCL_ERROR;
      }

      float center[3], normal[3], color[3];
      int status;
      if (argc == 6) {
        if (!app->molrep_get_clipplane(molid, repid, clipid, center, normal, color, &status)) {
          Tcl_AppendResult(interp, "Unable to get clip plane information", NULL);
          return TCL_ERROR;
        }
        if (!strupncmp(argv[2], "center", CMDLEN)) {
          // return center
          Tcl_Obj *result = Tcl_NewListObj(0, NULL);
          for (int i=0; i<3; i++) {
            Tcl_ListObjAppendElement(interp, result, 
              Tcl_NewDoubleObj(center[i]));
          }
          Tcl_SetObjResult(interp, result);
          return TCL_OK;
        } else if (!strupncmp(argv[2], "normal", CMDLEN)) { 
          // return normal
          Tcl_Obj *result = Tcl_NewListObj(0, NULL);
          for (int i=0; i<3; i++) {
            Tcl_ListObjAppendElement(interp, result, 
              Tcl_NewDoubleObj(normal[i]));
          }
          Tcl_SetObjResult(interp, result);
          return TCL_OK;
        } else if (!strupncmp(argv[2], "color", CMDLEN)) { 
          // return color 
          Tcl_Obj *result = Tcl_NewListObj(0, NULL);
          for (int i=0; i<3; i++) {
            Tcl_ListObjAppendElement(interp, result, 
              Tcl_NewDoubleObj(color[i]));
          }
          Tcl_SetObjResult(interp, result);
          return TCL_OK;
        } else if (!strupncmp(argv[2], "status", CMDLEN)) { 
          // return status 
          Tcl_SetObjResult(interp, Tcl_NewIntObj(status));
          return TCL_OK; 
        }
      } else if (argc == 7) {
        if (!strupncmp(argv[2], "center", CMDLEN)) {
          // set center
          if (tcl_get_vector(argv[6], center, interp) != TCL_OK) {
            return TCL_ERROR;
          }
          if (!app->molrep_set_clipcenter(molid, repid, clipid, center)) {
            Tcl_AppendResult(interp, "Unable to set clip center\n", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[2], "normal", CMDLEN)) { 
          // set normal
          if (tcl_get_vector(argv[6], normal, interp) != TCL_OK) {
            return TCL_ERROR;
          }
          if (!app->molrep_set_clipnormal(molid, repid, clipid, normal)) {
            Tcl_AppendResult(interp, "Unable to set clip normal\n", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[2], "color", CMDLEN)) { 
          // set color
          if (tcl_get_vector(argv[6], color, interp) != TCL_OK) {
            return TCL_ERROR;
          }
          if (!app->molrep_set_clipcolor(molid, repid, clipid, color)) {
            Tcl_AppendResult(interp, "Unable to set clip color\n", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[2], "status", CMDLEN)) { 
          // set status 
          if (Tcl_GetInt(interp, argv[6], &status) != TCL_OK) {
            return TCL_ERROR;
          }
          if (!app->molrep_set_clipstatus(molid, repid, clipid, status)) {
            Tcl_AppendResult(interp, "Unable to set clip status\n", NULL);
            return TCL_ERROR;
          }
        }
        return TCL_OK;
      }
    } 
    Tcl_AppendResult(interp, "Usage: \n",
      "mol clipplane center <clipid> <repid> <molid> [<vector>]\n",
      "mol clipplane normal <clipid> <repid> <molid> [<vector>]\n",
      "mol clipplane color  <clipid> <repid> <molid> [<vector>]\n",
      "mol clipplane status <clipid> <repid> <molid> [<mode>]\n",
      "mol clipplane num\n",
      NULL);
    return TCL_ERROR;
  } else if (argc == 4 && !strupncmp(argv[1], "rename", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
    if (!app->molecule_rename(molid, argv[3])) {
      Tcl_AppendResult(interp, "Unable to rename molecule.", NULL);
      return TCL_ERROR;
    }
  } else if (argc==4 && !strupncmp(argv[1], "repname", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], argv[1], 
                       " operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid = atoi(argv[3]);
    const char *name = app->molrep_get_name(idList[0], repid);
    if (!name) {
      Tcl_AppendResult(interp, "mol repname: invalid repid ", argv[3], NULL);
      return TCL_ERROR;
    }
    Tcl_SetObjResult(interp, Tcl_NewStringObj(name, -1));
  } else if (argc == 4 && !strupncmp(argv[1], "repindex", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) > 1) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1], 
                       " operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid = -1;
    if (idList.num() == 1)
      repid = app->molrep_get_by_name(idList[0], argv[3]);
    Tcl_SetObjResult(interp, Tcl_NewIntObj(repid));
// mol fromsels <selection list>
  } else if (argc >= 2 && !strupncmp(argv[1], "fromsels", CMDLEN)) {
    int newmolid = -1;
    if (argc == 3) {
      // build selection list
      int numsels=0;

      Tcl_Obj **sel_list = NULL;
      Tcl_Obj *selparms = Tcl_NewStringObj(argv[2], -1);
#if 1
      if (Tcl_ListObjGetElements(interp, selparms, &numsels, &sel_list) != TCL_OK) {
        Tcl_AppendResult(interp, "mol fromsels: bad selection list", NULL);
        return TCL_ERROR;
      }
#endif

// printf("mol fromsels: numsels %d\n", numsels);
      AtomSel **asels = (AtomSel **) calloc(1, numsels * sizeof(AtomSel *));
      int s;
      for (s=0; s<numsels; s++) {
        asels[s] = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(sel_list[s], NULL));
        if (!asels[s]) {
// printf("mol fromsels: invalid selection[%d]\n", s);
          Tcl_AppendResult(interp, "mol fromsels: invalid atom selection list element", NULL);
          return TCL_ERROR;
        }
      }

      newmolid = app->molecule_from_selection_list(NULL, 0, numsels, asels);
      free(asels);

      if (newmolid < 0) {
        Tcl_AppendResult(interp, "Unable to create new molecule.", NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_AppendResult(interp, "Atom selection list missing.", NULL);
      return TCL_ERROR;
    } 

    Tcl_SetObjResult(interp, Tcl_NewIntObj(newmolid));


// mol new [filename] [options...]
// mol addfile <filename> [options...]
// options: 
//   atoms <numatoms> (for "mol new" only and when no filename is given)
//   type <filetype> 
//   filebonds <onoff> 
//   autobonds <onoff> 
//   first <firstframe> 
//   last <lastframe>
//   step <frame stride>
//   waitfor <all | number>
//   volsets <list of set ids>
//   molid   (for addfile only; must be the last item)
  } else if ((argc >= 2 && !strupncmp(argv[1], "new", CMDLEN)) ||
             (argc >= 3 && !strupncmp(argv[1], "addfile", CMDLEN))) {
    int molid = -1;
    const char *type = NULL;
    if (argc == 2) {
      molid = app->molecule_new(NULL, 0);
      if (molid < 0) {
        Tcl_AppendResult(interp, "Unable to create new molecule.", NULL);
        return TCL_ERROR;
      }
    } else if ((argc == 4) && (!strupncmp(argv[2], "atoms", CMDLEN))) {
      int natoms;
      Tcl_GetInt(interp, argv[3], &natoms);
      molid = app->molecule_new(NULL, natoms);
      if (molid < 0) {
        Tcl_AppendResult(interp, "Unable to create new molecule.", NULL);
        return TCL_ERROR;
      }
    } else if (argc >= 3) {
      if (!strupncmp(argv[1], "addfile", CMDLEN)) {
        molid = app->molecule_top();
      } // otherwise just -1
      // Check for optional parameters
      FileSpec spec;
      int a;
      for (a=3; a<argc; a+=2) {
        if (!strupncmp(argv[a], "type", CMDLEN)) {
          if ((a+1) < argc) {
            type = argv[a+1];
          } else {
            Tcl_AppendResult(interp, "Error, missing type parameter", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[a], "autobonds", CMDLEN)) {
          if (((a+1) >= argc) || (Tcl_GetBoolean(interp, argv[a+1], &spec.autobonds) != TCL_OK)) {
            Tcl_AppendResult(interp, "Error, missing/bad autobonds parameter", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[a], "filebonds", CMDLEN)) {
          if (((a+1) >= argc) || (Tcl_GetBoolean(interp, argv[a+1], &spec.filebonds) != TCL_OK)) {
            Tcl_AppendResult(interp, "Error, missing/bad filebonds parameter", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[a], "first", CMDLEN)) {
          if (((a+1) >= argc) || (Tcl_GetInt(interp, argv[a+1], &spec.first) != TCL_OK)) {
            Tcl_AppendResult(interp, "Error, missing/bad first parameter", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[a], "last", CMDLEN)) {
          if (((a+1) >= argc) || (Tcl_GetInt(interp, argv[a+1], &spec.last) != TCL_OK)) {
            Tcl_AppendResult(interp, "Error, missing/bad first parameter", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[a], "step", CMDLEN)) {
          if (((a+1) >= argc) || (Tcl_GetInt(interp, argv[a+1], &spec.stride) != TCL_OK)) {
            Tcl_AppendResult(interp, "Error, missing/bad step parameter", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[a], "waitfor", CMDLEN)) {
          if ((a+1) < argc) {
            if (!strupncmp(argv[a+1], "all", CMDLEN)) {
              spec.waitfor = FileSpec::WAIT_ALL;
            } else {
              if (Tcl_GetInt(interp, argv[a+1], &spec.waitfor) != TCL_OK)
                return TCL_ERROR;
            }
          } else {
            Tcl_AppendResult(interp, "Error, missing waitfor parameter", NULL);
            return TCL_ERROR;
          }
        } else if (!strupncmp(argv[a], "volsets", CMDLEN)) {
          int nsets;
          const char **sets;

          if ((a+1) >= argc) {
            Tcl_AppendResult(interp, "Error, missing volsets parameters", NULL);
            return TCL_ERROR;
          }
          if (Tcl_SplitList(interp, argv[a+1], &nsets, &sets) != TCL_OK) {
            Tcl_AppendResult(interp, "Cannot parse list of volsets.", NULL);
            return TCL_ERROR;
          }
          if (nsets > 0) {
            spec.nvolsets = nsets;
            spec.setids = new int[nsets];
            for (int i=0; i<nsets; i++) {
              if (Tcl_GetInt(interp, sets[i], spec.setids+i) != TCL_OK) {
                return TCL_ERROR;
              }
            }
          }
          Tcl_Free((char *)sets);
        }
      }
      if (!type) {
        type = app->guess_filetype(argv[2]);
        if (!type) {
          Tcl_AppendResult(interp, "Could not determine file type for file '",
                           argv[2], "' from its extension.", NULL);
          return TCL_ERROR;
        }
      }
      if (a == argc+1) {
        IdList idList;
        if (idList.find(interp, app, argv[argc-1]) != 1) {
          Tcl_AppendResult(interp, argv[0], " ", argv[1], 
                           " operates on one molecule only", NULL);
          return TCL_ERROR;
        }
        molid = idList[0];
      }
      molid = app->molecule_load(molid, argv[2], type, &spec);
      if (molid < 0) {
        Tcl_AppendResult(interp, "Unable to load file '", argv[2], "' using file type '", type, "'.", NULL);
        return TCL_ERROR;
      }
    }
    Tcl_SetObjResult(interp, Tcl_NewIntObj(molid));
    return TCL_OK;
  } else if ( (argc == 4 || argc == 5) && !strupncmp(argv[1], "smoothrep", CMDLEN)) {
    // smoothrep <molid> <repid> [<smoothness>]
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1],
                       " operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid;
    if (Tcl_GetInt(interp, argv[3], &repid) != TCL_OK) return TCL_ERROR;
    if (argc == 4) {
      int smooth = app->molrep_get_smoothing(idList[0], repid);
      if (smooth < 0) {
        Tcl_AppendResult(interp, "mol smoothrep: invalid rep", NULL);
        return TCL_ERROR;
      }
      Tcl_SetObjResult(interp, Tcl_NewIntObj(smooth));
    } else {
      int smooth;
      if (Tcl_GetInt(interp, argv[4], &smooth) != TCL_OK) return TCL_ERROR;
      if (smooth < 0) {
        Tcl_AppendResult(interp, "mol smoothrep: smoothness must be nonnegative", NULL);
        return TCL_ERROR;
      }
      if (!app->molrep_set_smoothing(idList[0], repid, smooth)) {
        Tcl_AppendResult(interp, "mol smoothrep: Unable to set smoothing for this rep", NULL);
        return TCL_ERROR;
      }
    }
    return TCL_OK;

    // mol showperiodic <molid> <repid> <string>
    // where string is x, y, z, xy, xz, yz, or xyz
  } else if ((argc == 4 || argc == 5) && !strupncmp(argv[1], "showperiodic", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1],
                       (char *)" operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid;
    if (Tcl_GetInt(interp, argv[3], &repid) != TCL_OK) return TCL_ERROR;
    if (argc == 5) {
      int pbc = PBC_NONE;  // defaults to PBC_NONE if no 5th argument is given
      if (strchr(argv[4], 'x')) pbc |= PBC_X;
      if (strchr(argv[4], 'y')) pbc |= PBC_Y;
      if (strchr(argv[4], 'z')) pbc |= PBC_Z;
      if (strchr(argv[4], 'X')) pbc |= PBC_OPX;
      if (strchr(argv[4], 'Y')) pbc |= PBC_OPY;
      if (strchr(argv[4], 'Z')) pbc |= PBC_OPZ;
      if (strchr(argv[4], 'n')) pbc |= PBC_NOSELF;
      if (!app->molrep_set_pbc(idList[0], repid, pbc)) {
        Tcl_AppendResult(interp, "mol setpbc: Unable to set periodic images for this rep", NULL);
        return TCL_ERROR;
      }
    } else {
      int pbc = app->molrep_get_pbc(idList[0], repid);
      if (pbc < 0) {
        Tcl_AppendResult(interp, "mol showperiodic: Unable to get periodic info for this rep", NULL);
        return TCL_ERROR;
      }
      char buf[10];
      buf[0] = '\0';
      if (pbc & PBC_X) strcat(buf, "x");
      if (pbc & PBC_Y) strcat(buf, "y");
      if (pbc & PBC_Z) strcat(buf, "z");
      if (pbc & PBC_OPX) strcat(buf, "X");
      if (pbc & PBC_OPY) strcat(buf, "Y");
      if (pbc & PBC_OPZ) strcat(buf, "Z");
      if (pbc & PBC_NOSELF) strcat(buf, "n");
      Tcl_SetResult(interp, buf, TCL_VOLATILE);
    }
  } else if ((argc == 4 || argc == 5) && !strupncmp(argv[1], "numperiodic", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1],
          (char *)" operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid;
    if (Tcl_GetInt(interp, argv[3], &repid) != TCL_OK) return TCL_ERROR;
    if (argc == 5) {
      int npbc;
      if (Tcl_GetInt(interp, argv[4], &npbc) != TCL_OK) return TCL_ERROR;
      if (!app->molrep_set_pbc_images(idList[0], repid, npbc)) {
        Tcl_AppendResult(interp, "mol numperiodic: Unable to set number of replicas for this rep", NULL);
        return TCL_ERROR;
      }
    } else {
      int npbc = app->molrep_get_pbc_images(idList[0], repid);
      if (npbc < 0) {
        Tcl_AppendResult(interp, "mol numperiodic: Unable to get number of replicas for this rep", NULL);
        return TCL_ERROR;
      }
      Tcl_SetObjResult(interp, Tcl_NewIntObj(npbc));
    }

    // mol instances 
  } else if (argc >= 2 && !strupncmp(argv[1], "instances", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
    // 'mol instances x' query, return number of instances
    if (argc == 3) {
      int ninstances = app->molecule_num_instances(molid); 
      if (ninstances < 0) {
        Tcl_AppendResult(interp, "mol instances: Unable to get number of replicas for specified molecule", NULL);
        return TCL_ERROR;
      }
      Tcl_SetObjResult(interp, Tcl_NewIntObj(ninstances));
    } 

    // mol addinstance xform
  } else if (argc >= 2 && !strupncmp(argv[1], "addinstance", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];

    Matrix4 mat;
    Tcl_Obj *matobj = Tcl_NewStringObj(argv[3], -1);
    int mrc=tcl_get_matrix("mol addinstance:", interp, matobj , mat.mat);
    Tcl_DecrRefCount(matobj);
    if (mrc != TCL_OK)
      return TCL_ERROR;

    if (!app->molecule_add_instance(molid, mat)) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1], (char *)" failed to add instance", NULL);
      return TCL_ERROR;
    }

    // mol showinstances <molid> <repid> <string>
    // where string is none, all, noself
  } else if ((argc == 4 || argc == 5) && !strupncmp(argv[1], "showinstances", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1],
                       (char *)" operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid;
    if (Tcl_GetInt(interp, argv[3], &repid) != TCL_OK) return TCL_ERROR;
    if (argc == 5) {
      // defaults to INSTANCE_NONE if no 5th argument is given
      int instances = INSTANCE_NONE;

      if (!strcmp(argv[4], "all")) 
        instances |= INSTANCE_ALL;

      if (!strcmp(argv[4], "noself")) 
        instances |= INSTANCE_ALL | INSTANCE_NOSELF;

      if (!app->molrep_set_instances(idList[0], repid, instances)) {
        Tcl_AppendResult(interp, "mol setinstances: Unable to set instances for this rep", NULL);
        return TCL_ERROR;
      }
    } else {
      int instances = app->molrep_get_instances(idList[0], repid);
      if (instances < 0) {
        Tcl_AppendResult(interp, "mol showinstances: Unable to get instance info for this rep", NULL);
        return TCL_ERROR;
      }
      if (instances & INSTANCE_NONE) Tcl_AppendResult(interp, "none", NULL);
      else if (instances & INSTANCE_ALL) Tcl_AppendResult(interp, "all", NULL);
      else if (instances & INSTANCE_NOSELF) Tcl_AppendResult(interp, "noself", NULL);
    }

  } else if ((argc >= 4 && argc <= 6) && !strupncmp(argv[1], "scaleminmax", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1],
                       (char *)" operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid;
    if (Tcl_GetInt(interp, argv[3], &repid) != TCL_OK) return TCL_ERROR;
    if (argc == 4) {
      float min, max;
      if (!app->molrep_get_scaleminmax(idList[0], repid, &min, &max)) {
        Tcl_AppendResult(interp, "mol scaleminmax: Unable to get color range for this rep", NULL);
        return TCL_ERROR;
      }

      char tmpstring[128];
      sprintf(tmpstring, "%f %f", min, max);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    } else if (argc == 5 && !strupncmp(argv[4], "auto", CMDLEN)) {
      if (!app->molrep_reset_scaleminmax(idList[0], repid)) {
        Tcl_AppendResult(interp, "mol scaleminmax: Unable to reset color range for this rep", NULL);
        return TCL_ERROR;
      }
    } else {
      double min, max;
      if (Tcl_GetDouble(interp, argv[4], &min) != TCL_OK ||
          Tcl_GetDouble(interp, argv[5], &max) != TCL_OK)
        return TCL_ERROR;
      if (!app->molrep_set_scaleminmax(idList[0],repid,(float)min,(float)max)) {
        Tcl_AppendResult(interp, "mol scaleminmax: Unable to set color range for this rep", NULL);
        return TCL_ERROR;
      }
    }
    // mol showrep <molid> <repid> [<onoff>]
  } else if ((argc == 4 || argc == 5) && !strupncmp(argv[1], "showrep", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1],
                       (char *)" operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid;
    if (Tcl_GetInt(interp, argv[3], &repid) != TCL_OK) return TCL_ERROR;
    if (argc == 4) {
      Tcl_SetObjResult(interp, Tcl_NewIntObj(app->molrep_is_shown(idList[0], repid)));
    } else {
      int onoff;
      if (Tcl_GetBoolean(interp, argv[4], &onoff) != TCL_OK) return TCL_ERROR;
      if (!app->molrep_show(idList[0], repid, onoff)) {
        Tcl_AppendResult(interp, "Unable to show/hide this rep", NULL);
        return TCL_ERROR;
      }
    }
  } else if ((argc == 4 || argc == 5) && !strupncmp(argv[1], "drawframes", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " ", argv[1],
                       (char *)" operates on one molecule only", NULL);
      return TCL_ERROR;
    }
    int repid;
    if (Tcl_GetInt(interp, argv[3], &repid) != TCL_OK) return TCL_ERROR;

    if (argc == 4) {
      Tcl_SetObjResult(interp, Tcl_NewStringObj(app->molrep_get_drawframes(idList[0], repid), -1));
    } else {
      if (!app->molrep_set_drawframes(idList[0], repid, argv[4])) {
        Tcl_AppendResult(interp, "Set drawframes failed.", NULL);
        return TCL_ERROR;
      }
    }
    // mol orblocalize <molid> <waveid>
  } else if (argc == 4 && !strupncmp(argv[1], "orblocalize", CMDLEN)) {
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
    int waveid;
    if (Tcl_GetInt(interp, argv[3], &waveid) != TCL_OK) return TCL_ERROR;
    if (!app->molecule_orblocalize(molid, waveid)) {
      Tcl_AppendResult(interp, "Unable to localize orbitals.", NULL);
      return TCL_ERROR;
    }
  } else if ((argc == 3 || argc == 4) && !strupncmp(argv[1], "default", CMDLEN)) {
    if (argc == 3) {
      if (!strupncmp(argv[2], "color", CMDLEN)) {
        Tcl_SetResult(interp, (char *)app->moleculeList->default_color(), TCL_VOLATILE);
      } else if (!strupncmp(argv[2], "style", CMDLEN) || !strupncmp(argv[2], "representation", CMDLEN)) {

        Tcl_SetResult(interp, (char *)app->moleculeList->default_representation(), TCL_VOLATILE);
      } else if (!strupncmp(argv[2], "selection", CMDLEN)) {
        Tcl_SetResult(interp, (char *)app->moleculeList->default_selection(), TCL_VOLATILE);
      } else if (!strupncmp(argv[2], "material", CMDLEN)) {
        Tcl_SetResult(interp, (char *)app->moleculeList->default_material(), TCL_VOLATILE);
      } else {
        Tcl_SetResult(interp, (char *) "Usage: mol default [color | style | selection | material]", TCL_STATIC);
        return TCL_ERROR;
      }
      return TCL_OK;
    } else {
      if (!strupncmp(argv[2], "color", CMDLEN)) {
        if (!app->moleculeList->set_default_color(argv[3])) {
          Tcl_AppendResult(interp, "Could not set default color to ", argv[3], NULL);
          return TCL_ERROR;
        }
      } else if (!strupncmp(argv[2], "style", CMDLEN) || !strupncmp(argv[2], "representation", CMDLEN)) {
        if (!app->moleculeList->set_default_representation(argv[3])) {
          Tcl_AppendResult(interp, "Could not set default style to ", argv[3], NULL);
          return TCL_ERROR;
        }
      } else if (!strupncmp(argv[2], "selection", CMDLEN)) {
        if (!app->moleculeList->set_default_selection(argv[3])) {
          Tcl_AppendResult(interp, "Could not set default selection to ", argv[3], NULL);
          return TCL_ERROR;
        }
      } else if (!strupncmp(argv[2], "material", CMDLEN)) {
        if (!app->moleculeList->set_default_material(argv[3])) {
          Tcl_AppendResult(interp, "Could not set default material to ", argv[3], NULL);
          return TCL_ERROR;
        }
      } else {
        Tcl_SetResult(interp, (char *) "Usage: mol default [color | style | selection | material] <value>", TCL_STATIC);
        return TCL_ERROR;
      }
      return TCL_OK;
    }

    /// delete a volumetric object
  } else if ((argc == 4) && !strupncmp(argv[1], "voldelete", CMDLEN)) {
    int volset = 0;
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
    Molecule *mol = app->moleculeList->mol_from_id(molid);
    if (!mol) {
      Tcl_SetResult(interp, (char *) "mol voldelete: molecule was deleted", TCL_STATIC);
      return TCL_ERROR;
    }
    if (Tcl_GetInt(interp, argv[3], &volset) != TCL_OK)
      return TCL_ERROR;
    if (volset >= mol->num_volume_data() || volset < 0) {
      char tmpstring[128];
      sprintf(tmpstring, "mol voldelete: no volume set %d", volset);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_ERROR;
    }
    mol->remove_volume_data(volset); 
    return TCL_OK;
 
    /// move a specified volumetric data object by transforming the 
    /// origin and axes with a 4x4 matrix
  } else if (argc == 5 && !strupncmp(argv[1], "volmove", CMDLEN)) {
    // volmove <molid> <4x4 matrix, e.g., from trans*> [<which volset to move>]
    int volset = 0;
    IdList idList;
    if (idList.find(interp, app, argv[2]) != 1) {
      Tcl_AppendResult(interp, argv[0], " operates on one molecule only.", NULL);
      return TCL_ERROR;
    }
    int molid = idList[0];
    Molecule *mol = app->moleculeList->mol_from_id(molid);
    if (!mol) {
      Tcl_SetResult(interp, (char *) "mol volmove: molecule was deleted", TCL_STATIC);
      return TCL_ERROR;
    }
    if (Tcl_GetInt(interp, argv[4], &volset) != TCL_OK)
      return TCL_ERROR;
    if (volset >= mol->num_volume_data() || volset < 0) {
      char tmpstring[128];
      sprintf(tmpstring, "mol volmove: no volume set %d", volset);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_ERROR;
    }
    VolumetricData *v = mol->modify_volume_data(volset);
    Matrix4 mat;
    Tcl_Obj *matobj = Tcl_NewStringObj(argv[3], -1);
    if (tcl_get_matrix("mol volmove:", interp, 
                       matobj , mat.mat) != TCL_OK) {
      Tcl_DecrRefCount(matobj); 
      return TCL_ERROR;
    }
    Tcl_DecrRefCount(matobj); 
    float tmp[4];
    float input[4];
    int idx;
    input[3] = 0;
    for (idx = 0; idx < 3; idx++) {
      input[idx] = float(v->origin[idx]);
    }
    //printf("Orig. orign: %.3f %.3f %.3f\n", v->origin[0], v->origin[1],v->origin[2]);
    mat.multpoint3d(&input[0], &tmp[0]);
    for (idx = 0; idx < 3; idx++) {
      v->origin[idx] = double(tmp[idx]);
      input[idx] = float(v->xaxis[idx]);
    }
    //printf("Final orign: %.3f %.3f %.3f\n", v->origin[0], v->origin[1],v->origin[2]);
    //printf("Orig. xaxis: %.3f %.3f %.3f\n", v->xaxis[0], v->xaxis[1],v->xaxis[2]);
    mat.multpoint4d(&input[0], &tmp[0]);
    for (idx = 0; idx < 3; idx++) {
      v->xaxis[idx] = double(tmp[idx]);
      input[idx] = float(v->yaxis[idx]);
    }
    //printf("Final xaxis: %.3f %.3f %.3f\n", v->xaxis[0], v->xaxis[1],v->xaxis[2]);
    mat.multpoint4d(&input[0], &tmp[0]);
    //printf("Orig. yaxis: %.3f %.3f %.3f\n", v->yaxis[0], v->yaxis[1],v->yaxis[2]);
    for (idx = 0; idx < 3; idx++) {
      v->yaxis[idx] = double(tmp[idx]);
      input[idx] = float(v->zaxis[idx]);
    }
    //printf("Final yaxis: %.3f %.3f %.3f\n", v->yaxis[0], v->yaxis[1],v->yaxis[2]);
    mat.multpoint4d(&input[0], &tmp[0]);
    //printf("Orig. zaxis: %.3f %.3f %.3f\n", v->zaxis[0], v->zaxis[1],v->zaxis[2]);
    for (idx = 0; idx < 3; idx++) {
      v->zaxis[idx] = double(tmp[idx]);
    }
    //printf("Final zaxis: %.3f %.3f %.3f\n", v->zaxis[0], v->zaxis[1],v->zaxis[2]);
    mol->force_recalc(DrawMolItem::COL_REGEN);
    return TCL_OK;
  } else {
    cmd_mol_usage(interp);
    return TCL_ERROR;
  }
  return TCL_OK;
}


