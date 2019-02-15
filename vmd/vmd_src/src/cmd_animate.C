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
 *      $RCSfile: cmd_animate.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.52 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for animation control
 ***************************************************************************/

#include <tcl.h>
#include <ctype.h>
#include "config.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "TclCommands.h"
#include "Animation.h"

static void cmd_animate_usage(Tcl_Interp *interp) {
  Tcl_AppendResult(interp,
      "usage: animate <command> [args...]"
      "animate styles\n"
      "animate style [once|rock|loop]\n"
      "animate dup [|frame <number>] <molecule id>\n"
      "animate goto [start|end|<num]\n"
      "animate [reverse|rev|forward|for|prev|next|pause]\n"
      "animate [speed|skip] [|<value>]\n"
      "animate delete all\n"
      "animate delete [|beg <num>] [|end <num>] [|skip <num>] <molecule id>\n"
      "animate [read|write] <file type> <filename>\n"
      "    [|beg <num>] [|end <num>] [|skip <num>] [|waitfor <num/all>]\n"
      "    [|sel <atom selection>] [|<molecule id>]",
      NULL);
}

int text_cmd_animate(ClientData cd, Tcl_Interp *interp, int argc, 
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  // since I cannot easily add this to the rest of the code, I put
  // new code here:
  if (argc == 1) {
    cmd_animate_usage(interp);
  }

  if (argc == 2) {
    if (!strupncmp(argv[1], "styles", CMDLEN)) {
      // enumerate the styles 
      size_t loop = 0;
      while(loop < sizeof(animationStyleName) / sizeof(char*)) {
        Tcl_AppendElement(interp, animationStyleName[loop++]);
      }
      return TCL_OK;
    }
    if (!strupncmp(argv[1], "skip", CMDLEN)) {
      Tcl_SetObjResult(interp, Tcl_NewIntObj(app->anim->skip()));
      return TCL_OK;
    }
    if (!strupncmp(argv[1], "speed", CMDLEN)) {
      Tcl_SetObjResult(interp, Tcl_NewDoubleObj(app->anim->speed()));
      return TCL_OK;
    }
    if (!strupncmp(argv[1], "style", CMDLEN)) {
      int style = app->anim->anim_style();
      Tcl_AppendElement(interp, animationStyleName[style]);
      return TCL_OK;
    }
    // fall through and let the rest of the code catch the other 2 word cmds
  }

  if ((argc == 3 || argc == 5) && !strupncmp(argv[1], "dup", CMDLEN)) {
    // option is: animate dup [frame <number>] <molecule id>
    // default frame is "now"
    // It adds a new animation frame to the molecule, which is a copy of
    // the given frame.  If there is no frame, {0, 0, 0} is added.
    int frame = -1;
    int molid = -1;
    // get frame number
    if (argc == 3) {
      if (!strcmp(argv[2], "top")) {
        if (app->moleculeList -> top()) {
          molid = app->moleculeList -> top() -> id();
        } else {
          molid = -1;
        }        
      } else if (Tcl_GetInt(interp, argv[2], &molid) != TCL_OK) {
        Tcl_AppendResult(interp, " in animate dup", NULL);
        return TCL_ERROR;
      }

    } else {
      if (strcmp(argv[2], "frame")) {
        // error
        Tcl_AppendResult(interp,
          "format is: animate dup [frame <number>] <molecule id>", NULL);
        return TCL_ERROR;
      }
      if (!strcmp(argv[3], "now")) { // check special cases
        frame = -1;
      } else if (!strcmp(argv[3], "null")) {
        frame = -2;
      } else {
        if (Tcl_GetInt(interp, argv[3], &frame) != TCL_OK) {
          Tcl_AppendResult(interp, " in animate dup frame", NULL);
          return TCL_ERROR;
        }
      }
      if (Tcl_GetInt(interp, argv[4], &molid) != TCL_OK) {
        Tcl_AppendResult(interp, " in animate dup", NULL);
        return TCL_ERROR;
      }
    }
    if (!app->molecule_dupframe(molid, frame)) return TCL_ERROR;
    return TCL_OK;
  }
 

  if (argc == 2) {
    Animation::AnimDir newDir;
    if(!strupncmp(argv[1], "reverse", CMDLEN) ||
            !strupncmp(argv[1], "rev", CMDLEN))
      newDir = Animation::ANIM_REVERSE;
    else if(!strupncmp(argv[1], "forward", CMDLEN) ||
            !strupncmp(argv[1], "for", CMDLEN))
      newDir = Animation::ANIM_FORWARD;
    else if(!strupncmp(argv[1], "prev", CMDLEN))
      newDir = Animation::ANIM_REVERSE1;
    else if(!strupncmp(argv[1], "next", CMDLEN))
      newDir = Animation::ANIM_FORWARD1;
    else if(!strupncmp(argv[1], "pause", CMDLEN))
      newDir = Animation::ANIM_PAUSE;
    else {
      cmd_animate_usage(interp);
      return TCL_ERROR;                // error
    }
    app->animation_set_dir(newDir);
  } else if(argc == 3) {
    if(!strupncmp(argv[1], "skip", CMDLEN)) {
      int tmp;
      if (Tcl_GetInt(interp, argv[2], &tmp) != TCL_OK) {
        Tcl_AppendResult(interp, " in animate skip", NULL);
        return TCL_ERROR;
      }
      app->animation_set_stride(tmp);
    } else if(!strupncmp(argv[1], "delete", CMDLEN)) {
      if(!strupncmp(argv[2], "all", CMDLEN)) {
        int molid = app->molecule_top();
        int last = app->molecule_numframes(molid)-1;
        int rc = app->molecule_deleteframes(molid, 0, last, -1);
        return rc ? TCL_OK : TCL_ERROR;
      } else {
        cmd_animate_usage(interp);
        return TCL_ERROR;                // error
      }
    } else if(!strupncmp(argv[1], "speed", CMDLEN))
      app->animation_set_speed((float) atof(argv[2]));
    else if(!strupncmp(argv[1], "style", CMDLEN)) {
      int newStyle = Animation::ANIM_ONCE;
      Animation::AnimStyle enumVal;
      while(newStyle < Animation::ANIM_TOTAL_STYLES) {
        if(!strupncmp(argv[2], animationStyleName[newStyle], CMDLEN))
          break;
        newStyle++;
      }
      if(newStyle == Animation::ANIM_ONCE)
        enumVal = Animation::ANIM_ONCE;
      else if(newStyle == Animation::ANIM_ROCK)
        enumVal = Animation::ANIM_ROCK;
      else if(newStyle == Animation::ANIM_LOOP)
        enumVal = Animation::ANIM_LOOP;
      else {
        Tcl_AppendResult(interp, 
        "Unknown animate style '" ,argv[2] ,"'\n", NULL);
        Tcl_AppendResult(interp, "Valid styles are: ", NULL);
        newStyle = Animation::ANIM_ONCE;
        while(newStyle < Animation::ANIM_TOTAL_STYLES) {
          Tcl_AppendElement(interp, animationStyleName[newStyle]); 
          newStyle ++;
        }
        return TCL_ERROR;                // error, unknown style
      }
      app->animation_set_style(enumVal);
    } else if(!strupncmp(argv[1], "goto", CMDLEN)) {
      int newframe;
      if(!strupncmp(argv[2], "start", CMDLEN))
        newframe = -1;
      else if(!strupncmp(argv[2], "end", CMDLEN))
        newframe = -2;
      else if(isdigit(argv[2][0]))
        newframe = atoi(argv[2]);
      else {
        Tcl_AppendResult(interp, "Bad goto parameter '" ,argv[2] ,"'\n", NULL);
        Tcl_AppendResult(interp, 
          "Valid values are a non-negative number, 'start', or 'end'.", NULL);
        return TCL_ERROR;                // error, bad frame goto command
      }
      app->animation_set_frame(newframe);
    } else {
      cmd_animate_usage(interp);
      return TCL_ERROR; // 3 option parameter, didn't understand 3rd term
    }
  } else if(argc >= 4) {
    int bf = 0, ef = (-1), fs = 1, mid = (-1);
    const char *fileType = NULL; 
    const char *fileName = NULL;
    int do_action = (-1);
    int currarg = 1;
    int waitfor = FileSpec::WAIT_BACK;

    // find out what to do first
    if(!strupncmp(argv[currarg], "read", CMDLEN)) {
      do_action = 0;
    } else if(!strupncmp(argv[currarg], "write", CMDLEN)) {
      do_action = 1;
      waitfor = FileSpec::WAIT_ALL; // waitfor 'all' by default
    } else if(!strupncmp(argv[currarg], "delete", CMDLEN)) {
      do_action = 2;
      fs = -1; // for "delete", fs=1 means do not delete any frames.
    } else {
      cmd_animate_usage(interp);
      return TCL_ERROR;
    }
    currarg++;

    // if reading or writing, get file type and name
    if(do_action == 0 || do_action == 1) {
      fileType = argv[currarg++];
      fileName = argv[currarg++];
    }
    
    AtomSel *selection = NULL;
    ResizeArray<int> volsets;
    mid = app->molecule_top();
    // find if any beg, end, or skip specifiers
    while(currarg < argc) {
      if(currarg < (argc - 1)) {
        if(!strupncmp(argv[currarg], "beg", CMDLEN)) {
          bf = atoi(argv[currarg+1]);
          currarg += 2;
        } else if(!strupncmp(argv[currarg], "end", CMDLEN)) {
          ef = atoi(argv[currarg+1]);
          currarg += 2;
        } else if(!strupncmp(argv[currarg], "skip", CMDLEN)) {
          fs = atoi(argv[currarg+1]);
          currarg += 2;
        } else if(do_action == 2 && argc == 4 && currarg == 2 &&
                        !strupncmp(argv[currarg], "all", CMDLEN)) {
          if (strcmp(argv[currarg+1], "top"))
            mid = atoi(argv[currarg+1]);
          else
            mid = app->molecule_top();
          currarg += 2;
        } else if ((do_action == 0 || do_action == 1) && 
                   !strupncmp(argv[currarg], "waitfor", CMDLEN)) {
          const char *arg = argv[currarg+1];
          if (!strupncmp(arg, "all", CMDLEN))
            waitfor = FileSpec::WAIT_ALL;
          else
            waitfor = atoi(arg);
          currarg += 2;
        } else if (do_action == 1 && // writing
                   !strupncmp(argv[currarg], "sel", CMDLEN)) {
          // interpret the next argument as an atom selection
          const char *selstr = argv[currarg+1];
          if (!(selection = tcl_commands_get_sel(interp, selstr))) {
            Tcl_AppendResult(interp, "Invalid atom selection ", selstr, NULL);
            return TCL_ERROR;
          }
          currarg += 2;
        } else if (do_action == 1 && // writing
                   !strupncmp(argv[currarg], "volsets", CMDLEN)) {
          // interpret the next argument as a list of volsets
          const char *volstr = argv[currarg+1];
          int nsets;
          const char **setstrs;
          if (Tcl_SplitList(interp, volstr, &nsets, &setstrs) != TCL_OK) {
              Tcl_AppendResult(interp, "Invalid volset argument: ", volstr, NULL);
              return TCL_ERROR;
          }
          for (int i=0; i<nsets; i++) {
              int tmp;
              if (Tcl_GetInt(interp, setstrs[i], &tmp) != TCL_OK) {
                  Tcl_Free((char *)setstrs);
                  return TCL_ERROR;
              }
              volsets.append(tmp);
          }
          Tcl_Free((char *)setstrs);
          currarg += 2;
        } else
          return TCL_ERROR;
      } else {
        // only one item left; it must be the molecule id
        if (strcmp(argv[currarg], "top")) {
          mid = atoi(argv[currarg++]);
        } else {
          mid = app->molecule_top();
          currarg++;
        }
      }
    }
    
    // if a selection was given, make sure the molid of the selection matches
    // the molid for the write command.  This ensures that the number of atoms
    // in the selection matches those in the molecule.
    if (selection) {
      if (mid != selection->molid()) {
        Tcl_SetResult(interp, (char *) "ERROR: animate: Molecule in selection must match animation molecule.", TCL_STATIC);
        return TCL_ERROR;
      }
    }
    
    // do action now
    FileSpec spec;
    spec.first = bf; 
    spec.last = ef; 
    spec.stride = fs; 
    spec.waitfor = waitfor;
    if (do_action == 0) {
      int rc = app->molecule_load(mid, fileName, fileType, &spec);
      if (rc < 0) return TCL_ERROR;

    } else if (do_action == 1) {
      spec.selection = selection ? selection->on : NULL;
      if (volsets.num()) {
        // make a copy of the setids, since FileSpec frees its setids pointer
        spec.nvolsets = volsets.num();
        spec.setids = new int[spec.nvolsets];
        for (int i=0; i<spec.nvolsets; i++) spec.setids[i] = volsets[i];
      }
      int numwritten = app->molecule_savetrajectory(mid, fileName, fileType, 
              &spec);
      if (numwritten < 0) return TCL_ERROR;
      Tcl_SetObjResult(interp, Tcl_NewIntObj(numwritten));

    } else if (do_action == 2)
      app->molecule_deleteframes(mid, bf, ef, fs);

    else
      return TCL_ERROR;

  } else
    return TCL_ERROR;
    
  // if here, everything worked out ok
  return TCL_OK;
}

// read raw bytes into a typestep
// Usage; rawtimestep <molid> <bytearray> -start index -frame whichframe 
// Optional start parameter specifies where in the byte array to start reading
// The array must contain at least (12*numatoms) bytes beginning from start 
// or an error will be returned.
// -frame parameter can be last, current, append, or a valid frame number.
//   If 'last' and there are no frames, a new frame will be created.
//   If 'current' and there are no frames, error.
//   If 'append', always append.

int cmd_rawtimestep(ClientData cd, Tcl_Interp *interp, int argc,
    Tcl_Obj * const objv[]) {
  VMDApp *app = (VMDApp *)cd;
  Molecule *mol;
  Timestep *ts;
  int molid=-1, start=0, frame=-1, length, neededLength;
  unsigned char *bytes;

  if (argc != 3 && argc != 5 && argc != 7) {
    Tcl_WrongNumArgs(interp, 1,objv, 
        "<molid> <bytearray> ?-start index? ?-frame whichframe?");
    return TCL_ERROR;
  }

  // get molid, either "top" or a number.
  if (!strcmp(Tcl_GetStringFromObj(objv[1], NULL), "top")) 
    molid = app->molecule_top();
  else if (Tcl_GetIntFromObj(interp, objv[1], &molid) != TCL_OK)
    return TCL_ERROR;
  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    Tcl_SetResult(interp, (char *) "rawtimestep: invalid molid", TCL_STATIC);
    return TCL_ERROR;
  }

  // Read raw bytes and get length
  if (!(bytes = Tcl_GetByteArrayFromObj(objv[2], &length))) {
    Tcl_SetResult(interp, (char *) "rawtimestep: could not read bytearray", TCL_STATIC);
    return TCL_ERROR;
  }

  // Read optional frame and start otions
  for (int iarg=3; iarg<argc; iarg += 2) {
    const char *opt = Tcl_GetStringFromObj(objv[iarg], NULL);
    if (!strcmp(opt, "-start")) {
      if (Tcl_GetIntFromObj(interp, objv[iarg+1], &start) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "-frame")) {
      // check for "last", "current", "append", otherwise must be numeric and
      // in the correct range
      const char *strframe = Tcl_GetStringFromObj(objv[iarg+1], NULL);
      if (!strcmp(strframe, "last")) {
        // allow frame to be -1 if the number of frames is zero, which 
        // corresponds to "append".
        frame = mol->numframes()-1;
      } else if (!strcmp(strframe, "current")) {
        frame = mol->frame();
      } else if (!strcmp(strframe, "append")) {
        frame = -1;
      } else {
        int tmpframe = -1;
        if (Tcl_GetIntFromObj(interp, objv[iarg+1], &tmpframe) != TCL_OK) 
          return TCL_ERROR;
        if (tmpframe < 0 || tmpframe >= mol->numframes()) {
          Tcl_SetResult(interp, (char *) "rawtimestep: invalid frame specified.",
              TCL_STATIC);
          return TCL_ERROR;
        }
        frame = tmpframe;
      }
    } else {
      Tcl_SetResult(interp, (char *) "rawtimestep: valid options are -frame and -start",
          TCL_STATIC);
      return TCL_ERROR;
    }
  }

  // Check that the size of the byte array and the start option are valid
  neededLength = 12L*mol->nAtoms;
  if (length-start < neededLength) {
    Tcl_SetResult(interp, (char *) "rawtimestep: not enough bytes!", TCL_STATIC);
    return TCL_ERROR;
  }

  // Get the timestep - either existing or new
  ts = (frame < 0) ? new Timestep(mol->nAtoms) 
                   : mol->get_frame(frame);
  if (!ts) {
    Tcl_SetResult(interp, (char *) "rawtimestep: Unable to find timestep!", TCL_STATIC);
    return TCL_ERROR;
  }
  memcpy(ts->pos, bytes+start, neededLength);
  if (frame < 0) {
    mol->append_frame(ts);
  } else {
    mol->force_recalc(DrawMolItem::MOL_REGEN);
  }
  return TCL_OK;
}


// return timestep as byte array.
// arguments: molid, frame
int cmd_gettimestep(ClientData cd, Tcl_Interp *interp, int argc,
    Tcl_Obj * const objv[]) {
  if (argc != 3) {
    Tcl_WrongNumArgs(interp, 1, objv, "molid frame");
    return TCL_ERROR;
  }

  VMDApp *app = (VMDApp *)cd;
  int molid = -1;
  const char *molidstr = Tcl_GetStringFromObj(objv[1], NULL);
  if (!strcmp(molidstr, "top")) {
    molid = app->molecule_top();
  } else if (Tcl_GetIntFromObj(interp, objv[1], &molid) != TCL_OK) {
    return TCL_ERROR;
  }

  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    Tcl_AppendResult(interp, "Invalid molid: ", molidstr, NULL);
    return TCL_ERROR;
  }

  int frame;
  if (Tcl_GetIntFromObj(interp, objv[2], &frame) != TCL_OK)
    return TCL_ERROR;

  if (frame < 0 || frame >= mol->numframes()) {
    Tcl_AppendResult(interp, "Invalid frame for molecule ", molidstr, NULL);
    return TCL_ERROR;
  }

  Timestep *ts = mol->get_frame(frame);
  Tcl_SetObjResult(interp, Tcl_NewByteArrayObj(
        (const unsigned char *)(ts->pos),   // bytes
        3L*mol->nAtoms*sizeof(float)));      // length

  return TCL_OK;
}

