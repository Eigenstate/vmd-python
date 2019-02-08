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
 *      $RCSfile: TclTextInterp.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.118 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   The Tcl-based text command interpreter implementation
 ***************************************************************************/

#include <tcl.h>
#include <stdlib.h>
#include <ctype.h>  // for toupper/tolower

#ifdef VMDTK
#if defined(_MSC_VER)
// XXX prototype, skip problems with tk.h.
EXTERN int              Tk_Init _ANSI_ARGS_((Tcl_Interp *interp));
#else
#include <tk.h>         // Tk extensions
#endif
#endif

#include "TclTextInterp.h"
#include "Inform.h"
#include "TclCommands.h"
#include "VMDApp.h"
#include "DisplayDevice.h" 

#include "config.h"
#if defined(VMDTKCON)
#include "vmdconsole.h"
#endif

#if !defined(_MSC_VER)
#include <unistd.h>
static int vmd_isatty(int fd) {
  // Check for console tty override in case we're running on a cluster node
  // on Clustermatic or Scyld, which cause isatty() to return false even when
  // we do have a tty.  This makes it possible to get the normal VMD prompts
  // in an interactive bpsh session if we want.
  if (getenv("VMDFORCECONSOLETTY") != NULL)
    return 1;

  return isatty(fd);
}

#else
static int vmd_isatty(int) {
  return 1;
}
#endif

static int text_cmd_wait(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  TclTextInterp *ttinterp = (TclTextInterp *)cd;
  if(argc == 2) {
    ttinterp->wait((float)atof(argv[1]));
  } else {
    Tcl_AppendResult(interp, "wait: Usage: wait <seconds>",NULL);
    return TCL_ERROR;
  }
  return TCL_OK;
}

static int text_cmd_quit(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;
  // Trigger exit sequence on next display update.  
  // Avoid calling VMDexit more than once.
  if (!app->exitFlag) app->VMDexit("",0,0);

  // return TCL_ERROR so that execution of procs or sourcing of files
  // stops here as well.
  return TCL_ERROR;
}

static int text_cmd_play(ClientData cd, Tcl_Interp *interp, int argc,
                                const char *argv[]) {

  TclTextInterp *ttinterp = (TclTextInterp *)cd;
  if (argc != 2) {
    Tcl_AppendResult(interp, "Usage: play <filename>", NULL);
    return TCL_ERROR;
  }
  if (ttinterp->evalFile(argv[1])) return TCL_ERROR;
  return TCL_OK;
}

 __attribute__((no_sanitize_address))
TclTextInterp::TclTextInterp(VMDApp *vmdapp, int guienabled, int mpienabled)
: app(vmdapp) {
  
  interp = Tcl_CreateInterp();
#if 0
  Tcl_InitMemory(interp); // enable Tcl memory debugging features
                          // when compiled with TCL_MEM_DEBUG
#endif

  commandPtr = Tcl_NewObj();
  Tcl_IncrRefCount(commandPtr);
  consoleisatty = vmd_isatty(0); // whether we're interactive or not
  ignorestdin = 0;
  gotPartial = 0;
  needPrompt = 1;
  callLevel = 0;
  starttime = delay = 0;

#if defined(VMDMPI)
  //
  // MPI builds of VMD cannot try to read any command input from the 
  // console because it creates shutdown problems, at least with MPICH.
  // File-based command input is fine however.
  //
  // don't check for interactive console input if running in parallel
  if (mpienabled)
    ignorestdin = 1;
#endif

#if defined(ANDROIDARMV7A)
  //
  // For the time being, the Android builds won't attempt to get any
  // console input.  Any input we're going to get is going to come via
  // some means other than stdin, such as a network socket, text box, etc.
  //
  // Don't check for interactive console input if compiled for Android
  ignorestdin = 1;
#endif

  // set tcl_interactive, lets us run unix commands as from a shell
#if !defined(VMD_NANOHUB)
  Tcl_SetVar(interp, "tcl_interactive", "1", 0);
#else
  Tcl_SetVar(interp, "tcl_interactive", "0", 0);

  Tcl_Channel channel;
#define CLIENT_READ	(3)
#define CLIENT_WRITE	(4)
  channel = Tcl_MakeFileChannel((ClientData)CLIENT_READ, TCL_READABLE);
  if (channel != NULL) {
      const char *result;

      Tcl_RegisterChannel(interp, channel);
      result = Tcl_SetVar2(interp, "vmd_client", "read", 
		Tcl_GetChannelName(channel), 
		TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
      if (result == NULL) {
	  fprintf(stderr, "can't create variable for client read channel\n");
      }
  }
  channel = Tcl_MakeFileChannel((ClientData)CLIENT_WRITE, TCL_WRITABLE);
  if (channel != NULL) {
      const char *result;

      Tcl_RegisterChannel(interp, channel);
      result = Tcl_SetVar2(interp, "vmd_client", "write", 
		Tcl_GetChannelName(channel), 
		TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
      if (result == NULL) {
	  fprintf(stderr, "can't create variable for client write channel\n");
      }
  }
  write(CLIENT_WRITE, "vmd 1.0\n", 8);
#endif


  // pass our instance of VMDApp to a hash table assoc. with the interpreter 
  Tcl_SetAssocData(interp, "VMDApp", NULL, app);
 
  // Set up argc, argv0, and argv variables
  {
    char argcbuf[20];
    sprintf(argcbuf, "%d", app->argc_m);
    Tcl_SetVar(interp, "argc", argcbuf, TCL_GLOBAL_ONLY);
    // it might be better to use the same thing that was passed to
    // Tcl_FindExecutable, but this is now
    Tcl_SetVar(interp, "argv0", app->argv_m[0], TCL_GLOBAL_ONLY);
    char *args = Tcl_Merge(app->argc_m-1, app->argv_m+1);
    Tcl_SetVar(interp, "argv", args, TCL_GLOBAL_ONLY);
    Tcl_Free(args);
  }

#if defined(_MSC_VER) && TCL_MINOR_VERSION >= 4
  // The Windows versions of Tcl 8.5.x have trouble finding
  // the Tcl library subdirectory for unknown reasons.
  // We force the appropriate env variables to be set in Tcl, 
  // despite Windows.
  {
    char vmdinitscript[4096];
    char * tcl_library = getenv("TCL_LIBRARY");
    char * tk_library = getenv("TK_LIBRARY");

    if (tcl_library) {
      sprintf(vmdinitscript, "set env(TCL_LIBRARY) {%s}", tcl_library);
      if (Tcl_Eval(interp, vmdinitscript) != TCL_OK) {
        msgErr << Tcl_GetStringResult(interp) << sendmsg;
      }
    }
    if (tk_library) {
      sprintf(vmdinitscript, "set env(TK_LIBRARY) {%s}", tk_library);
      if (Tcl_Eval(interp, vmdinitscript) != TCL_OK) {
        msgErr << Tcl_GetStringResult(interp) << sendmsg;
      }
    }
  }
#endif

  if (Tcl_Init(interp) == TCL_ERROR) {  // new with 7.6
    msgErr << "Tcl startup error: " << Tcl_GetStringResult(interp) << sendmsg;
  }

#ifdef VMDTK
  // and the Tk commands (but only if a GUI is available!)
  if (guienabled) {
    if (Tk_Init(interp) == TCL_ERROR) {
      msgErr << "Tk startup error: " << Tcl_GetStringResult(interp) << sendmsg;
    } else {
      Tcl_StaticPackage(interp,  "Tk",
                        (Tcl_PackageInitProc *) Tk_Init,
                        (Tcl_PackageInitProc *) NULL);
    }
  } // end of check that GUI is allowed
#endif
  add_commands();
}

void TclTextInterp::add_commands() {
  Vmd_Init(interp);

  Atomsel_Init(interp);

  Tcl_CreateCommand(interp,  "molinfo", molecule_tcl,
                      (ClientData) app, (Tcl_CmdDeleteProc *) NULL);

  Tcl_CreateCommand(interp,  "graphics", graphics_tcl,
                      (ClientData) app, (Tcl_CmdDeleteProc *) NULL);

  Tcl_CreateCommand(interp,  "colorinfo", tcl_colorinfo,
                      (ClientData) app, (Tcl_CmdDeleteProc *) NULL);

  Tcl_CreateCommand(interp,  "wait", text_cmd_wait,
                      (ClientData) this, (Tcl_CmdDeleteProc *) NULL);

  Tcl_CreateCommand(interp,  "play", text_cmd_play,
                      (ClientData) this, (Tcl_CmdDeleteProc *) NULL);

  Tcl_CreateCommand(interp,  "exit", text_cmd_quit,
                      (ClientData) app, (Tcl_CmdDeleteProc *) NULL);

  Tcl_CreateCommand(interp,  "quit", text_cmd_quit,
                      (ClientData) app, (Tcl_CmdDeleteProc *) NULL);

  Vec_Init(interp);
}
  
  
void TclTextInterp::doInit() {
  int startuperror = 0;
  const char *vmddir;
  char vmdinitscript[4096];
  
  vmddir = getenv("VMDDIR");

  // read the VMD initialization script
  if (vmddir == NULL) {
    msgErr << "VMDDIR undefined, startup failure likely." << sendmsg;
#if defined(_MSC_VER)
  vmddir = "c:/program file/university of illinoid/vmd";
#else
  vmddir = "/usr/local/lib/vmd";
#endif
    startuperror = 1;
  } 

  // force VMDDIR env variable to be set in Tcl, despite Windows.
  sprintf(vmdinitscript, "set env(VMDDIR) {%s}", vmddir);
  if (Tcl_Eval(interp, vmdinitscript) != TCL_OK) {
    msgErr << Tcl_GetStringResult(interp) << sendmsg;
    startuperror = 1;
  }

  sprintf(vmdinitscript, "source {%s/scripts/vmd/vmdinit.tcl}", vmddir);
  if (Tcl_Eval(interp, vmdinitscript) != TCL_OK) {
    startuperror = 1;
  }

  if (startuperror) {
    msgErr << "Could not read the vmd initialization file -" << sendmsg;
    msgErr << "  " << vmdinitscript << sendmsg;
    msgErr << Tcl_GetStringResult(interp) << sendmsg;

#if defined(_MSC_VER)
    msgErr << "The VMDDIR variable in the Windows registry is missing or" 
           << " incorrect. " << sendmsg;
#else
    msgErr << "The VMDDIR environment variable is set by the startup"
           << sendmsg;
    msgErr << "script and should point to the top of the VMD hierarchy." 
           << sendmsg;
#endif
    msgErr << "VMD will continue with limited functionality." << sendmsg;
  }
}

TclTextInterp::~TclTextInterp() {
  // Set callback variable, giving a chance for Tcl to do some clean-ups
  // (for example, if external jobs have been run and need to be halted...)
  setString("vmd_quit", "1");
  
  // DeleteInterp must precede Finalize!
  Tcl_DeleteInterp(interp);
  interp = NULL; // prevent use by Python if Tcl_Finalize() invokes
                 // shutdown scripts
}

int TclTextInterp::doTkUpdate() {
  // Loop on the Tcl event notifier
  while (Tcl_DoOneEvent(TCL_DONT_WAIT));
  return 1; 
}  

void TclTextInterp::doEvent() {
  if (!done_waiting())
    return;

  // no recursive calls to TclEvalObj; this prevents  
  // display update ui from messing up Tcl. 
  if (callLevel) 
    return;

  Tcl_Channel inChannel = Tcl_GetStdChannel(TCL_STDIN);
  Tcl_Channel outChannel = Tcl_GetStdChannel(TCL_STDOUT);

  if (needPrompt && consoleisatty) {
#if TCL_MINOR_VERSION >= 4
    if (gotPartial) {
      Tcl_WriteChars(outChannel, "? ", -1);
    } else { 
      Tcl_WriteChars(outChannel, VMD_CMD_PROMPT, -1);
    }
#else
    if (gotPartial) {
      Tcl_Write(outChannel, "? ", -1);
    } else { 
      Tcl_Write(outChannel, VMD_CMD_PROMPT, -1);
    }
#endif
#if defined(VMDTKCON)
    vmdcon_purge();
#endif
    Tcl_Flush(outChannel);
    needPrompt = 0;
  }

#if defined(VMD_NANOHUB)  
  return;
#endif

  //
  // MPI builds of VMD cannot try to read any command input from the 
  // console because it creates shutdown problems, at least with MPICH.
  // File-based command input is fine however.
  //
  // For the time being, the Android builds won't attempt to get any
  // console input.  Any input we're going to get is going to come via
  // some means other than stdin, such as a network socket, text box, etc.
  //
  if (ignorestdin)
    return;
 
  if (!vmd_check_stdin())
    return;

  //
  // event loop based on tclMain.c
  //
  // According to the Tcl docs, GetsObj returns -1 on error or EOF.
    
  int length = Tcl_GetsObj(inChannel, commandPtr);
  if (length < 0) {
    if (Tcl_Eof(inChannel)) {
      // exit if we're not a tty, or if eofexit is set
      if ((!consoleisatty) || app->get_eofexit())
        app->VMDexit("", 0, 0);
    } else {
      msgErr << "Error reading Tcl input: " << Tcl_ErrnoMsg(Tcl_GetErrno()) 
             << sendmsg;
    }
    return;
  }
  
  needPrompt = 1;
  // add the newline removed by Tcl_GetsObj
  Tcl_AppendToObj(commandPtr, "\n", 1);

  char *stringrep = Tcl_GetStringFromObj(commandPtr, NULL);
  if (!Tcl_CommandComplete(stringrep)) {
    gotPartial = 1;
    return;
  }
  gotPartial = 0;

  callLevel++;
#if defined(VMD_NANOHUB)
  Tcl_EvalObjEx(interp, commandPtr, 0);
#else
  Tcl_RecordAndEvalObj(interp, commandPtr, 0);
#endif
  callLevel--;

#if TCL_MINOR_VERSION >= 4
  Tcl_DecrRefCount(commandPtr);
  commandPtr = Tcl_NewObj();
  Tcl_IncrRefCount(commandPtr);
#else
  // XXX this crashes Tcl 8.5.[46] with an internal panic
  Tcl_SetObjLength(commandPtr, 0);
#endif
    
  // if ok, send to stdout; if not, send to stderr
  Tcl_Obj *resultPtr = Tcl_GetObjResult(interp);
  char *bytes = Tcl_GetStringFromObj(resultPtr, &length);
#if defined(VMDTKCON)
  if (length > 0) {
    vmdcon_append(VMDCON_ALWAYS, bytes,length);
    vmdcon_append(VMDCON_ALWAYS, "\n", 1);
  }
  vmdcon_purge();
#else
  if (length > 0) {
#if TCL_MINOR_VERSION >= 4
    Tcl_WriteChars(outChannel, bytes, length);
    Tcl_WriteChars(outChannel, "\n", 1);
#else
    Tcl_Write(outChannel, bytes, length);
    Tcl_Write(outChannel, "\n", 1);
#endif
  }
  Tcl_Flush(outChannel);
#endif
}

int TclTextInterp::evalString(const char *s) {
#if defined(VMD_NANOHUB)
  if (Tcl_Eval(interp, s) != TCL_OK) {
#else
  if (Tcl_RecordAndEval(interp, s, 0) != TCL_OK) {
#endif
    // Don't print error message if there's nothing to show.
    if (strlen(Tcl_GetStringResult(interp))) 
      msgErr << Tcl_GetStringResult(interp) << sendmsg;
    return FALSE;
  }
  return TRUE;
}

void TclTextInterp::setString(const char *name, const char *val) {
  if (interp)
    Tcl_SetVar(interp, name, val, 
      TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
}

void TclTextInterp::setMap(const char *name, const char *key, 
                           const char *val) { 
  if (interp)
    Tcl_SetVar2(interp, name, key, val, 
      TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    
}

// There's a fair amount of code duplication between doEvent and evalFile,
// maybe these could be combined somehow, say by having TclTextInterp keep 
// track of its Tcl_Channel objects.
// 
// Side note: Reading line-by-line gives different Tcl semantics than 
// just calling Tcl_EvalFile. Shell commands (e.g., stty) are properly
// parsed when read line-by-line and passed to Tcl_RecordAndEval, but are
// unrecognized when contained in a file read by Tcl_EvalFile.  I would 
// consider this a bug.  

int TclTextInterp::evalFile(const char *fname) {
  Tcl_Channel inchannel = Tcl_OpenFileChannel(interp, fname, "r", 0644);
  Tcl_Channel outchannel = Tcl_GetStdChannel(TCL_STDOUT);
  if (inchannel == NULL) {
    msgErr << "Error opening file " << fname << sendmsg;
    msgErr << Tcl_GetStringResult(interp) << sendmsg;
    return 1;
  }

  Tcl_Obj *cmdPtr = Tcl_NewObj();
  Tcl_IncrRefCount(cmdPtr);
  int length = 0;
  while ((length = Tcl_GetsObj(inchannel, cmdPtr)) >= 0) {
    Tcl_AppendToObj(cmdPtr, "\n", 1);
    char *stringrep = Tcl_GetStringFromObj(cmdPtr, NULL);
    if (!Tcl_CommandComplete(stringrep)) {
      continue;
    }

    // check if "exit" was called
    if (app->exitFlag) break;

#if defined(VMD_NANOHUB)
    Tcl_EvalObjEx(interp, cmdPtr, 0);
#else
    Tcl_RecordAndEvalObj(interp, cmdPtr, 0);
#endif

#if TCL_MINOR_VERSION >= 4
    Tcl_DecrRefCount(cmdPtr);
    cmdPtr = Tcl_NewObj();
    Tcl_IncrRefCount(cmdPtr);
#else
    // XXX this crashes Tcl 8.5.[46] with an internal panic
    Tcl_SetObjLength(cmdPtr, 0);
#endif

    // XXX this makes sure the display is updated 
    // after each line read from the file or pipe
    // So, this is also where we'd optimise reading multiple
    // lines at once
    //
    // In VR modes (CAVE, FreeVR, VR Juggler) the draw method will 
    // not be called from app->display_update(), so multiple lines
    // of input could be combined in one frame, if possible
    app->display_update();

    Tcl_Obj *resultPtr = Tcl_GetObjResult(interp);
    char *bytes = Tcl_GetStringFromObj(resultPtr, &length);
#if defined(VMDTKCON)
    if (length > 0) {
      vmdcon_append(VMDCON_ALWAYS, bytes,length);
      vmdcon_append(VMDCON_ALWAYS, "\n", 1);
    }
    vmdcon_purge();
#else
    if (length > 0) {
#if TCL_MINOR_VERSION >= 4
      Tcl_WriteChars(outchannel, bytes, length);
      Tcl_WriteChars(outchannel, "\n", 1);
#else
      Tcl_Write(outchannel, bytes, length);
      Tcl_Write(outchannel, "\n", 1);
#endif
    }
    Tcl_Flush(outchannel);
#endif
  }
  Tcl_Close(interp, inchannel);
  Tcl_DecrRefCount(cmdPtr);
  return 0;
}

void TclTextInterp::wait(float wd) {
  delay = wd;
  starttime = time_of_day();
}
int TclTextInterp::done_waiting() {
  if (delay > 0) {
    double elapsed = time_of_day() - starttime;
    if (elapsed > delay) {
      delay = -1;     // done waiting
    } else {
      return 0;       // not done yet
    }
  }
  return 1; // done
}


void TclTextInterp::frame_cb(int molid, int frame) {
  Tcl_ObjSetVar2(interp, Tcl_NewStringObj("vmd_frame", -1),
                         Tcl_NewIntObj(molid),
                         Tcl_NewIntObj(frame),
                         TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
}

void TclTextInterp::help_cb(const char *topic) {
  JString cmd("help ");
  cmd += topic;
  evalString((const char *)cmd);
}

void TclTextInterp::molecule_changed_cb(int molid, int code) {
  char molstr[30];
  sprintf(molstr, "%d", molid);
  char codestr[30];
  sprintf(codestr, "%d", code);
  setMap("vmd_molecule", molstr, codestr);
}

void TclTextInterp::initialize_structure_cb(int molid, int code) {
  char molstr[30];
  sprintf(molstr, "%d", molid);
  char codestr[30];
  sprintf(codestr, "%d", code);
  setMap("vmd_initialize_structure", molstr, codestr);
}


void TclTextInterp::logfile_cb(const char *str) {
  setString("vmd_logfile", (const char *)str);
}

void TclTextInterp::pick_atom_cb(int molid, int atom, int ss, bool is_pick) {
  char s[40];
  sprintf(s, "%d",ss);
  setString("vmd_pick_shift_state", s);
  sprintf(s, "%d", molid);
  setString("vmd_pick_mol", s);
  sprintf(s, "%d", atom);
  setString("vmd_pick_atom", s);
  
  // only set this callback variable for a user pick event
  if (is_pick)
    setString("vmd_pick_event", "1");
}

void TclTextInterp::pick_atom_callback_cb(int molid, int atom, const char *client) {
  char s[40];
  sprintf(s, "%s", (const char *)client);
  setString("vmd_pick_client", s);
  sprintf(s, "%d", molid);
  setString("vmd_pick_mol_silent", s);
  sprintf(s, "%d", atom);
  setString("vmd_pick_atom_silent", s);
} 

void TclTextInterp::pick_graphics_cb(int molid, int tag, int btn, int shift_state) {
  char s[300];
  sprintf(s, "%d %d %d %d", molid, tag, btn, shift_state);
  setString("vmd_pick_graphics", s);
}

void TclTextInterp::pick_selection_cb(int num, const int *atoms) {
  JString s;
  if (num > 0) {
    s = "index";
    for (int i=0; i<num; i++) {
      char buf[20];
      sprintf(buf, " %d", atoms[i]);
      s += buf;
    }
  } else {
    s = "none";
  }
  setString("vmd_pick_selection", (const char *)s);
}
 
void TclTextInterp::pick_value_cb(float value) {
  char buf[20];
  sprintf(buf, "%f", value);
  setString("vmd_pick_value", buf);
}

void TclTextInterp::timestep_cb(int molid, int frame) {
  char mol[10];
  char n[10];
  sprintf(mol, "%d", molid);
  sprintf(n, "%d", frame);
  setMap("vmd_timestep", mol, n);
}

void TclTextInterp::graph_label_cb(const char *type, const int *ids, int n) {
  Tcl_Obj *itemlist = Tcl_NewListObj(0, NULL);
  for (int i=0; i<n; i++) {
    Tcl_Obj *item = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, item, Tcl_NewStringObj(type, -1));
    Tcl_ListObjAppendElement(interp, item, Tcl_NewIntObj(ids[i]));
    Tcl_ListObjAppendElement(interp, itemlist, item);
  }
  Tcl_Obj *varname = Tcl_NewStringObj("vmd_graph_label", -1);
  if (!Tcl_ObjSetVar2(interp, varname, NULL, itemlist, 
        TCL_LEAVE_ERR_MSG | TCL_GLOBAL_ONLY)) {
    msgErr << "Error graphing labels: " << Tcl_GetStringResult(interp) << sendmsg;
  }
}

void TclTextInterp::trajectory_cb(int molid, const char *name) {
  char s[10];
  if (!name) return;
  sprintf(s, "%d", molid);
  setMap("vmd_trajectory_read", s, name);
}

void TclTextInterp::tcl_cb(const char *cmd) {
  evalString(cmd);
}

void TclTextInterp::mousemode_cb(const char *mode, int submode) {
  char tmp[20];
  sprintf(tmp, "%d", submode);
  setString("vmd_mouse_mode", (const char *)mode);
  setString("vmd_mouse_submode", tmp);
}

void TclTextInterp::mouse_pos_cb(float x, float y, int buttondown) {
  Tcl_Obj *poslist = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(x));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(y));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewIntObj(buttondown));
  Tcl_Obj *varname = Tcl_NewStringObj("vmd_mouse_pos", -1);
  Tcl_ObjSetVar2(interp, varname, NULL, poslist, TCL_LEAVE_ERR_MSG | TCL_GLOBAL_ONLY);
}

void TclTextInterp::mobile_state_changed_cb() {
  setString("vmd_mobile_state_changed", "1");
}

void TclTextInterp::mobile_device_command_cb(const char *str) {
  setString("vmd_mobile_device_command", (const char *)str);
}

void TclTextInterp::mobile_cb(float tx, float ty, float tz,
                              float rx, float ry, float rz, int buttondown) {
  Tcl_Obj *poslist = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(tx));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(ty));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(tz));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(rx));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(ry));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(rz));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewIntObj(buttondown));
  Tcl_Obj *varname = Tcl_NewStringObj("vmd_mobile", -1);
  Tcl_ObjSetVar2(interp, varname, NULL, poslist, TCL_LEAVE_ERR_MSG | TCL_GLOBAL_ONLY);
}


void TclTextInterp::spaceball_cb(float tx, float ty, float tz,
                                 float rx, float ry, float rz, int buttondown) {
  Tcl_Obj *poslist = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(tx));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(ty));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(tz));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(rx));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(ry));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewDoubleObj(rz));
  Tcl_ListObjAppendElement(interp, poslist, Tcl_NewIntObj(buttondown));
  Tcl_Obj *varname = Tcl_NewStringObj("vmd_spaceball", -1);
  Tcl_ObjSetVar2(interp, varname, NULL, poslist, TCL_LEAVE_ERR_MSG | TCL_GLOBAL_ONLY);
}

void TclTextInterp::userkey_cb(const char *key_desc) {
  int indx = app->userKeys.typecode(key_desc);
  if(indx >= 0) {
    const char *cmd = app->userKeys.data(indx);
    evalString(cmd);
  }
}

