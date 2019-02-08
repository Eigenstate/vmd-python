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
 *      $RCSfile: TclTextInterp.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.46 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   The Tcl-based text command interpreter
 ***************************************************************************/

#ifndef TCL_TEXT_INTERP_H
#define TCL_TEXT_INTERP_H

#include "TextInterp.h"
#include "JString.h"
#include "TextEvent.h"

struct Tcl_Interp;
struct Tcl_Obj;

class VMDApp;
class Inform;

/// TextInterp subclass implementing the Tcl-based text command interpreter
class TclTextInterp : public TextInterp {
private:
  Tcl_Interp *interp;
  Tcl_Obj *commandPtr;
  int callLevel;
  int gotPartial;
  int needPrompt;
  int ignorestdin; ///< flag to avoid checking stdin for command input
  VMDApp *app;

  // XXX: not used
  double starttime, delay;
  int consoleisatty;

  /// Add a bunch of commands to the interpreter.  Eventually, most of these
  /// commands will be dynamically loaded by the interpreter instead of 
  /// hard-coded like they are here.
  void add_commands();

  void setMap(const char *, const char *, const char *);

public:
  TclTextInterp(VMDApp *, int guienabled, int mpienabled);
  ~TclTextInterp();

  virtual void doInit();
  virtual void doEvent();
  virtual int doTkUpdate();

  /// Evaluate the given string in the Tcl interpreter.  Return success,
  /// or false on Tcl error.
  virtual int evalString(const char *);

  /// this evalFile reads lines one at a time and calls VMD display update
  /// every time it reads a complete command.  Returns success.
  virtual int evalFile(const char *);

  virtual void setString(const char *, const char *);

  Tcl_Interp* get_interp() {
    return interp;
  }

  /// set the text processor to wait for the given number of seconds before
  /// reading another text command
  void wait(float wd);
  int done_waiting();

  //
  // callbacks for various VMD events
  //
  //virtual void display_update_cb();
  virtual void frame_cb(int molid, int frame);
  virtual void help_cb(const char *topic);
  virtual void initialize_structure_cb(int molid, int create_or_destroy);
  virtual void molecule_changed_cb(int molid, int code); 
  virtual void logfile_cb(const char *cmd);
  virtual void mousemode_cb(const char *mode, int submode);
  virtual void mouse_pos_cb(float x, float y, int buttondown);
  virtual void mobile_cb(float tx, float ty, float tz,
                         float rx, float ry, float rz, int buttondown);
  virtual void mobile_state_changed_cb();
  virtual void mobile_device_command_cb(const char *str);
  virtual void spaceball_cb(float tx, float ty, float tz,
                            float rx, float ry, float rz, int buttondown);
  virtual void pick_atom_cb(int molid, int atomid, int shift_state, bool is_pick);
  virtual void pick_atom_callback_cb(int molid, int atm, const char *client);
  virtual void pick_selection_cb(int n, const int *atoms);
  virtual void pick_value_cb(float value);
  virtual void pick_graphics_cb(int molid, int tag, int btn, int shift_state);
  //virtual void python_cb(const char *cmd) {}
  virtual void tcl_cb(const char *cmd);
  virtual void timestep_cb(int molid, int frame);
  virtual void trajectory_cb(int molid, const char *fname);
  virtual void graph_label_cb(const char *type, const int *ids, int n);
  virtual void userkey_cb(const char *canonical_key_desc);
};

#endif
