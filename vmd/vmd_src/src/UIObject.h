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
 *	$RCSfile: UIObject.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.40 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * User Interface Object base class.  All user interface modules are derived
 * from this; it provides methods for registering with the command processor
 * and 'signing up' for which commands it is interested in, as well as
 * generating commands and events.
 *
 ***************************************************************************/
#ifndef UIOBJECT_H
#define UIOBJECT_H

class CommandQueue;
class Command;
class VMDApp;

/// User Interface Object base class.  All user interface modules are derived
/// from this; it provides methods for registering with the command processor
/// and 'signing up' for which commands it is interested in, as well as
/// generating commands and events.
class UIObject {
  /// number of commands in flag array
  int maxCmds;
  /// flag array for the commands we are interested in.  Starts out all set
  /// to false, subclasses set the flags for interesting commands.
  char *doCmd;

  /// is the UIObject active or not
  int is_on;

protected:
  /// pointer to parent instance of VMD
  VMDApp *app;

  /// Command Queue to use for new commands
  CommandQueue *cmdQueue;

  /// send a command to the command queue.  
  void runcommand(Command *);

  /// indicate that we are/are not interested in a command
  void command_wanted(int cmd);
  
  /// reset the user interface (force update of all info displays)
  virtual void reset() {}
  
  /// send callbacks whenever the object moves (true) or don't (false)
  int make_callbacks;

  /// virtual methods for performing on/off actions
  virtual void do_on() {}
  virtual void do_off() {}
 
public:
  UIObject(VMDApp *);       ///< constructor
  virtual ~UIObject(void);  ///< destructor
  
  /// Turns the object on or off. When off, the check_event method will not 
  /// be called for the object.  act_on_command will be called because many
  /// UIObjects still depend on being kept current with VMD state and don't
  /// reset themselves when switched on; we should remove this limitation in
  /// the future.
  void On() {
    do_on();
    is_on = 1;
  }
  void Off() {
    do_off();
    is_on = 0;
  }

  /// is UIObject active or not
  int active() const { return is_on; }

  /// is the given command one we're interested in?
  int want_command(int cmd) {
    return !(cmd >= maxCmds || cmd < 0 || !doCmd[cmd]);
  }

  /// check for an event, and queue it if found.  Return TRUE if an event
  /// was generated.
  virtual int check_event() { return 0; }

  /// update the display due to a command being executed.  Return whether
  /// any action was taken on this command.
  /// Arguments are the command type, command object, and the 
  /// success of the command (T or F).
  virtual int act_on_command(int, Command *) { return 0; }

  /// send callbacks whenever the object moves (on=true) or don't (on=false)
  void set_callbacks(int on) { make_callbacks = on; }
};

#endif

