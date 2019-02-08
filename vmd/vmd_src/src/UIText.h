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
 *	$RCSfile: UIText.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.79 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * This is the User Interface for text commands.  It reads characters from
 * the console and from files, and executes the commands.
 *
 ***************************************************************************/
#ifndef UITEXT_H
#define UITEXT_H

#include "UIObject.h"
#include "Inform.h"

#ifdef VMDTCL
#include "TclTextInterp.h" // use Tcl interp if we were compiled with Tcl.
#endif
class TextInterp;

/// UIObject class providing text based user interfaces for scripting etc
class UIText : public UIObject {
private:
  TextInterp *interp;          ///< text interpreter object
  TextInterp *tclinterp;       ///< Tcl interpreter
  TextInterp *pythoninterp;    ///< Python interpreter
  
  Inform *cmdbufstr;

public:
  /// constructor
  UIText(VMDApp *, int guienabled, int mpienabled);

  /// destructor
  virtual ~UIText(void);

#ifdef VMDVRJUGGLER
private:
  bool _isInitialized;
#endif

public:
  /// set up the interpreter environment
  void read_init(void); 

  /// change to the text interpreter with the given name.  Currently 
  /// "tcl" and "python" are supported.  Return success.
  int change_interp(const char *interpname);

  /// specify new file to read commands from
  void read_from_file(const char *);

  /// save state to Tcl script.  Return success.
  int save_state(const char *fname);

  /// check for an event, and queue it if found.  Return TRUE if an event
  /// was generated.
  virtual int check_event(void);

  virtual int act_on_command(int, Command *);  

#ifdef VMDTCL
  Tcl_Interp* get_tcl_interp() {
    if (tclinterp == NULL) 
      return NULL;

    return ((TclTextInterp *) tclinterp)->get_interp();
  }
#else
  void* get_tcl_interp() {
    return NULL;
  }
#endif
 
#ifdef VMDVRJUGGLER
  bool isInitialized(void);

  TextInterp* get_interp(){
	  return interp;
  }
#endif

};

#endif

