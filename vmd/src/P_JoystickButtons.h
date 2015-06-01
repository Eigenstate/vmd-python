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
 *	$RCSfile: P_JoystickButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.17 $	$Date: 2010/12/16 04:08:31 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * This is a Buttons that gets its info from the Win32 joystick API
 *
 ***************************************************************************/

#ifdef WINGMAN

/// Buttons subclass that gets its info from the Win32 joystick API
class JoystickButtons : public Buttons {
private:
  JoyHandle joy;

protected:
  virtual int do_start(const SensorConfig *);

public:
  JoystickButtons();
  ~JoystickButtons();
  
  virtual const char *device_name() { return "joystickbuttons"; }
  virtual Buttons *clone() { return new JoystickButtons; }

  virtual void update();
  inline virtual int alive() { return joy!=NULL; }
};

#endif

