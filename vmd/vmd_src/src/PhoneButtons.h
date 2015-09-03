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
 *	$RCSfile: PhoneButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.2 $	$Date: 2010/12/16 04:08:34 $
 *
 ***************************************************************************/

/// Buttons subclass that gets its info from a WiFi SmartPhone
class PhoneButtons : public Buttons {
private:
  VMDApp *app;
  int numButtons;
   
protected:
  virtual int do_start(const SensorConfig *);

public:
  PhoneButtons(VMDApp *);
  
  virtual const char *device_name() const { return "phonebuttons"; }
  virtual Buttons *clone() { return new PhoneButtons(app); }

  virtual void update();
  inline virtual int alive() { return 1; }
};

