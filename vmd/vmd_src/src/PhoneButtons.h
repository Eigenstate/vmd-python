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
 *	$RCSfile: PhoneButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.4 $	$Date: 2019/01/17 21:21:01 $
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

