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
 *	$RCSfile: MobileButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2011/06/22 19:02:47 $
 *
 ***************************************************************************/

/// Buttons subclass that gets its info from a WiFi SmartMobile
class MobileButtons : public Buttons {
private:
  VMDApp *app;
  int numButtons;
   
protected:
  virtual int do_start(const SensorConfig *);

public:
  MobileButtons(VMDApp *);
  
  virtual const char *device_name() const { return "mobilebuttons"; }
  virtual Buttons *clone() { return new MobileButtons(app); }

  virtual void update();
  inline virtual int alive() { return 1; }
};

