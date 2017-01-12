/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: SpaceballButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.4 $	$Date: 2016/11/28 03:05:04 $
 *
 ***************************************************************************/

/// Buttons subclass that gets its info from the local Spaceball.
class SpaceballButtons : public Buttons {
private:
  VMDApp *app;
  int numButtons;
   
protected:
  virtual int do_start(const SensorConfig *);

public:
  SpaceballButtons(VMDApp *);
  
  virtual const char *device_name() const { return "sballbuttons"; }
  virtual Buttons *clone() { return new SpaceballButtons(app); }

  virtual void update();
  inline virtual int alive() { return 1; }
};

