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
 *	$RCSfile: SpaceballButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.5 $	$Date: 2019/01/17 21:21:01 $
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

