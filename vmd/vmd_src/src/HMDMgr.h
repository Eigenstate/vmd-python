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
*      $RCSfile: OptiXDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.5 $         $Date: 2019/01/17 21:20:59 $
*
***************************************************************************
* DESCRIPTION:
*   VMD head mounted display (HMD) interface class
*
***************************************************************************/

#ifndef HMDMGR_H
#define HMDMGR_H

#if defined(VMDUSEOPENHMD)
class ohmd_context;
class ohmd_device;
#endif

class HMDMgr {
private: 
#if defined(VMDUSEOPENHMD)
  ohmd_context *ctx;
  ohmd_device  *hmd;
#endif

  int hmdcount;

public: 
  HMDMgr(void);
  ~HMDMgr(void);

  void reset_orientation(void);
  int device_count(void);
  void update(void);
  void get_rot_quat(float *, int doupdate);
  void rot_point_quat(float *p, const float *op);
  void rot_basis_quat(float *u, float *v, float *w, 
                      const float *ou, const float *ov, const float *ow);
#if 0
  int device_list(int **, char ***);
#endif

}; 

#endif

