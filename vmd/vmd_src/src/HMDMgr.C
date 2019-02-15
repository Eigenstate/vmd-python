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
*      $Revision: 1.8 $         $Date: 2019/01/17 21:20:59 $
*
***************************************************************************
* DESCRIPTION:
*   VMD head mounted display (HMD) interface class
*
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HMDMgr.h"
#include "Inform.h"
#include "Matrix4.h"
#include "VMDQuat.h"

#if defined(VMDUSEOPENHMD)
#include <openhmd.h>
#endif

#if 0
// helper routine
static void quat_rot_matrix(float *m, const float *q) {
  m[ 0] = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
  m[ 1] = 2.0 * (q[0] * q[1] - q[2] * q[3]);
  m[ 2] = 2.0 * (q[2] * q[0] + q[1] * q[3]);
  m[ 3] = 0.0;

  m[ 4] = 2.0 * (q[0] * q[1] + q[2] * q[3]);
  m[ 5] = 1.0 - 2.0 * (q[2] * q[2] + q[0] * q[0]);
  m[ 6] = 2.0 * (q[1] * q[2] - q[0] * q[3]);
  m[ 7] = 0.0;

  m[ 8] = 2.0 * (q[2] * q[0] - q[1] * q[3]);
  m[ 9] = 2.0 * (q[1] * q[2] + q[0] * q[3]);
  m[10] = 1.0 - 2.0 * (q[1] * q[1] + q[0] * q[0]);
  m[11] = 0.0;

  m[12] = 0.0;
  m[13] = 0.0;
  m[14] = 0.0;
  m[15] = 1.0;
}
#endif

HMDMgr::HMDMgr(void) {
  hmdcount = 0;

#if defined(VMDUSEOPENHMD)
  int i;
  ctx = ohmd_ctx_create();

  hmdcount = ohmd_ctx_probe(ctx);

  if (hmdcount < 1) {
    msgInfo << "No HMD device detected." << sendmsg;
  } else {
    msgInfo << "Found " << hmdcount << " HMD device" 
            << ((hmdcount > 1) ? "s" : "") << sendmsg;
  }

  for (i=0; i < hmdcount; i++) {
    printf("HMD[%d] ", i);
    printf("%s", ohmd_list_gets(ctx, i, OHMD_PRODUCT));
    printf(", %s", ohmd_list_gets(ctx, i, OHMD_VENDOR));
    printf(", USB dev: %s\n", ohmd_list_gets(ctx, i, OHMD_PATH));
  }

  // open first HMD we find...
  hmd = ohmd_list_open_device(ctx, 0);

  if (!hmd) {
    printf("failed to open device: %s\n", ohmd_ctx_get_error(ctx));
  }
#endif
}

HMDMgr::~HMDMgr(void) {
#if defined(VMDUSEOPENHMD)
  if (ctx)
    ohmd_ctx_destroy(ctx);
#endif
}


int HMDMgr::device_count(void) {
  return hmdcount;
}


void HMDMgr::update(void) {
#if defined(VMDUSEOPENHMD)
  ohmd_ctx_update(ctx);
#endif
}


void HMDMgr::get_rot_quat(float *q, int doupdate) {
#if defined(VMDUSEOPENHMD)
  if (doupdate)
    update();
  ohmd_device_getf(hmd, OHMD_ROTATION_QUAT, q);
#else
  memset(q, 0, 4 * sizeof(float));
#endif
}


void HMDMgr::reset_orientation(void) {
#if defined(VMDUSEOPENHMD)
  // reset rotation and position
  float zero[] = {0, 0, 0, 1};
  ohmd_device_setf(hmd, OHMD_ROTATION_QUAT, zero);
  ohmd_device_setf(hmd, OHMD_POSITION_VECTOR, zero);
#endif
}


void HMDMgr::rot_point_quat(float *p, const float *op) {
  float q[4];
  get_rot_quat(q, 0);

  // manipulate the HMD pose quaternion to account
  // for reversed coordinate system handedness and 
  // reversing the direction of rotation we apply to
  // the camera basis vectors vs. what the HMD reports
  q[0] = -q[0];
  q[1] = -q[1];

#if 0
  Matrix4 m;
  q[0] = -q[0];
  q[1] = -q[1];
  q[2] = -q[2];
  q[3] =  q[3];
  quat_rot_matrix(&m.mat[0], q);
  m.multnorm3d(op, p);
#else
  Quat Q(q[0], q[1], q[2], q[3]);
  Q.multpoint3(op, p);
#endif
}


void HMDMgr::rot_basis_quat(float *u, float *v, float *w,                    
                            const float *ou, const float *ov, const float *ow) {
  float q[4];
  get_rot_quat(q, 0);

  // manipulate the HMD pose quaternion to account
  // for reversed coordinate system handedness and 
  // reversing the direction of rotation we apply to
  // the camera basis vectors vs. what the HMD reports
  q[0] = -q[0];
  q[1] = -q[1];

#if 0
  Matrix4 m;
  q[0] = -q[0];
  q[1] = -q[1];
  q[2] = -q[2];
  q[3] =  q[3];
  quat_rot_matrix(&m.mat[0], q);
  m.multnorm3d(ou, u);
  m.multnorm3d(ov, v);
  m.multnorm3d(ow, w);
#else
  Quat Q(q[0], q[1], q[2], q[3]);
  Q.multpoint3(ou, u);
  Q.multpoint3(ov, v);
  Q.multpoint3(ow, w);
#endif
}


#if 0
int HMDMgr::device_list(int **, char ***) {
  return 0;
}
#endif

