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
 *	$RCSfile: P_VRPNFeedback.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.35 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#if defined(VMDVRPN)

#include "P_VRPNFeedback.h"
#include "utilities.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>    // for sqrt()

VRPNFeedback::VRPNFeedback() {
  fdv = NULL;
}

int VRPNFeedback::do_start(const SensorConfig *config) {
  if (fdv) return 0;
  if (!config->have_one_sensor()) return 0;
  char myUSL[100];
  config->make_vrpn_address(myUSL);
  fdv = new vrpn_ForceDevice_Remote(myUSL);
  set_maxforce(config->getmaxforce());

  // 
  // XXX we wish this worked, but apparently not implemented.
  //     Have haptic device interpolate the force field between 
  //     updates.  This would help eliminate jerky updates.
  //  fdv->setRecoveryTime(10); // make haptic constraint update smoother,
  //                            // by stepping N times between points...
  forceoff();
  return 1;
}

VRPNFeedback::~VRPNFeedback() {
  forceoff();
  //  if(fdv) delete fdv; // XXX VRPN has broken destructors
}

void VRPNFeedback::update() {
  if(!alive()) return;
  //  fdv->mainloop();
}

void VRPNFeedback::addconstraint(float k, const float *location) {
  int i;
  for(i=0;i<3;i++) {
    // the jacobian is k*I
    jac[i][i] -= k;

    // there is a force due to offset from the origin
    F[i] += k*location[i];
  }
}

void VRPNFeedback::addforcefield(const float *origin, const float *force,
				 const float *jacobian) {
  int i;
  for(i=0;i<3;i++) {
    F[i] += force[i];

    // force due to the offset
    F[i] -=   origin[0] * jacobian[0+3*i]
            + origin[1] * jacobian[1+3*i]
            + origin[2] * jacobian[2+3*i];
  }

  for(i=0;i<9;i++) jac[i/3][i%3] += jacobian[i];

}

void VRPNFeedback::sendforce(const float *initial_pos) {
  float disp[3],Finitial[3],Fmag;
  float origin[3]={0,0,0};

  if (!fdv) return;
  vec_sub(disp,initial_pos,origin);
  Finitial[0] = dot_prod(jac[0],disp);
  Finitial[1] = dot_prod(jac[1],disp);
  Finitial[2] = dot_prod(jac[2],disp);
  vec_add(Finitial,F,Finitial);

  Fmag=norm(Finitial);
  if(Fmag>=maxforce && maxforce>=0) { // send a constant force
    float newjac[3][3];
    float newFinitial[3];
    int i, j;

    vec_scale(newFinitial,maxforce/Fmag,Finitial); // scale the force
    for(i=0; i<3; i++) { // now fix up the jacobian
      for(j=0; j<3; j++) {
	float FJj = Finitial[0]*jac[0][j] +
	            Finitial[1]*jac[1][j] +
	            Finitial[2]*jac[2][j];
	newjac[i][j] = jac[i][j] - Finitial[i]*FJj/(Fmag*Fmag);
      }
    }
    for(i=0; i<3; i++) { // scale the jacobian
      vec_scale(newjac[i],maxforce/Fmag,newjac[i]);
    }

    vec_copy(origin,initial_pos); // use the point as an origin
    fdv->sendForceField(origin, newFinitial, newjac, 1000); // send the force
  }
  else {
    // send the requested force
    fdv->sendForceField(origin, F, jac, 1000);
  }
}

inline float sqr(float x) {
  return x*x;
}

void VRPNFeedback::addplaneconstraint(float k, const float *point,
				      const float *thenormal) {
  int i,j;
  float jacobian[9], normal[3], force[3]={0,0,0}, len=0;

  for(i=0;i<3;i++) len += sqr(thenormal[i]); // normalize the normal
  len = sqrtf(len);
  for(i=0;i<3;i++) normal[i] = thenormal[i]/len;

  /* the part of a vector V that is normal to the plane is given by
     N*(V.N)  (when V is a coordinate vector this is very simple)
     And (dFx/dx, dFy/dx, dFz/dx) = N*(i.N) for a plane! */
  for(i=0;i<3;i++) { // V = ith basis vector
    for(j=0;j<3;j++) // jth component of N
      jacobian[i+j*3] = -k * normal[j] * normal[i];
  }
  addforcefield(point,force,jacobian);
}
 
inline void VRPNFeedback::zeroforce() {
  int i,j;
  for(i=0;i<3;i++) {
    F[i]=0;
    for(j=0;j<3;j++) jac[i][j]=0;
  }
}

inline void VRPNFeedback::forceoff() {
  zeroforce();
  if(fdv) fdv->stopForceField();
}

#endif

