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
 *	$RCSfile: DrawRingsUtils.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.14 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Ulities for calculating ring axes, ring puckering and displacement of
 * atoms from the mean ring plane.
 *
 ***************************************************************************/

#ifndef DRAWRINGUTILS_H
#define DRAWRINGUTILS_H

#include "SmallRing.h"

// Calculate Hill-Reilly pucker sum for a given ring
float hill_reilly_ring_pucker(SmallRing &ring, float *framepos);

// Calculate Hill-Reilly puckering parameters and convert these to a ring colour
void hill_reilly_ring_color(SmallRing &ring, float *framepos, float *rgb);

void hill_reilly_ring_colorscale(SmallRing &ring, float *framepos, float vmin, float vmax, const Scene *scene, float *rgb);

// Calculate Cremer-Pople puckering parameters and convert these to a ring colour
void cremer_pople_ring_color(SmallRing &ring, float *framepos, float *rgb);

// helper functions for Cremer-Pople puckering calculations
void atom_displ_from_mean_plane(float * X, float * Y, float * Z,
                                float * displ, int N);

int cremer_pople_params(int N_ring_atoms, float * displ, float * q,
                        float * phi, int  & m , float & Q);

// Calculates the position at point t along the spline with co-efficients
// A, B, C and D.
// spline(t) = ((A * t + B) * t + C) * t + D
void ribbon_spline(float *pos, const float * const A, const float * const B,
                               const float * const C, const float * const D, const float t);

/*
 * Ribbon Frame: A frame of reference at a point along a ribbon being drawn by
 *               using Twister algorithm.
 * A frame has an origin and 3 basis vectors, plus and approximate cumulative arc
 * length (used for texturing).
 */
struct RibbonFrame {
    float forward[3];
    float right[3];
    float up[3];
    float origin[3];
    float arclength;
};



#endif
