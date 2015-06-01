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
 *  $RCSfile: DrawRingsUtils.C,v $
 *  $Author: johns $  $Locker:  $    $State: Exp $
 *  $Revision: 1.35 $  $Date: 2011/03/16 15:18:05 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Ulities for calculating ring axes, ring puckering and displacement of
 * atoms from the mean ring plane.
 *
 ***************************************************************************/

#include <math.h>
#include "utilities.h"
#include "AtomColor.h"
#include "Scene.h"
#include "DrawRingsUtils.h"

/*
 * Hill-Reilly pucker-paramter based coloring
 */
void lerp_color_range(float *newcol, float val, float rmin, float rmax,
                      float *mincol, float *maxcol) {
  float range = rmax-rmin;
  float lerpval = (val - rmin) / range;
  vec_lerp(newcol, mincol, maxcol, lerpval);
}


void hotcold_gradient_lerp(float pucker_sum, float *rgb) {
  vec_zero(rgb); // set default color to black

  // hot to cold color map
  // Red (1, 0, 0) -> Yellow (1, 1, 0) -> Green (0, 1, 0) -> Cyan (0, 1, 1) -> blue (0, 0, 1)
  float     red[3] = {1.0f, 0.0f, 0.0f};
  float  yellow[3] = {1.0f, 1.0f, 0.0f};
  float yellow2[3] = {0.8f, 1.0f, 0.0f};
  float   green[3] = {0.0f, 1.0f, 0.0f};
  float  green2[3] = {0.6f, 1.0f, 0.0f};
  float    cyan[3] = {0.0f, 1.0f, 1.0f};
  float   cyan2[3] = {0.0f, 1.0f, 0.8f};
  float    blue[3] = {0.0f, 0.0f, 1.0f};

  if (pucker_sum < 0.25f) {
    lerp_color_range(rgb, pucker_sum, 0.00f, 0.25f, red, yellow);
  } else if (pucker_sum < 0.45f) {
    vec_copy(rgb, yellow);
  } else if (pucker_sum < 0.55f) {
    lerp_color_range(rgb, pucker_sum, 0.45f, 0.55f, yellow, green2);
  } else if (pucker_sum < 0.75f) {
    lerp_color_range(rgb, pucker_sum, 0.55f, 0.75f, green, cyan2);
  } else {
    lerp_color_range(rgb, pucker_sum, 0.75f, 1.00f, cyan, blue);
  }

  clamp_color(rgb); // clamp color values to legal range
}


void hotcold_gradient(float pucker_sum, float *rgb) {
  vec_zero(rgb); // set default color to black

  // hot to cold color map
  // Red (1, 0, 0) -> Yellow (1, 1, 0) -> Green (0, 1, 0) -> Cyan (0, 1, 1) -> blue (0, 0, 1) -> magenta (1, 0, 1)
  if (pucker_sum < 0.40f) {  //MK - envelopes here
    rgb[0] = 1.0f;  // red
    rgb[1] = pucker_sum * 2.5f;  // MK from red increasing green -> yellow -  adjusted multiplier for large range
    rgb[2] = 0.0f;
  } else if (pucker_sum < 0.56f) {
    rgb[0] = 1.0f - (pucker_sum - 0.40f) * 6.25f; // from Yellow, decrease red -> green adjusted multiplier for small range
    rgb[1] = 1.0f;
    rgb[2] = 0.0f;
  } else if (pucker_sum < 0.64f) {
    rgb[0] = 0.0f;
    rgb[1] = 1.0f; //green
    rgb[2] = (pucker_sum - 0.56f) * 12.5f; // from green, increasing blue ->  cyan,  adjusted multiplier for small range
  } else if (pucker_sum < 0.76f) {
    rgb[0] = 0.0f;
    rgb[1] = 1.0f - (pucker_sum - 0.64f) * 5.0f; // from cyan, decrease green -> blue, adjusted multiplier for small range
    rgb[2] = 1.0f;
  } else {
    rgb[0] = (pucker_sum - 0.76f) * 0.8f; // from blue, increase red to get magenta, adjusted multiplier for very large range
    rgb[1] = 0.0f;
    rgb[2] = 1.0f;
  }

  clamp_color(rgb); // clamp color values to legal range
}


float hill_reilly_ring_pucker(SmallRing &ring, float *framepos) {
  int N = ring.num(); // the number of atoms in the current ring

#if 0
  // return the default color if this isn't a 5 or 6 ring atom
  if (N != 5 && N != 6)
    return 0.0;
    //MK added
    if (N==6) {
      //MK do Hill-Reilly for 6-membered rings
      int NP = N-3; // number of puckering parameters
      float *X = new float[N*3]; // atom co-ordinates
      float *r = new float[N*3]; // bond vectors
      float *a = new float[NP*3]; // puckering axes
      float *q = new float[NP*3]; // normalized puckering vectors
      float *n = new float[3]; // normal to reference plane
      float *p = new float[3]; // a flap normal
      float *theta = new float[NP]; // puckering parameters
      float pucker_sum;
      float max_pucker_sum;
      float *atompos;
      int curatomid, i, j, k, l;
    
      // load ring co-ordinates
      for (i=0; i<N; i++) {
        curatomid = ring[i];
        atompos = framepos + 3*curatomid;
        X[3*i] = atompos[0];
        X[3*i+1] = atompos[1];
        X[3*i+2] = atompos[2];
      }     
    
      // calculate bond vectors
      for (i=0; i<N; i++) {
        j = (i+1) % N;
        vec_sub(r+3*i,X+3*j,X+3*i);
      }
    
      // calculate puckering axes, flap normals and puckering vectors
      for (i=0; i<NP; i++) {
        k = (2*(i+1)) % N;
        j = (2*i) % N;
        l = (2*i+1) % N;
        vec_sub(a+3*i,X+3*k,X+3*j);
        cross_prod(p,r+3*j,r+3*l);
        cross_prod(q+3*i,a+3*i,p);
        vec_normalize(q+3*i);
      }
    
      // reference normal
      cross_prod(n,a+3*0,a+3*1);
      vec_normalize(n);
    
      // calculate the puckering parameters
      pucker_sum = 0.0;
    
      for (i=0; i<NP; i++) {
        theta[i] = (float(VMD_PI)/2.0f) - acosf(dot_prod(q+3*i, n));
        pucker_sum += theta[i];
      }
    
    
      // 0.6154 radians (35.26 degrees) has significance for perfect tetrahedral bond geometry (see Hill paper)
      max_pucker_sum = NP * 0.6154f;
      float pucker_scaled = pucker_sum/max_pucker_sum;
      pucker_sum = fabsf((pucker_scaled < 1.0f) ? pucker_scaled : 1.0f);
      pucker_sum = (pucker_sum < 1.0f) ? pucker_sum : 1.0f;
    
      delete [] X;
      delete [] r;
      delete [] a;
      delete [] q;
      delete [] n;
      delete [] p;
      delete [] theta;
      return pucker_sum;
    }  //end MK if N==6
    else {  //N==5 
#endif
    float *xring = new float[N];
    float *yring = new float[N];
    float *zring = new float[N];
    float *displ = new float[N];
    float *q = new float[N];
    float *phi = new float[N];
    float Q;
    int m;
    float *atompos;
    int curatomid;
    
    for (int i=0; i<N; i++) {
      curatomid = ring[i];
      atompos = framepos + 3*curatomid; // pointer arithmetic is evil :)
      xring[i] = atompos[0];
      yring[i] = atompos[1];
      zring[i] = atompos[2];
    }     
    
    atom_displ_from_mean_plane(xring, yring, zring, displ, N);

    delete [] xring;
    delete [] yring;
    delete [] zring;

    if (cremer_pople_params(N, displ, q, phi, m, Q)) {
      // Q is the puckering amplitude - i.e. the intensity of the pucker.
      Q = (Q < 2.0f) ? Q : 2.0f;  //truncate amplitude at 2
      delete [] displ;
      delete [] q;
      delete [] phi;
      return Q;
    } else {
      delete [] displ;
      delete [] q;
      delete [] phi;
      return 0.0;
    }    
#if 0
  }
#endif
}


// Calculate Hill-Reilly Pucker Parameters and convert these to a ring colour
void hill_reilly_ring_color(SmallRing &ring, float *framepos, float *rgb) {
  float pucker_sum = hill_reilly_ring_pucker(ring, framepos); 
  hotcold_gradient(pucker_sum, rgb);
  //MK added and now removed   hotcold_gradient(pucker_sum/0.8, rgb);  //scale, assuming amplitude value not bigger than 0.8
}

void hill_reilly_ring_colorscale(SmallRing &ring, float *framepos, 
                                 float vmin, float vmax,
                                 const Scene *scene, float *rgb) {
  float pucker_sum = hill_reilly_ring_pucker(ring, framepos); 

  // map data min/max to range 0->1
  // values must be clamped before use, since user-specified
  // min/max can cause out-of-range color indices to be generated
  float vscale;
  float vrange = vmax - vmin;
  if (fabsf(vrange) < 0.00001f)
    vscale = 0.0f;
  else
    vscale = 1.00001f / vrange;

  float level = (pucker_sum - vmin) * vscale;
  int colindex = (int)(level * MAPCLRS-1);
  if (colindex < 0)
    colindex = 0;
  else if (colindex >= MAPCLRS)
    colindex = MAPCLRS-1;
 
  const float *scrgb = scene->color_value(MAPCOLOR(colindex));
  
  rgb[0] = scrgb[0];
  rgb[1] = scrgb[1];
  rgb[2] = scrgb[2];
}



/*
 * Cremer-Pople pucker-parameter based coloring
 */

// return sum + (x,y,z)
void vec_incr(float sum[3], const float x, const float y, const float z) {
  sum[0] += x;
  sum[1] += y;
  sum[2] += z;
}

// Calculates cartesian axes based on coords of nuclei in ring
// using cremer-pople algorithm. It is assumed that the
// centre of geometry is the centre of the ring.
void ring_axes(const float * X, const float * Y, const float * Z, int N, 
             float x[3], float y[3], float z[3]) {
  float Rp[3] = {0.0, 0.0, 0.0}; float Rpp[3] = {0.0, 0.0, 0.0};
  int j;
  for (j=0; j<N; j++) {
    float ze_angle = 2.0f * float(VMD_PI) * float(j-1) / float(N);
    float ze_sin = sinf(ze_angle);
    float ze_cos = cosf(ze_angle);
    vec_incr(Rp, X[j]*ze_sin, Y[j]*ze_sin, Z[j]*ze_sin);
    vec_incr(Rpp, X[j]*ze_cos, Y[j]*ze_cos, Z[j]*ze_cos);
  }

  cross_prod(z, Rp, Rpp);
  vec_normalize(z);

  /* 
   * OK, now we have z, the norm to the central plane, we need
   * to calculate y as the projection of Rp onto the plane
   * and x as y x z
   */
  float lambda = dot_prod(z, Rp);
  y[0] = Rp[0] - z[0]*lambda;
  y[1] = Rp[1] - z[1]*lambda;
  y[2] = Rp[2] - z[2]*lambda;
  vec_normalize(y);
  cross_prod(x, y, z);    // voila !
}


// Calculate distances of atoms from the mean ring plane
void atom_displ_from_mean_plane(float * X, float * Y, float * Z, 
                                float * displ, int N) {
  float cog[3] = {0.0, 0.0, 0.0};
  float x_axis[3], y_axis[3], z_axis[3];
  int i;

  // calculate centre of geometry
  for (i=0; i<N; i++) {
      cog[0] += X[i]; cog[1] += Y[i]; cog[2] += Z[i];
  }
  cog[0] /= float(N); 
  cog[1] /= float(N); 
  cog[2] /= float(N); 

  // centre the ring
  for (i=0; i<N; i++) {
    X[i] -= cog[0]; 
    Y[i] -= cog[1]; 
    Z[i] -= cog[2];
  }

  ring_axes( X, Y, Z, N, x_axis, y_axis, z_axis );

  // calculate displacement from mean plane
  for (i=0; i<N; i++) {
    displ[i] = X[i]*z_axis[0] + Y[i]*z_axis[1] + Z[i]*z_axis[2];
  }
}


// Calculate Cremer-Pople puckering parameters
int cremer_pople_params(int N_ring_atoms, float * displ, float * q, 
                        float * phi, int  & m , float & Q) {
  int i, j, k;
  if (N_ring_atoms<3)
    return -1;

  float N = float(N_ring_atoms);
  phi[0]=0;  q[0]=0;  //no puckering parameters for m=1

  // if even no ring atoms, first calculate unpaired q puck parameter
  if (fmod(N, 2.0f)==0) {
    float sum =0;
    m = N_ring_atoms/2 -1;

    for (i=0; i<N_ring_atoms; i++)         
      sum += displ[i]*cosf(i*float(VMD_PI));

    q[int(N_ring_atoms/2)-1]=sqrtf(1.0f/N)*sum;
  } else {
    m = int(N_ring_atoms-1)/2;
  }

  // calculate paired puckering parameters
  for (i=1; i<m; i++) {
    float q_cosphi=0, q_sinphi=0;
    for (j=0; j<N_ring_atoms; j++) {
      q_cosphi += displ[j]*cosf(2.0f*float(VMD_PI)*float((i+1)*j)/N);
      q_sinphi += displ[j]*sinf(2.0f*float(VMD_PI)*float((i+1)*j)/N);
    }

    q_cosphi *=  sqrtf(2.0f/N);
    q_sinphi *= -sqrtf(2.0f/N);
    phi[i]=atanf(q_sinphi/q_cosphi);

    if (q_cosphi < 0)
      phi[i]+=float(VMD_PI);
    else if (q_sinphi < 0) 
      phi[i]+=2.0f*float(VMD_PI);

    q[i]=q_cosphi/phi[i];
    //phi[i]*=180.0f/VMD_PI;  //convert to degrees

    //calculate puckering amplitude
    Q=0.0f;
    for (k=0; k<N_ring_atoms; k++)
      Q+=displ[k]*displ[k];
    Q=sqrtf(Q);  
  }

  return 1;
}


// Calculate Cremer-Pople Pucker Parameters and convert these to a ring colour
void cremer_pople_ring_color(SmallRing &ring, float *framepos, float *rgb) {
  int N = ring.num(); //the number of atoms in the current ring
  float *xring = new float[N];
  float *yring = new float[N];
  float *zring = new float[N];
  float *displ = new float[N];
  float *q = new float[N];
  float *phi = new float[N];
  float Q;
  int m;
  float *atompos;
  int curatomid;

  vec_zero(rgb); // set default color to black

  for (int i=0; i<N; i++) {
    curatomid = ring[i];
    atompos = framepos + 3*curatomid; // pointer arithmetic is evil :)
    xring[i] = atompos[0];
    yring[i] = atompos[1];
    zring[i] = atompos[2];
  }     

  atom_displ_from_mean_plane(xring, yring, zring, displ, N);
         
  if (N==6) { //special case - pyranose rings
    if (cremer_pople_params(N, displ, q, phi, m, Q)) {
      float cosTheta = q[2]/Q;
      float theta = acosf(cosTheta); 
      float sinTheta = sinf(theta);

      // Q is the puckering amplitude - i.e. the intensity of the pucker.
      // multiply by Q to show intensity, particularly for rings with 
      // little pucker (black)
      // NOTE -using abs - polar positions therefore equivalent
      float intensity = Q;
      
      rgb[0] = fabsf(sinTheta)*intensity;
      rgb[1] = fabsf(cosTheta)*intensity;
      rgb[2] = fabsf(sinf(3.0f*phi[1])*sinTheta)*intensity;
    }
  } else if (N==5) { //special case - furanose rings
    if (cremer_pople_params(N, displ, q, phi, m, Q)) {
      rgb[0] = 0;
      rgb[1] = 0;
      rgb[2] = Q;
    }
  }

  // clamp color values to legal range
  clamp_color(rgb);
 
  delete [] xring;
  delete [] yring;
  delete [] zring;
  delete [] displ;
  delete [] q;
  delete [] phi;
}


/*
 * Ribbon spline handling
 */
// Calculates the position at point t along the spline with co-efficients
// A, B, C and D.
// spline(t) = ((A * t + B) * t + C) * t + D
void ribbon_spline(float *pos, const float * const A, const float * const B,
                   const float * const C, const float * const D, const float t) {
  vec_copy(pos,D);
  vec_scaled_add(pos,t,C);
  vec_scaled_add(pos,t*t,B);
  vec_scaled_add(pos,t*t*t,A);
}




