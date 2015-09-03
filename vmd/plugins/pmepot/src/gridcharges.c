/*
 * routines for computing charges on a grid 
 * 
 * $Id: gridcharges.c,v 1.3 2005/07/20 15:37:39 johns Exp $
 *
 */

#include <string.h>
#include <math.h>

void compute_b_spline(float *frac, float *M) {
  int j;
  float x,y,z,x1,y1,z1, div;
  float *Mx, *My, *Mz;
  Mx=M-1; My=M+4-1; Mz=M+2*4-1;
  x=frac[0];
  y=frac[1];
  z=frac[2];
  x1=1.0-x; y1=1.0-y; z1=1.0-z;
  /* Do n=3 case first */
  Mx[1]=.5*x1*x1;
  Mx[2]=x1*x + .5;
  Mx[3]=0.5*x*x;
  Mx[4]=0.0;
  My[1]=.5*y1*y1;
  My[2]=y1*y + .5;
  My[3]=0.5*y*y;
  My[4]=0.0;
  Mz[1]=.5*z1*z1;
  Mz[2]=z1*z + .5;
  Mz[3]=0.5*z*z;
  Mz[4]=0.0;
  /* Now finish the job!    */
  div=1.0/(4-1);
  Mx[4] = x*div*Mx[4-1];
  My[4] = y*div*My[4-1];
  Mz[4] = z*div*Mz[4-1];
  for (j=1; j<=4-2; j++) {
    Mx[4-j] = ((x+j)*Mx[4-j-1] + (4-x-j)*Mx[4-j])*div;
    My[4-j] = ((y+j)*My[4-j-1] + (4-y-j)*My[4-j])*div;
    Mz[4-j] = ((z+j)*Mz[4-j-1] + (4-z-j)*Mz[4-j])*div;
  }
  Mx[1] *= (1.0-x)*div;
  My[1] *= (1.0-y)*div;
  Mz[1] *= (1.0-z)*div;
}

static float dot_product(const float *v1, const float *v2) {
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

static void scale_vector(float *v, float s) {
  v[0] *= s;
  v[1] *= s;
  v[2] *= s;
}

static void scale_add_vector(float *c, float s, const float *v) {
  c[0] += s * v[0];
  c[1] += s * v[1];
  c[2] += s * v[2];
}

static void copy_vector(float *c, const float *v) {
  c[0] = v[0];
  c[1] = v[1];
  c[2] = v[2];
}

static void cross_product(float *c, const float *v1, const float *v2) {
  c[0] = v1[1]*v2[2]-v2[1]*v1[2];
  c[1] = v2[0]*v1[2]-v1[0]*v2[2];
  c[2] = v1[0]*v2[1]-v2[0]*v1[1];
}

void reciprocal_lattice(const float *cell, float *rcell) {
  const float *a1, *a2, *a3;
  float *b1, *b2, *b3;
  rcell[0] = cell[0]; rcell[1] = cell[1]; rcell[2] = cell[2];
  a1 = &cell[1*3]; a2 = &cell[2*3]; a3 = &cell[3*3];
  b1 = &rcell[1*3]; b2 = &rcell[2*3]; b3 = &rcell[3*3];
  cross_product(b1,a2,a3);  scale_vector(b1, 1./dot_product(a1,b1));
  cross_product(b2,a3,a1);  scale_vector(b2, 1./dot_product(a2,b2));
  cross_product(b3,a1,a2);  scale_vector(b3, 1./dot_product(a3,b3));
}


int fill_charges(const int *dims, const float *cell, int natoms,
		const float *xyzq, float *q_arr, float *rcell, float *oddd) {
  
  int i, j, k, l;
  int K1, K2, K3, dim2, dim3;
  float frac[3], Mi[12];
  float ox,oy,oz,r1x,r1y,r1z,r2x,r2y,r2z,r3x,r3y,r3z,kx,ky,kz;

  K1=dims[0]; K2=dims[1]; K3=dims[2]; dim2=dims[3]; dim3=dims[4];

  memset( (void*) q_arr, 0, K1*dim2*dim3 * sizeof(float) );

  reciprocal_lattice(cell,rcell);
  scale_add_vector(&rcell[0], -0.5*(K1-1.)/K1, &cell[1*3]);
  scale_add_vector(&rcell[0], -0.5*(K2-1.)/K2, &cell[2*3]);
  scale_add_vector(&rcell[0], -0.5*(K3-1.)/K3, &cell[3*3]);

  copy_vector(&oddd[0*3],&rcell[0*3]);
  copy_vector(&oddd[1*3],&cell[1*3]);
  copy_vector(&oddd[2*3],&cell[2*3]);
  copy_vector(&oddd[3*3],&cell[3*3]);
  scale_vector(&oddd[1*3],1./K1);
  scale_vector(&oddd[2*3],1./K2);
  scale_vector(&oddd[3*3],1./K3);

  ox = rcell[0];
  oy = rcell[1];
  oz = rcell[2];
  r1x = rcell[3];
  r1y = rcell[4];
  r1z = rcell[5];
  r2x = rcell[6];
  r2y = rcell[7];
  r2z = rcell[8];
  r3x = rcell[9];
  r3y = rcell[10];
  r3z = rcell[11];
  kx = 2./K1;  /* cancels position shifts below */
  ky = 2./K2;
  kz = 2./K3;

  for (i=0; i<natoms; i++) {
    float px,py,pz,sx,sy,sz;
    float x,y,z,q;
    int u1, u2, u2i, u3i;

    px = xyzq[4*i+0] - ox;
    py = xyzq[4*i+1] - oy;
    pz = xyzq[4*i+2] - oz;
    sx = px*r1x + py*r1y + pz*r1z + kx;
    sy = px*r2x + py*r2y + pz*r2z + ky;
    sz = px*r3x + py*r3y + pz*r3z + kz;
    x = K1 * ( sx - floor(sx) );
    y = K2 * ( sy - floor(sy) );
    z = K3 * ( sz - floor(sz) );
    /*  Check for rare rounding condition where K * ( 1 - epsilon ) == K */
    /*  which was observed with g++ on Intel x86 architecture.           */
    if ( x == K1 ) x = 0;
    if ( y == K2 ) y = 0;
    if ( z == K3 ) z = 0;

    q = xyzq[4*i+3];

    u1 = (int)x;
    u2i = (int)y;
    u3i = (int)z;
    frac[0] = x - u1;
    frac[1] = y - u2i;
    frac[2] = z - u3i;
    compute_b_spline(frac,Mi);
    u1 -= 4;
    u2i -= 4;
    u3i -= 4;
    u3i++;
    for (j=0; j<4; j++) {
      float m1;
      int ind1;
      m1 = Mi[j]*q;
      u1++;
      ind1 = u1 + (u1 < 0 ? K1 : 0);
      u2 = u2i;
      for (k=0; k<4; k++) {
        float m1m2;
	int ind2;
        m1m2 = m1*Mi[4+k];
	u2++;
	ind2 = ind1*dim2 + (u2 + (u2 < 0 ? K2 : 0));
        for (l=0; l<4; l++) {
	  float m3;
	  int ind;
	  int u3 = u3i + l;
	  m3 = Mi[2*4 + l];
          ind = ind2*dim3 + (u3 + (u3 < 0 ? K3 : 0));
          q_arr[ind] += m1m2*m3;
        }
      }
    }
  }

  return 0;
}

