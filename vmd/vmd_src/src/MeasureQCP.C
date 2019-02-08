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
 *      $RCSfile: MeasureQCP.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $       $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Code to compute RMSD values for unaligned structures without 
 *   actually performing the alginment, particularly useful for 
 *   computing large dissimilarity matrices required for 
 *   trajectory clustering analysis
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#define VMDQCPUSESSE  1
// #define VMDQCPUSEAVX2 1
#if defined(VMDUSEAVX512)
#define VMDQCPUSEAVX512 1
#endif

#define VMDQCPUSETHRPOOL 1

#if VMDQCPUSESSE && defined(__SSE2__)
#include <emmintrin.h>
#endif
#if VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
#include <immintrin.h>
#endif
#if VMDQCPUSEAVX512 && defined(__AVX512F__)
#include <immintrin.h>
#endif
#if (defined(VMDQCPUSEVSX) && defined(__VSX__))
#if defined(__GNUC__) && defined(__VEC__)
#include <altivec.h>
#endif
#endif

#include <math.h>
#include "Measure.h"
#include "AtomSel.h"
#include "utilities.h"
#include "ResizeArray.h"
#include "MoleculeList.h"
#include "Inform.h"
#include "Timestep.h"
#include "VMDApp.h"
#include "WKFThreads.h"
#include "WKFUtils.h"

#if VMDQCPUSEAVX512 && defined(__AVX512F__)

static double hadd8_m512d(__m512d sum8) {
//  __m512d tmp = sum8;
//  __m512d hsum4 = _mm512_add_pd(tmp, _mm512_permute2f128_pd(tmp, tmp, 0x1));
//  __m512d hsum2 = _mm256_castpd256_pd128(hsum4);
//  __m512d sum2 = _mm_hadd_pd(hsum2, hsum2);
//  return _mm_cvtsd_f64(sum2);
  return 0.0;
}


// AVX2 + FMA + 32-byte-aligned SOA-format memory buffers
static double InnerProductSOA_avx512(double *A,
                                     float *crdx1, float *crdy1, float *crdz1,
                                     float *crdx2, float *crdy2, float *crdz2,
                                     const int cnt, const float *weight) {
  __m512d va0 = _mm512_set1_pd(0.0);
  __m512d va1 = _mm512_set1_pd(0.0);
  __m512d va2 = _mm512_set1_pd(0.0);
  __m512d va3 = _mm512_set1_pd(0.0);
  __m512d va4 = _mm512_set1_pd(0.0);
  __m512d va5 = _mm512_set1_pd(0.0);
  __m512d va6 = _mm512_set1_pd(0.0);
  __m512d va7 = _mm512_set1_pd(0.0);
  __m512d va8 = _mm512_set1_pd(0.0);
  __m512d vG1 = _mm512_set1_pd(0.0);
  __m512d vG2 = _mm512_set1_pd(0.0);

  if (weight != NULL) {
    for (int i=0; i<cnt; i+=8) {
      __m256 xa8f = _mm256_load_ps(crdx1 + i); // load 8-float vectors
      __m256 ya8f = _mm256_load_ps(crdy1 + i);
      __m256 za8f = _mm256_load_ps(crdz1 + i);

      __m512d xa8 = _mm512_cvtps_pd(xa8f); // convert from float to doubles
      __m512d ya8 = _mm512_cvtps_pd(ya8f);
      __m512d za8 = _mm512_cvtps_pd(za8f);

      __m512d gatmp = _mm512_mul_pd(xa8, xa8);
      gatmp = _mm512_fmadd_pd(ya8, ya8, gatmp);
      gatmp = _mm512_fmadd_pd(za8, za8, gatmp);

      __m256 xb8f = _mm256_load_ps(crdx2 + i); // load 8-float vectors
      __m256 yb8f = _mm256_load_ps(crdy2 + i);
      __m256 zb8f = _mm256_load_ps(crdz2 + i);

      __m512d xb8 = _mm512_cvtps_pd(xb8f); // convert from float to doubles
      __m512d yb8 = _mm512_cvtps_pd(yb8f);
      __m512d zb8 = _mm512_cvtps_pd(zb8f);

      __m512d gbtmp = _mm512_mul_pd(xb8, xb8);
      gbtmp = _mm512_fmadd_pd(yb8, yb8, gbtmp);
      gbtmp = _mm512_fmadd_pd(zb8, zb8, gbtmp);

      __m256 w8f = _mm256_load_ps(weight + i); // load 8-float vector
      __m512d w8 = _mm512_cvtps_pd(w8f); // convert from float to double

      vG1 = _mm512_fmadd_pd(w8, gatmp, vG1);
      vG2 = _mm512_fmadd_pd(w8, gbtmp, vG2);

      va0 = _mm512_fmadd_pd(xa8, xb8, va0);
      va1 = _mm512_fmadd_pd(xa8, yb8, va1);
      va2 = _mm512_fmadd_pd(xa8, zb8, va2);

      va3 = _mm512_fmadd_pd(ya8, xb8, va3);
      va4 = _mm512_fmadd_pd(ya8, yb8, va4);
      va5 = _mm512_fmadd_pd(ya8, zb8, va5);

      va6 = _mm512_fmadd_pd(za8, xb8, va6);
      va7 = _mm512_fmadd_pd(za8, yb8, va7);
      va8 = _mm512_fmadd_pd(za8, zb8, va8);
    }
  } else {
    for (int i=0; i<cnt; i+=8) {
      __m256 xa8f = _mm256_load_ps(crdx1 + i); // load 8-float vectors
      __m256 ya8f = _mm256_load_ps(crdy1 + i);
      __m256 za8f = _mm256_load_ps(crdz1 + i);

      __m512d xa8 = _mm512_cvtps_pd(xa8f); // convert from float to doubles
      __m512d ya8 = _mm512_cvtps_pd(ya8f);
      __m512d za8 = _mm512_cvtps_pd(za8f);

      vG1 = _mm512_fmadd_pd(xa8, xa8, vG1);
      vG1 = _mm512_fmadd_pd(ya8, ya8, vG1);
      vG1 = _mm512_fmadd_pd(za8, za8, vG1);

      __m256 xb8f = _mm256_load_ps(crdx2 + i); // load 8-float vectors
      __m256 yb8f = _mm256_load_ps(crdy2 + i);
      __m256 zb8f = _mm256_load_ps(crdz2 + i);

      __m512d xb8 = _mm512_cvtps_pd(xb8f); // convert from float to doubles
      __m512d yb8 = _mm512_cvtps_pd(yb8f);
      __m512d zb8 = _mm512_cvtps_pd(zb8f);

      vG2 = _mm512_fmadd_pd(xb8, xb8, vG2);
      vG2 = _mm512_fmadd_pd(yb8, yb8, vG2);
      vG2 = _mm512_fmadd_pd(zb8, zb8, vG2);

      va0 = _mm512_fmadd_pd(xa8, xb8, va0);
      va1 = _mm512_fmadd_pd(xa8, yb8, va1);
      va2 = _mm512_fmadd_pd(xa8, zb8, va2);

      va3 = _mm512_fmadd_pd(ya8, xb8, va3);
      va4 = _mm512_fmadd_pd(ya8, yb8, va4);
      va5 = _mm512_fmadd_pd(ya8, zb8, va5);

      va6 = _mm512_fmadd_pd(za8, xb8, va6);
      va7 = _mm512_fmadd_pd(za8, yb8, va7);
      va8 = _mm512_fmadd_pd(za8, zb8, va8);
    }
  }

  A[0] = hadd8_m512d(va0);
  A[1] = hadd8_m512d(va1);
  A[2] = hadd8_m512d(va2);
  A[3] = hadd8_m512d(va3);
  A[4] = hadd8_m512d(va4);
  A[5] = hadd8_m512d(va5);
  A[6] = hadd8_m512d(va6);
  A[7] = hadd8_m512d(va7);
  A[8] = hadd8_m512d(va8);

  double G1 = hadd8_m512d(vG1);
  double G2 = hadd8_m512d(vG2);

  return (G1 + G2) * 0.5;
}

#endif

#if VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__)

static double hadd4_m256d(__m256d sum4) {
  __m256d tmp = sum4;
  __m256d hsum4 = _mm256_add_pd(tmp, _mm256_permute2f128_pd(tmp, tmp, 0x1));
  __m128d hsum2 = _mm256_castpd256_pd128(hsum4);
  __m128d sum2 = _mm_hadd_pd(hsum2, hsum2);
  return _mm_cvtsd_f64(sum2);
}


// AVX2 + FMA + 32-byte-aligned SOA-format memory buffers
static double InnerProductSOA_avx2(double *A,
                                   float *crdx1, float *crdy1, float *crdz1,
                                   float *crdx2, float *crdy2, float *crdz2,
                                   const int cnt, const float *weight) {
  __m256d va0 = _mm256_set1_pd(0.0);
  __m256d va1 = _mm256_set1_pd(0.0);
  __m256d va2 = _mm256_set1_pd(0.0);
  __m256d va3 = _mm256_set1_pd(0.0);
  __m256d va4 = _mm256_set1_pd(0.0);
  __m256d va5 = _mm256_set1_pd(0.0);
  __m256d va6 = _mm256_set1_pd(0.0);
  __m256d va7 = _mm256_set1_pd(0.0);
  __m256d va8 = _mm256_set1_pd(0.0);
  __m256d vG1 = _mm256_set1_pd(0.0);
  __m256d vG2 = _mm256_set1_pd(0.0);

  if (weight != NULL) {
    for (int i=0; i<cnt; i+=4) {
      __m128 xa4f = _mm_load_ps(crdx1 + i); // load 4-float vectors
      __m128 ya4f = _mm_load_ps(crdy1 + i);
      __m128 za4f = _mm_load_ps(crdz1 + i);

      __m256d xa4 = _mm256_cvtps_pd(xa4f); // convert from float to doubles
      __m256d ya4 = _mm256_cvtps_pd(ya4f);
      __m256d za4 = _mm256_cvtps_pd(za4f);

      __m256d gatmp = _mm256_mul_pd(xa4, xa4);
      gatmp = _mm256_fmadd_pd(ya4, ya4, gatmp);
      gatmp = _mm256_fmadd_pd(za4, za4, gatmp);

      __m128 xb4f = _mm_load_ps(crdx2 + i); // load 4-float vectors
      __m128 yb4f = _mm_load_ps(crdy2 + i);
      __m128 zb4f = _mm_load_ps(crdz2 + i);

      __m256d xb4 = _mm256_cvtps_pd(xb4f); // convert from float to doubles
      __m256d yb4 = _mm256_cvtps_pd(yb4f);
      __m256d zb4 = _mm256_cvtps_pd(zb4f);

      __m256d gbtmp = _mm256_mul_pd(xb4, xb4);
      gbtmp = _mm256_fmadd_pd(yb4, yb4, gbtmp);
      gbtmp = _mm256_fmadd_pd(zb4, zb4, gbtmp);

      __m128 w4f = _mm_load_ps(weight + i); // load 4-float vector
      __m256d w4 = _mm256_cvtps_pd(w4f); // convert from float to double

      vG1 = _mm256_fmadd_pd(w4, gatmp, vG1);
      vG2 = _mm256_fmadd_pd(w4, gbtmp, vG2);

      va0 = _mm256_fmadd_pd(xa4, xb4, va0);
      va1 = _mm256_fmadd_pd(xa4, yb4, va1);
      va2 = _mm256_fmadd_pd(xa4, zb4, va2);

      va3 = _mm256_fmadd_pd(ya4, xb4, va3);
      va4 = _mm256_fmadd_pd(ya4, yb4, va4);
      va5 = _mm256_fmadd_pd(ya4, zb4, va5);

      va6 = _mm256_fmadd_pd(za4, xb4, va6);
      va7 = _mm256_fmadd_pd(za4, yb4, va7);
      va8 = _mm256_fmadd_pd(za4, zb4, va8);
    }
  } else {
    for (int i=0; i<cnt; i+=4) {
      __m128 xa4f = _mm_load_ps(crdx1 + i); // load 4-float vectors
      __m128 ya4f = _mm_load_ps(crdy1 + i);
      __m128 za4f = _mm_load_ps(crdz1 + i);

      __m256d xa4 = _mm256_cvtps_pd(xa4f); // convert from float to doubles
      __m256d ya4 = _mm256_cvtps_pd(ya4f);
      __m256d za4 = _mm256_cvtps_pd(za4f);

      vG1 = _mm256_fmadd_pd(xa4, xa4, vG1);
      vG1 = _mm256_fmadd_pd(ya4, ya4, vG1);
      vG1 = _mm256_fmadd_pd(za4, za4, vG1);

      __m128 xb4f = _mm_load_ps(crdx2 + i); // load 4-float vectors
      __m128 yb4f = _mm_load_ps(crdy2 + i);
      __m128 zb4f = _mm_load_ps(crdz2 + i);

      __m256d xb4 = _mm256_cvtps_pd(xb4f); // convert from float to doubles
      __m256d yb4 = _mm256_cvtps_pd(yb4f);
      __m256d zb4 = _mm256_cvtps_pd(zb4f);

      vG2 = _mm256_fmadd_pd(xb4, xb4, vG2);
      vG2 = _mm256_fmadd_pd(yb4, yb4, vG2);
      vG2 = _mm256_fmadd_pd(zb4, zb4, vG2);

      va0 = _mm256_fmadd_pd(xa4, xb4, va0);
      va1 = _mm256_fmadd_pd(xa4, yb4, va1);
      va2 = _mm256_fmadd_pd(xa4, zb4, va2);

      va3 = _mm256_fmadd_pd(ya4, xb4, va3);
      va4 = _mm256_fmadd_pd(ya4, yb4, va4);
      va5 = _mm256_fmadd_pd(ya4, zb4, va5);

      va6 = _mm256_fmadd_pd(za4, xb4, va6);
      va7 = _mm256_fmadd_pd(za4, yb4, va7);
      va8 = _mm256_fmadd_pd(za4, zb4, va8);
    }
  }

  A[0] = hadd4_m256d(va0);
  A[1] = hadd4_m256d(va1);
  A[2] = hadd4_m256d(va2);
  A[3] = hadd4_m256d(va3);
  A[4] = hadd4_m256d(va4);
  A[5] = hadd4_m256d(va5);
  A[6] = hadd4_m256d(va6);
  A[7] = hadd4_m256d(va7);
  A[8] = hadd4_m256d(va8);

  double G1 = hadd4_m256d(vG1);
  double G2 = hadd4_m256d(vG2);

  return (G1 + G2) * 0.5;
}

#endif


// plain C++ version of inner product for SOA coordinate storage
static double InnerProductSOA(double *A,
                              float *crdx1, float *crdy1, float *crdz1,
                              float *crdx2, float *crdy2, float *crdz2,
                              const int cnt, const float *weight) {
  double G1=0.0, G2 = 0.0;
  memset(A, 0, sizeof(double) * 9);

  double x1, x2, y1, y2, z1, z2;
  double a0, a1, a2, a3, a4, a5, a6, a7, a8;
  a0=a1=a2=a3=a4=a5=a6=a7=a8=0.0;
  if (weight != NULL) {
    for (int i=0; i<cnt; i++) {
      double w = weight[i];
      x1 = crdx1[i];
      y1 = crdy1[i];
      z1 = crdz1[i];

      G1 += w * (x1*x1 + y1*y1 + z1*z1);

      x2 = crdx2[i];
      y2 = crdy2[i];
      z2 = crdz2[i];

      G2 += w * (x2*x2 + y2*y2 + z2*z2);

      a0 += x1 * x2;
      a1 += x1 * y2;
      a2 += x1 * z2;

      a3 += y1 * x2;
      a4 += y1 * y2;
      a5 += y1 * z2;

      a6 += z1 * x2;
      a7 += z1 * y2;
      a8 += z1 * z2;
    }
  } else {
    for (int i=0; i<cnt; i++) {
      x1 = crdx1[i];
      y1 = crdy1[i];
      z1 = crdz1[i];

      G1 += x1*x1 + y1*y1 + z1*z1;

      x2 = crdx2[i];
      y2 = crdy2[i];
      z2 = crdz2[i];

      G2 += x2*x2 + y2*y2 + z2*z2;

      a0 += x1 * x2;
      a1 += x1 * y2;
      a2 += x1 * z2;

      a3 += y1 * x2;
      a4 += y1 * y2;
      a5 += y1 * z2;

      a6 += z1 * x2;
      a7 += z1 * y2;
      a8 += z1 * z2;
    }
  }

  A[0] = a0;
  A[1] = a1;
  A[2] = a2;

  A[3] = a3;
  A[4] = a4;
  A[5] = a5;

  A[6] = a6;
  A[7] = a7;
  A[8] = a8;

  return (G1 + G2) * 0.5;
}

//
// OpenACC version of inner product for SOA coordinate storage
//
// use pgc++ -m64 -Minfo=accel -ta=nvidia -O -acc
#if defined(__PGIC__) && defined(_OPENACC)

#if 0
static void vecadd_acc(void) {
  printf("****** OpenACC test vecadd_acc()...\n");

  // Size of vectors
  int n = 10000;
 
  // Input vectors
  double *restrict a;
  double *restrict b;

  // Output vector
  double *restrict c;
 
  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(double);
 
  // Allocate memory for each vector
  a = (double*)malloc(bytes);
  b = (double*)malloc(bytes);
  c = (double*)malloc(bytes);
 
  // Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
  int i;
  for (i=0; i<n; i++) {
    a[i] = sin(i)*sin(i);
    b[i] = cos(i)*cos(i);
  }   
 
  // sum component wise and save result into vector c
#pragma acc kernels copyin(a[0:n],b[0:n]), copyout(c[0:n])
  for (i=0; i<n; i++) {
    c[i] = a[i] + b[i];
  }
 
  // Sum up vector c and print result divided by n, this should equal 1 within error
  double sum = 0.0;
  for(i=0; i<n; i++) {
    sum += c[i];
  }
  sum = sum/n;
  printf("****** final result: %f *******\n", sum);
 
  // Release memory
  free(a);
  free(b);
  free(c);

  printf("****** OpenACC test vecadd_acc() done.\n");
}
#endif

//
// Use 1-D loop rather than 2-D to please PGI OpenACC so it doesn't
// complain about loop-carried dependencies, this appears to be the only
// successful method to achieve a good quality parallelization.
//
#define LOOP1D 1

#if defined(LOOP1D)
#if defined(__PGIC__) && defined(_OPENACC)
#pragma acc routine seq
#endif
static void acc_idx2sub_tril(long N, long ind, long *J, long *I) {
  long i, j;
  i = long(floor((2*N+1 - sqrt((2*N+1)*(2*N+1) - 8*ind)) / 2));
  j = ind - N*i + i*(i-1)/2 + i;
 
  *I = i;
  *J = j;
}
#endif

static void rmsdmat_qcp_acc(int cnt, int padcnt, int framecrdsz, 
                            int framecount, 
#if defined(LOOP1D)
                            const float * restrict crds, 
#else
                            const float * crds, 
#endif
                            // const float *weight,
                            float * rmsdmat) {
  printf("OpenACC rmsdmat_qcp_acc()...\n");
  printf("ACC cnt: %d padcnt: %d\n", cnt, padcnt);

  printf("Copying input arrays to accelerators...\n");
  long totalsz = 3L * framecrdsz * framecount;
  printf("ACC copysz: %ld  (3 * %d * %d)\n", totalsz, framecrdsz, framecount);

  long matcnt = framecount * framecount;
  printf("ACC matcnt: %ld\n", matcnt);

  printf("Running OpenACC kernels...\n");
#if defined(LOOP1D)
  long i, j, k;
#pragma acc kernels copyin(crds[0:totalsz]), copy(rmsdmat[0:matcnt])
  for (k=0; k<(framecount*(framecount-1))/2; k++) {
    acc_idx2sub_tril(long(framecount-1), k, &i, &j);
    long x1addr = j * 3L * framecrdsz;
    {
#else
  long i, j;
#pragma acc kernels copyin(crds[0:totalsz]), copy(rmsdmat[0:matcnt])
  for (j=0; j<framecount; j++) {
    long x1addr = j * 3L * framecrdsz;

    for (i=0; i<j; i++) {
#endif
      // calculate the (weighted) inner product of two structures
      long x2addr = i * 3L * framecrdsz;

      double G1=0.0, G2=0.0;

      double a0, a1, a2, a3, a4, a5, a6, a7, a8;
      a0=a1=a2=a3=a4=a5=a6=a7=a8=0.0;
#if 0
      if (weight != NULL) {
        double x1, x2, y1, y2, z1, z2;
#pragma acc loop
        for (long l=0; l<cnt; l++) {
          double w = weight[l];
          x1 = crds[l + x1addr];
          y1 = crds[l + x1addr + framecrdsz];
          z1 = crds[l + x1addr + framecrdsz*2];

          G1 += w * (x1*x1 + y1*y1 + z1*z1);

          x2 = crds[l + x2addr];
          y2 = crds[l + x2addr + framecrdsz];
          z2 = crds[l + x2addr + framecrdsz*2];

          G2 += w * (x2*x2 + y2*y2 + z2*z2);

          a0 += x1 * x2;
          a1 += x1 * y2;
          a2 += x1 * z2;

          a3 += y1 * x2;
          a4 += y1 * y2;
          a5 += y1 * z2;

          a6 += z1 * x2;
          a7 += z1 * y2;
          a8 += z1 * z2;
        }
      } else {
#endif
        double x1, x2, y1, y2, z1, z2;
#pragma acc loop vector(256)
//#pragma acc loop vector(256) reduction(+:a0),reduction(+:a1),reduction(+:a2),reduction(+:a3),reduction(+:a4),reduction(+:a5),reduction(+:a6),reduction(+:a7),reduction(+:a8),reduction(+:G1),reduction(+:G2)
        for (long l=0; l<cnt; l++) {
          x1 = crds[l + x1addr];
          y1 = crds[l + x1addr + framecrdsz];
          z1 = crds[l + x1addr + framecrdsz*2];

          G1 += x1*x1 + y1*y1 + z1*z1;

          x2 = crds[l + x2addr];
          y2 = crds[l + x2addr + framecrdsz];
          z2 = crds[l + x2addr + framecrdsz*2];

          G2 += x2*x2 + y2*y2 + z2*z2;

          a0 += x1 * x2;
          a1 += x1 * y2;
          a2 += x1 * z2;

          a3 += y1 * x2;
          a4 += y1 * y2;
          a5 += y1 * z2;

          a6 += z1 * x2;
          a7 += z1 * y2;
          a8 += z1 * z2;
        }
#if 0
      }
#endif

      double A[9];
      A[0] = a0;
      A[1] = a1;
      A[2] = a2;

      A[3] = a3;
      A[4] = a4;
      A[5] = a5;

      A[6] = a6;
      A[7] = a7;
      A[8] = a8;

      double E0 = (G1 + G2) * 0.5;

      // calculate the RMSD & rotational matrix
      float rmsd;
      FastCalcRMSDAndRotation(NULL, A, &rmsd, E0, cnt, -1);
#if defined(LOOP1D)
      rmsdmat[k]=rmsd; // store linearized triangle
#else
      rmsdmat[j*framecount + i]=rmsd;
#endif
    }
  }

  printf("ACC done.\n");
}

#endif


#if 0
static double InnerProductAOS(double *A, double *coords1, double *coords2,
                              const int cnt, const double *weight) {
  double G1=0.0, G2=0.0;
  memset(A, 0, sizeof(double) * 9);

  long i;
  double x1, x2, y1, y2, z1, z2;
  if (weight != NULL) {
    for (i=0; i<cnt; i++) {
      double w = weight[i];
      long idx = i*3;
      x1 = coords1[idx  ];
      y1 = coords1[idx+1];
      z1 = coords1[idx+2];

      G1 += w * (x1*x1 + y1*y1 + z1*z1);

      x2 = coords2[idx  ];
      y2 = coords2[idx+1];
      z2 = coords2[idx+2];

      G2 += w * (x2*x2 + y2*y2 + z2*z2);

      A[0] +=  (x1 * x2);
      A[1] +=  (x1 * y2);
      A[2] +=  (x1 * z2);

      A[3] +=  (y1 * x2);
      A[4] +=  (y1 * y2);
      A[5] +=  (y1 * z2);

      A[6] +=  (z1 * x2);
      A[7] +=  (z1 * y2);
      A[8] +=  (z1 * z2);
    }
  } else {
    for (i=0; i<cnt; i++) {
      long idx = i*3;
      x1 = coords1[idx  ];
      y1 = coords1[idx+1];
      z1 = coords1[idx+2];

      G1 += x1*x1 + y1*y1 + z1*z1;

      x2 = coords2[idx  ];
      y2 = coords2[idx+1];
      z2 = coords2[idx+2];

      G2 += x2*x2 + y2*y2 + z2*z2;

      A[0] +=  (x1 * x2);
      A[1] +=  (x1 * y2);
      A[2] +=  (x1 * z2);

      A[3] +=  (y1 * x2);
      A[4] +=  (y1 * y2);
      A[5] +=  (y1 * z2);

      A[6] +=  (z1 * x2);
      A[7] +=  (z1 * y2);
      A[8] +=  (z1 * z2);
    }
  }

  return (G1 + G2) * 0.5;
}
#endif


void com_soa(int cnt, 
             float *&soax, float *&soay, float *&soaz,
             double &comx, double &comy, double &comz,
             const float *weight) {
  comx=comy=comz=0.0;

  if (weight != NULL) {
    double wsum = 0.0;

    // compute weighted center of mass
    int i;
    for (i=0; i<cnt; i++) {
      double w = weight[i];
      wsum += w;

      comx += soax[i] * w;
      comy += soay[i] * w;
      comz += soaz[i] * w;
    }
    double wsumnorm = 1.0 / wsum;
    comx *= wsumnorm; 
    comy *= wsumnorm;
    comz *= wsumnorm;
  } else {
    // compute unweighted center of mass
    int i;
    for (i=0; i<cnt; i++) {
      comx += soax[i];
      comy += soay[i];
      comz += soaz[i];
    }
    double avenorm = 1.0 / ((double) cnt);
    comx *= avenorm; 
    comy *= avenorm;
    comz *= avenorm;
  }
}



int center_convert_soa(const AtomSel *sel, int num, const float *framepos,
                       const float *weight, 
                       float *&soax, float *&soay, float *&soaz) {
  // allocate temporary working arrays, plus required SIMD padding
  int cnt  = sel->selected;
  soax = (float *) calloc(1, (cnt + 16)*sizeof(float));
  soay = (float *) calloc(1, (cnt + 16)*sizeof(float));
  soaz = (float *) calloc(1, (cnt + 16)*sizeof(float));

  int selind = sel->firstsel; // start from the first selected atom
  double comx=0.0, comy=0.0, comz=0.0;

  int i;
  for (i=0; i<cnt; i++) {
    // find next 'on' atom in selection
    // loop is safe since we already stop the on cnt > 0 above
    while (!sel->on[selind])
      selind++;

    // compact selection and convert AOS to SOA storage on-the-fly
    long addr = 3*selind;
    float tx = framepos[addr    ];
    float ty = framepos[addr + 1];
    float tz = framepos[addr + 2];

    comx += tx;
    comy += ty;
    comz += tz;

    soax[i] = tx;
    soay[i] = ty; 
    soaz[i] = tz;

    selind++; // advance to next atom
  }

  double avenorm = 1.0 / ((double) cnt);
  comx *= avenorm; // compute unweighted center of mass
  comy *= avenorm;
  comz *= avenorm;

#if 0
  printf("center_convert_soa(): structure com: %g %g %g\n", comx, comy, comz);
#endif

  // translate center of mass to the origin
  for (i=0; i<cnt; i++) {
    soax[i] -= comx;
    soay[i] -= comy;
    soaz[i] -= comz;
  }

#if 0
  // check post-translation com 
  com_soa(cnt, soax, soay, soaz, comx, comy, comz, weight); 
  printf("center_convert_soa():  centered com: %lg %lg %lg\n", comx, comy, comz);
#endif   

  return 0;
}


int center_convert_single_soa(const AtomSel *sel, int num, 
                              const float *framepos,
                              const float *weight, 
                              float *soax, float *soay, float *soaz) {
  // allocate temporary working arrays, plus required SIMD padding
  int cnt = sel->selected;
  int selind = sel->firstsel; // start from the first selected atom
  double comx=0.0, comy=0.0, comz=0.0;

  int i;
  for (i=0; i<cnt; i++) {
    // find next 'on' atom in selection
    // loop is safe since we already stop the on cnt > 0 above
    while (!sel->on[selind])
      selind++;

    // compact selection and convert AOS to SOA storage on-the-fly
    long addr = 3*selind;
    float tx = framepos[addr    ];
    float ty = framepos[addr + 1];
    float tz = framepos[addr + 2];

    comx += tx;
    comy += ty;
    comz += tz;

    soax[i] = tx;
    soay[i] = ty; 
    soaz[i] = tz;

    selind++; // advance to next atom
  }

  double avenorm = 1.0 / ((double) cnt);
  comx *= avenorm; // compute unweighted center of mass
  comy *= avenorm;
  comz *= avenorm;

  // translate center of mass to the origin
  for (i=0; i<cnt; i++) {
    soax[i] -= comx;
    soay[i] -= comy;
    soaz[i] -= comz;
  }

  return 0;
}


int measure_rmsd_qcp(VMDApp *app,
                     const AtomSel *sel1, const AtomSel *sel2,
                     int num, const float *framepos1, const float *framepos2,
                     float *weight, float *rmsd) {
  if (!sel1 || !sel2)   return MEASURE_ERR_NOSEL;
  if (sel1->selected < 1 || sel2->selected < 1) return MEASURE_ERR_NOSEL;
  if (!weight || !rmsd) return MEASURE_ERR_NOWEIGHT;

  // the number of selected atoms must be the same
  if (sel1->selected != sel2->selected) return MEASURE_ERR_MISMATCHEDCNT;

#if 0
  // need to know how to traverse the list of weights
  // there could be 1 weight per atom (sel_flg == 1) or
  // 1 weight per selected atom (sel_flg == 0)
  int sel_flg;
  if (num == sel1->num_atoms) {
    sel_flg = 1; // using all elements
  } else {
    sel_flg = 0; // using elements from selection
  }
#endif

  //
  // compute CoM for each selection while copying them into target bufs 
  //
  float *sel1x, *sel1y, *sel1z, *sel2x, *sel2y, *sel2z;
  center_convert_soa(sel1, num, framepos1, weight, sel1x, sel1y, sel1z);
  center_convert_soa(sel2, num, framepos2, weight, sel2x, sel2y, sel2z);

  // calculate the (weighted) inner product of two structures
  double E0 = 0;
  double A[9];
  E0 = InnerProductSOA(A, 
                       sel1x, sel1y, sel1z,
                       sel2x, sel2y, sel2z,
                       sel1->selected, NULL /* weight */);

#if 0
  printf("QCP inner product results:\n");
  printf("  E0: %g\n", E0);
  int i;
  for (i=0; i<9; i+=3) 
    printf("A[%d-%d]: %g %g %g\n", i, i+2, A[i], A[i+1], A[i+2]);
  printf("\n");
#endif

  // calculate the RMSD & rotational matrix
  FastCalcRMSDAndRotation(NULL, A, rmsd, E0, sel1->selected, -1);

  free(sel1x);
  free(sel1y);
  free(sel1z);

  free(sel2x);
  free(sel2y);
  free(sel2z);

  return MEASURE_NOERR; // and say rmsd is OK
}


#if 0
// compute linear array index from lower-triangular indices i,j 
static int sub2idx_tril(long N, long i, long j, long *ind) {
//  *ind = i + (j-1)*N - j*(j-1)/2;
  *ind = j + N*i - i*(i-1)/2;
  return 0;
}
#endif

// compute lower-triangular indices i,j from linear array index
static int idx2sub_tril(long N, long ind, long *J, long *I) {
  long i, j;

  if (ind > (N*(N+1)/2)) {
    return -1; // out of bounds
  }

  // XXX deal with ambiguous types for sqrt() on Solaris
  double tmp2np1 = 2*N+1;
  i = long(floor((tmp2np1 - sqrt(tmp2np1*tmp2np1 - 8.0*ind)) / 2));
  // i = long(floor((2*N+1 - sqrt((2*N+1)*(2*N+1) - 8*ind)) / 2));
  j = ind - N*i + i*(i-1)/2 + i;
  
  *I = i;
  *J = j+1;

  return 0;
}


typedef struct {
  const AtomSel *sel;
  int first;
  int last;
  int step;
  float *rmsdmat;
  int padcnt;
  int framecrdsz;
  float *crds;
#if (VMDQCPUSEAVX512 && defined(__AVX512F__))
  int useavx512;
#endif
#if (VMDQCPUSESSE && defined(__SSE2__)) || (VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__))
  int useavx2;
#endif
#if (VMDQCPUSEVSX && defined(__VEC__))
  int usevsx;
#endif
} qcprmsdthreadparms;


static void * measure_rmsdmat_qcp_thread(void *voidparms) {
  int threadid;
  qcprmsdthreadparms *parms = NULL;
#if defined(VMDQCPUSETHRPOOL)
  wkf_threadpool_worker_getdata(voidparms, (void **) &parms);
  wkf_threadpool_worker_getid(voidparms, &threadid, NULL);
#else
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);
  wkf_threadlaunch_getid(voidparms, &threadid, NULL);
#endif

  //
  // copy in per-thread parameters
  //
  const AtomSel *sel = parms->sel;
  float *rmsdmat = parms->rmsdmat;

  // XXX array padding not universally honored yet...
  // int padcnt = parms->padcnt;

  int framecrdsz = parms->framecrdsz;
  float *crds = parms->crds;
  int first  = parms->first;
  int last   = parms->last;
  int step   = parms->step;
#if VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
  int useavx2 = parms->useavx2;
#endif
#if (VMDQCPUSEAVX512 && defined(__AVX512F__))
  int useavx512 = parms->useavx512;
#endif
#if (VMDQCPUSEVSX && defined(__VEC__))
  int usevsx = parms->usevsx;
#endif

#if 0
printf("qcpthread[%d] running... %s\n", threadid, 
#if (VMDQCPUSEAVX512 && defined(__AVX512F__))
         (useavx512) ? "(AVX512)" : "(C++)");
#elif (VMDQCPUSESSE && defined(__SSE2__)) || (VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__))
         (useavx2) ? "(AVX2)" : "(C++)");
#elif (VMDQCPUSEVSX && defined(__VEC__))
         (useavsx) ? "(VSX)" : "(C++)");
#else  
       "(C++)");
#endif
#endif

  int framecount = (last - first + 1) / step;

  wkf_tasktile_t tile;
#if defined(VMDQCPUSETHRPOOL)
  while (wkf_threadpool_next_tile(voidparms, 1, &tile) != WKF_SCHED_DONE) {
#else
  while (wkf_threadlaunch_next_tile(voidparms, 8, &tile) != WKF_SCHED_DONE) {
#endif
    long idx;

    for (idx=tile.start; idx<tile.end; idx++) {
      long i, j;

      // compute i,j from idx...
      // only compute off-diagonal elements, so we use (framecount-1)
      if (idx2sub_tril(framecount-1, idx, &i, &j)) {
        printf("qcpthread[%d]: work idx %ld out of triangle!\n", threadid, idx);
        break;
      }

      // calculate the (weighted) inner product of two structures
      double A[9];
      double E0 = 0;

      float *xj = crds + (j * 3 * framecrdsz);
      float *yj = xj + framecrdsz;
      float *zj = xj + framecrdsz*2;

      float *xi = crds + (i * 3 * framecrdsz);
      float *yi = xi + framecrdsz;
      float *zi = xi + framecrdsz*2;

#if VMDQCPUSEAVX512 && defined(__AVX512F__)
      if (useavx512) {
        E0 = InnerProductSOA_avx512(A, xj, yj, zj, xi, yi, zi,
                                    sel->selected, NULL /* weight */);
      } else 
#endif
#if VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
      if (useavx2) {
        E0 = InnerProductSOA_avx2(A, xj, yj, zj, xi, yi, zi,
                                  sel->selected, NULL /* weight */);
      } else 
#endif
        E0 = InnerProductSOA(A, xj, yj, zj, xi, yi, zi, 
                             sel->selected, NULL /* weight */);

      // calculate the RMSD & rotational matrix
      FastCalcRMSDAndRotation(NULL, A, &rmsdmat[j*framecount + i], 
                              E0, sel->selected, -1);

      // reflect the outcome of the lower triangle into the upper triangle
      rmsdmat[i*framecount + j] = rmsdmat[j*framecount + i];
    } 
  }

  return NULL;
}


int measure_rmsdmat_qcp(VMDApp *app,
                        const AtomSel *sel, MoleculeList *mlist,
                        int num, float *weight, 
                        int first, int last, int step,
                        float *rmsdmat) {
  if (!sel) return MEASURE_ERR_NOSEL;
  if (sel->selected < 1) return MEASURE_ERR_NOSEL;
//  if (!weight || !rmsd) return MEASURE_ERR_NOWEIGHT;

  Molecule *mymol = mlist->mol_from_id(sel->molid());
  int maxframes = mymol->numframes();

  // accept value of -1 meaning "all" frames
  if (last == -1)
    last = maxframes-1;

  if (maxframes == 0 || first < 0 || first > last ||
      last >= maxframes || step <= 0)
    return MEASURE_ERR_BADFRAMERANGE;

  // XXX replace with calls to centralized control system
#if (VMDQCPUSESSE && defined(__SSE2__)) || (VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__))
// XXX there's no SSE-specific code path 
//  int usesse=1;
//  if (getenv("VMDNOSSE")) {
//    usesse=0;
//  }
#endif
#if (VMDQCPUSESSE && defined(__SSE2__)) || (VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__))
  int useavx2=1;
  if (getenv("VMDNOAVX2")) {
    useavx2=0;
  }
#endif
#if (VMDQCPUSEAVX512 && defined(__AVX512F__))
  int useavx512=1;
  if (getenv("VMDNOAVX512")) {
    useavx512=0;
  }
#endif
#if (VMDQCPUSEVSX && defined(__VEC__))
  int usevsx=1;
  if (getenv("VMDNOVSX")) {
    usevsx=0;
  }
#endif

#if 0
  // need to know how to traverse the list of weights
  // there could be 1 weight per atom (sel_flg == 1) or
  // 1 weight per selected atom (sel_flg == 0)
  int sel_flg;
  if (num == sel->num_atoms) {
    sel_flg = 1; // using all elements
  } else {
    sel_flg = 0; // using elements from selection
  }
#endif

  // start timers
  wkf_timerhandle timer;
  timer=wkf_timer_create();
  wkf_timer_start(timer);


  //
  // compute CoM for frame/selection while copying them into SOA target bufs 
  //
  int framecount = (last - first + 1) / step;

  int padcnt = (num + 255) & ~255;
  int framecrdsz = padcnt + 256;
  float *crds = (float *) calloc(1, (framecount * 3L * framecrdsz + 256) * sizeof(float));
    
  int frame;
  for (frame=first; frame<=last; frame+=step) {
    const float *framepos = (mymol->get_frame(frame))->pos;
    float *xc = crds + (frame * 3L * framecrdsz);
    float *yc = xc + framecrdsz;
    float *zc = xc + framecrdsz*2;

    center_convert_single_soa(sel, num, framepos, weight, xc, yc, zc);
  }

  double converttime = wkf_timer_timenow(timer);

#if !(defined(__PGIC__) && defined(_OPENACC))
#if defined(VMDTHREADS)
  int numprocs = wkf_thread_numprocessors();
#else
  int numprocs = 1;
#endif

  //
  // copy in per-thread parameters
  //
  qcprmsdthreadparms parms;
  memset(&parms, 0, sizeof(parms));
  parms.sel = sel;
  parms.rmsdmat = rmsdmat;
  parms.padcnt = padcnt;
  parms.framecrdsz = framecrdsz;
  parms.crds = crds;
  parms.first = first;
  parms.last = last;
  parms.step = step;
#if (VMDQCPUSESSE && defined(__SSE2__)) || (VMDQCPUSEAVX2 && defined(__AVX__) && defined(__AVX2__))
  parms.useavx2 = useavx2;
#endif
#if (VMDQCPUSEAVX512 && defined(__AVX512F__))
  parms.useavx512 = useavx512;
#endif
#if (VMDQCPUSEVSX && defined(__VEC__))
  parms.usevsx = usevsx;
#endif
  
  // spawn child threads to do the work
  wkf_tasktile_t tile;
  tile.start=0;
  tile.end=(framecount-1)*(framecount-1)/2; // only compute off-diag elements

#if defined(VMDORBUSETHRPOOL)
  wkf_threadpool_sched_dynamic(app->thrpool, &tile);
  rc = wkf_threadpool_launch(app->thrpool, measure_rmsdmat_qcp_thread, &parms, 1);
#else
  wkf_threadlaunch(numprocs, &parms, measure_rmsdmat_qcp_thread, &tile);
#endif
#elif defined(__PGIC__) && defined(_OPENACC)
  // OpenACC variant
  rmsdmat_qcp_acc(sel->selected, padcnt, framecrdsz, framecount, crds, 
//                  NULL /* weight */, 
                  rmsdmat);
#else
  int i, j;
  for (j=0; j<framecount; j++) {
    float *xj = crds + (j * 3 * framecrdsz);
    float *yj = xj + framecrdsz;
    float *zj = xj + framecrdsz*2;
    for (i=0; i<j; i++) {
      // calculate the (weighted) inner product of two structures
      double A[9];

      float *xi = crds + (i * 3 * framecrdsz);
      float *yi = xi + framecrdsz;
      float *zi = xi + framecrdsz*2;

      double E0 = InnerProductSOA(A, xj, yj, zj, xi, yi, zi, 
                                  sel->selected, NULL /* weight */);

      // calculate the RMSD & rotational matrix
      FastCalcRMSDAndRotation(NULL, A, &rmsdmat[j*framecount + i], 
                              E0, sel->selected, -1);

      // reflect the outcome of the lower triangle into the upper triangle
      rmsdmat[i*framecount + j] = rmsdmat[j*framecount + i];
    }
  }
#endif

  // mark all self-RMSDs with a value of 1.0
  for (long l=0; l<framecount; l++) {
    rmsdmat[l*framecount + l] = 1.0;
  }

  double rmsdtime = wkf_timer_timenow(timer) - converttime;

  // free all temporary buffers
  free(crds);

#if 1
  double totaltime = wkf_timer_timenow(timer);
  printf("QCP RMSD Matrix calculation time: SOA selection: %.3f  RMSD solve: %.3f  total: %.3f\n", converttime, rmsdtime, totaltime); 
#endif

  wkf_timer_destroy(timer);

  return MEASURE_NOERR; // and say rmsd is OK
}






//
// Copyright notice for original QCP FastCalcRMSDAndRotation() routine
//
// If you use this QCP rotation calculation method in a publication, please
// reference:
//   Douglas L. Theobald (2005)
//   "Rapid calculation of RMSD using a quaternion-based characteristic
//   polynomial."
//   Acta Crystallographica A 61(4):478-480.
//
//   Pu Liu, Dmitris K. Agrafiotis, and Douglas L. Theobald (2009)
//   "Fast determination of the optimal rotational matrix for macromolecular
//   superpositions."
//   Journal of Computational Chemistry 31(7):1561-1563.
//
//  Copyright (c) 2009-2013 Pu Liu and Douglas L. Theobald
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without modification, are permitted
//  provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the <ORGANIZATION> nor the names of its 
//    contributors may be used to endorse or promote products derived from 
//    this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#if defined(__PGIC__)
#pragma acc routine seq
#endif
int FastCalcRMSDAndRotation(double *rot, double *A, float *rmsd, 
                            double E0, int len, double minScore) {
  double Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz;
  double Szz2, Syy2, Sxx2, Sxy2, Syz2, Sxz2, Syx2, Szy2, Szx2,
         SyzSzymSyySzz2, Sxx2Syy2Szz2Syz2Szy2, Sxy2Sxz2Syx2Szx2,
         SxzpSzx, SyzpSzy, SxypSyx, SyzmSzy,
         SxzmSzx, SxymSyx, SxxpSyy, SxxmSyy;
  double C[4];
  int i;
  double mxEigenV; 
  double oldg = 0.0;
  double b, a, delta, rms, qsqr;
  double q1, q2, q3, q4, normq;
  double a11, a12, a13, a14, a21, a22, a23, a24;
  double a31, a32, a33, a34, a41, a42, a43, a44;
  double a2, x2, y2, z2; 
  double xy, az, zx, ay, yz, ax; 
  double a3344_4334, a3244_4234, a3243_4233, a3143_4133,a3144_4134, a3142_4132; 
  double evecprec = 1e-6;
  double evalprec = 1e-11;

  Sxx = A[0]; Sxy = A[1]; Sxz = A[2];
  Syx = A[3]; Syy = A[4]; Syz = A[5];
  Szx = A[6]; Szy = A[7]; Szz = A[8];

  Sxx2 = Sxx * Sxx;
  Syy2 = Syy * Syy;
  Szz2 = Szz * Szz;

  Sxy2 = Sxy * Sxy;
  Syz2 = Syz * Syz;
  Sxz2 = Sxz * Sxz;

  Syx2 = Syx * Syx;
  Szy2 = Szy * Szy;
  Szx2 = Szx * Szx;

  SyzSzymSyySzz2 = 2.0*(Syz*Szy - Syy*Szz);
  Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2;

  C[2] = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2);
  C[1] = 8.0 * (Sxx*Syz*Szy + Syy*Szx*Sxz + Szz*Sxy*Syx - Sxx*Syy*Szz - Syz*Szx*Sxy - Szy*Syx*Sxz);

  SxzpSzx = Sxz + Szx;
  SyzpSzy = Syz + Szy;
  SxypSyx = Sxy + Syx;
  SyzmSzy = Syz - Szy;
  SxzmSzx = Sxz - Szx;
  SxymSyx = Sxy - Syx;
  SxxpSyy = Sxx + Syy;
  SxxmSyy = Sxx - Syy;
  Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2;

  C[0] = Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2
       + (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2) * (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2)
       + (-(SxzpSzx)*(SyzmSzy)+(SxymSyx)*(SxxmSyy-Szz)) * (-(SxzmSzx)*(SyzpSzy)+(SxymSyx)*(SxxmSyy+Szz))
       + (-(SxzpSzx)*(SyzpSzy)-(SxypSyx)*(SxxpSyy-Szz)) * (-(SxzmSzx)*(SyzmSzy)-(SxypSyx)*(SxxpSyy+Szz))
       + (+(SxypSyx)*(SyzpSzy)+(SxzpSzx)*(SxxmSyy+Szz)) * (-(SxymSyx)*(SyzmSzy)+(SxzpSzx)*(SxxpSyy+Szz))
       + (+(SxypSyx)*(SyzmSzy)+(SxzmSzx)*(SxxmSyy-Szz)) * (-(SxymSyx)*(SyzpSzy)+(SxzmSzx)*(SxxpSyy-Szz));

  /* Newton-Raphson */
  mxEigenV = E0;
  for (i = 0; i < 50; ++i) {
    oldg = mxEigenV;
    x2 = mxEigenV*mxEigenV;
    b = (x2 + C[2])*mxEigenV;
    a = b + C[1];
    delta = ((a*mxEigenV + C[0])/(2.0*x2*mxEigenV + b + a));
    mxEigenV -= delta;
#if 0
    printf("QCP diff[%3d]: %16g %16g %16g\n", i, mxEigenV - oldg, evalprec*mxEigenV, mxEigenV);
#endif
    if (fabs(mxEigenV - oldg) < fabs(evalprec*mxEigenV))
      break;
  }

#if !defined(__PGIC__)
  if (i == 50) 
    printf("MeasureQCP: More than %d iterations needed!\n", i);
#endif

  // the fabs() is to guard against extremely small, 
  // but *negative* numbers due to floating point error 
  rms = sqrt(fabs(2.0 * (E0 - mxEigenV)/len));
  (*rmsd) = rms;
  /* printf("\n\n %16g %16g %16g \n", rms, E0, 2.0 * (E0 - mxEigenV)/len); */

  if (minScore > 0) 
    if (rms < minScore)
      return (-1); // Don't bother with rotation. 

  // only perform rotation related calculations if we have a non-NULL
  // pointer for the output rotation matrix
  if (rot != NULL) {
    a11 = SxxpSyy + Szz-mxEigenV; a12 = SyzmSzy; a13 = - SxzmSzx; a14 = SxymSyx;
    a21 = SyzmSzy; a22 = SxxmSyy - Szz-mxEigenV; a23 = SxypSyx; a24= SxzpSzx;
    a31 = a13; a32 = a23; a33 = Syy-Sxx-Szz - mxEigenV; a34 = SyzpSzy;
    a41 = a14; a42 = a24; a43 = a34; a44 = Szz - SxxpSyy - mxEigenV;
    a3344_4334 = a33 * a44 - a43 * a34; a3244_4234 = a32 * a44-a42*a34;
    a3243_4233 = a32 * a43 - a42 * a33; a3143_4133 = a31 * a43-a41*a33;
    a3144_4134 = a31 * a44 - a41 * a34; a3142_4132 = a31 * a42-a41*a32;
    q1 =  a22*a3344_4334-a23*a3244_4234+a24*a3243_4233;
    q2 = -a21*a3344_4334+a23*a3144_4134-a24*a3143_4133;
    q3 =  a21*a3244_4234-a22*a3144_4134+a24*a3142_4132;
    q4 = -a21*a3243_4233+a22*a3143_4133-a23*a3142_4132;

    qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

    // The following code tries to calculate another column in the 
    // adjoint matrix when the norm of the current column is too small.
    // Usually this block will never be activated.  
    // To be absolutely safe this should be
    // uncommented, but it is most likely unnecessary.
    if (qsqr < evecprec) {
      q1 =  a12*a3344_4334 - a13*a3244_4234 + a14*a3243_4233;
      q2 = -a11*a3344_4334 + a13*a3144_4134 - a14*a3143_4133;
      q3 =  a11*a3244_4234 - a12*a3144_4134 + a14*a3142_4132;
      q4 = -a11*a3243_4233 + a12*a3143_4133 - a13*a3142_4132;
      qsqr = q1*q1 + q2 *q2 + q3*q3+q4*q4;

      if (qsqr < evecprec) {
        double a1324_1423 = a13*a24 - a14*a23, a1224_1422 = a12*a24 - a14*a22;
        double a1223_1322 = a12*a23 - a13*a22, a1124_1421 = a11*a24 - a14*a21;
        double a1123_1321 = a11*a23 - a13*a21, a1122_1221 = a11*a22 - a12*a21;

        q1 =  a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322;
        q2 = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321;
        q3 =  a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221;
        q4 = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221;
        qsqr = q1*q1 + q2 *q2 + q3*q3+q4*q4;

        if (qsqr < evecprec) {
          q1 =  a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322;
          q2 = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321;
          q3 =  a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221;
          q4 = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221;
          qsqr = q1*q1 + q2 *q2 + q3*q3 + q4*q4;
                
          if (qsqr < evecprec) {
            // if qsqr is still too small, return the identity matrix.
            rot[0] = rot[4] = rot[8] = 1.0;
            rot[1] = rot[2] = rot[3] = rot[5] = rot[6] = rot[7] = 0.0;

            return(0);
          }
        }
      }
    }

    normq = sqrt(qsqr);
    q1 /= normq;
    q2 /= normq;
    q3 /= normq;
    q4 /= normq;

    a2 = q1 * q1;
    x2 = q2 * q2;
    y2 = q3 * q3;
    z2 = q4 * q4;

    xy = q2 * q3;
    az = q1 * q4;
    zx = q4 * q2;
    ay = q1 * q3;
    yz = q3 * q4;
    ax = q1 * q2;

    rot[0] = a2 + x2 - y2 - z2;
    rot[1] = 2 * (xy + az);
    rot[2] = 2 * (zx - ay);
    rot[3] = 2 * (xy - az);
    rot[4] = a2 - x2 + y2 - z2;
    rot[5] = 2 * (yz + ax);
    rot[6] = 2 * (zx + ay);
    rot[7] = 2 * (yz - ax);
    rot[8] = a2 - x2 - y2 + z2;
  }

  return 1;
}

