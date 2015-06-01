/*
  Copyright (c) 2003-2008, Mark Borgerding

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the author nor the names of any contributors may be used
      to endorse or promote products derived from this software without
      specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* this code is adapted from kiss_fft_v1_2_8. http://kissfft.sf.net/ */

#ifndef KISS_FFT_H
#define KISS_FFT_H

#ifdef __cplusplus
extern "C" {
#endif

/* SSE is not working currently */
#if defined(USE_SIMD)
#undef USE_SIMD    
#endif

/* we use doubles for simplicity, as tcl uses them as well */
#define kiss_fft_scalar double

#define FFT_FORWARD  0
#define FFT_BACKWARD 1

#ifdef USE_SIMD
# define _XOPEN_SOURCE 600
# include <malloc.h>
# include <xmmintrin.h>
# define kiss_fft_scalar __m128
# define KISS_FFT_MALLOC(nbytes) memalign(16,nbytes)
#else	
# define KISS_FFT_MALLOC malloc
#endif	

#ifdef FIXED_POINT
# include <sys/types.h>	
# if (FIXED_POINT == 32)
#  define kiss_fft_scalar int32_t
# else	
#  define kiss_fft_scalar int16_t
# endif
#else
# ifndef kiss_fft_scalar
/*  default is float */
#  define kiss_fft_scalar float
# endif
#endif

typedef struct {
    kiss_fft_scalar r;
    kiss_fft_scalar i;
} kiss_fft_cpx;

#define MAXFACTORS 32
/* e.g. an fft of length 128 has 4 factors 
 as far as kissfft is concerned
 4*4*4*2
 */

struct kiss_fft_state {
    int nfft;
    int inverse;
    int factors[2*MAXFACTORS];
    kiss_fft_cpx twiddles[1];
};
typedef struct kiss_fft_state * kiss_fft_cfg;


struct kiss_fftnd_state {
    int dimprod; /* dimsum would be mighty tasty right now */
    int ndims; 
    int *dims;
    kiss_fft_cfg *states; /* cfg states for each dimension */
    kiss_fft_cpx *tmpbuf; /*buffer capable of hold the entire input */
};
typedef struct kiss_fftnd_state * kiss_fftnd_cfg;

struct kiss_fftr_state {
    kiss_fft_cfg substate;
    kiss_fft_cpx *tmpbuf;
    kiss_fft_cpx *super_twiddles;
#ifdef USE_SIMD    
    long pad;
#endif
};
typedef struct kiss_fftr_state * kiss_fftr_cfg;

/* 
 *  kiss_fft_alloc
 *  
 *  Initialize a FFT (or IFFT) algorithm's cfg/state buffer.
 *
 *  typical usage:      kiss_fft_cfg mycfg=kiss_fft_alloc(1024,0,NULL,NULL);
 *
 *  The return value from fft_alloc is a cfg buffer used internally
 *  by the fft routine or NULL.
 *
 *  If lenmem is NULL, then kiss_fft_alloc will allocate a cfg buffer using malloc.
 *  The returned value should be free()d when done to avoid memory leaks.
 *  
 *  The state can be placed in a user supplied buffer 'mem':
 *  If lenmem is not NULL and mem is not NULL and *lenmem is large enough,
 *      then the function places the cfg in mem and the size used in *lenmem
 *      and returns mem.
 *  
 *  If lenmem is not NULL and ( mem is NULL or *lenmem is not large enough),
 *      then the function returns NULL and places the minimum cfg 
 *      buffer size in *lenmem.
 */

kiss_fft_cfg kiss_fft_alloc(int nfft, int inverse_fft, void *mem, size_t *lenmem); 

/*
 * kiss_fft(cfg,in_out_buf)
 *
 * Perform an FFT on a complex input buffer.
 * for a forward FFT,
 * fin should be  f[0] , f[1] , ... ,f[nfft-1]
 * fout will be   F[0] , F[1] , ... ,F[nfft-1]
 * Note that each element is complex and can be accessed like
    f[k].r and f[k].i
 * */
void kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout);

/*
 A more generic version of the above function. It reads its input from every Nth sample.
 * */
void kiss_fft_stride(kiss_fft_cfg cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout, int fin_stride);

/* If kiss_fft_alloc allocated a buffer, it is one contiguous 
   buffer and can be simply free()d when no longer needed*/
#define kiss_fft_free free

/*
 Cleans up some memory that gets managed internally. Not necessary to call, but it might clean up 
 your compiler output to call this before you exit.
*/
void kiss_fft_cleanup(void);

/*
 * Returns the smallest integer k, such that k>=n and k has only "fast" factors (2,3,5)
 */
int kiss_fft_next_fast_size(int n);

/* for real ffts, we need an even size */
#define kiss_fftr_next_fast_size_real(n)  (kiss_fft_next_fast_size( ((n)+1)>>1)<<1)

/* 
 *  kiss_fftnd_alloc
 *  
 *  Initialize a FFT (or IFFT) algorithm's cfg/state buffer.
 *  similar to kiss_fft_alloc, but for multi-dimensional fft.
 */
kiss_fftnd_cfg kiss_fftnd_alloc(const int *dims, int ndims, int inverse_fft, 
                                void *mem, size_t *lenmem);

/*
 * kiss_fftnd(cfg,in_out_buf)
 *
 * Perform an n-dimensional FFT on a complex input buffer.
 */
void kiss_fftnd(kiss_fftnd_cfg cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout);

/*
 * note: nfft must be even
 */
kiss_fftr_cfg kiss_fftr_alloc(int nfft, int inverse_fft, void * mem, size_t * lenmem);

/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points
*/
void kiss_fftr(kiss_fftr_cfg cfg, const kiss_fft_scalar *timedata, kiss_fft_cpx *freqdata);

/*
 input freqdata has  nfft/2+1 complex points
 output timedata has nfft scalar points
*/
void kiss_fftri(kiss_fftr_cfg cfg, const kiss_fft_cpx *freqdata, kiss_fft_scalar *timedata);

#ifdef __cplusplus
} 
#endif

#endif
