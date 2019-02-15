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
 *	$RCSfile: utilities.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.111 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * General utility routines and definitions.
 *
 ***************************************************************************/
#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h> // need RAND_MAX

#ifndef FALSE
#define FALSE 0
#define TRUE  1
#endif

#ifndef NULL
#define NULL 0
#endif

#ifndef ABS
#define ABS(A) ((A)>0?(A):-(A))
#endif

// various PI macros not available on all machines
#define VMD_PI      3.14159265358979323846
#define VMD_TWOPI   (2.0 * VMD_PI)
#define VMD_1_PI    0.31830988618379067154

// Convert Angstroms to Bohr
#define VMD_ANGS_TO_BOHR 1.88972612478289694072

// Degree-to-Radians and Radians-to-Degrees Conversion macros
#define DEGTORAD(a)     (a*VMD_PI/180.0)
#define RADTODEG(a)     (a*180.0/VMD_PI)

/// given an argc, argv pair, take all the arguments from the Nth one on
/// and combine them into a single string with spaces separating words.  This
/// allocates space for the string, which must be freed by the user.
extern char *combine_arguments(int, const char **, int);


/// make a copy of a string using c++ new routine for memory alloc
extern char *stringdup(const char *);


/// convert the given string to upper case
extern char *stringtoupper(char *);

/// strip trailing '/' characters from a string.
void stripslashes(char *str);

/// do case-insensitive string comparisons
extern int strupcmp(const char *, const char *);
/// do case-insensitive string comparisons
extern int strupncmp(const char *, const char *, int);


///  break a file name up into path + name, returning both in the specified
///      character pointers.  This creates storage for the new strings
///      by allocating space for them.
extern void breakup_filename(const char *, char **, char **);

/// tokenize a given string; return char ptr if ok, NULL if error
extern char *str_tokenize(const char *, int *, char **);

/// get the time of day from the system clock, and return it (in seconds)
///	(This is supposedly accurate to within about 1 millisecond
extern double time_of_day(void);

/// check for input on stdin
extern int vmd_check_stdin(void);

/// return the username of the currently logged-on user
extern char * vmd_username(void);
/// return the uid of the currently logged-on user
extern int vmd_getuid(void);


// Find the first selected atom, return 0 on success, return -1 if no selection
extern int find_first_selection_aligned(int n, const int *on, int *firstsel);

// Find the last selected atom, return 0 on success, return -1 if no selection
extern int find_last_selection_aligned(int n, const int *on, int *lastsel);

// Find the first selected atom, the last selected atom,
// and the total number of selected atoms.
// return 0 on success, return -1 if no selection
extern int analyze_selection_aligned(int n, const int *on,
                                     int *firstsel, int *lastsel, int *selected);

/// find min/max/mean values for an array of floats
extern void minmaxmean_1fv_aligned(const float *f, long n, 
                                   float *fmin, float *fmax, float *fmean);

/// find min/max values for an array of floats
extern void minmax_1fv_aligned(const float *f, long n, float *min, float *max);

// Compute min/max values for a 16-byte-aligned array of floats
// input value n3 is the number of 3-element vectors to process
extern void minmax_3fv_aligned(const float *f, const long n3, float *fmin, float *fmax);

// Compute min/max values for a 16-byte-aligned array of floats
// input value on is the array of atom selection flags, and firstsel
// and lastsel indicate the first and last 3-element vectors to process
// return 0 on success, -1 on no selected atoms or other problems
extern int minmax_selected_3fv_aligned(const float *f, const int *on, 
                                       const long n3, const long firstsel, 
                                       const long lastsel, 
                                       float *fmin, float *fmax);

/// clamp an integer value to the range min->max
inline int clamp_int(int val, int min, int max) {
  return   (val > min) ? ((val <= max) ? val : max) : 0;
}

/// compute the cross product, assumes that x1 memory is _different_ 
/// than both x2 and x3, and returns the pointer to x1
extern float * cross_prod(float *x1, const float *x2, const float *x3);

/// compute the inner dot product
inline float dot_prod(const float *v1, const float *v2) {
  return v1[0]* v2[0] + v1[1]* v2[1] + v1[2] * v2[2];
}

inline double dot_prod(const double *v1, const double *v2) {
  return v1[0]* v2[0] + v1[1]* v2[1] + v1[2] * v2[2];
}

/// copy the first 3 elements from v2 to v1
inline void vec_copy(float *v1, const float *v2) {
  v1[0] = v2[0];
  v1[1] = v2[1];
  v1[2] = v2[2];
}

/// copy the first 3 elements from v2 to v1
inline void vec_copy(double *v1, const double *v2) {
  v1[0] = v2[0];
  v1[1] = v2[1];
  v1[2] = v2[2];
}

/// normalizes the 3-vector to length one and returns the pointer
/// note that this changes the vector
extern float * vec_normalize(float *);

/// subtract 3rd vector from 2nd and put into 1st
/// in other words, a = b - c
inline void vec_sub(float *a, const float *b, const float *c) {
  a[0]=b[0]-c[0];
  a[1]=b[1]-c[1];
  a[2]=b[2]-c[2];
}

/// subtract 3rd vector from 2nd and put into 1st
/// in other words, a = b - c
inline void vec_sub(double *a, const double *b, const double *c) {
  a[0]=b[0]-c[0];
  a[1]=b[1]-c[1];
  a[2]=b[2]-c[2];
}

/// add 2nd and 3rd elements, put into 1st
inline void vec_add(float *a, const float *b, const float *c) {
  a[0]=b[0]+c[0];
  a[1]=b[1]+c[1];
  a[2]=b[2]+c[2];
}

/// increment 1st vector by 2nd vector
inline void vec_incr(float *a, const float *b) {
  a[0] += b[0];
  a[1] += b[1];
  a[2] += b[2];
}

/// a = b*c
inline void vec_scale(float *a, float b, const float *c) {
  a[0] = b*c[0];
  a[1] = b*c[1];
  a[2] = b*c[2];
}

/// a = b*c
inline void vec_scale(float *a, float b, const double *c) {
  a[0] = b * (float) c[0];
  a[1] = b * (float) c[1];
  a[2] = b * (float) c[2];
}

/// a = -b
inline void vec_negate(float *a, const float *b) {
  a[0] = -b[0];
  a[1] = -b[1];
  a[2] = -b[2];
}

/// a += c*d
inline void vec_scaled_add(float *a, float b, const float *c) {
  a[0] += b*c[0];
  a[1] += b*c[1];
  a[2] += b*c[2];
}

/// a += c*d
inline void vec_scaled_add(double *a, float b, const double *c) {
  a[0] += b*c[0];
  a[1] += b*c[1];
  a[2] += b*c[2];
}

/// a = b + c*d (name taken from STREAM benchmark routine)
inline void vec_triad(float *a, const float *b, float c, const float *d) {
  a[0] = b[0] + c*d[0];
  a[1] = b[1] + c*d[1];
  a[2] = b[2] + c*d[2];
}

/// perform linear interpolation between two vectors a = b + frac*(c-b)
inline void vec_lerp(float *a, const float *b, const float *c, float frac) {
  float diff[3], tmp[3];
  vec_sub(diff, c, b);
  vec_scale(tmp, frac, diff);
  vec_add(a, b, tmp);
}

// multiply the matrix mat with the vector vec (length 3)
inline void vectrans(float *npoint, float *mat, double *vec){
  npoint[0]=vec[0]*mat[0]+vec[1]*mat[4]+vec[2]*mat[8];
  npoint[1]=vec[0]*mat[1]+vec[1]*mat[5]+vec[2]*mat[9];
  npoint[2]=vec[0]*mat[2]+vec[1]*mat[6]+vec[2]*mat[10];
}

inline void vec_zero(float *a) {
  a[0] = 0.0f;
  a[1] = 0.0f;
  a[2] = 0.0f;
}

inline void vec_zero(double *a) {
  a[0] = 0.0f;
  a[1] = 0.0f;
  a[2] = 0.0f;
}

inline void clamp_color(float *rgb) {
  // clamp color values to legal range
  if (rgb[0] < 0.0f)
    rgb[0] = 0.0f;
  if (rgb[0] > 1.0f)
    rgb[0] = 1.0f;

  if (rgb[1] < 0.0f)
    rgb[1] = 0.0f;
  if (rgb[1] > 1.0f)
    rgb[1] = 1.0f;

  if (rgb[2] < 0.0f)
    rgb[2] = 0.0f;
  if (rgb[2] > 1.0f)
    rgb[2] = 1.0f;
}


/// compute the midpoint a between two vectors b & c (a = (b + c)/2)
inline void midpoint(float *a, const float *b, const float *c) {
  a[0] = 0.5f * (b[0]+c[0]);
  a[1] = 0.5f * (b[1]+c[1]);
  a[2] = 0.5f * (b[2]+c[2]);
}


// These define a cubic spline with various bases
// see Foley and Van Dam, et. al., Computer Graphics, p505 or so
//
// this one was too sharply curved for my tastes
//
// float CatmullRom_basis[4][4]={{-1.0/2.0,  3.0/2.0, -3.0/2.0,  1.0/2.0},
//                               { 2.0/2.0, -5.0/2.0,  4.0/2.0, -1.0/2.0},
//                               {-1.0/2.0,  0.0/2.0,  1.0/2.0,  0.0/2.0},
//                               { 0.0/2.0,  2.0/2.0,  0.0/2.0,  0.0/2.0}};
//
// this define makes the next basis identical to CatmullRom
// #define SLOPE 2.0f
//
// This deemphasizes the slope and makes things look nicer (IMHO)
// #define SLOPE 1.25f
//
// float modified_CR_basis[4][4] = {
//    {-1.0f/SLOPE,  -1.0f/SLOPE + 2.0f,  1.0f/SLOPE - 2.0f,  1.0f/SLOPE},
//    { 2.0f/SLOPE,   1.0f/SLOPE - 3.0f,   -2.0f/SLOPE+3.0f, -1.0f/SLOPE},
//    {-1.0f/SLOPE,                0.0f,         1.0f/SLOPE,        0.0f},
//    {       0.0f,                1.0f,               0.0f,        0.0f}
// };
//
// This doesn't describe the system very nicely as the lines don't
// go through the control points (which are the C-alphas)
// float Bspline_basis[4][4]={{-1.0/6.0,  3.0/6.0, -3.0/6.0,  1.0/6.0},
//                            { 3.0/6.0, -6.0/6.0,  3.0/6.0,  0.0/6.0},
//                            {-3.0/6.0,  0.0/6.0,  3.0/6.0,  0.0/6.0},
//                            { 1.0/6.0,  4.0/6.0,  1.0/6.0,  0.0/6.0}};

/// define a cubic spline with a B-Spline basis 
inline void create_Bspline_basis(float array[4][4]) {
  array[0][0] = -1.0f/6.0f;
  array[0][1] =  3.0f/6.0f;
  array[0][2] = -3.0f/6.0f;
  array[0][3] =  1.0f/6.0f;
  array[1][0] =  3.0f/6.0f;
  array[1][1] = -6.0f/6.0f;
  array[1][2] =  3.0f/6.0f;
  array[1][3] =  0.0f/6.0f;
  array[2][0] = -3.0f/6.0f;
  array[2][1] =  0.0f/6.0f;
  array[2][2] =  3.0f/6.0f;
  array[2][3] =  0.0f/6.0f;
  array[3][0] =  1.0f/6.0f;
  array[3][1] =  4.0f/6.0f;
  array[3][2] =  1.0f/6.0f;
  array[3][3] =  0.0f/6.0f;
}

/// define a cubic spline with a Catmull-Rom basis 
inline void create_modified_CR_spline_basis(float array[4][4], float slope) {
  array[0][0] = -1.0f / slope;
  array[0][1] = -1.0f / slope + 2.0f;
  array[0][2] =  1.0f / slope - 2.0f;
  array[0][3] =  1.0f / slope;
  array[1][0] =  2.0f / slope;
  array[1][1] =  1.0f / slope - 3.0f;
  array[1][2] = -2.0f / slope + 3.0f;
  array[1][3] = -1.0f / slope;
  array[2][0] = -1.0f / slope;
  array[2][1] =  0.0f;
  array[2][2] =  1.0f / slope;
  array[2][3] =  0.0f;
  array[3][0] =  0.0f;
  array[3][1] =  1.0f;
  array[3][2] =  0.0f;
  array[3][3] =  0.0f;
}

/// Builds the spline matrix "Q" from the basis matrix "M" and the
/// geometry matrix "G".  The geometry matrix in this case is the pts
/// parameter, which contains the previous, current, and next two points
/// defining the curve.  For Catmull-Rom splines the tangent at the current
/// point is the same as the direction from the previous point to the next 
/// point.
inline void make_spline_Q_matrix(float q[4][3], float basis[4][4], const float *pts) {
  int i, j;
  for (i = 0; i<4; i++) {
    float a, b, c;
    a = b = c = 0.0;

    for (j = 0; j<4; j++) {
      a += basis[i][j] * pts[j*3L    ];
      b += basis[i][j] * pts[j*3L + 1];
      c += basis[i][j] * pts[j*3L + 2];
    }

    q[i][0] = a; 
    q[i][1] = b; 
    q[i][2] = c; 
  }
}

/// Builds the spline matrix "Q" from the basis matrix "M" and the
/// geometry matrix "G".  The geometry matrix in this case is the pts
/// parameter, which contains the previous, current, and next two points
/// defining the curve.  For Catmull-Rom splines the tangent at the current
/// point is the same as the direction from the previous point to the next 
/// point.  This one works with non-contiguous memory layouts.
inline void make_spline_Q_matrix_noncontig(float q[4][3], float basis[4][4], 
                                        const float *pts1, const float *pts2,
                                        const float *pts3, const float *pts4) {
  int i;

  for (i = 0; i<4; i++) {
    float a, b, c;
    a = b = c = 0.0;

    a += basis[i][0] * pts1[0];
    b += basis[i][0] * pts1[1];
    c += basis[i][0] * pts1[2];

    a += basis[i][1] * pts2[0];
    b += basis[i][1] * pts2[1];
    c += basis[i][1] * pts2[2];

    a += basis[i][2] * pts3[0];
    b += basis[i][2] * pts3[1];
    c += basis[i][2] * pts3[2];

    a += basis[i][3] * pts4[0];
    b += basis[i][3] * pts4[1];
    c += basis[i][3] * pts4[2];

    q[i][0] = a; 
    q[i][1] = b; 
    q[i][2] = c; 
  }
}


/// Evaluate the spline to return a point on the curve specified by the
/// w parameter, in the range 0 to 1.
///
/// XXX an improved implementation might use forward differences to  
///     find the points on the curve rather than explicitly evaluating
///     points one at a time.  Forward differences should be much faster,
///     since it can be done with 9 additions and no multiplies, whereas 
///     this code has to do 9 multiplies as well as 9 additions for each point.
///     The forward difference method requires setup, and some extra 
///     storage for it's position/velocity/acceleration accumulators 
///     however, so it may be an even trade-off.
inline void make_spline_interpolation(float out[3], float w, float q[4][3]) {
  out[0] = w * (w * (w * q[0][0] + q[1][0]) + q[2][0]) + q[3][0];
  out[1] = w * (w * (w * q[0][1] + q[1][1]) + q[2][1]) + q[3][1];
  out[2] = w * (w * (w * q[0][2] + q[1][2]) + q[2][2]) + q[3][2];
}

/// determine if a triangle is degenerate or not
extern int tri_degenerate(const float *, const float *, const float *);

/// compute the angle between two vectors a & b (0 to 180 deg)
extern float angle(const float *, const float *);

/// Compute the dihedral angle for the given atoms, returning a value between
/// -180 and 180. 
extern float dihedral(const float *, const float *, const float *, 
                      const float *);

/// compute the distance between two points a & b
extern float distance(const float *, const float *);

/// compute the squared distance between two points a & b
inline float distance2(const float *a, const float *b) {
  float delta = a[0] - b[0];
  float r2 = delta*delta;
  delta = a[1] - b[1];
  r2 += delta*delta;
  delta = a[2] - b[2];
  return r2 + delta*delta;
}

/// find and return the norm of a 3-vector
extern float norm(const float *);

/// VMD temp file (portable)
/// given a string, return a new one with the temp dir name prepended.
/// The returned string must be deleted.
char *vmd_tempfile(const char *);

/// VMD file deletion function (portable)
int vmd_delete_file(const char *);

/// VMD process sleep functions (portable)
void vmd_sleep(int);   // sleeps for N seconds
void vmd_msleep(int);  // sleeps for N milliseconds

/// a buffer function to system() call to be replaced by 
/// a different implementation in console-free Win32 applications
extern int vmd_system(const char* cmd);


/// portable random number generation, NOT thread-safe however
extern void vmd_srandom(unsigned int);
extern long vmd_random();
extern float vmd_random_gaussian();

// select the right VMD_RAND_MAX value depending on implementation and platform
#if defined(__linux) || defined(_MSC_VER)
// Linux uses RAND_MAX for both rand() and random()
// Windows is currently implemented with rand()
#define VMD_RAND_MAX RAND_MAX
#else
// All other platforms I've seen use 2^31-1 for their max random() return val
#if defined(LONG_MAX)
#define VMD_RAND_MAX LONG_MAX
#else
// just in case no LONG_MAX, 
// or in cases where they choose to update it for 64-bits or something
#define VMD_RAND_MAX 2147483647L
#endif
#endif

/// return the number of MB of physical memory installed in the system
long vmd_get_total_physmem_mb(void);

/// return the number of MB of physical memory "free" (no VM/swap counted...)
long vmd_get_avail_physmem_mb(void);

/// return the percentage of physical memory available
long vmd_get_avail_physmem_percent(void);

#endif
