/* common definitions for the specden plugin */
#ifndef _SPECDEN_PLUGIN_H
#define _SPECDEN_PLUGIN_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum specden_norm_type {
  NORM_FOURIER, NORM_CLASSIC, NORM_KUBO, NORM_HARMONIC, NORM_SCHOFIELD
};

#ifdef __cplusplus
extern "C" 
{
#endif

extern int calc_specden(const int ndat, double *input, double *output,
                        const int normtype, const int specr, 
                        const double maxfreq, const double deltat, 
                        const double temp);

extern double *calc_sgsmooth(const int ndat, double *input,
                             const int window, const int order);

extern double *calc_sgsderiv(const int ndat, double *input,
                             const int window, const int order, 
                             const double delta);
#ifdef __cplusplus
}
#endif

#endif
