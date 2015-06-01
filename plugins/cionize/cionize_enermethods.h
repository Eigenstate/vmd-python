#ifndef CIONIZE_ENERMETHODS_H
#define CIONIZE_ENERMETHODS_H

#ifdef __cplusplus
extern "C" {
#endif

  enum {
    STANDARD   = 0x00000,   /* standard (direct summation) */
    DOUBLEPREC = 0x00001,   /* use double precision */
    DDD        = 0x00002,   /* distance dependent dielectric */

    GPU        = 0x00008,   /* gpu acceleration flag? */

    MULTIGRID  = 0x10000,   /* use multilevel summation */

    MLATCUT01  = 0x00010,   /* msm with lattice cutoff kernel 01 */
    MLATCUT02  = 0x00020,   /* msm with lattice cutoff kernel 02 */
    MLATCUT03  = 0x00030,   /* msm with lattice cutoff kernel 03 */
    MLATCUT04  = 0x00040,   /* msm with lattice cutoff kernel 04 */
    MLATCUTMASK= 0x000F0,   /* mask */

    MBINLARGE  = 0x00100,   /* msm with short-range large bin kernel */
    MBINSMALL  = 0x00200,   /* msm with short-range small bin kernel */
    MBINMASK   = 0x00F00,   /* mask */

    MDEV0      = 0x00000,   /* msm long-range using device 0 (default) */
    MDEV1      = 0x01000,   /* msm long-range using device 1 */
    MDEV2      = 0x02000,   /* msm long-range using device 2 */
    MDEV3      = 0x03000,   /* msm long-range using device 3 */
    MDEVMASK   = 0x0F000,   /* mask */
    MDEVSHIFT  = 12,        /* right-shift by 12 to get device number */

    MGMASK     = 0x1FFF0,   /* full method mask */
  };

#ifdef __cplusplus
}
#endif

#endif
