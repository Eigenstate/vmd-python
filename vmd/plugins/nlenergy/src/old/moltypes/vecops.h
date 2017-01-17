/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/vecops.h
 * @brief   Vector operations.
 * @author  David J. Hardy
 * @date    Mar. 2008
 */

#ifndef MOLTYPES_VECOPS_H
#define MOLTYPES_VECOPS_H

#ifdef __cplusplus
extern "C" {
#endif

  /* set u=0 */
#define VECZERO(u) \
  ((u).x = 0, (u).y = 0, (u).z = 0)

  /* set u=(a,b,c) */
#define VECINIT(u, a, b, c) \
  ((u).x = a, (u).y = b, (u).z = c)

  /* set u = v + w */
#define VECADD(u, v, w) \
  ((u).x = (v).x + (w).x, (u).y = (v).y + (w).y, (u).z = (v).z + (w).z)

  /* set u = v - w */
#define VECSUB(u, v, w) \
  ((u).x = (v).x - (w).x, (u).y = (v).y - (w).y, (u).z = (v).z - (w).z)

  /* set u = c*v, c is scalar */
#define VECMUL(u, c, v) \
  ((u).x = (c)*(v).x, (u).y = (c)*(v).y, (u).z = (c)*(v).z)

  /* set u = c*v + w, c is scalar */
#define VECMADD(u, c, v, w) \
  ( (u).x = (c) * (v).x + (w).x, \
    (u).y = (c) * (v).y + (w).y, \
    (u).z = (c) * (v).z + (w).z )

  /* set u = c*v - w, c is scalar */
#define VECMSUB(u, c, v, w) \
  ( (u).x = (c) * (v).x - (w).x, \
    (u).y = (c) * (v).y - (w).y, \
    (u).z = (c) * (v).z - (w).z )
 
  /* compute u as cross product of vectors v and w */
#define VECCROSS(w, u, v) \
   ( (w).x = (u).y*(v).z - (u).z*(v).y, \
     (w).y = (u).z*(v).x - (u).x*(v).z, \
     (w).z = (u).x*(v).y - (u).y*(v).x )

  /* return dot product of vectors u and v */
#define VECDOT(u, v) \
  ((u).x*(v).x + (u).y*(v).y + (u).z*(v).z)

  /* return square of Euclidean norm of vector u */
#define VECLEN2(u)  VECDOT(u,u)

#ifdef __cplusplus
}
#endif

#endif /* MOLTYPES_VECOPS_H */
