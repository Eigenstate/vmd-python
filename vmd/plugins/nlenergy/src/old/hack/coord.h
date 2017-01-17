/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/coord.h
 * @brief   Contains arrays of position, velocity, and force coordinates.
 * @author  David J. Hardy
 * @date    Apr. 2008
 */

#ifndef MOLTYPES_COORD_H
#define MOLTYPES_COORD_H

#include "nlbase/nlbase.h"
#include "moltypes/domain.h"
#include "moltypes/topology.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief Coordinate data structure.
   *
   * Contains position, velocity, and force arrays.
   * The external position R is stored internally as (pos,image):
   *   R[i] = pos[i] + A * image[i],
   * where A is matrix of basis vectors defined by periodic domain.
   */
  typedef struct Coord_t {
    Array pos;     /**< atom positions */
    Array image;   /**< image offset into periodic domain for each atom
                    *   array of ivec (Z^3) counts domain basis vectors */
    Array vel;     /**< atom velocities */
    Array force;   /**< atom forces */

    Array trans;   /**< calculate translation of clusters for wrapping */
    Array output;  /**< output positions produced by _wrap_output() */

    Domain domain; /**< defines the (semi-)periodic domain */
    const Topology *topo;
  } Coord;

  int Coord_init(Coord *);
  void Coord_done(Coord *);

  /* to be called once after _init() */
  int Coord_setup(Coord *,
      const dvec *domain_center,
      const dvec *domain_basis_v1,
      const dvec *domain_basis_v2,
      const dvec *domain_basis_v3,
      const Topology *);

  /* establish a new basis for the domain */
  int Coord_setup_basis(Coord *,
      const dvec *domain_center,
      const dvec *domain_basis_v1,
      const dvec *domain_basis_v2,
      const dvec *domain_basis_v3);

  /* apply a transformation to the domain, updates domain basis and pos */
  int Coord_rescale_domain(Coord *, const Transform *);

  /* access functions */
  int32 Coord_numatoms(const Coord *);
  const Domain *Coord_domain(const Coord *);

  const dvec *Coord_force_const(const Coord *);
  const dvec *Coord_vel_const(const Coord *);
  const dvec *Coord_pos_const(const Coord *);

  const ivec *Coord_image_const(const Coord *);
  const dvec *Coord_output_const(const Coord *);

  dvec *Coord_force(Coord *);  /* write access for force computation */
  dvec *Coord_vel(Coord *);    /* write access for integration routines */
  dvec *Coord_pos(Coord *);    /* write access for integration routines */

  enum {
    UPDATE_ALL    = 0,    /**< each pos[i] is wrapped into domain */
    UPDATE_PARENT = 1     /**< each parent pos[i] is wrapped into domain,
                           *   lighter atoms follow their parent atom */
  };

  /* normalize internal position (pos,image) so that pos[i] is either
   * inside periodic domain or nearby (if following its parent) */
  int Coord_update_pos(Coord *, int32 updating);

  /* set pos, vel, and force arrays from external arrays */
  int Coord_set_pos(Coord *, const dvec *, int32 n, int32 updating);
  int Coord_set_vel(Coord *, const dvec *, int32 n);
  int Coord_set_force(Coord *, const dvec *, int32 n);

  /* output positions by wrapping atom clusters into periodic domain */
  int Coord_wrap_output(Coord *, int32 wrapping);

  enum {
    WRAP_NONE    = 0x00,  /**< don't wrap to periodic boundaries */
    WRAP_WATER   = 0x01,  /**< wrap only water, leave others unwrapped */
    WRAP_ALL     = 0x03,  /**< ALL includes WATER */
    WRAP_NEAREST = 0x04   /**< wrap to image closest to domain center,
                           *   this allows parallelepiped to provide
                           *   other periodic geometries depending on
                           *   choice of basis vectors */
  };

  /* set velocities to desired temperature distribution */
  int Coord_set_temperature(Coord *, Random *, dreal temperature);

  /* remove center-of-mass motion of atomic system, adjusting velocities */
  int Coord_remove_com_motion(Coord *);


#ifdef __cplusplus
}
#endif

#endif /* MOLTYPES_COORD_H */
