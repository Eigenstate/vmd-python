/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "moltypes/coord.h"


int Coord_init(Coord *p) {
  int s;  /* error status */
  if ((s=Domain_init(&(p->domain))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->pos), sizeof(dvec))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->image), sizeof(ivec))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->trans), sizeof(dvec))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->vel), sizeof(dvec))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->force), sizeof(dvec))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->output), sizeof(dvec))) != OK) return ERROR(s);
  p->topo = NULL;
  return OK;
}


int Coord_setup(Coord *p, const dvec *center,
    const dvec *basis_v1, const dvec *basis_v2, const dvec *basis_v3,
    const Topology *topo) {
  int32 natoms;
  int s;  /* error status */
  if (NULL==topo) return ERROR(ERR_EXPECT);
  p->topo = topo;
  natoms = Topology_atom_array_length(topo);
  if (natoms <= 0) return ERROR(ERR_VALUE);
  if ((s=Coord_setup_basis(p, center, basis_v1, basis_v2, basis_v3)) != OK) {
    return ERROR(s);
  }
  if ((s=Array_resize(&(p->pos), natoms)) != OK) return ERROR(s);
  if ((s=Array_resize(&(p->image), natoms)) != OK) return ERROR(s);
  if ((s=Array_resize(&(p->trans), natoms)) != OK) return ERROR(s);
  if ((s=Array_resize(&(p->vel), natoms)) != OK) return ERROR(s);
  if ((s=Array_resize(&(p->force), natoms)) != OK) return ERROR(s);
  if ((s=Array_resize(&(p->output), natoms)) != OK) return ERROR(s);
  return OK;
}


/*
 * Change the domain center and basis vectors.
 * Internal coordinates are completely unwrapped wrt old basis,
 * so you can follow up with call to Coord_update_pos().
 */
int Coord_setup_basis(Coord *p, const dvec *center,
    const dvec *bv0, const dvec *bv1, const dvec *bv2) {
  Domain *domain = &(p->domain);
  const int32 natoms = Array_length(&(p->pos));
  int s;

  if (natoms > 0) {  /* unwrap existing coordinates */
    const dvec a0 = domain->basis[0];
    const dvec a1 = domain->basis[1];
    const dvec a2 = domain->basis[2];
    dvec *pos = Array_data(&(p->pos));
    ivec *image = Array_data(&(p->image));
    ivec m;
    int32 i;
    for (i = 0;  i < natoms;  i++) {
      m = image[i];
      pos[i].x += (m.x * a0.x + m.y * a1.x + m.z * a2.x);
      pos[i].y += (m.x * a0.y + m.y * a1.y + m.z * a2.y);
      pos[i].z += (m.x * a0.z + m.y * a1.z + m.z * a2.z);
      VECZERO(image[i]);
    }
  }
  if ((s=Domain_setup(domain, center, bv0, bv1, bv2)) != OK) {
    return ERROR(s);
  }
  return OK;
}


void Coord_done(Coord *p) {
  Domain_done(&(p->domain));
  Array_done(&(p->pos));
  Array_done(&(p->image));
  Array_done(&(p->trans));
  Array_done(&(p->vel));
  Array_done(&(p->force));
  Array_done(&(p->output));
}


int32 Coord_numatoms(const Coord *p) {
  return Topology_atom_array_length(p->topo);
}

const Domain *Coord_domain(const Coord *p) {
  return &(p->domain);
}

const dvec *Coord_force_const(const Coord *p) {
  return (const dvec *) Array_data_const(&(p->force));
}

const dvec *Coord_vel_const(const Coord *p) {
  return (const dvec *) Array_data_const(&(p->vel));
}

const dvec *Coord_pos_const(const Coord *p) {
  return (const dvec *) Array_data_const(&(p->pos));
}

const ivec *Coord_image_const(const Coord *p) {
  return (const ivec *) Array_data_const(&(p->image));
}

const dvec *Coord_output_const(const Coord *p) {
  return (const dvec *) Array_data_const(&(p->output));
}

dvec *Coord_force(Coord *p) {
  return (dvec *) Array_data(&(p->force));
}

dvec *Coord_vel(Coord *p) {
  return (dvec *) Array_data(&(p->vel));
}

dvec *Coord_pos(Coord *p) {
  return (dvec *) Array_data(&(p->pos));
}

int Coord_set_force(Coord *p, const dvec *f, int32 n) {
  if (n != Array_length(&(p->force))) return ERROR(ERR_VALUE);
  memcpy(Array_data(&(p->force)), f, n*sizeof(dvec));
  return OK;
}

int Coord_set_vel(Coord *p, const dvec *v, int32 n) {
  if (n != Array_length(&(p->vel))) return ERROR(ERR_VALUE);
  memcpy(Array_data(&(p->vel)), v, n*sizeof(dvec));
  return OK;
}

int Coord_set_pos(Coord *p, const dvec *r, int32 n, int32 updating) {
  int s;
  if (n != Array_length(&(p->pos))) return ERROR(ERR_VALUE);
  memcpy(Array_data(&(p->pos)), r, n*sizeof(dvec));
  ASSERT(Array_length(&(p->image)) == n);
  if ((s=Array_erase(&(p->image))) != OK) return ERROR(s);
  return Coord_update_pos(p, updating);
}


int Coord_update_pos(Coord *p, int32 updating) {
  const Domain *domain = &(p->domain);
  const Atom *atom = Topology_atom_array(p->topo);
  const int32 natoms = Topology_atom_array_length(p->topo);
  dvec *pos = Array_data(&(p->pos));
  ivec *image = Array_data(&(p->image));
  dvec w;
  ivec n;
  int32 i;

  if ( !(domain->periodic_x || domain->periodic_y || domain->periodic_z)) {
    return OK;  /* aperiodic in all dimensions, nothing to do */
  }
  if (UPDATE_ALL == updating) {
    for (i = 0;  i < natoms;  i++) {
      Domain_wrap_vec(domain, &w, &n, &pos[i]);
      VECADD(pos[i], pos[i], w);
      VECADD(image[i], image[i], n);
    }
  }
  else if (UPDATE_PARENT == updating) {
    int32 pid;
    for (i = 0;  i < natoms;  i++) {
      pid = atom[i].parentID;
      ASSERT(0 <= pid && pid < natoms);
      if (pid >= i) {
        /* performed twice for an atom in cases where pid > i */
        Domain_wrap_vec(domain, &w, &n, &pos[pid]);
        VECADD(pos[pid], pos[pid], w);
        VECADD(image[pid], image[pid], n);
      }
      if (i != pid) {
        Domain_wrap_vec_location(domain, &w, &n, &pos[i], &pos[pid]);
        VECADD(pos[i], pos[i], w);
        VECADD(image[i], image[i], n);
      }
    } /* for */
  }
  else {
    return ERROR(ERR_VALUE);
  } 
  return OK;
}


int Coord_rescale_domain(Coord *p, const Transform *t) {
  const dvec t0 = t->row[0];
  const dvec t1 = t->row[1];
  const dvec t2 = t->row[2];
  dvec r, tr;
  const dvec center = p->domain.center;
  dvec *pos = Array_data(&(p->pos));
  const int32 natoms = Array_length(&(p->pos));
  int32 i;
  int s;

  if ((s=Domain_rescale(&(p->domain), t)) != OK) return ERROR(s);
  for (i = 0;  i < natoms;  i++) {  /* apply transformation to atoms */
    VECSUB(r, pos[i], center);
    tr.x = VECDOT(t0, r);
    tr.y = VECDOT(t1, r);
    tr.z = VECDOT(t2, r);
    VECADD(pos[i], tr, center);
  }
  return OK;
}


int Coord_wrap_output(Coord *p, int32 wrapping) {
  const Domain *domain = &(p->domain);
  const Atom *atom = Topology_atom_array(p->topo);
  const int32 natoms = Topology_atom_array_length(p->topo);
  dvec *pos = Array_data(&(p->pos));
  ivec *image = Array_data(&(p->image));
  dvec *trans = Array_data(&(p->trans));
  dvec *output = Array_data(&(p->output));
  const boolean wrap_none = (wrapping == WRAP_NONE);
  const boolean water_only = ((wrapping & WRAP_ALL) == WRAP_WATER);
  const boolean wrap_nearest = ((wrapping & WRAP_NEAREST) != 0);
  const dvec a0 = domain->basis[0];
  const dvec a1 = domain->basis[1];
  const dvec a2 = domain->basis[2];
  ivec m;
  int32 i, j;
  dvec shift;
  dreal scaling;
  int s;

  for (i = 0;  i < natoms;  i++) {
    if (wrap_none || (water_only && (atom[i].atomInfo & ATOM_WATER)==0)) {
      /* calculate full unwrapping:  output[i] = pos[i] + A*image[i] */
      m = image[i];
    }
    else {
      /* translate (pos,image)->(output,0) relative to first atom in cluster:
       * output[i] = pos[i] + A*(image[i]-image[clusterID]) */
      j = atom[i].clusterID;
      VECSUB(m, image[i], image[j]);
    }
    output[i].x = pos[i].x + (m.x * a0.x + m.y * a1.x + m.z * a2.x);
    output[i].y = pos[i].y + (m.x * a0.y + m.y * a1.y + m.z * a2.y);
    output[i].z = pos[i].z + (m.x * a0.z + m.y * a1.z + m.z * a2.z);
  }
  if (wrap_none) return OK;
  if ((s=Array_erase(&(p->trans))) != OK) return ERROR(s);  /* trans[i]=0 */
  for (i = 0;  i < natoms;  i++) {
    if (water_only && (atom[i].atomInfo & ATOM_WATER)==0) continue;
    j = atom[i].clusterID;
    VECADD(trans[j], trans[j], output[i]);  /* use trans to find geom center */
  }
  for (i = 0;  i < natoms;  i++) {
    if (water_only && (atom[i].atomInfo & ATOM_WATER)==0) continue;
    j = atom[i].clusterID;
    if (j == i) {
      /* take advantage that clusterID is assigned to lowest index,
       * finish calculating geometric center then use this location to wrap,
       * compute translation for entire cluster */
      scaling = 1.0 / (dreal) atom[i].clusterSize;
      VECMUL(trans[i], scaling, trans[i]);
      if (wrap_nearest) {
        Domain_wrap_vec_center(domain, &shift, &m, &trans[i]);
      }
      else {
        Domain_wrap_vec(domain, &shift, &m, &trans[i]);
      }
      trans[i] = shift;
    }
    VECADD(output[i], output[i], trans[j]);
  }
  return OK;
}


#if 0
int Coord_wrap_pos(Coord *p, const Topology *t, int32 wrapping) {
  const Domain *domain = &(p->domain);
  const Atom *atom = Topology_atom_array(t);
  dvec *gcen = Array_data(&(p->gcen));
  ivec *gimage = Array_data(&(p->gimage));
  dvec *pos = Array_data(&(p->pos));
  ivec *image = Array_data(&(p->image));
  const int32 natoms = Array_length(&(p->pos));
  const boolean water_only = ((wrapping & WRAP_ALL) == WRAP_WATER);
  const boolean wrap_nearest = ((wrapping & WRAP_NEAREST) != 0);
  int32 i, j;
  int s;

  if (Topology_atom_array_length(t) != natoms) return ERROR(ERR_VALUE);
  if (WRAP_NONE == wrapping) return OK;  /* nothing to do! */
  else if ((wrapping & ~(WRAP_NEAREST | WRAP_ALL)) != 0) {
    return ERROR(ERR_VALUE);  /* test for invalid wrapping flags */
  }

  if ((s=Array_erase(&(p->gcen))) != OK) return ERROR(s);  /* gcen[i]=0 */
  for (i = 0;  i < natoms;  i++) {
    if (water_only && (atom[i].atomInfo & ATOM_WATER) == 0) continue;
    j = atom[i].clusterID;
    VECADD(gcen[j], gcen[j], pos[i]);
    VECADD(gimage[j], gimage[j], image[i]);  /* hope it doesn't overflow */
  }
  for (i = 0;  i < natoms;  i++) {
    if (water_only && (atom[i].atomInfo & ATOM_WATER) == 0) continue;
    j = atom[i].clusterID;
    if (j == i) {
      /* take advantage that clusterID is assigned to lowest index,
       * finish calculating geometric center then use this location to wrap,
       * compute vector shift and image offset to add to other atom positions
       */
      dvec shift;
      dreal scal = 1.0 / (dreal) atom[i].clusterSize;
      VECMUL(delta[i], scal, delta[i]);
      if (wrap_nearest) {
       	Domain_wrap_vec_center(domain, &shift, &delta_offset[i], &delta[i]);
      }
      else {
       	Domain_wrap_vec(domain, &shift, &delta_offset[i], &delta[i]);
      }
      delta[i] = shift;
    }
    VECADD(pos[i], delta[j], pos[i]);
    VECADD(offset[i], delta_offset[j], offset[i]);
  }
  return OK;
}


int Coord_normalize_pos(Coord *p) {
  const Domain *domain = &(p->domain);
  dvec *scalpos = Array_data(&(p->scalpos));
  const dvec *pos = Array_data(&(p->pos));
  const int32 natoms = Array_length(&(p->pos));
  int32 i;
  ivec n;

  for (i = 0;  i < natoms;  i++) {
    Domain_normalize_vec(domain, scalpos+i, &n, pos+i);
  }
  return OK;
}
#endif


int Coord_set_temperature(Coord *p, Random *ran, dreal temperature) {
  const dreal kbtemp = BOLTZMANN * temperature;
  dreal sqrt_kbtemp_div_mass;
#ifndef STEP_ALT_INITTEMP
  dreal rnum;
#endif
  const Atom *atom = Topology_atom_array(p->topo);
  dvec *vel = Array_data(&(p->vel));
  const int32 natoms = Array_length(&(p->vel));
  int32 i, k;

  /* make sure initial temperature is valid */
  if (temperature < 0.0) return ERROR(ERR_VALUE);

  for (i = 0;  i < natoms;  i++) {
    if ((atom[i].atomInfo & ATOM_TYPE) == ATOM_LONEPAIR) {
      VECZERO(vel[i]);
      continue;
    }
    sqrt_kbtemp_div_mass = sqrt(kbtemp / atom[i].m);

#ifndef STEP_ALT_INITTEMP
    /*
     * The following method and comments taken from NAMD WorkDistrib.C:
     *
     * //  The following comment was stolen from X-PLOR where
     * //  the following section of code was adapted from.
     *
     * //  This section generates a Gaussian random
     * //  deviate of 0.0 mean and standard deviation RFD for
     * //  each of the three spatial dimensions.
     * //  The algorithm is a "sum of uniform deviates algorithm"
     * //  which may be found in Abramowitz and Stegun,
     * //  "Handbook of Mathematical Functions", pg 952.
     */
    rnum = -6.0;
    for (k = 0;  k < 12;  k++) {
      rnum += Random_uniform(ran);
    }
    vel[i].x = sqrt_kbtemp_div_mass * rnum;

    rnum = -6.0;
    for (k = 0;  k < 12;  k++) {
      rnum += Random_uniform(ran);
    }
    vel[i].y = sqrt_kbtemp_div_mass * rnum;

    rnum = -6.0;
    for (k = 0;  k < 12;  k++) {
      rnum += Random_uniform(ran);
    }
    vel[i].z = sqrt_kbtemp_div_mass * rnum;
#else
    /*
     * Alternate method from NAMD Sequencer.C:
     */
    vel[i].x = sqrt_kbtemp_div_mass * Random_gaussian(ran);
    vel[i].y = sqrt_kbtemp_div_mass * Random_gaussian(ran);
    vel[i].z = sqrt_kbtemp_div_mass * Random_gaussian(ran);
#endif
  }
  return OK;
}


int Coord_remove_com_motion(Coord *p) {
  const Atom *atom = Topology_atom_array(p->topo);
  dvec *vel = Array_data(&(p->vel));
  const int32 natoms = Array_length(&(p->vel));
  int32 i;
  dvec mv = { 0.0, 0.0, 0.0 };     /* accumulate net momentum */
  dreal mass = 0.0;                /* accumulate total mass */
  dreal inv_mass;

  /* compute net momentum and total mass */
  for (i = 0;  i < natoms;  i++) {
    VECMADD(mv, atom[i].m, vel[i], mv);
    mass += atom[i].m;
  }

  /* scale net momentum by total mass */
  inv_mass = 1.0 / mass;
  VECMUL(mv, inv_mass, mv);

  /* remove from atom velocities */
  for (i = 0;  i < natoms;  i++) {
    if ((atom[i].atomInfo & ATOM_TYPE) == ATOM_LONEPAIR) continue;
    VECSUB(vel[i], vel[i], mv);
  }
  return OK;
}
