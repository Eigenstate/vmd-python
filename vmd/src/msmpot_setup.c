/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: msmpot_setup.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.5 $      $Date: 2013/04/09 14:15:55 $
 *
 ***************************************************************************/

#include "msmpot_internal.h"


int Msmpot_check_params(Msmpot *msm, const float *epotmap,
    int mx, int my, int mz, float lx, float ly, float lz,
    float vx, float vy, float vz, const float *atom, int natoms) {

  if (NULL == epotmap) {
    return ERRMSG(MSMPOT_ERROR_PARAM, "map buffer is NULL");
  }
  else if (NULL == atom) {
    return ERRMSG(MSMPOT_ERROR_PARAM, "atom array is NULL");
  }
  else if (natoms <= 0) {
    return ERRMSG(MSMPOT_ERROR_PARAM, "number of atoms is not positive");
  }
  else if (mx <= 0 || my <= 0 || mz <= 0) {
    return ERRMSG(MSMPOT_ERROR_PARAM, "number of map points is not positive");
  }
  else if (lx <= 0 || ly <= 0 || lz <= 0) {
    return ERRMSG(MSMPOT_ERROR_PARAM, "map lengths are not positive");
  }
  else if ((vx <= msm->hmin && vx != 0) || (vy <= msm->hmin && vy != 0)
      || (vz <= msm->hmin && vz != 0)) {
    return ERRMSG(MSMPOT_ERROR_PARAM,
        "periodic cell lengths must be greater than MSM spacing");
  }
  else if ((lx > vx && vx != 0) || (ly > vy && vy != 0)
      || (lz > vz && vz != 0)) {
    return ERRMSG(MSMPOT_ERROR_PARAM,
        "map lengths must be within periodic cell");
  }

  return MSMPOT_SUCCESS;
}


/* called by Msmpot_create() */
void Msmpot_set_defaults(Msmpot *msm) {
  /* setup default parameters */
  msm->hmin = DEFAULT_HMIN;
  msm->a = DEFAULT_CUTOFF;
  msm->interp = DEFAULT_INTERP;
  msm->split = DEFAULT_SPLIT;
  msm->bindepth = DEFAULT_BINDEPTH;
  msm->binfill = DEFAULT_BINFILL;
  msm->density = DEFAULT_DENSITY;
  msm->errtol = ((float) DEFAULT_ERRTOL);

#ifdef MSMPOT_CUDA
  /* attempt to use CUDA but fall back on CPU if unable */
  msm->use_cuda = 1;
  msm->cuda_optional = 1;
#endif
}


int Msmpot_configure(Msmpot *msm,
    int interp,
    int split,
    float cutoff,
    float hmin,
    int nlevels,
    float density,
    float binfill,
    float errtol,
    int usecuda
    ) {
  if (interp < 0 || interp >= MSMPOT_INTERPMAX) {
    return ERROR(MSMPOT_ERROR_PARAM);
  }
  else msm->interp = interp;  /* default is 0 */

  if (split < 0 || split >= MSMPOT_SPLITMAX) {
    return ERROR(MSMPOT_ERROR_PARAM);
  }
  else msm->split = split;  /* default is 0 */

  if (cutoff < 0) return ERROR(MSMPOT_ERROR_PARAM);
  else if (cutoff > 0) msm->a = cutoff;
  else msm->a = DEFAULT_CUTOFF;

  if (hmin < 0) return ERROR(MSMPOT_ERROR_PARAM);
  else if (hmin > 0) msm->hmin = hmin;
  else msm->hmin = DEFAULT_HMIN;

  if (nlevels < 0) return ERROR(MSMPOT_ERROR_PARAM);
  else msm->nlevels = nlevels;  /* 0 is default */

  if (density < 0) return ERROR(MSMPOT_ERROR_PARAM);
  else if (density > 0) msm->density = density;
  else msm->density = DEFAULT_DENSITY;

  if (binfill < 0) return ERROR(MSMPOT_ERROR_PARAM);
  else if (binfill > 0) msm->binfill = binfill;
  else msm->binfill = DEFAULT_BINFILL;

  if (errtol < 0) return ERROR(MSMPOT_ERROR_PARAM);
  else if (errtol > 0) msm->errtol = errtol;
  else msm->errtol = ((float) DEFAULT_ERRTOL);

#ifdef MSMPOT_CUDA
  msm->use_cuda = (usecuda != 0);
#else
  if (usecuda != 0) {
    return ERROR(MSMPOT_ERROR_SUPPORT);
  }
#endif

  return MSMPOT_SUCCESS;
}


/* called by Msmpot_destroy() */
void Msmpot_cleanup(Msmpot *msm) {
  int i;
  for (i = 0;  i < msm->maxlevels;  i++) {
    GRID_DONE( &(msm->qh[i]) );
    GRID_DONE( &(msm->eh[i]) );
    GRID_DONE( &(msm->gc[i]) );
  }
  free(msm->qh);
  free(msm->eh);
  free(msm->gc);
  free(msm->ezd);
  free(msm->eyzd);
  free(msm->lzd);
  free(msm->lyzd);
  free(msm->phi_x);
  free(msm->phi_y);
  free(msm->phi_z);

  free(msm->first_atom_index);
  free(msm->next_atom_index);

  free(msm->bin);
  free(msm->bincount);
  free(msm->over);
  free(msm->boff);
}


static int setup_domain(Msmpot *msm);
static int setup_bins(Msmpot *msm);
static int setup_origin(Msmpot *msm);

static int setup_hierarchy(Msmpot *msm);

/* called by setup_hierarchy() */
static int setup_periodic_hlevelparams_1d(Msmpot *msm,
    float len,            /* domain length */
    float *hh,            /* determine h */
    int *nn,              /* determine number grid spacings covering domain */
    int *aindex,          /* determine smallest lattice index */
    int *bindex           /* determine largest lattice index */
    );

/* called by setup_hierarchy() */
static int setup_nonperiodic_hlevelparams_1d(Msmpot *msm,
    float len,            /* measure to furthest point interpolated from grid */
    float *hh,            /* determine h */
    int *nn,              /* determine number grid spacings covering domain */
    int *aindex,          /* determine smallest lattice index */
    int *bindex           /* determine largest lattice index */
    );

static int setup_mapinterp(Msmpot *msm);

static int setup_mapinterpcoef_1d(Msmpot *msm,
    float h,             /* spacing of MSM h-level lattice */
    float delta,         /* spacing of epotmap lattice */
    int n,               /* number of MSM h spacings to cover domain */
    int m,               /* number of epotmap lattice points */
    float *p_h_delta,    /* calculate ratio h/delta */
    int *p_cycle,        /* number of MSM points until next map alignment */
    int *p_rmap,         /* radius of map points about MSM point */
    float **p_phi,       /* coefficients that weight map about MSM points */
    int *p_max_phi       /* size of phi memory allocation */
    );


#ifdef MSMPOT_VERBOSE
static int print_status(Msmpot *msm);
#endif


/* called by Msmpot_compute() */
int Msmpot_setup(Msmpot *msm) {
  int err = 0;

  REPORT("Setting up for MSM computation.");

  /* set domain lengths, needed for any non-periodic dimensions */
  err = setup_domain(msm);
  if (err) return ERROR(err);

  /* determine bin subdivision across domain */
  err = setup_bins(msm);
  if (err) return ERROR(err);

  /* given bin subdivision, find good domain origin for periodic dimensions */
  err = setup_origin(msm);
  if (err) return ERROR(err);

  /* set up hierarchy of lattices for long-range part */
  err = setup_hierarchy(msm);
  if (err) return ERROR(err);


#if ! defined(MSMPOT_SHORTRANGE_ONLY)
  if (msm->px == msm->lx && msm->py == msm->ly && msm->pz == msm->lz) {
    /* determine map interpolation parameters
     * and MSM lattice spacings hx, hy, hz */
    err = setup_mapinterp(msm);
    if (err) return ERROR(err);
  }
#endif


#ifdef MSMPOT_VERBOSE
  err = print_status(msm);
  if (err) return ERROR(err);
#endif

#ifdef MSMPOT_CUDA
  /* set up CUDA device */
  if (msm->use_cuda) {
    err = Msmpot_cuda_setup(msm->msmcuda, msm);
    if (err) return ERROR(err);
  }
#endif

  return MSMPOT_SUCCESS;
}


typedef struct InterpParams_t {
  int nu;
  int stencil;
  int omega;
} InterpParams;

static InterpParams INTERP_PARAMS[] = {
  { 1, 4, 6 },    /* cubic */
  { 2, 6, 10 },   /* quintic */
  { 2, 6, 10 },   /* quintic, C2 */
  { 3, 8, 14 },   /* septic */
  { 3, 8, 14 },   /* septic, C3 */
  { 4, 10, 18 },  /* nonic */
  { 4, 10, 18 },  /* nonic, C4 */
};


#ifdef MSMPOT_VERBOSE
int print_status(Msmpot *msm) {
  int j, k;
  int ispx = (IS_SET_X(msm->isperiodic) != 0);
  int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  int ispz = (IS_SET_Z(msm->isperiodic) != 0);
  int ispany = (IS_SET_ANY(msm->isperiodic) != 0);
  printf("#MSMPOT STATUS\n");
  printf("#  natoms= %d\n", msm->natoms);
  printf("#  domain lengths= %g %g %g\n", msm->px, msm->py, msm->pz);
  printf("#  domain origin= %g %g %g\n", msm->px0, msm->py0, msm->pz0);
  printf("#  map size= %d x %d x %d\n", msm->mx, msm->my, msm->mz);
  printf("#  map spacing= %g, %g, %g\n", msm->dx, msm->dy, msm->dz);
  printf("#  map lengths= %g, %g, %g\n", msm->lx, msm->ly, msm->lz);
  printf("#  map origin= %g, %g, %g\n", msm->lx0, msm->ly0, msm->lz0);
  printf("#  hmin= %g\n", msm->hmin);
  printf("#  cutoff= %g\n", msm->a);
  printf("#  MSM size= %d x %d x %d\n", msm->nx, msm->ny, msm->nz);
  printf("#  MSM spacing= %g, %g, %g\n", msm->hx, msm->hy, msm->hz);
  printf("#  MSM interpolation= %d\n", msm->interp);
  printf("#  MSM splitting= %d\n", msm->split);
  printf("#  MSM number of levels= %d\n", msm->nlevels);
  printf("#  MSM lattice hierarchy:\n");
  for (k = 0;  k < msm->nlevels;  k++) {
    floatGrid *qh = &(msm->qh[k]);
    int ia = qh->i0;
    int ib = ia + qh->ni - 1;
    int ja = qh->j0;
    int jb = ja + qh->nj - 1;
    int ka = qh->k0;
    int kb = ka + qh->nk - 1;
    printf("#  level= %d:  [%d..%d] x [%d..%d] x [%d..%d]\n",
        k, ia, ib, ja, jb, ka, kb);
  }
  printf("#  ispx= %d  ispy= %d  ispz= %d  ispany= %d\n",
      ispx, ispy, ispz, ispany);
  printf("#  hx_dx= %g  hy_dy= %g  hz_dz= %g\n",
      msm->hx_dx, msm->hy_dy, msm->hz_dz);
  printf("#  cycle_x= %d  rmap_x= %d\n", msm->cycle_x, msm->rmap_x);
  for (k = 0;  k < msm->cycle_x;  k++) {
    int jtotal = 2*msm->rmap_x + 1;
    float *phi = msm->phi_x + k*jtotal;
    float phisum = 0;
    for (j = 0;  j < jtotal;  j++) phisum += phi[j];
    printf("#  %d:  sum= %g  (", k, phisum);
    for (j = 0;  j < jtotal;  j++) {
      printf("%s%g", (0==j ? "= " : " + "), phi[j]);
    }
    printf(")\n");
  }
  printf("#  cycle_y= %d  rmap_y= %d\n", msm->cycle_y, msm->rmap_y);
  for (k = 0;  k < msm->cycle_y;  k++) {
    int jtotal = 2*msm->rmap_y + 1;
    float *phi = msm->phi_y + k*jtotal;
    float phisum = 0;
    for (j = 0;  j < jtotal;  j++) phisum += phi[j];
    printf("#  %d:  sum= %g  (", k, phisum);
    for (j = 0;  j < jtotal;  j++) {
      printf("%s%g", (0==j ? "= " : " + "), phi[j]);
    }
    printf(")\n");
  }
  printf("#  cycle_z= %d  rmap_z= %d\n", msm->cycle_z, msm->rmap_z);
  for (k = 0;  k < msm->cycle_z;  k++) {
    int jtotal = 2*msm->rmap_z + 1;
    float *phi = msm->phi_z + k*jtotal;
    float phisum = 0;
    for (j = 0;  j < jtotal;  j++) phisum += phi[j];
    printf("#  %d:  sum= %g  (", k, phisum);
    for (j = 0;  j < jtotal;  j++) {
      printf("%s%g", (0==j ? "= " : " + "), phi[j]);
    }
    printf(")\n");
  }
  printf("#  bin size= %g %g %g\n", msm->bx, msm->by, msm->bz);
  printf("#  atom bins= %d x %d x %d\n",
      msm->nbx, msm->nby, msm->nbz);
  return MSMPOT_SUCCESS;
}
#endif


int setup_mapinterp(Msmpot *msm) {
  int mymz = msm->my * msm->mz;
  int err = 0;

  ASSERT(msm->mx > 0);
  ASSERT(msm->my > 0);
  ASSERT(msm->mz > 0);
  if (msm->max_eyzd < mymz) {
    float *t;
    t = (float *) realloc(msm->eyzd, mymz * sizeof(float));
    if (NULL == t) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->eyzd = t;
    msm->max_eyzd = mymz;
  }
  if (msm->max_ezd < msm->mz) {
    float *t;
    t = (float *) realloc(msm->ezd, msm->mz * sizeof(float));
    if (NULL == t) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->ezd = t;
    msm->max_ezd = msm->mz;
  }

  if (msm->px != msm->lx
      || msm->py != msm->ly
      || msm->pz != msm->lz) {
    /* must bail, can't do interpolation right yet */
    printf("px = %f  lx = %f\n", msm->px, msm->lx);
    return ERRMSG(MSMPOT_ERROR_SUPPORT,
        "can't do interpolation for map lengths "
        "not equal to atom domain lengths");
  }


  err |= setup_mapinterpcoef_1d(msm,
      msm->hx, msm->dx, msm->nx, msm->mx, &(msm->hx_dx),
      &(msm->cycle_x), &(msm->rmap_x), &(msm->phi_x), &(msm->max_phi_x));
  err |= setup_mapinterpcoef_1d(msm,
      msm->hy, msm->dy, msm->ny, msm->my, &(msm->hy_dy),
      &(msm->cycle_y), &(msm->rmap_y), &(msm->phi_y), &(msm->max_phi_y));
  err |= setup_mapinterpcoef_1d(msm,
      msm->hz, msm->dz, msm->nz, msm->mz, &(msm->hz_dz),
      &(msm->cycle_z), &(msm->rmap_z), &(msm->phi_z), &(msm->max_phi_z));
  if (err) return ERROR(err);


  return MSMPOT_SUCCESS;
}


static int gcd(int a, int b) {
  /* subtraction-based Euclidean algorithm, from Knuth */
  if (0 == a) return b;
  while (b != 0) {
    if (a > b) a -= b;
    else       b -= a;
  }
  return a;
}


int setup_mapinterpcoef_1d(Msmpot *msm,
    float h,             /* spacing of MSM h-level lattice */
    float delta,         /* spacing of epotmap lattice */
    int n,               /* number of MSM h spacings to cover domain */
    int m,               /* number of epotmap lattice points */
    float *p_h_delta,    /* calculate ratio h/delta */
    int *p_cycle,        /* number of MSM points until next map alignment */
    int *p_rmap,         /* radius of map points about MSM point */
    float **p_phi,       /* coefficients that weight map about MSM points */
    int *p_max_phi       /* size of phi memory allocation */
    ) {
  float *phi = NULL;
  const int nu = INTERP_PARAMS[msm->interp].nu;
  float delta_h, h_delta, t;
  int cycle, rmap, diam, nphi;
  int i, k;

  *p_h_delta = h_delta = h / delta;
  *p_cycle = cycle = n / gcd(m, n);
  *p_rmap = rmap = (int) ceilf(h_delta * (nu+1));

  delta_h = delta / h;
  diam = 2*rmap + 1;
  nphi = diam * cycle;

  if (*p_max_phi < nphi) {  /* allocate more memory if we need it */
    phi = (float *) realloc(*p_phi, nphi * sizeof(float));
    if (NULL == phi) return ERROR(MSMPOT_ERROR_MALLOC);
    *p_phi = phi;
    *p_max_phi = nphi;
  }
  ASSERT(*p_phi != NULL);

  for (k = 0;  k < cycle;  k++) {
    float offset = floorf(k * h_delta) * delta_h - k;
    phi = *p_phi + k * diam + rmap;  /* center of this weight stencil */
    switch (msm->interp) {
      case MSMPOT_INTERP_CUBIC:
        for (i = -rmap;  i <= rmap;  i++) {
          t = fabsf(i * delta_h + offset);
          if (t <= 1) {
            phi[i] = (1 - t) * (1 + t - 1.5f * t * t);
          }
          else if (t <= 2) {
            phi[i] = 0.5f * (1 - t) * (2 - t) * (2 - t);
          }
          else {
            phi[i] = 0;
          }
        }
        break;
      case MSMPOT_INTERP_QUINTIC:
        for (i = -rmap;  i <= rmap;  i++) {
          t = fabsf(i * delta_h);
          if (t <= 1) {
            phi[i] = (1-t*t) * (2-t) * (0.5f + t * (0.25f - (5.f/12)*t));
          }
          else if (t <= 2) {
            phi[i] = (1-t)*(2-t)*(3-t) * ((1.f/6) + t*(0.375f - (5.f/24)*t));
          }
          else if (t <= 3) {
            phi[i] = (1.f/24) * (1-t) * (2-t) * (3-t) * (3-t) * (4-t);
          }
          else {
            phi[i] = 0;
          }
        }
        break;
      default:
        return ERRMSG(MSMPOT_ERROR_SUPPORT,
            "interpolation method not implemented");
    } /* end switch on interp */
  } /* end loop k over cycles */
  return MSMPOT_SUCCESS;
}


int setup_domain(Msmpot *msm) {
  const float *atom = msm->atom;
  const int natoms = msm->natoms;
  int n;
  float xmin, xmax, ymin, ymax, zmin, zmax;

  /* find extent of atoms */
  ASSERT(natoms > 0);
  xmin = xmax = atom[ATOM_X(0)];
  ymin = ymax = atom[ATOM_Y(0)];
  zmin = zmax = atom[ATOM_Z(0)];
  for (n = 1;  n < natoms;  n++) {
    float x = atom[ATOM_X(n)];
    float y = atom[ATOM_Y(n)];
    float z = atom[ATOM_Z(n)];
    if (xmin > x)      xmin = x;
    else if (xmax < x) xmax = x;
    if (ymin > y)      ymin = y;
    else if (ymax < y) ymax = y;
    if (zmin > z)      zmin = z;
    else if (zmax < z) zmax = z;
  }

  /* store maximum extent of atoms, regardless of periodicity */
  msm->xmin = xmin;
  msm->xmax = xmax;
  msm->ymin = ymin;
  msm->ymax = ymax;
  msm->zmin = zmin;
  msm->zmax = zmax;

#if 1
  /* domain for non-periodic dimensions is to include both epotmap and atoms */
  if ( ! IS_SET_X(msm->isperiodic) ) {  /* non-periodic in x */
    float lx1 = msm->lx0 + msm->lx;  /* contains last epotmap point */
    if (xmin >= msm->lx0 && xmax < lx1) {
      msm->px = msm->lx;    /* assignment can enable factored interpolation */
      msm->px0 = msm->lx0;
    }
    else {
      if (xmin > msm->lx0)  xmin = msm->lx0;
      msm->px0 = xmin;
      if (xmax < lx1) {
        xmax = lx1;
        msm->px = xmax - xmin;
      }
      else {
        msm->px = xmax - xmin + msm->dx;
      }
    }
  }
  if ( ! IS_SET_Y(msm->isperiodic) ) {  /* non-periodic in y */
    float ly1 = msm->ly0 + msm->ly;  /* contains last epotmap point */
    if (ymin >= msm->ly0 && ymax < ly1) {
      msm->py = msm->ly;    /* assignment can enable factored interpolation */
      msm->py0 = msm->ly0;
    }
    else {
      if (ymin > msm->ly0)  ymin = msm->ly0;
      msm->py0 = ymin;
      if (ymax < ly1) {
        ymax = ly1;
        msm->py = ymax - ymin;
      }
      else {
        msm->py = ymax - ymin + msm->dy;
      }
    }
  }
  if ( ! IS_SET_Z(msm->isperiodic) ) {  /* non-periodic in z */
    float lz1 = msm->lz0 + msm->lz;  /* contains last epotmap point */
    if (zmin >= msm->lz0 && zmax < lz1) {
      msm->pz = msm->lz;    /* assignment can enable factored interpolation */
      msm->pz0 = msm->lz0;
    }
    else {
      if (zmin > msm->lz0)  zmin = msm->lz0;
      msm->pz0 = zmin;
      if (zmax < lz1) {
        zmax = lz1;
        msm->pz = zmax - zmin;
      }
      else {
        msm->pz = zmax - zmin + msm->dz;
      }
    }
  }
#else
  /* domain for non-periodic dimensions is to include both epotmap and atoms */
  if ( ! IS_SET_X(msm->isperiodic) ) {  /* non-periodic in x */
    float lx1 = msm->lx0 + msm->lx;  /* contains last epotmap point */
    if (xmin >= msm->lx0 && xmax < lx1) {
      msm->px = msm->lx;    /* assignment can enable factored interpolation */
      msm->px0 = msm->lx0;
    }
    else {
      if (xmin > msm->lx0)  xmin = msm->lx0;
      if (xmax < lx1)       xmax = lx1;
      msm->px = xmax - xmin;
      msm->px0 = xmin;
    }
  }
  if ( ! IS_SET_Y(msm->isperiodic) ) {  /* non-periodic in y */
    float ly1 = msm->ly0 + msm->ly;  /* contains last epotmap point */
    if (ymin >= msm->ly0 && ymax < ly1) {
      msm->py = msm->ly;    /* assignment can enable factored interpolation */
      msm->py0 = msm->ly0;
    }
    else {
      if (ymin > msm->ly0)  ymin = msm->ly0;
      if (ymax < ly1)       ymax = ly1;
      msm->py = ymax - ymin;
      msm->py0 = ymin;
    }
  }
  if ( ! IS_SET_Z(msm->isperiodic) ) {  /* non-periodic in z */
    float lz1 = msm->lz0 + msm->lz;  /* contains last epotmap point */
    if (zmin >= msm->lz0 && zmax < lz1) {
      msm->pz = msm->lz;    /* assignment can enable factored interpolation */
      msm->pz0 = msm->lz0;
    }
    else {
      if (zmin > msm->lz0)  zmin = msm->lz0;
      if (zmax < lz1)       zmax = lz1;
      msm->pz = zmax - zmin;
      msm->pz0 = zmin;
    }
  }
#endif

  return MSMPOT_SUCCESS;
}


/*
 * Bins are set up based on domain side lengths.
 * This is done independently from selecting origin for periodic systems.
 */
int setup_bins(Msmpot *msm) {
  /* vol is measure of local volume */
  float vol = msm->binfill * msm->bindepth / msm->density;
  float blen = powf(vol, 1.f/3);  /* ideal bin side length */
  int maxbin;
  int nbx = (int) ceilf(msm->px / blen);  /* using ceilf to count bins */
  int nby = (int) ceilf(msm->py / blen);  /* makes bin vol <= desired vol */
  int nbz = (int) ceilf(msm->pz / blen);

  ASSERT(nbx > 0);
  ASSERT(nby > 0);
  ASSERT(nbz > 0);

  maxbin = nbx * nby * nbz;
  if (msm->maxbin < maxbin) {  /* grab more memory if we need it */
    void *v;
    size_t floatsperbin = msm->bindepth * ATOM_SIZE;
    v = realloc(msm->bin, maxbin * floatsperbin * sizeof(float));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->bin = (float *) v;
    v = realloc(msm->bincount, maxbin * sizeof(int));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->bincount = (int *) v;
    msm->maxbin = maxbin;

    /* for cursor linked list implementation */
    v = realloc(msm->first_atom_index, maxbin * sizeof(int));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->first_atom_index = (int *) v;
    /*   */
  }

  /* for cursor linked list implmentation */
  if (msm->maxatoms < msm->natoms) {
    void *v;
    v = realloc(msm->next_atom_index, msm->natoms * sizeof(int));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->next_atom_index = (int *) v;
    msm->maxatoms = msm->natoms;
  }
  /*   */

  msm->nbx = nbx;
  msm->nby = nby;
  msm->nbz = nbz;

  msm->bx = msm->px / nbx;
  msm->by = msm->py / nby;
  msm->bz = msm->pz / nbz;

  msm->invbx = 1 / msm->bx;
  msm->invby = 1 / msm->by;
  msm->invbz = 1 / msm->bz;

  if (msm->maxover < DEFAULT_OVER) {
    void *vover = realloc(msm->over, DEFAULT_OVER * ATOM_SIZE * sizeof(float));
    if (NULL == vover) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->over = (float *) vover;
    msm->maxover = DEFAULT_OVER;
  }

  return MSMPOT_SUCCESS;
}


int setup_origin(Msmpot *msm) {
  /* we get to choose the origin (lower-left corner) of the atom domain,
   * which can be advantageous to reduce wrapping for periodic boundaries */

  msm->isbinwrap = 0;     /* reset flags */
  msm->islongcutoff = 0;

  if (IS_SET_X(msm->isperiodic)) {
    /* how many bins does cutoff extend? */
    int mbx = (int) ceilf(msm->a * msm->invbx);
    if (msm->lx < (msm->px - 2*mbx*msm->bx)) {
      /* epotmap fits inside of domain with thick shell of bins */
      msm->px0 = msm->lx0 - mbx * msm->bx;
    }
    else {
      /* we will have to wrap bin neighborhood */
      msm->px0 = msm->lx0;
      SET_X(msm->isbinwrap);
      if (mbx > msm->nbx) SET_X(msm->islongcutoff);  /* wraps more than once */
    }
  }

  if (IS_SET_Y(msm->isperiodic)) {
    /* how many bins does cutoff extend? */
    int mby = (int) ceilf(msm->a * msm->invby);
    if (msm->ly < (msm->py - 2*mby*msm->by)) {
      /* epotmap fits inside of domain with thick shell of bins */
      msm->py0 = msm->ly0 - mby * msm->by;
    }
    else {
      /* we will have to wrap bin neighborhood */
      msm->py0 = msm->ly0;
      SET_Y(msm->isbinwrap);
      if (mby > msm->nby) SET_Y(msm->islongcutoff);  /* wraps more than once */
    }
  }

  if (IS_SET_Z(msm->isperiodic)) {
    /* how many bins does cutoff extend? */
    int mbz = (int) ceilf(msm->a * msm->invbz);
    if (msm->lz < (msm->pz - 2*mbz*msm->bz)) {
      /* epotmap fits inside of domain with thick shell of bins */
      msm->pz0 = msm->lz0 - mbz * msm->bz;
    }
    else {
      /* we will have to wrap bin neighborhood */
      msm->pz0 = msm->lz0;
      SET_Z(msm->isbinwrap);
      if (mbz > msm->nbz) SET_Z(msm->islongcutoff);  /* wraps more than once */
    }
  }

  return MSMPOT_SUCCESS;
}


int setup_nonperiodic_hlevelparams_1d(Msmpot *msm,
    float len,        /* measure to furthest point interpolated from grid */
    float *hh,        /* determine h */
    int *nn,          /* determine number grid spacings covering domain */
    int *aindex,      /* determine smallest lattice index */
    int *bindex       /* determine largest lattice index */
    ) {
  const float hmin = msm->hmin;  /* minimum bound on h */
  const int nu = INTERP_PARAMS[msm->interp].nu;  /* interp stencil radius */
    /* make sure RH grid point is beyond farthest atom or epotmap point */
  int n = (int) floorf(len / hmin) + 1;
  *hh = hmin;
  *nn = n;
  *aindex = -nu;
  *bindex = n + nu;
  return MSMPOT_SUCCESS;
}


int setup_periodic_hlevelparams_1d(Msmpot *msm,
    float len,        /* domain length */
    float *hh,        /* determine h */
    int *nn,          /* determine number grid spacings covering domain */
    int *aindex,      /* determine smallest lattice index */
    int *bindex       /* determine largest lattice index */
    ) {
  const float hmin = msm->hmin;  /* minimum bound on h */
  const float hmax = 1.5f * hmin;
  float h = len;
  int n = 1;    /* start with one grid point across domain */
  while (h >= hmax) {
    h *= 0.5f;  /* halve h */
    n <<= 1;    /* double grid points */
  }
  if (h < hmin) {
    if (n < 4) {  /* error: either len is too small or hmin is too large */
      return ERRMSG(MSMPOT_ERROR_PARAM,
          "ratio of domain length to hmin is too small");
    }
    h *= (4.f/3); /* scale h by 4/3 */
    n >>= 2;      /* scale n by 3/4 */
    n *= 3;
  }
  /* now we have:  hmin <= h < hmax */
  /* now we have:  n is power of two times no more than one power of 3 */
  *hh = h;
  *nn = n;
  *aindex = 0;
  *bindex = n-1;
  return MSMPOT_SUCCESS;
}


int setup_hierarchy(Msmpot *msm) {
  const int nu = INTERP_PARAMS[msm->interp].nu;
  const int omega = INTERP_PARAMS[msm->interp].omega;
  const int split = msm->split;
  const int ispx = IS_SET_X(msm->isperiodic);
  const int ispy = IS_SET_Y(msm->isperiodic);
  const int ispz = IS_SET_Z(msm->isperiodic);
  const int ispany = IS_SET_ANY(msm->isperiodic);
  const float a = msm->a;
  float hx, hy, hz;
  float scaling;

  floatGrid *p = NULL;
  int ia, ib, ja, jb, ka, kb, ni, nj, nk;
  int nx, ny, nz;  /* counts the grid points that span just the domain */

  int i, j, k, n;
  int index;
  int level, toplevel, nlevels, maxlevels;
  int lastnelems = 1;
  int isclamped = 0;
  int done, alldone;

  int err = 0;

  if (ispx) {
    err = setup_periodic_hlevelparams_1d(msm, msm->px, &hx, &nx, &ia, &ib);
  }
  else {
    float xmax = msm->lx0 + msm->lx - msm->dx;  /* furthest epotmap point */
    if (xmax < msm->xmax) xmax = msm->xmax;     /* furthest atom */
    err = setup_nonperiodic_hlevelparams_1d(msm, xmax - msm->px0,
        &hx, &nx, &ia, &ib);
  }
  if (err) return ERROR(err);

  if (ispy) {
    err = setup_periodic_hlevelparams_1d(msm, msm->py, &hy, &ny, &ja, &jb);
  }
  else {
    float ymax = msm->ly0 + msm->ly - msm->dy;  /* furthest epotmap point */
    if (ymax < msm->ymax) ymax = msm->ymax;     /* furthest atom */
    err = setup_nonperiodic_hlevelparams_1d(msm, ymax - msm->py0,
        &hy, &ny, &ja, &jb);
  }
  if (err) return ERROR(err);

  if (ispz) {
    err = setup_periodic_hlevelparams_1d(msm, msm->pz, &hz, &nz, &ka, &kb);
  }
  else {
    float zmax = msm->lz0 + msm->lz - msm->dz;  /* furthest epotmap point */
    if (zmax < msm->zmax) zmax = msm->zmax;     /* furthest atom */
    err = setup_nonperiodic_hlevelparams_1d(msm, zmax - msm->pz0,
        &hz, &nz, &ka, &kb);
  }
  if (err) return ERROR(err);

  msm->hx = hx;
  msm->hy = hy;
  msm->hz = hz;

  msm->nx = nx;
  msm->ny = ny;
  msm->nz = nz;

  ni = ib - ia + 1;
  nj = jb - ja + 1;
  nk = kb - ka + 1;

  /* allocate temp buffer space for factored grid transfer */
  n = (nk > omega ? nk : omega);  /* row along z-dimension */
  if (msm->max_lzd < n) {
    float *t;
    t = (float *) realloc(msm->lzd, n * sizeof(float));
    if (NULL == t) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->lzd = t;
    msm->max_lzd = n;
  }
  n *= (nj > omega ? nj : omega);  /* plane along yz-dimensions */
  if (msm->max_lyzd < n) {
    float *t;
    t = (float *) realloc(msm->lyzd, n * sizeof(float));
    if (NULL == t) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->lyzd = t;
    msm->max_lyzd = n;
  }

  nlevels = msm->nlevels;
  if (nlevels <= 0) {
    /* automatically set number of levels */
    n = ni;
    if (n < nj) n = nj;
    if (n < nk) n = nk;
    for (maxlevels = 1;  n > 0;  n >>= 1)  maxlevels++;
    nlevels = maxlevels;
    if ( ! ispany ) {  /* no periodicity */
      int omega3 = omega * omega * omega;
      int nhalf = (int) sqrtf(ni*nj*nk);  /* scale down for performance? */
      lastnelems = (nhalf > omega3 ? nhalf : omega3);
      isclamped = 1;
    }
  }
  else {
    /* user-defined number of levels */
    maxlevels = nlevels;
  }

  /* allocate any additional levels that may be needed */
  if (msm->maxlevels < maxlevels) {
    void *vqh, *veh, *vgc;
    vqh = realloc(msm->qh, maxlevels * sizeof(floatGrid));
    if (NULL == vqh) return ERROR(MSMPOT_ERROR_MALLOC);
    veh = realloc(msm->eh, maxlevels * sizeof(floatGrid));
    if (NULL == veh) return ERROR(MSMPOT_ERROR_MALLOC);
    vgc = realloc(msm->gc, maxlevels * sizeof(floatGrid));
    if (NULL == vgc) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->qh = (floatGrid *) vqh;
    msm->eh = (floatGrid *) veh;
    msm->gc = (floatGrid *) vgc;
    /* initialize the newest grids appended to array */
    for (level = msm->maxlevels;  level < maxlevels;  level++) {
      GRID_INIT( &(msm->qh[level]) );
      GRID_INIT( &(msm->eh[level]) );
      GRID_INIT( &(msm->gc[level]) );
    }
    msm->maxlevels = maxlevels;
  }

  level = 0;
  done = 0;
  alldone = 0;
  do {
    GRID_RESIZE( &(msm->qh[level]), ia, ni, ja, nj, ka, nk);
    GRID_RESIZE( &(msm->eh[level]), ia, ni, ja, nj, ka, nk);

    if (++level == nlevels)    done |= 0x07;  /* user limit on levels */

    alldone = (done == 0x07);  /* make sure all dimensions are done */

    if (isclamped) {
      int nelems = ni * nj * nk;
      if (nelems <= lastnelems)  done |= 0x07;
    }

    if (ispx) {
      ni >>= 1;
      ib = ni-1;
      if (ni & 1)              done |= 0x07;  /* == 3 or 1 */
      else if (ni == 2)        done |= 0x01;  /* can do one more */
    }
    else {
      ia = -((-ia+1)/2) - nu;
      ib = (ib+1)/2 + nu;
      ni = ib - ia + 1;
      if (ni <= omega)         done |= 0x01;  /* can do more restrictions */
    }

    if (ispy) {
      nj >>= 1;
      jb = nj-1;
      if (nj & 1)              done |= 0x07;  /* == 3 or 1 */
      else if (nj == 2)        done |= 0x02;  /* can do one more */
    }
    else {
      ja = -((-ja+1)/2) - nu;
      jb = (jb+1)/2 + nu;
      nj = jb - ja + 1;
      if (nj <= omega)         done |= 0x02;  /* can do more restrictions */
    }

    if (ispz) {
      nk >>= 1;
      kb = nk-1;
      if (nk & 1)              done |= 0x07;  /* == 3 or 1 */
      else if (nk == 2)        done |= 0x04;  /* can do one more */
    }
    else {
      ka = -((-ka+1)/2) - nu;
      kb = (kb+1)/2 + nu;
      nk = kb - ka + 1;
      if (nk <= omega)         done |= 0x04;  /* can do more restrictions */
    }

  } while ( ! alldone );
  msm->nlevels = level;

  toplevel = (ispany ? msm->nlevels : msm->nlevels - 1);

  /* ellipsoid axes for lattice cutoff weights */
  ni = (int) ceilf(2*a/hx) - 1;
  nj = (int) ceilf(2*a/hy) - 1;
  nk = (int) ceilf(2*a/hz) - 1;
  scaling = 1;
  for (level = 0;  level < toplevel;  level++) {
    p = &(msm->gc[level]);
    GRID_RESIZE(p, -ni, 2*ni+1, -nj, 2*nj+1, -nk, 2*nk+1);

    if (0 == level) {
      index = 0;
      for (k = -nk;  k <= nk;  k++) {
        for (j = -nj;  j <= nj;  j++) {
          for (i = -ni;  i <= ni;  i++) {
            float s, t, gs, gt, g;
            s = ( (i*hx)*(i*hx) + (j*hy)*(j*hy) + (k*hz)*(k*hz) ) / (a*a);
            t = 0.25f * s;
            if (t >= 1) {
              g = 0;
            }
            else if (s >= 1) {
              gs = 1/sqrtf(s);
              SPOLY(&gt, t, split);
              g = (gs - 0.5f * gt) / a;
            }
            else {
              SPOLY(&gs, s, split);
              SPOLY(&gt, t, split);
              g = (gs - 0.5f * gt) / a;
            }
            GRID_INDEX_CHECK(p, i, j, k);
            ASSERT( p->buffer + index == p->data + GRID_INDEX(p, i, j, k) );
            p->buffer[index] = g;
            index++;
          }
        }
      } /* end loops over k-j-i */
    }
    else {
      /* set each level as scaling of h-level */
      const floatGrid *first = &(msm->gc[0]);
      scaling *= 0.5f;
      index = 0;
      for (k = -nk;  k <= nk;  k++) {
        for (j = -nj;  j <= nj;  j++) {
          for (i = -ni;  i <= ni;  i++) {
            GRID_INDEX_CHECK(p, i, j, k);
            ASSERT( p->buffer + index == p->data + GRID_INDEX(p, i, j, k) );
            p->buffer[index] = scaling * first->buffer[index];
            index++;
          }
        }
      }
    }
  } /* end loop over levels */

  if (toplevel < msm->nlevels) {
    /* nonperiodic in all dimensions,
     * calculate top level weights, ellipsoid axes are length of lattice */
    const floatGrid *qhtop = &(msm->qh[toplevel]);
    ni = qhtop->ni - 1;
    nj = qhtop->nj - 1;
    nk = qhtop->nk - 1;
    p = &(msm->gc[toplevel]);
    GRID_RESIZE(p, -ni, 2*ni+1, -nj, 2*nj+1, -nk, 2*nk+1);
    scaling *= 0.5f;
    index = 0;
    for (k = -nk;  k <= nk;  k++) {
      for (j = -nj;  j <= nj;  j++) {
        for (i = -ni;  i <= ni;  i++) {
          float s, gs;
          s = ( (i*hx)*(i*hx) + (j*hy)*(j*hy) + (k*hz)*(k*hz) ) / (a*a);
          if (s >= 1) {
            gs = 1/sqrtf(s);
          }
          else {
            SPOLY(&gs, s, split);
          }
          GRID_INDEX_CHECK(p, i, j, k);
          ASSERT( p->buffer + index == p->data + GRID_INDEX(p, i, j, k) );
          p->buffer[index] = scaling * gs/a;
          index++;
        }
      }
    } /* end loops over k-j-i for coarsest level weights */
  }

  return 0;
}
