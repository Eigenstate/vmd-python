#if defined(MGRID)

#include "energythr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include "mgrid/mgrid.h"

#include "util.h"    /* timer code taken from Tachyon */
#include "threads.h" /* threads code taken from Tachyon */

#define ENERGY(x)  printf("DEBUG: %g\n", (double)x)

#define CUBIC_TAYLOR2
#undef QUINTIC1_TAYLOR3
#undef HEPTIC1_TAYLOR4
#undef HERMITE_TAYLOR3

#define CUTOFF 12.0

#define ASSERT(expr) \
  do { \
    if (!(expr)) { \
      printf("ASSERT(%s, line %d): %s\n", __FILE__, __LINE__, #expr); \
      abort(); \
    } \
  } while (0)


typedef struct {
  int threadid;
  int threadcount;
  float* atoms;
  float* grideners;
  long int numplane;
  long int numcol;
  long int numpt;
  long int natoms;
  float gridspacing;
  unsigned char* excludepos;
} enthrparms;

/* thread prototype */
static void * energythread(void *);

int calc_grid_energies_excl_mgrid(float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, long int natoms, float gridspacing, unsigned char* excludepos, int maxnumprocs) {
// cionize_params* params, cionize_molecule* molecule, cionize_grid* grid) {
  int i;
  enthrparms *parms;
  rt_thread_t * threads;

#if defined(THR)
  int numprocs;
  int availprocs = rt_thread_numprocessors();
  if (params->maxnumprocs <= availprocs) {
    numprocs = params->maxnumprocs;
  } else {
    numprocs = availprocs;
  }
#else
  int numprocs = 1;
#endif

  printf("calc_grid_energies_excl_mgrid()\n");

  /* DH: setup and compute long-range */
  MgridParam mg_prm;
  MgridSystem mg_sys;
  Mgrid mg;
  //MgridLattice *lattice;
  int j, k;
  int gridfactor;

  double h, h_1;
  int nspacings;

  double *eh;
  int ndim;
  int ii, jj, kk;
  int im, jm, km;
  int ilo, jlo, klo;
  double dx_h, dy_h, dz_h;
  double xphi[4], yphi[4], zphi[4];
  double t, en, c;
  int koff, jkoff, index;

  rt_timerhandle timer;
  float totaltime;

  memset(&mg_prm, 0, sizeof(mg_prm));
  memset(&mg_sys, 0, sizeof(mg_sys));

  /*
   * set mgrid parameters
   *
   * make sure origin of mgrid's grid is at ionize's grid origin (0,0,0)
   *
   * length is (number of grid spacings) * (grid spacing),
   * where the number of spacings is one less than number of grid points
   */
  mg_prm.length = (numpt-1) * gridspacing;
  mg_prm.center.x = 0.5 * mg_prm.length;
  mg_prm.center.y = 0.5 * mg_prm.length;
  mg_prm.center.z = 0.5 * mg_prm.length;

  /* make sure domain and grid are both cubic */
  if (numpt != numcol || numcol != numplane) {
    printf("ERROR: grid must be cubic\n");
    return -1;
  }

  /*
   * grid used by mgrid needs spacing h >= 2
   *
   * determine grid factor:  (2^gridfactor)*h_ionize = h_mgrid
   */
  gridfactor = 0;
  //nspacings = numpt - 1;
  nspacings = numpt;
    /* add one more spacing so that interpolation loop below will work */
  h = gridspacing;
  while (h < 2.0) {
    h *= 2;
    nspacings = ((nspacings & 1) ? nspacings/2 + 1 : nspacings/2);
    gridfactor++;
  }
  mg_prm.nspacings = nspacings;

  /* have to modify mgrid length */
  mg_prm.length += h;
  mg_prm.center.x = 0.5 * mg_prm.length;
  mg_prm.center.y = 0.5 * mg_prm.length;
  mg_prm.center.z = 0.5 * mg_prm.length;
  h_1 = 1.0/h;

  mg_prm.cutoff = CUTOFF;
  mg_prm.boundary = MGRID_NONPERIODIC;

  /* choice of splitting must be consistent with short-range below */
#if defined(CUBIC_TAYLOR2)
  mg_prm.approx = MGRID_CUBIC;
  mg_prm.split = MGRID_TAYLOR2;
#elif defined(QUINTIC1_TAYLOR3)
  mg_prm.approx = MGRID_QUINTIC1;
  mg_prm.split = MGRID_TAYLOR3;
#elif defined(HEPTIC1_TAYLOR4)
  mg_prm.approx = MGRID_HEPTIC1;
  mg_prm.split = MGRID_TAYLOR4;
  /*
#elif defined(HERMITE_TAYLOR3)
  mg_prm.approx = MGRID_HERMITE;
  mg_prm.split = MGRID_TAYLOR3;
  */
#endif

  mg_prm.natoms = natoms;
  printf("natom = %d\n", mg_prm.natoms);
  printf("mgrid center = %g %g %g\n",
      mg_prm.center.x, mg_prm.center.y, mg_prm.center.z);
  printf("mgrid length = %g\n", mg_prm.length);
  printf("mgrid nspacings = %d\n", mg_prm.nspacings);

  /* setup mgrid system */
  mg_sys.f_elec = (MD_Dvec *) calloc(mg_prm.natoms, sizeof(MD_Dvec));
  mg_sys.pos = (MD_Dvec *) calloc(mg_prm.natoms, sizeof(MD_Dvec));
  mg_sys.charge = (double *) calloc(mg_prm.natoms, sizeof(double));
  for (i = 0;  i < mg_prm.natoms;  i++) {
    mg_sys.pos[i].x  = atoms[4*i    ];
    mg_sys.pos[i].y  = atoms[4*i + 1];
    mg_sys.pos[i].z  = atoms[4*i + 2];
    mg_sys.charge[i] = atoms[4*i + 3];
  }

  /* setup mgrid solver and compute */
  if (mgrid_param_config(&mg_prm)) {
    printf("ERROR: mgrid_param_config() failed\n");
    return -1;
  }
  printf("spacing = %g\n", mg_prm.spacing);
  if (mgrid_init(&mg)) {
    printf("ERROR: mgrid_init() failed\n");
    return -1;
  }
  if (mgrid_setup(&mg, &mg_sys, &mg_prm)) {
    printf("ERROR: mgrid_setup() failed\n");
    return -1;
  }

  timer = rt_timer_create();
  rt_timer_start(timer);

  if (mgrid_force(&mg, &mg_sys)) {
    printf("ERROR: mgrid_force() failed\n");
    return -1;
  }
  /* DH: end setup and compute long-range */

  printf("  using %d processors\n", numprocs);  

  /* allocate array of threads */
  threads = (rt_thread_t *) calloc(numprocs * sizeof(rt_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (enthrparms *) malloc(numprocs * sizeof(enthrparms));
  for (i=0; i<numprocs; i++) {
    parms[i].threadid = i;
    parms[i].threadcount = numprocs;
    parms[i].atoms = atoms;
    parms[i].grideners = grideners;
    parms[i].numplane = numplane;
    parms[i].numcol = numcol;
    parms[i].numpt = numpt;
    parms[i].natoms = natoms;
    parms[i].gridspacing = gridspacing;
    parms[i].excludepos = excludepos;
  }

#if defined(THR)
  /* spawn child threads to do the work */
  for (i=0; i<numprocs; i++) {
    rt_thread_create(&threads[i], energythread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numprocs; i++) {
    rt_thread_join(threads[i], NULL);
  } 
#else
  /* single thread does all of the work */
  energythread((void *) &parms[0]);
#endif

  /* DH: tabulate and cleanup long-range */

  /* interpolate from mgrid potential lattice */

  eh = (double *)(mg.egrid[0].data); /* mgrid's long-range potential lattice */
  ndim = mg.egrid[0].ni;  /* number of points in each dimension of lattice */

  for (kk = 0;  kk < numplane;  kk++) {
    for (jj = 0;  jj < numcol;  jj++) {
      for (ii = 0;  ii < numpt;  ii++) {

        /* distance between atom and corner measured in grid points */
        dx_h = (ii*gridspacing) * h_1;
        dy_h = (jj*gridspacing) * h_1;
        dz_h = (kk*gridspacing) * h_1;

        /* find closest mgrid lattice point less than or equal to */
        im = ii >> gridfactor;
        jm = jj >> gridfactor;
        km = kk >> gridfactor;

#if defined(CUBIC_TAYLOR2)
        ilo = im-1;
        jlo = jm-1;
        klo = km-1;

        /* find t for x dimension and compute xphi */
        t = dx_h - ilo;
        xphi[0] = 0.5 * (1 - t) * (2 - t) * (2 - t);
        t--;
        xphi[1] = (1 - t) * (1 + t - 1.5 * t * t);
        t--;
        xphi[2] = (1 + t) * (1 - t - 1.5 * t * t);
        t--;
        xphi[3] = 0.5 * (1 + t) * (2 + t) * (2 + t);

        /* find t for y dimension and compute yphi */
        t = dy_h - jlo;
        yphi[0] = 0.5 * (1 - t) * (2 - t) * (2 - t);
        t--;
        yphi[1] = (1 - t) * (1 + t - 1.5 * t * t);
        t--;
        yphi[2] = (1 + t) * (1 - t - 1.5 * t * t);
        t--;
        yphi[3] = 0.5 * (1 + t) * (2 + t) * (2 + t);

        /* find t for z dimension and compute zphi */
        t = dz_h - klo;
        zphi[0] = 0.5 * (1 - t) * (2 - t) * (2 - t);
        t--;
        zphi[1] = (1 - t) * (1 + t - 1.5 * t * t);
        t--;
        zphi[2] = (1 + t) * (1 - t - 1.5 * t * t);
        t--;
        zphi[3] = 0.5 * (1 + t) * (2 + t) * (2 + t);

        /* determine 64=4*4*4 eh grid stencil contribution to potential */
        en = 0;
        for (k = 0;  k < 4;  k++) {
          koff = (k + klo) * ndim;
          for (j = 0;  j < 4;  j++) {
            jkoff = (koff + (j + jlo)) * ndim;
            c = yphi[j] * zphi[k];
            for (i = 0;  i < 4;  i++) {
              index = jkoff + (i + ilo);
              /*
              ASSERT(&eh[index]
                  == mgrid_lattice_elem(&(mg.egrid[0]), i+ilo, j+jlo, k+klo));
                  */
              if (&eh[index] !=
                  mgrid_lattice_elem(&(mg.egrid[0]), i+ilo, j+jlo, k+klo)) {
                printf("ndim=%d  index=%d  i+ilo=%d  j+jlo=%d  k+klo=%d\n"
                    "ia=%d  ib=%d  ni=%d\n"
                    "ja=%d  jb=%d  nj=%d\n"
                    "ka=%d  kb=%d  nk=%d\n",
                    ndim, index, i+ilo, j+jlo, k+klo,
                    mg.egrid[0].ia, mg.egrid[0].ib, mg.egrid[0].ni,
                    mg.egrid[0].ja, mg.egrid[0].jb, mg.egrid[0].nj,
                    mg.egrid[0].ka, mg.egrid[0].kb, mg.egrid[0].nk
                    );
                abort();
              }

              en += eh[index] * xphi[i] * c;
            }
          }
        }
        /* end CUBIC */
#endif

        //ENERGY(grid->eners[k*numcol*numpt + j*numpt + i]);

        grideners[kk*numcol*numpt + jj*numpt + ii] += (float)en;

        //  (float) *((double *)mgrid_lattice_elem(lattice, i, j, k));

        //ENERGY((float) *((double *)mgrid_lattice_elem(lattice, i, j, k)));
        //ENERGY(grid->eners[k*numcol*numpt + j*numpt + i]*560.47254);
      }
    }
  }

  totaltime = rt_timer_timenow(timer);
  printf("total time for mgrid: %.1f\n", totaltime);
  rt_timer_destroy(timer);

  /* cleanup mgrid */
  mgrid_done(&mg);
  free(mg_sys.f_elec);
  free(mg_sys.pos);
  free(mg_sys.charge);
  /* DH: tabulate and cleanup long-range */

  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}


static void * energythread(void *voidparms) {
  enthrparms *parms = (enthrparms *) voidparms;
  /* 
   * copy in per-thread parameters 
   */
  const float *atoms = parms->atoms;
  float* grideners = parms->grideners;
  const long int numplane = parms->numplane;
  const long int numcol = parms->numcol;
  const long int numpt = parms->numpt;
  const long int natoms = parms->natoms;
  const float gridspacing = parms->gridspacing;
  const unsigned char* excludepos = parms->excludepos;
  const int threadid = parms->threadid;
  const int threadcount = parms->threadcount;

  /* Calculate the coulombic energy at each grid point from each atom
   * This is by far the most time consuming part of the process
   * We iterate over z,y,x, and then atoms
   * This function is the same as the original calc_grid_energies, except
   * that it utilizes the exclusion grid
   */
  int i,j,k,n; //Loop counters
  //float lasttime;
  float totaltime;

  // Holds atom x, dy**2+dz**2, and atom q as x/r/q/x/r/q...
  float * xrq = (float *) malloc(3*natoms * sizeof(float)); 
  int maxn = natoms * 3;

  rt_timerhandle timer = rt_timer_create();
  rt_timer_start(timer);

  //printf("thread %d started...\n", threadid);

  // For each point in the cube...
  for (k=threadid; k<numplane; k+= threadcount) {
    const float z = gridspacing * (float) k;
    //lasttime = rt_timer_timenow(timer);
    for (j=0; j<numcol; j++) {
      const float y = gridspacing * (float) j;
      long int voxaddr = numcol*numpt*k + numpt*j;

      // Prebuild a table of dy and dz values on a per atom basis
      for (n=0; n<natoms; n++) {
        int addr3 = n*3;
        int addr4 = n*4;
        float dy = y - atoms[addr4 + 1];
        float dz = z - atoms[addr4 + 2];
        xrq[addr3    ] = atoms[addr4];
        xrq[addr3 + 1] = dz*dz + dy*dy;
        xrq[addr3 + 2] = atoms[addr4 + 3];
      }

#if defined(__INTEL_COMPILER)
// help the vectorizer make reasonable decisions (used prime to keep it honest)
#pragma loop count(1009)
#endif
      for (i=0; i<numpt; i++) {
        // Check if we're on an excluded point, and skip it if we are
        if (excludepos[voxaddr + i] == 0) {
          float energy = 0; // Energy of current grid point
          const float x = gridspacing * (float) i;

#if defined(__INTEL_COMPILER)
// help the vectorizer make reasonable decisions
#pragma vector always 
#endif
          // Calculate the interaction with each atom
#if 1
          for (n=0; n<maxn; n+=3) {
            float dx = x - xrq[n];
            // DH: short-range part: 1/r - g(r)
            float r2 = dx*dx + xrq[n+1];
            if (r2 < CUTOFF*CUTOFF) {
              float s = r2/(CUTOFF*CUTOFF);
#if defined(CUBIC_TAYLOR2)
              energy += xrq[n + 2]*(1./sqrtf(r2) - (1./CUTOFF)*(1+(s-1)*(-1./2+(s-1)*(3./8))));
#elif defined(QUINTIC1_TAYLOR3)
              energy += xrq[n + 2]*(1./sqrtf(r2) - (1./CUTOFF)*(1 + (s-1)*(-1./2 + (s-1)*(3./8 + (s-1)*(-5./16)))));
#elif defined(HEPTIC1_TAYLOR4)
              energy += xrq[n + 2]*(1./sqrtf(r2) - (1./CUTOFF)*(1 + (s-1)*(-1./2 + (s-1)*(3./8 + (s-1)*(-5./16 + (s-1)*(35./128))))));
#elif defined(HERMITE_TAYLOR3)
              energy += xrq[n + 2]*(1./sqrtf(r2) - (1./CUTOFF)*(1 + (s-1)*(-1./2 + (s-1)*(3./8 + (s-1)*(-5./16)))));
#endif
            }
          }
#endif
          //ENERGY(energy);
          grideners[voxaddr + i] = energy;
        }
      }
    }
#if 0
    totaltime = rt_timer_timenow(timer);
    printf("thread[%d] plane %d/%ld time %.2f, elapsed %.1f, est. total: %.1f\n",
           threadid, k, numplane,

           totaltime - lasttime, totaltime, 
           totaltime * numplane / (k+1));
#endif
  }

  totaltime = rt_timer_timenow(timer);
  printf("total time for short-range part: %.1f\n", totaltime);
  rt_timer_destroy(timer);
  free(xrq);

  return NULL;
}

#endif

