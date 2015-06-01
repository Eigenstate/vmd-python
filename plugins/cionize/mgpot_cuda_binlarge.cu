#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEBUG
#undef DEBUG
#include "mgpot_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "util.h"


/*** domain decomposition parameters ***/
/*
 * VX, VY, VZ - Coulombic lattice is subdivided into blocks of voxels,
 *              block is data set operated on by CUDA kernel invocation,
 *              voxel is smallest unit of work (here it is a lattice point)
 *
 * BX, BY, BZ - dimensions of thread block,
 *              each thread is assigned at least one voxel
 *
 * NBX, NBY, NBZ - voxel block is subdivided into thread blocks,
 *                 these are the dimensions of this decomposition
 *
 * GX, GY, GZ - thread blocks are officially grouped into grids
 *              unfortunately, GZ is always 1 (i.e. this is a *video* card)
 *              so for a 3D problem, these need to be remapped into the
 *              NBX*NBY*NBZ space
 *
 * UNROLL     - unrolling factor for voxels along z-direction
 */

/* unrolling factor */
#define UNROLL  3

/* voxel block dimensions */
#if defined(MGPOT_SPACING_1_0)
#define VX  24
#define VY  24
#define VZ  24
#elif defined(MGPOT_SPACING_0_2)
#define VX  96
#define VY  96
#define VZ  96
#else
#define VX  48
#define VY  48
#define VZ  48
#endif

/* thread block dimensions */
#define BX  4
#define BY  4
#define BZ  8

/* number of thread blocks across voxel block */
#define NBX  (VX/BX)
#define NBY  (VY/BY)
#define NBZ  (VZ/BZ/UNROLL)

/* grid dimensions */
#define GX  NBX
#define GY  NBY*NBZ
#define GZ  1


/*** method parameters ***/

#if defined(MGPOT_SPACING_1_0)
#define GRIDSPACING  1.0f
#define CUTOFF       12.f
#elif defined(MGPOT_SPACING_0_2)
#define GRIDSPACING  0.2f
#define CUTOFF       9.6f
#else
#define GRIDSPACING  0.5f
#define CUTOFF       12.f
#endif

#define CUTOFF2      (CUTOFF*CUTOFF)
#define INV_CUTOFF   (1.f/CUTOFF)
#define INV_CUTOFF2  (1.f/CUTOFF2)
#define GC0  (1.875f*INV_CUTOFF)
#define GC1  (-1.25f*INV_CUTOFF*INV_CUTOFF2)
#define GC2  (0.375f*INV_CUTOFF*INV_CUTOFF2*INV_CUTOFF2)



/* CUDA version of multilevel short-range contribution to grid potential */

static __constant__ float4 atominfo[MAXATOMS];

__global__ static void mgpot_shortrng_energy(
    int natoms,                       /* number of atoms */
    int ncx, int ncy,                 /* x and y sizes of output lattice */
    int xlo, int ylo, int zlo,        /* lowest numbered point in subgrid */
    float *outenergy) {               /* the output lattice */
  /*
   * find 3D index (xg,yg,zg) for this grid block
   *
   * this could be made faster by choosing nbx and (nbx*nby)
   * to be powers of 2 and then passing their respective exponents
   */
  unsigned int zg = blockIdx.y / NBY;
  unsigned int yg = blockIdx.y % NBY;
  unsigned int xg = blockIdx.x;

  /* find 3D index (xi,yi,zi) within the output lattice for this thread */
  unsigned int xi = (__umul24(xg,blockDim.x) + threadIdx.x) + xlo;
  unsigned int yi = (__umul24(yg,blockDim.y) + threadIdx.y) + ylo;
  // unsigned int zi = (__umul24(zg,blockDim.z) + threadIdx.z) + zlo;
  unsigned int zi = (__umul24(zg,blockDim.z) + threadIdx.z);

  /* find corresponding index into flat output array */
  /* can't represent the second multiply using 24-bit integers */
  unsigned int index_z0 = (__umul24((zi),ncy) + yi)*ncx + xi;
  unsigned int index_z1 = (__umul24((zi+NBZ*BZ),ncy) + yi)*ncx + xi;
  unsigned int index_z2 = (__umul24((zi+2*NBZ*BZ),ncy) + yi)*ncx + xi;

  /* start read early */
  float current_energy_z0 = outenergy[index_z0];
  float current_energy_z1 = outenergy[index_z1];
  float current_energy_z2 = outenergy[index_z2];

  /* coordinate of this grid point */
  float coorx = GRIDSPACING * xi;
  float coory = GRIDSPACING * yi;
  // float coorz = GRIDSPACING * zi;
  float coorz = GRIDSPACING * (zi + zlo);

  int n;
  float accum_energy_z0 = 0.f;
  float accum_energy_z1 = 0.f;
  float accum_energy_z2 = 0.f;

  /* add interactions from all provided atoms into this grid point */
  for (n = 0;  n < natoms;  n++) {
    float dx = coorx - atominfo[n].x;
    float dy = coory - atominfo[n].y;
    float dz = coorz - atominfo[n].z;
    float q = atominfo[n].w;
    float dxdy2 = dx*dx + dy*dy;

    float r2 = dxdy2 + dz*dz;

    if (r2 < CUTOFF2) {
      /*
       * this is a good place for the branch, we appear to save
       * much work by having the warp all fail test together
       */
      float gr2 = GC0 + r2*(GC1 + r2*GC2);  /* TAYLOR2 softening */
      float r_1 = 1.f/sqrtf(r2);
      accum_energy_z0 += q * (r_1 - gr2);
    }

    dz += (NBZ*BZ*GRIDSPACING);
    r2 = dxdy2 + dz*dz;

    if (r2 < CUTOFF2) {
      /*
       * this is a good place for the branch, we appear to save
       * much work by having the warp all fail test together
       */
      float gr2 = GC0 + r2*(GC1 + r2*GC2);  /* TAYLOR2 softening */
      float r_1 = 1.f/sqrtf(r2);
      accum_energy_z1 += q * (r_1 - gr2);
    }

    dz += (NBZ*BZ*GRIDSPACING);
    r2 = dxdy2 + dz*dz;

    if (r2 < CUTOFF2) {
      /*
       * this is a good place for the branch, we appear to save
       * much work by having the warp all fail test together
       */
      float gr2 = GC0 + r2*(GC1 + r2*GC2);  /* TAYLOR2 softening */
      float r_1 = 1.f/sqrtf(r2);
      accum_energy_z2 += q * (r_1 - gr2);
    }

  }
  outenergy[index_z0] = current_energy_z0 + accum_energy_z0;
  outenergy[index_z1] = current_energy_z1 + accum_energy_z1;
  outenergy[index_z2] = current_energy_z2 + accum_energy_z2;
}


/*****************************************************************************/

struct sender_t {
  float atom[4*MAXATOMS];
  int atomcnt;
};

static void reset_atoms(struct sender_t *s)
{
  s->atomcnt = 0;
}

static int append_atoms(struct sender_t *s, float *a, int n)
{
  if (n + s->atomcnt > MAXATOMS) return -1;
  memcpy(s->atom + 4*s->atomcnt, a, 4*n*sizeof(float));
  s->atomcnt += n;
  return 0;
}

static int send_atoms(struct sender_t *s)
{
  DEBUG( printf("sending %d atoms\n", s->atomcnt); );
  cudaMemcpyToSymbol(atominfo, s->atom, s->atomcnt*4*sizeof(float), 0);
  CUERR;  /* check and clear any existing errors */
  return 0;
}


/*****************************************************************************/

int mgpot_cuda_binlarge(Mgpot *mg) {

  /* for clustering grid cells */
  struct sender_t sender;

  float *host_epot = mg->host_epot;
  float *device_epot_slab = mg->device_epot_slab;

  MgpotLargeBin *bin = mg->largebin;

  dim3 gsize, bsize;

  const long allnx = mg->allnx;
  const long allny = mg->allny;
  const long slabsz = mg->slabsz;
  long memsz;

  const int nxsub = mg->nxsub;
  const int nysub = mg->nysub;
  const int nzsub = mg->nzsub;

  const int nxbin = mg->nxbin;
  const int nybin = mg->nybin;

  int xlo, ylo, zlo;
  int i, j, k, ib, jb, kb, index;

  /* timers */
  rt_timerhandle runtimer, hashtimer;
  float runtotal, hashtotal;
  int iterations, invocations;

  gsize.x = GX;  /* cuda kernel grid size */
  gsize.y = GY;
  gsize.z = GZ;
  bsize.x = BX;  /* cuda kernel block size */
  bsize.y = BY;
  bsize.z = BZ;

  ASSERT(VX == slabsz);

  ASSERT(GRIDSPACING == mg->gridspacing);
  ASSERT(CUTOFF == mg->a);

  /* initialize wall clock timers */
  runtimer = rt_timer_create();
  hashtimer = rt_timer_create();
  runtotal = 0;
  hashtotal = 0;
  iterations = 0;
  invocations = 0;

  memsz = sizeof(float) * allnx * allny * slabsz;

  /*
   * loop over subcubes
   */
  rt_timer_start(hashtimer);
  reset_atoms(&sender);
  for (k = 0;  k < nzsub;  k++) {

    /* clear memory buffer to accumulate another slab */
    cudaMemset(device_epot_slab, 0, memsz);
    CUERR;  /* check and clear any existing errors */

    for (j = 0;  j < nysub;  j++) {
      for (i = 0;  i < nxsub;  i++) {

        /* lowest numbered point in subcube */
        xlo = i * slabsz;
        ylo = j * slabsz;
        zlo = k * slabsz;

        /*
         * for each subcube, calculate interactions with atoms in the
         * 8 grid cell regions that contain this subcube + cutoff margin
         */
        for (kb = k;  kb <= k+1;  kb++) {
          for (jb = j;  jb <= j+1;  jb++) {
            for (ib = i;  ib <= i+1;  ib++) {
              index = (kb*nybin + jb)*nxbin + ib;
              if (append_atoms(&sender, bin[index].atom, bin[index].atomcnt)) {
                /* buffer is full, so send atoms to GPU buffer and run */
                send_atoms(&sender);
                rt_timer_start(runtimer);
                mgpot_shortrng_energy<<<gsize, bsize, 0>>>(
                    sender.atomcnt,
                    allnx, allny,
                    xlo, ylo, zlo,
                    device_epot_slab);
                CUERR;  /* check and clear any existing errors */
                rt_timer_stop(runtimer);
                runtotal += rt_timer_time(runtimer);
                invocations++;
                reset_atoms(&sender);
                /* have to append again since first try failed */
                append_atoms(&sender, bin[index].atom, bin[index].atomcnt);
              }
              iterations++;
            }
          }
        }
        /* have atoms in buffer, send and run */
        send_atoms(&sender);
        rt_timer_start(runtimer);
        mgpot_shortrng_energy<<<gsize, bsize, 0>>>(
            sender.atomcnt, 
            allnx, allny,
            xlo, ylo, zlo,
            device_epot_slab);
        CUERR;  /* check and clear any existing errors */
        rt_timer_stop(runtimer);
        runtotal += rt_timer_time(runtimer);
        invocations++;
        reset_atoms(&sender);
      }
    }

    /* copy epot slab from device memory back to host */
    cudaMemcpy(host_epot + k*slabsz*allny*allnx, device_epot_slab, memsz,
      cudaMemcpyDeviceToHost);
    CUERR;  /* check and clear any existing errors */

  } /* end subcube loop */
  rt_timer_stop(hashtimer);
  hashtotal += rt_timer_time(hashtimer);

  printf("Number of loop iterations:  %d\n", iterations);
  printf("Number of kernel invocations:  %d\n", invocations);
  printf("Kernel time: %f seconds, %f per iteration\n",
    runtotal, runtotal / (float) iterations);
  printf("Hash & Loop:  %f seconds\n", hashtotal);
  printf("CPU overhead: %f seconds\n", hashtotal-runtotal);

  return 0;
}

#ifdef __cplusplus
}
#endif
