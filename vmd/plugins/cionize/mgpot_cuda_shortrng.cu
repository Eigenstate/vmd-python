#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "util.h"
}

#define DEBUG
#undef DEBUG
#include "mgpot_cuda.h"


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

/* cuda code */
#if 0
#define CUERR \
  do { \
    cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
      printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
      return FAIL; \
    } \
  } while (0)
#endif

#define MAXATOMS 4000
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

/*
 * used for very coarse hashing of 24 A^3 regions of space;
 * tile these in space, offset by 12 A cutoff along each dimension
 * from the grid point blocks being updated by the GPU,
 * so that blocks of 8 grid cells completely cover the grid
 * point block plus a 12 A cutoff margin in each direction
 *
 * note that 24^3/10 = 1382.4 expected atoms;
 * the extra memory below, although seeming excessive, is still
 * less than 10% of the total memory required for a large grid,
 * e.g. the virus test problem
 */
struct gridcell_t {
  int atomcnt;
  float x0, y0, z0;    /* lowest spatial corner */
  float atom[4*2000];  /* coordinates stored x/y/z/q */
};


/*
 * create the grid cell array and hash the atoms
 * dimensions are set according to properly covering the lattice,
 * as described above
 */
int gridcell_atom_hashing(
    int allnx, int allny, int allnz,          /* lattice origin at (0,0,0) */
    float gridspacing,
    float cutoff,
    int natoms,
    const float *atom,                        /* coordinates stored x/y/z/q */
    int *nxcells, int *nycells, int *nzcells,
    struct gridcell_t **cell)
{
  float cellen = 2 * cutoff;
  float xlen = allnx * gridspacing + cutoff;
  float ylen = allny * gridspacing + cutoff;
  float zlen = allnz * gridspacing + cutoff;
  float xmin = -cutoff;
  float ymin = -cutoff;
  float zmin = -cutoff;
  int ncells, nx, ny, nz;
  int n, i, j, k, index, aindex;
  struct gridcell_t *c;

  /* allocate cells */
  *nxcells = nx = (int) ceilf(xlen / cellen);
  *nycells = ny = (int) ceilf(ylen / cellen);
  *nzcells = nz = (int) ceilf(zlen / cellen);
  ncells = nx * ny * nz;
  *cell = c = (struct gridcell_t *) calloc(ncells, sizeof(gridcell_t));
  if (NULL==c) {
    return ERROR("calloc() failed\n");
  }

  /* prepare cells */
  for (k = 0;  k < nz;  k++) {
    for (j = 0;  j < ny;  j++) {
      for (i = 0;  i < nx;  i++) {
        index = (k*ny + j)*nx + i;
        c[index].x0 = xmin + i*cellen;
        c[index].y0 = ymin + j*cellen;
        c[index].z0 = zmin + k*cellen;
      }
    }
  }

  /* hash atoms */
  for (n = 0;  n < 4*natoms;  n += 4) {
    i = (int) floorf((atom[n  ] - xmin) / cellen);
    j = (int) floorf((atom[n+1] - ymin) / cellen);
    k = (int) floorf((atom[n+2] - zmin) / cellen);
    if (i < 0) i = 0;
    else if (i >= nx) i = nx-1;
    if (j < 0) j = 0;
    else if (j >= ny) j = ny-1;
    if (k < 0) k = 0;
    else if (k >= nz) k = nz-1;
    index = (k*ny + j)*nx + i;
    aindex = 4*c[index].atomcnt;
    c[index].atom[aindex  ] = atom[n  ];
    c[index].atom[aindex+1] = atom[n+1];
    c[index].atom[aindex+2] = atom[n+2];
    c[index].atom[aindex+3] = atom[n+3];
    c[index].atomcnt++;
  }

#ifdef DEBUGGING
  /*
   * some sanity checks:
   * loop over cells and make sure that
   *   1. all atoms are within the correct cell
   *   2. the sum over all atomcnt counters equals natoms
   */
  {
    int atomsum = 0;
    for (k = 0;  k < nz;  k++) {
      for (j = 0;  j < ny;  j++) {
        for (i = 0;  i < nx;  i++) {
          int index = (k*ny + j)*nx + i;
          float xmin = c[index].x0;
          float ymin = c[index].y0;
          float zmin = c[index].z0;
          atomsum += c[index].atomcnt;
          for (n = 0;  n < 4*c[index].atomcnt;  n += 4) {
            float x = c[index].atom[n  ];
            float y = c[index].atom[n+1];
            float z = c[index].atom[n+2];
            if ((x < xmin && i != 0)
                || (x >= xmin+cellen && i != nx-1)
                || (y < ymin && j != 0)
                || (y >= ymin+cellen && j != ny-1)
                || (z < zmin && k != 0)
                || (z >= zmin+cellen && k != nz-1)) {
              return FAIL;
            }
          } /* loop n */

        }
      }
    } /* loop i, j, k */

  } /* end block */
#endif

  return 0;
}


/*****************************************************************************/

struct sender_t {
  float atom[4*MAXATOMS];
  int atomcnt;
};

void reset_atoms(struct sender_t *s)
{
  s->atomcnt = 0;
}

int append_atoms(struct sender_t *s, float *a, int n)
{
  if (n + s->atomcnt > MAXATOMS) return -1;
  memcpy(s->atom + 4*s->atomcnt, a, 4*n*sizeof(float));
  s->atomcnt += n;
  return 0;
}

int send_atoms(struct sender_t *s)
{
  DEBUG( printf("sending %d atoms\n", s->atomcnt); );
  cudaMemcpyToSymbol(atominfo, s->atom, s->atomcnt*4*sizeof(float), 0);
  CUERR;  /* check and clear any existing errors */
  return 0;
}


#ifdef MAIN
/* generate random test case */
int init_atoms(float **atombuf, int count, dim3 volsize, float gridspacing)
{
  dim3 size;
  int i;
  float *atom;

  atom = (float *) malloc(count * 4 * sizeof(float));
  *atombuf = atom;

  /* compute grid dimensions in Angstroms */
  size.x = gridspacing * volsize.x;
  size.y = gridspacing * volsize.y;
  size.z = gridspacing * volsize.z;

  for (i = 0;  i < count;  i++) {
    int addr = i * 4;
    atom[addr  ] = (rand() / (float) RAND_MAX) * size.x;
    atom[addr+1] = (rand() / (float) RAND_MAX) * size.y;
    atom[addr+2] = (rand() / (float) RAND_MAX) * size.z;
    atom[addr+3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  /* charge */
  }

  return 0;
}
#endif


/*****************************************************************************/

#ifdef MAIN
int main(void)
#else
extern "C"
int mgpot_cuda_largebin(Mgpot *mg, const float *atom, float *gridener,
    long int numplane, long int numcol, long int numpt, long int natoms)
#endif
{
  /* use with these method parameters */
  const float gridspacing = GRIDSPACING;
  const float cutoff = CUTOFF;

#ifdef MAIN
  long int allnx = 96;
  long int allny = 96;
  long int allnz = 96;
  long int natoms = 5000;
  float *atom = NULL;
#else
  /* have to round up lattice dimensions to next multiple of 48 */
  long int allnx = (long int) (48.f * ceilf(numpt / 48.f));
  long int allny = (long int) (48.f * ceilf(numcol / 48.f));
  long int allnz = (long int) (48.f * ceilf(numplane / 48.f));
#endif

  float *outenergy = NULL;
  float *savenergy = NULL;

  dim3 allsize, volsize, gsize, bsize;

#if 0
  int deviceCount, dev;
#endif
  long int memsz;
  int xlo, ylo, zlo;
  int nsubx, nsuby, nsubz;
  int i, j, k, ic, jc, kc, index;

  /* for grid cell hashing */
  int nxcells, nycells, nzcells;
  struct gridcell_t *cell;

  /* for clustering grid cells */
  struct sender_t sender;

  /* timers */
  rt_timerhandle runtimer, hashtimer;
  float runtotal, hashtotal;
  int iterations, invocations;

#if 0
  cudaError_t rc;

  deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Detected %d CUDA accelerators:\n", deviceCount);
  for (dev = 0;  dev < deviceCount;  dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("  CUDA device[%d]: '%s'  Mem: %dMB  Rev: %d.%d\n",
        dev, deviceProp.name, deviceProp.totalGlobalMem / (1024*1024),
        deviceProp.major, deviceProp.minor);
  }
  dev = 0;  /* use the default device */

if (getenv("CUDADEV"))
  sscanf(getenv("CUDADEV"), "%d", &dev);
  

  printf("  Opening CUDA device %d...\n", dev);
  rc = cudaSetDevice(dev);
  if (rc != cudaSuccess) {
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError(); // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess)
      return FAIL; // abort and return an error
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }
  CUERR;  /* check and clear any existing errors */
#endif

  /* 3D grid */
  allsize.x = allnx;
  allsize.y = allny;
  allsize.z = allnz;

  /*
   * set up CUDA grid and block sizes to access subgrid
   */
  volsize.x = VX;
  volsize.y = VY;
  volsize.z = VZ;
  bsize.x = BX;
  bsize.y = BY;
  bsize.z = BZ;
  gsize.x = GX;
  gsize.y = GY;
  gsize.z = GZ;

  /* dimensions of subcubes within lattice */
  nsubx = allsize.x / volsize.x;
  nsuby = allsize.y / volsize.y;
  nsubz = allsize.z / volsize.z;

  printf("CUDA subgrid size: %d x %d x %d\n", volsize.x, volsize.y, volsize.z);
  printf("Entire grid size:  %d x %d x %d\n", allsize.x, allsize.y, allsize.z);
  printf("Number of atoms:  %d\n", natoms);

  /* initialize wall clock timers */
  runtimer = rt_timer_create();
  hashtimer = rt_timer_create();
  runtotal = 0;
  hashtotal = 0;
  iterations = 0;
  invocations = 0;

#ifdef MAIN
  /* allocate and initialize atom coordinates and charges */
  if (init_atoms(&atom, natoms, allsize, gridspacing)) {
    return FAIL;
  }
  printf("Allocating %.2fMB of memory for the atoms...\n",
      natoms*4*sizeof(float) / (1024.0 * 1024.0));
#endif

  /* allocate grid cells and hash the atoms */
  rt_timer_start(hashtimer);
  if (gridcell_atom_hashing(allnx, allny, allnz, gridspacing, cutoff,
        natoms, atom, &nxcells, &nycells, &nzcells, &cell)) {
    return FAIL;
  }
  rt_timer_stop(hashtimer);
  hashtotal += rt_timer_time(hashtimer);
  printf("Allocating %.2fMB of memory for the grid cells...\n",
      nxcells*nycells*nzcells*sizeof(gridcell_t) / (1024.0 * 1024.0));
  if (nsubx != nxcells-1 || nsuby != nycells-1 || nsubz != nzcells-1) {
    printf("grid cell dimensions are not as expected\n");
    return FAIL;
  }

  /* allocate and initialize the GPU output array */
  memsz = sizeof(float) * allsize.x * allsize.y * allsize.z;
  savenergy = (float *) malloc(memsz);

  memsz = sizeof(float) * allsize.x * allsize.y * volsize.z;
  printf("Allocating %.2fMB of memory for GPU data buffer...\n",
      memsz / (1024.0 * 1024.0));
  cudaMalloc((void **) &outenergy, memsz);
  CUERR;  /* check and clear any existing errors */

#if 0
  /* copy zeroed array space into gpu data space */
  cudaMemcpy(outenergy, savenergy, allmemsz, cudaMemcpyHostToDevice);
  CUERR;  /* check and clear any existing errors */

  cudaMemcpy(savenergy, outenergy, allmemsz, cudaMemcpyDeviceToHost);
  CUERR;  /* check and clear any existing errors */
#endif

  /*
   * loop over subcubes
   */
  rt_timer_start(hashtimer);
  reset_atoms(&sender);
  for (k = 0;  k < nsubz;  k++) {

    /* clear memory buffer to accumulate another slab */
    cudaMemset(outenergy, 0, memsz);
    CUERR;  /* check and clear any existing errors */

    for (j = 0;  j < nsuby;  j++) {
      for (i = 0;  i < nsubx;  i++) {

        /* lowest numbered point in subcube */
        xlo = i * volsize.x;
        ylo = j * volsize.y;
        zlo = k * volsize.z;

        /*
         * for each subcube, calculate interactions with atoms in the
         * 8 grid cell regions that contain this subcube + cutoff margin
         */
        for (kc = k;  kc <= k+1;  kc++) {
          for (jc = j;  jc <= j+1;  jc++) {
            for (ic = i;  ic <= i+1;  ic++) {
              index = (kc*nycells + jc)*nxcells + ic;
              if (append_atoms(&sender, cell[index].atom,
                               cell[index].atomcnt)) {
                /* buffer is full, so send atoms to GPU buffer and run */
                send_atoms(&sender);
                rt_timer_start(runtimer);
                mgpot_shortrng_energy<<<gsize, bsize, 0>>>(
                    sender.atomcnt,
                    allnx, allny,
                    xlo, ylo, zlo,
                    outenergy);
                CUERR;  /* check and clear any existing errors */
                rt_timer_stop(runtimer);
                runtotal += rt_timer_time(runtimer);
                invocations++;
                reset_atoms(&sender);
                /* have to append again since first try failed */
                append_atoms(&sender, cell[index].atom, cell[index].atomcnt);
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
            outenergy);
        CUERR;  /* check and clear any existing errors */
        rt_timer_stop(runtimer);
        runtotal += rt_timer_time(runtimer);
        invocations++;
        reset_atoms(&sender);
      }
    }

    /* copy GPU memory buffer data back to host */
    cudaMemcpy(savenergy + k*volsize.z*allny*allnx, outenergy, memsz,
      cudaMemcpyDeviceToHost);
    CUERR;  /* check and clear any existing errors */

  } /* end subcube loop */
  rt_timer_stop(hashtimer);
  hashtotal += rt_timer_time(hashtimer);

#ifdef DEBUGGING
  k = 0;
  printf("after copy back:  savenergy[%d] = %g\n", k, (double) savenergy[k]);
  k = 1;
  printf("after copy back:  savenergy[%d] = %g\n", k, (double) savenergy[k]);
#endif

#if 0
  for (k = 0;  k < allnx*allny*allnz;  k++) {
    printf("savenergy[%d] = %g\n", k, savenergy[k]);
  }
#endif

#ifndef MAIN
  /* copy host buffer into (likely smaller) gridener buffer */
  for (k = 0;  k < numplane;  k++) {
    for (j = 0;  j < numcol;  j++) {
      for (i = 0;  i < numpt;  i++) {
        gridener[(k*numcol+j)*numpt+i] += savenergy[(k*allny+j)*allnx+i];
      }
    }
  }
#endif

  printf("Number of loop iterations:  %d\n", iterations);
  printf("Number of kernel invocations:  %d\n", invocations);
  printf("Kernel time: %f seconds, %f per iteration\n",
    runtotal, runtotal / (float) iterations);
  printf("Hash & Loop:  %f seconds\n", hashtotal);
  printf("CPU overhead: %f seconds\n", hashtotal-runtotal);

#ifdef DEBUGGING
  /* verify solution by slow method */
  {
    float *checkenergy = (float *) calloc(allnx*allny*allnz, sizeof(float));
    float r2, dx, dy, dz;
    float maxabserr;
    int n;

    printf("Verifying CUDA calculation using brute force method\n");

    for (n = 0;  n < 4*natoms;  n += 4) {
      for (k = 0;  k < allnz;  k++) {
        for (j = 0;  j < allny;  j++) {
          for (i = 0;  i < allnx;  i++) {
            dx = i*gridspacing - atom[n  ];
            dy = j*gridspacing - atom[n+1];
            dz = k*gridspacing - atom[n+2];
            r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < CUTOFF2) {
              float s = r2 * INV_CUTOFF2;
              float gs = 1.f + (s-1.f)*(-0.5f + (s-1.f)*0.375f);
                /* gs is the TAYLOR2 softening */
              float r_1 = 1.f/sqrtf(r2);
              index = (k*allny + j)*allnx + i;
              checkenergy[index] += atom[n+3] * (r_1 - INV_CUTOFF * gs);
            }
          }
        }
      } /* end loop over grid points */
    } /* end loop over atoms */

    maxabserr = 0;
    for (k = 0;  k < allnz;  k++) {
      for (j = 0;  j < allny;  j++) {
        for (i = 0;  i < allnx;  i++) {
          index = (k*allny + j)*allnx + i;
          float abserr = fabsf(checkenergy[index] - savenergy[index]);
          if (maxabserr < abserr) maxabserr = abserr;
        }
      }
    }
    printf("max absolute error = %g\n", maxabserr);
  } /* end verify block */
#endif

  free(cell);
  free(savenergy);
  cudaFree(outenergy);

  return 0;
}
