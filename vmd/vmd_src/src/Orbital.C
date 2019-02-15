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
 *	$RCSfile: Orbital.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.150 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Orbital class, which stores orbitals, SCF energies, etc. for a
 * single timestep.
 *
 ***************************************************************************/

// pgcc 2016 has troubles with hand-vectorized x86 intrinsics presently
#if !defined(__PGIC__)
#if defined(VMDUSEAVX512) && defined(__AVX512F__) && defined(__AVX512ER__)
// AVX512F + AVX512ER for Xeon Phi
#define VMDORBUSEAVX512 1
#define VECPADSZ   16
#define VECPADMASK (VECPADSZ - 1)
#else
// fall-back to SSE 
#define VMDORBUSESSE 1
#define VECPADSZ   4
#define VECPADMASK (VECPADSZ - 1)
#endif
#endif


// The OpenPOWER VSX code path runs on POWER8 and later hardware, but is
// untested on older platforms that support VSX instructions.
// XXX GCC 4.8.5 breaks with conflicts between vec_xxx() routines 
//     defined in utilities.h vs. VSX intrinsics in altivec.h and similar.
//     For now, we disable VSX for GCC for this source file.
#if !defined(__GNUC__) && defined(__VSX__)
#define VMDORBUSEVSX 1
#define VECPADSZ   4
#define VECPADMASK (VECPADSZ - 1)
#endif

// #define DEBUGORBS 1

#include <math.h>
#include <stdio.h>

#if defined(VMDORBUSESSE) && defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(VMDORBUSEAVX512) && defined(__AVX512F__) && defined(__AVX512ER__)
#include <immintrin.h>
#endif
#if defined(VMDORBUSEVSX) && defined(__VSX__)
#if defined(__GNUC__) && defined(__VEC__)
#include <altivec.h>
#endif
#endif

#include "Orbital.h"
#include "DrawMolecule.h"
#include "utilities.h"
#include "Inform.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#if defined(VMDCUDA)
#include "CUDAOrbital.h"
#endif
#if defined(VMDOPENCL)
#include "OpenCLUtils.h"
#include "OpenCLKernels.h"
#endif
#include "ProfileHooks.h"

#define ANGS_TO_BOHR 1.88972612478289694072f

///  constructor  
Orbital::Orbital(const float *pos,
                 const float *wfn,
                 const float *barray,
                 const basis_atom_t *bset,
                 const int *types,
                 const int *asort,
                 const int *abasis,
                 const float **norm,
                 const int *nshells,
                 const int *nprimshell,
                 const int *shelltypes, 
                 int natoms, int ntypes, int numwave, int numbasis, 
                 int orbid) :
  numatoms(natoms), atompos(pos),
  num_wave_f(numwave),
  wave_f(NULL),
  num_basis_funcs(numbasis),
  basis_array(barray),
  numtypes(ntypes), 
  basis_set(bset),
  atom_types(types),
  atom_sort(asort),
  atom_basis(abasis),
  norm_factors(norm),
  num_shells_per_atom(nshells),
  num_prim_per_shell(nprimshell),
  shell_types(shelltypes), 
  grid_data(NULL)
{
  origin[0] = origin[1] = origin[2] = 0.0;

  // Multiply wavefunction coefficients with the
  // angular momentum dependent part of the basis set
  // normalization factors.
  normalize_wavefunction(wfn + num_wave_f*orbid);

  //print_wavefunction();
}

/// destructor  
Orbital::~Orbital() {
  if (wave_f) delete [] wave_f;
}


// Multiply wavefunction coefficients with the
// basis set normalization factors. We do this
// here rather than normalizing the basisset itself
// because we need different factors for the different
// cartesian components of a shell and the basis set
// stores data only per shell.
// By doing the multiplication here we save a lot of
// flops during orbital rendering.
void Orbital::normalize_wavefunction(const float *wfn) {
#ifdef DEBUGORBS
  char shellname[8] = {'S', 'P', 'D', 'F', 'G', 'H', 'I', 'K'};
#endif
  int i, j, k;
  // Get size of the symmetry-expanded wavefunction array
//   int wave_size = 0;
//   for (i=0; i<numatoms; i++) {
//     printf("atom[%d]: type = %d\n", i, atom_types[i]);
//     const basis_atom_t *basis_atom = &basis_set[atom_types[i]];
//     for (j=0; j<basis_atom->numshells; j++) {
//       wave_size += basis_atom->shell[j].num_cart_func;
//     }
//   }
//   printf("num_wave_f/wave_size = %d/%d\n", num_wave_f,  wave_size);

  wave_f = new float[num_wave_f];
  int ifunc = 0;
  for (i=0; i<numatoms; i++) {
    const basis_atom_t *basis_atom = &basis_set[atom_types[i]];
    for (j=0; j<basis_atom->numshells; j++) {
      int stype = basis_atom->shell[j].type;
#ifdef DEBUGORBS
      printf("atom %i/%i,  %i/%i %c-shell\n", i+1, numatoms, j+1, basis_atom->numshells, shellname[stype]);
#endif
      for (k=0; k<basis_atom->shell[j].num_cart_func; k++) {
        wave_f[ifunc] = wfn[ifunc] * norm_factors[stype][k];

#ifdef DEBUGORBS
        printf("%3i %c %2i wave_f[%3i]=% f  norm=%.3f  normwave=% f\n",
               i, shellname[stype], k, ifunc, wfn[ifunc],
               norm_factors[stype][k], wave_f[ifunc]);
#endif
        ifunc++;
      }
    }
  }
}


// Sets the grid dimensions to the bounding box of the given
// set of atoms *pos including a padding in all dimensions.
// The resulting grid dimensions will be rounded to a multiple
// of the voxel size.
int Orbital::set_grid_to_bbox(const float *pos, float padding,
                              float resolution) {
  int i = 0;
  float xyzdim[3];

  /* set initial values of temp values to the coordinates
   * of the first atom.  */
  origin[0] = xyzdim[0] = pos[0];
  origin[1] = xyzdim[1] = pos[1];
  origin[2] = xyzdim[2] = pos[2];

  /* now loop over the rest of the atoms to check if there's
   * something larger/smaller for the maximum and minimum
   * respectively */
  for(i=1; i<numatoms; i++) {
    if (pos[3*i  ] < origin[0]) origin[0] = pos[3*i];
    if (pos[3*i+1] < origin[1]) origin[1] = pos[3*i+1];
    if (pos[3*i+2] < origin[2]) origin[2] = pos[3*i+2];
    if (pos[3*i  ] > xyzdim[0]) xyzdim[0] = pos[3*i];
    if (pos[3*i+1] > xyzdim[1]) xyzdim[1] = pos[3*i+1];
    if (pos[3*i+2] > xyzdim[2]) xyzdim[2] = pos[3*i+2];
  }

  // Apply padding in each direction
  origin[0] -= padding;
  origin[1] -= padding;
  origin[2] -= padding;
  gridsize[0] = xyzdim[0] + padding - origin[0];
  gridsize[1] = xyzdim[1] + padding - origin[1];
  gridsize[2] = xyzdim[2] + padding - origin[2];  

  set_resolution(resolution);

  return TRUE;
}


// Set the dimensions and resolution of the grid for which 
// the orbital shall be computed.
// The given grid dimensions will be rounded to a multiple
// of the voxel size.
void Orbital::set_grid(float newori[3], float newdim[3], float newvoxelsize) {
  origin[0] = newori[0];
  origin[1] = newori[1];
  origin[2] = newori[2];
  gridsize[0] = newdim[0];
  gridsize[1] = newdim[1];
  gridsize[2] = newdim[2];
  set_resolution(newvoxelsize);
}

// Change the resolution of the grid
void Orbital::set_resolution(float resolution) {
  voxelsize = resolution;
  int i;
  for (i=0; i<3; i++) {
    numvoxels[i] = (int)(gridsize[i]/voxelsize) + 1;
    gridsize[i] = voxelsize*(numvoxels[i]-1);
  }
}

#define XNEG 0
#define YNEG 1
#define ZNEG 2
#define XPOS 3
#define YPOS 4
#define ZPOS 5

// Check if all values in the boundary plane given by dir 
// are below threshold.
// If not, jump back, decrease the stepsize and test again.
int Orbital::check_plane(int dir, float threshold, int minstepsize,
                         int &stepsize) {
  bool repeat=0;
  int u, v, w, nu, nv;
  // w is the dimension we want to adjust,
  // u and v are the other two, i.e. the plane in which we test
  // the orbital values. 
  u = (dir+1)%3;
  v = (dir+2)%3;
  w = dir%3;

  // for debugging
  //char axis[3] = {'X', 'Y', 'Z'};
  //char sign[2] = {'-', '+'};
  //printf("%c%c: ", sign[dir/3], axis[w]);

  do {
    int success = 0;
    int gridstep = stepsize;

    if (repeat) {
      // We are repeating the test on the previous slate but with
      // twice the resolution. Hence we only have to test the new
      // grid points lying between the old ones.
      gridstep = 2*stepsize;
    }


    float grid[3];
    grid[w] = origin[w] + (dir/3)*(numvoxels[w]-1) * voxelsize;

    // Search for a value of the wave function larger than threshold.
    for (nu=0; nu<numvoxels[u]; nu+=gridstep) {
      grid[u] = origin[u] + nu * voxelsize;
    
      for (nv=0; nv<numvoxels[v]; nv+=gridstep) {
        grid[v] = origin[v] + nv * voxelsize;
      
        if (fabs(evaluate_grid_point(grid[0], grid[1], grid[2])) > threshold) {
          success = 1;
          break;
        }
      }
      if (success) break;
    }

    if (success) {
      // Found an orbital value higher than the threshold.
      // We want the threshold isosurface to be completely inside the grid.
      // The boundary must be between the previous and this plane.
      if (!(dir/3)) origin[w] -= stepsize*voxelsize;
      numvoxels[w] += stepsize;
      if (stepsize<=minstepsize) {
        //printf("success!\n");
        return 1;
      }
      stepsize /=2;
      repeat = 1;
      //printf("increase by %i, reduce stepsize to %i.\n", 2*stepsize, stepsize);

    } else {
      // All values lower than threshold, we decrease the grid size.
      if (!(dir/3)) origin[w] += stepsize*voxelsize;
      numvoxels[w] -= stepsize;
      //printf("decrease by %i\n", stepsize);
      repeat = 0;
      if (numvoxels[w] <= 1) {
        // Here we ended up with a zero grid size.
        // We must have missed something. Let's increase grid again and 
        // try a smaller step size.
        numvoxels[w] = stepsize; 
        if (!(dir/3)) origin[w] -= stepsize*voxelsize;
        stepsize /=2;
        repeat = 1;
        //printf("zero grid size - increase to %i, reduce stepsize to %i.\n", 2*stepsize, stepsize);
      }
    }

  } while (repeat);

  return 0;
}


// Optimize position and dimension of current grid so that all orbital
// values higher than threshold are contained in the grid.
//
// Algorithm:
// Based on the idea that the wave function trails off in a distance of
// a few Angstroms from the molecule.
// We start from the current grid size (which could be for instance the
// molecular bounding box plus a padding region) and test the values on
// each of the six boundary planes. If there is no value larger than the
// given threshold in a plane then we shrink the system along the plane
// normal. In the distance the wave function tends to be smoother so we
// start the testing on a coarser grid. A parameter maxstepsize=4 means
// to begin with a grid using a four times higher voxel side length than
// the original grid. When we find the first value above the threshold we
// jump back one step and continue with half of the previous stepsize.
// When stepsize has reached minstepsize then we consider the corresponding
// boundary plane optimal. Note that starting out with a too coarse
// grid one might miss some features of the wave function.
// If you want to be sure not to miss anything then use the voxelsize
// for both minstepsize and maxstepsize.
void Orbital::find_optimal_grid(float threshold, 
				int minstepsize, int maxstepsize) {
  int optimal[6] = {0, 0, 0, 0, 0, 0};
  int stepsize[6];
  int i;
  for (i=0; i<6; i++) stepsize[i] = maxstepsize;

#ifdef DEBUGORBS
  printf("origin = {%f %f %f}\n", origin[0], origin[1], origin[2]);
  printf("gridsize = {%f %f %f}\n", gridsize[0], gridsize[1], gridsize[2]);
#endif  
  
  
  // Loop until we have optimal grid boundaries in all
  // dimensions
  int iter = 0;
  while ( !optimal[0] || !optimal[1] || !optimal[2] ||
	  !optimal[3] || !optimal[4] || !optimal[5] )
  {
    if (iter>100) {
      msgInfo << "WARNING: Could not optimize orbital grid boundaries in"
              << iter << "steps!" << sendmsg; 
      break;
    }
    iter++;

    // Examine the current grid boundaries and shrink if
    // all values are smaller than threshold    .
    if (!optimal[XNEG])
      optimal[XNEG] = check_plane(XNEG, threshold, minstepsize, stepsize[XNEG]);

    if (!optimal[XPOS])
      optimal[XPOS] = check_plane(XPOS, threshold, minstepsize, stepsize[XPOS]);

    if (!optimal[YNEG])
      optimal[YNEG] = check_plane(YNEG, threshold, minstepsize, stepsize[YNEG]);

    if (!optimal[YPOS])
      optimal[YPOS] = check_plane(YPOS, threshold, minstepsize, stepsize[YPOS]);

    if (!optimal[ZNEG])
      optimal[ZNEG] = check_plane(ZNEG, threshold, minstepsize, stepsize[ZNEG]);

    if (!optimal[ZPOS])
      optimal[ZPOS] = check_plane(ZPOS, threshold, minstepsize, stepsize[ZPOS]);

#if defined(DEBUGORBS)
    printf("origin {%f %f %f}\n", origin[0], origin[1], origin[2]);
    printf("ngrid {%i %i %i}\n", numvoxels[0], numvoxels[1], numvoxels[2]);
    printf("stepsize {%i %i %i %i %i %i}\n", stepsize[0], stepsize[1], stepsize[2],
	   stepsize[3], stepsize[4], stepsize[5]);
#endif
  }

  
  gridsize[0] = numvoxels[0]*voxelsize;
  gridsize[1] = numvoxels[1]*voxelsize;
  gridsize[2] = numvoxels[2]*voxelsize;
}


// this function creates the orbital grid given the system dimensions
int Orbital::calculate_mo(DrawMolecule *mol, int density) {
  PROFILE_PUSH_RANGE("Orbital", 4);

  wkf_timerhandle timer=wkf_timer_create();
  wkf_timer_start(timer);

  //
  // Force vectorized N-element padding for the X dimension to prevent
  // the possibility of an out-of-bounds orbital grid read/write operation
  //
  numvoxels[0] = (numvoxels[0] + VECPADMASK) & ~(VECPADMASK);
  gridsize[0] = numvoxels[0]*voxelsize;
 
  // Allocate memory for the volumetric grid
  int numgridpoints = numvoxels[0] * numvoxels[1] * numvoxels[2];
  grid_data = new float[numgridpoints];

#ifdef DEBUGORBS
  printf("num_wave_f=%i\n", num_wave_f);

  int i=0;
  for (i=0; i<num_wave_f; i++) {
    printf("wave_f[%i] = %f\n", i, wave_f[i]);
  }

  // perhaps give the user a warning, since the calculation
  // could take a while, otherwise they might think the system is borked 
  printf("Calculating %ix%ix%i orbital grid.\n", 
         numvoxels[0], numvoxels[1], numvoxels[2]);
#endif


  int rc=-1; // initialize to sentinel value

  // Calculate the value of the orbital at each gridpoint
#if defined(VMDCUDA)
  // The CUDA kernel currently only handles up to "G" shells,
  // and up to 32 primitives per basis function
  if ((max_shell_type() <= G_SHELL) &&
      (max_primitives() <= 32) &&
      (!getenv("VMDNOCUDA"))) {
    rc = vmd_cuda_evaluate_orbital_grid(mol->cuda_devpool(), 
                                        numatoms, wave_f, num_wave_f,
                                        basis_array, num_basis_funcs,
                                        atompos, atom_basis,
                                        num_shells_per_atom, 
                                        num_prim_per_shell,
                                        shell_types, total_shells(),
                                        numvoxels, voxelsize, 
                                        origin, density, grid_data);
  }
#endif
#if defined(VMDOPENCL)
  // The OpenCL kernel currently only handles up to "G" shells,
  // and up to 32 primitives per basis function
  if (rc!=0 &&
      (max_shell_type() <= G_SHELL) &&
      (max_primitives() <= 32) &&
      (!getenv("VMDNOOPENCL"))) {

#if 1
    // XXX this would be done during app startup normally...
    static vmd_opencl_orbital_handle *orbh = NULL;
    static cl_context clctx = NULL;
    static cl_command_queue clcmdq = NULL;
    static cl_device_id *cldevs = NULL;
    if (orbh == NULL) {
      printf("Attaching OpenCL device:\n");
      wkf_timer_start(timer);
      cl_int clerr = CL_SUCCESS;

      cl_platform_id clplatid = vmd_cl_get_platform_index(0);
      cl_context_properties clctxprops[] = {(cl_context_properties) CL_CONTEXT_PLATFORM, (cl_context_properties) clplatid, (cl_context_properties) 0};
      clctx = clCreateContextFromType(clctxprops, CL_DEVICE_TYPE_GPU, NULL, NULL, &clerr);

      size_t parmsz;
      clerr |= clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);
      if (clerr != CL_SUCCESS) return -1;
      cldevs = (cl_device_id *) malloc(parmsz);
      if (clerr != CL_SUCCESS) return -1;
      clerr |= clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);
      if (clerr != CL_SUCCESS) return -1;
      clcmdq = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr);
      if (clerr != CL_SUCCESS) return -1;
      wkf_timer_stop(timer);
      printf("  OpenCL context creation time: %.3f sec\n", wkf_timer_time(timer));

      wkf_timer_start(timer);
      orbh = vmd_opencl_create_orbital_handle(clctx, clcmdq, cldevs);
      wkf_timer_stop(timer);
      printf("  OpenCL kernel compilation time: %.3f sec\n", wkf_timer_time(timer));

      wkf_timer_start(timer);
    }
#endif

    rc = vmd_opencl_evaluate_orbital_grid(mol->cuda_devpool(), orbh,
                                        numatoms, wave_f, num_wave_f,
                                        basis_array, num_basis_funcs,
                                        atompos, atom_basis,
                                        num_shells_per_atom, 
                                        num_prim_per_shell,
                                        shell_types, total_shells(),
                                        numvoxels, voxelsize, 
                                        origin, density, grid_data);

#if 0
    // XXX this would normally be done at program shutdown
    vmd_opencl_destroy_orbital_handle(parms.orbh);
    clReleaseCommandQueue(clcmdq);
    clReleaseContext(clctx);
    free(cldevs);
#endif
  }
#endif
#if 0
  int numprocs = 1;
  if (getenv("VMDDUMPORBITALS")) {
    write_orbital_data(getenv("VMDDUMPORBITALS"), numatoms,
                       wave_f, num_wave_f, basis_array, num_basis,
                       atompos, atom_basis, num_shells_per_atom,
                       num_prim_per_shell, shell_types,
                       num_shells, numvoxels, voxelsize, origin);

    read_calc_orbitals(devpool, getenv("VMDDUMPORBITALS"));
  }
#endif


#if !defined(VMDORBUSETHRPOOL)
#if defined(VMDTHREADS)
  int numcputhreads = wkf_thread_numprocessors();
#else
  int numcputhreads = 1;
#endif
#endif
  if (rc!=0)  rc = evaluate_grid_fast(
#if defined(VMDORBUSETHRPOOL)
                                      mol->cpu_threadpool(), 
#else
                                      numcputhreads,
#endif
                                      numatoms, wave_f, basis_array,
                                      atompos, atom_basis,
                                      num_shells_per_atom, num_prim_per_shell,
                                      shell_types, numvoxels, voxelsize, 
                                      origin, density, grid_data);

  if (rc!=0) {
    msgErr << "Error computing orbital grid" << sendmsg;
    delete [] grid_data;
    grid_data=NULL;

    PROFILE_POP_RANGE(); // first return point

    return FALSE;
  }

  wkf_timer_stop(timer);

#if 1
  if (getenv("VMDORBTIMING") != NULL) { 
    double gflops = (numgridpoints * flops_per_gridpoint()) / (wkf_timer_time(timer) * 1000000000.0);

    char strbuf[1024];
    sprintf(strbuf, "Orbital calc. time %.3f secs, %.2f gridpoints/sec, %.2f GFLOPS",
            wkf_timer_time(timer), 
            (((double) numgridpoints) / wkf_timer_time(timer)),
            gflops);
    msgInfo << strbuf << sendmsg;
  }
#endif

  wkf_timer_destroy(timer);

  PROFILE_POP_RANGE(); // second return point

  return TRUE;
}


/*********************************************************
 *
 * This function calculates the value of the wavefunction
 * corresponding to a particular orbital at grid point
 * grid_x, grid_y, grid_z.


 Here's an example of a basis set definition for one atom:

  SHELL TYPE  PRIMITIVE        EXPONENT          CONTRACTION COEFFICIENT(S)

 Oxygen

      1   S       1          5484.6716600    0.001831074430
      1   S       2           825.2349460    0.013950172200
      1   S       3           188.0469580    0.068445078098
      1   S       4            52.9645000    0.232714335992
      1   S       5            16.8975704    0.470192897984
      1   S       6             5.7996353    0.358520852987

      2   L       7            15.5396162   -0.110777549525    0.070874268231
      2   L       8             3.5999336   -0.148026262701    0.339752839147
      2   L       9             1.0137618    1.130767015354    0.727158577316

      3   L      10             0.2700058    1.000000000000    1.000000000000

 *********************************************************/
float Orbital::evaluate_grid_point(float grid_x, float grid_y, float grid_z) {
  int at;
  int prim, shell;

  // initialize value of orbital at gridpoint
  float value = 0.0;

  // initialize the wavefunction and shell counters
  int ifunc = 0; 
  int shell_counter = 0;

  // loop over all the QM atoms
  for (at=0; at<numatoms; at++) {
    int maxshell = num_shells_per_atom[at];
    int prim_counter = atom_basis[at];

    // calculate distance between grid point and center of atom
    float xdist = (grid_x - atompos[3*at  ])*ANGS_TO_BOHR;
    float ydist = (grid_y - atompos[3*at+1])*ANGS_TO_BOHR;
    float zdist = (grid_z - atompos[3*at+2])*ANGS_TO_BOHR;
    float dist2 = xdist*xdist + ydist*ydist + zdist*zdist;

    // loop over the shells belonging to this atom
    // XXX this is maybe a misnomer because in split valence
    //     basis sets like 6-31G we have more than one basis
    //     function per (valence-)shell and we are actually
    //     looping over the individual contracted GTOs
    for (shell=0; shell < maxshell; shell++) {
      float contracted_gto = 0.0f;

      // Loop over the Gaussian primitives of this contracted 
      // basis function to build the atomic orbital
      int maxprim = num_prim_per_shell[shell_counter];
      int shelltype = shell_types[shell_counter];
      for (prim=0; prim < maxprim;  prim++) {
        float exponent       = basis_array[prim_counter    ];
        float contract_coeff = basis_array[prim_counter + 1];
        contracted_gto += contract_coeff * expf(-exponent*dist2);
        prim_counter += 2;
      }

      /* multiply with the appropriate wavefunction coefficient */
      float tmpshell=0;
      // Loop over the cartesian angular momenta of the shell.
      // avoid unnecessary branching and minimize use of pow()
      int i, j; 
      float xdp, ydp, zdp;
      float xdiv = 1.0f / xdist;
      for (j=0, zdp=1.0f; j<=shelltype; j++, zdp*=zdist) {
        int imax = shelltype - j; 
        for (i=0, ydp=1.0f, xdp=powf(xdist, float(imax)); i<=imax; i++, ydp*=ydist, xdp*=xdiv) {
          tmpshell += wave_f[ifunc++] * xdp * ydp * zdp;
        }
      }
      value += tmpshell * contracted_gto;

      shell_counter++;
    } 
  }

  /* return the final value at grid point */
  return value;
}


//
// Return the max number of primitives that occur in a basis function
//
int Orbital::max_primitives(void) {
  int maxprim=-1;

  int shell_counter = 0;
  for (int at=0; at<numatoms; at++) {
    for (int shell=0; shell < num_shells_per_atom[at]; shell++) {
      int numprim = num_prim_per_shell[shell_counter];
      if (numprim > maxprim)
        maxprim = numprim; 
    }
  }

  return maxprim;
}


//
// Return the maximum shell type used
//
int Orbital::max_shell_type(void) {
  int maxshell=-1;

  int shell_counter = 0;
  for (int at=0; at<numatoms; at++) {
    for (int shell=0; shell < num_shells_per_atom[at]; shell++) {
      int shelltype = shell_types[shell_counter];
      shell_counter++;
      if (shelltype > maxshell)
        maxshell=shelltype;
    }
  }

  return maxshell;
}


//
// count the maximum number of wavefunction coefficient accesses
// required for the highest shell types contained in this orbital
//
int Orbital::max_wave_f_count(void) {
  int maxcount=0;

  int shell_counter = 0;
  for (int at=0; at<numatoms; at++) {
    for (int shell=0; shell < num_shells_per_atom[at]; shell++) {
      int shelltype = shell_types[shell_counter];
      int i, j; 
      int count=0;
      for (i=0; i<=shelltype; i++) {
        int jmax = shelltype - i; 
        for (j=0; j<=jmax; j++) {
          count++;
        }
      }
      shell_counter++;
      if (count > maxcount)
        maxcount=count;
    }
  }

  return maxcount;
}


//
// compute the FLOPS per grid point for performance measurement purposes
//
double Orbital::flops_per_gridpoint() {
  double flops=0.0;

  int shell_counter = 0;
  for (int at=0; at<numatoms; at++) {
    flops += 7;

    for (int shell=0; shell < num_shells_per_atom[at]; shell++) {
      for (int prim=0; prim < num_prim_per_shell[shell_counter];  prim++)
        flops += 4; // expf() costs far more, but we count as one.

      int shelltype = shell_types[shell_counter];

      switch (shelltype) {
        // separately count for the hand-optimized cases
        case S_SHELL: flops += 2; break;
        case P_SHELL: flops += 8; break;
        case D_SHELL: flops += 17; break;
        case F_SHELL: flops += 30; break;
        case G_SHELL: flops += 50; break;

        // count up for catch-all loop
        default:
          int i, j; 
          for (i=0; i<=shelltype; i++) {
            int jmax = shelltype - i; 
            flops += 1;
            for (j=0; j<=jmax; j++) {
              flops += 6;
            }
          }
          break;
      }

      shell_counter++;
    } 
  }

  return flops;
}


//
// Fast single-precision expf() implementation
// Adapted from the free cephes math library on Netlib
//   http://www.netlib.org/cephes/
//
// Cephes Math Library Release 2.2:  June, 1992
// Copyright 1984, 1987, 1989 by Stephen L. Moshier
// Direct inquiries to 30 Frost Street, Cambridge, MA 02140
//
static const float MAXNUMF =    3.4028234663852885981170418348451692544e38f;
static const float MAXLOGF =   88.72283905206835f;
static const float MINLOGF = -103.278929903431851103f; /* log(2^-149) */
static const float LOG2EF  =    1.44269504088896341f;
static const float C1      =    0.693359375f;
static const float C2      =   -2.12194440e-4f;

static inline float cephesfastexpf(float x) {
  float z;
  int n;
  
  if(x > MAXLOGF) 
    return MAXNUMF;

  if(x < MINLOGF) 
    return 0.0;

  // Express e^x = e^g 2^n = e^g e^(n loge(2)) = e^(g + n loge(2))
  z = floorf( LOG2EF * x + 0.5f ); // floor() truncates toward -infinity.
  x -= z * C1;
  x -= z * C2;
  n = (int) z;

  z = x * x;
  // Theoretical peak relative error in [-0.5, +0.5] is 4.2e-9.
  z = ((((( 1.9875691500E-4f  * x + 1.3981999507E-3f) * x
          + 8.3334519073E-3f) * x + 4.1665795894E-2f) * x
          + 1.6666665459E-1f) * x + 5.0000001201E-1f) * z + x + 1.0f;

  x = ldexpf(z, n); // multiply by power of 2
  return x;
}




/*
 * David J. Hardy
 * 12 Dec 2008
 *
 * aexpfnx() - Approximate expf() for negative x.
 *
 * Assumes that x <= 0.
 *
 * Assumes IEEE format for single precision float, specifically:
 * 1 sign bit, 8 exponent bits biased by 127, and 23 mantissa bits.
 *
 * Interpolates exp() on interval (-1/log2(e), 0], then shifts it by
 * multiplication of a fast calculation for 2^(-N).  The interpolation
 * uses a linear blending of 3rd degree Taylor polynomials at the end
 * points, so the approximation is once differentiable.
 *
 * The error is small (max relative error per interval is calculated
 * to be 0.131%, with a max absolute error of -0.000716).
 *
 * The cutoff is chosen so as to speed up the computation by early
 * exit from function, with the value chosen to give less than the
 * the max absolute error.  Use of a cutoff is unnecessary, except
 * for needing to shift smallest floating point numbers to zero,
 * i.e. you could remove cutoff and replace by:
 *
 * #define MINXNZ  -88.0296919311130  // -127 * log(2)
 *
 *   if (x < MINXNZ) return 0.f;
 *
 * Use of a cutoff causes a discontinuity which can be eliminated
 * through the use of a switching function.
 *
 * We can obtain arbitrarily smooth approximation by taking k+1 nodes on
 * the interval and weighting their respective Taylor polynomials by the
 * kth order Lagrange interpolant through those nodes.  The wiggle in the
 * polynomial interpolation due to equidistant nodes (Runge's phenomenon)
 * can be reduced by using Chebyshev nodes.
 */

#define MLOG2EF    -1.44269504088896f

/*
 * Interpolating coefficients for linear blending of the
 * 3rd degree Taylor expansion of 2^x about 0 and -1.
 */
#define SCEXP0     1.0000000000000000f
#define SCEXP1     0.6987082824680118f
#define SCEXP2     0.2633174272827404f
#define SCEXP3     0.0923611991471395f
#define SCEXP4     0.0277520543324108f

/* for single precision float */
#define EXPOBIAS   127
#define EXPOSHIFT   23

/* cutoff is optional, but can help avoid unnecessary work */
#define ACUTOFF    -10

typedef union flint_t {
  float f;
  int n;
} flint;

float aexpfnx(float x) {
  /* assume x <= 0 */
  float mb;
  int mbflr;
  float d;
  float sy;
  flint scalfac;

  if (x < ACUTOFF) return 0.f;

  mb = x * MLOG2EF;    /* change base to 2, mb >= 0 */
  mbflr = (int) mb;    /* get int part, floor() */
  d = mbflr - mb;      /* remaining exponent, -1 < d <= 0 */
  sy = SCEXP0 + d*(SCEXP1 + d*(SCEXP2 + d*(SCEXP3 + d*SCEXP4)));
                       /* approx with linear blend of Taylor polys */
  scalfac.n = (EXPOBIAS - mbflr) << EXPOSHIFT;  /* 2^(-mbflr) */
  return (sy * scalfac.f);  /* scaled approx */
}



//
// Optimized molecular orbital grid evaluation code
//
#define S_SHELL 0
#define P_SHELL 1
#define D_SHELL 2
#define F_SHELL 3
#define G_SHELL 4
#define H_SHELL 5

int evaluate_grid(int numatoms,
                  const float *wave_f, const float *basis_array,
                  const float *atompos,
                  const int *atom_basis,
                  const int *num_shells_per_atom,
                  const int *num_prim_per_shell,
                  const int *shell_types,
                  const int *numvoxels,
                  float voxelsize,
                  const float *origin,
                  int density,
                  float * orbitalgrid) {
  if (!orbitalgrid)
    return -1;

  int nx, ny, nz;
  // Calculate the value of the orbital at each gridpoint and store in 
  // the current oribtalgrid array
  int numgridxy = numvoxels[0]*numvoxels[1];
  for (nz=0; nz<numvoxels[2]; nz++) {
    float grid_x, grid_y, grid_z;
    grid_z = origin[2] + nz * voxelsize;
    for (ny=0; ny<numvoxels[1]; ny++) {
      grid_y = origin[1] + ny * voxelsize;
      int gaddrzy = ny*numvoxels[0] + nz*numgridxy;
      for (nx=0; nx<numvoxels[0]; nx++) {
        grid_x = origin[0] + nx * voxelsize;

        // calculate the value of the wavefunction of the
        // selected orbital at the current grid point
        int at;
        int prim, shell;

        // initialize value of orbital at gridpoint
        float value = 0.0;

        // initialize the wavefunction and shell counters
        int ifunc = 0; 
        int shell_counter = 0;

        // loop over all the QM atoms
        for (at=0; at<numatoms; at++) {
          int maxshell = num_shells_per_atom[at];
          int prim_counter = atom_basis[at];

          // calculate distance between grid point and center of atom
          float xdist = (grid_x - atompos[3*at  ])*ANGS_TO_BOHR;
          float ydist = (grid_y - atompos[3*at+1])*ANGS_TO_BOHR;
          float zdist = (grid_z - atompos[3*at+2])*ANGS_TO_BOHR;

          float xdist2 = xdist*xdist;
          float ydist2 = ydist*ydist;
          float zdist2 = zdist*zdist;
          float xdist3 = xdist2*xdist;
          float ydist3 = ydist2*ydist;
          float zdist3 = zdist2*zdist;

          float dist2 = xdist2 + ydist2 + zdist2;

          // loop over the shells belonging to this atom
          // XXX this is maybe a misnomer because in split valence
          //     basis sets like 6-31G we have more than one basis
          //     function per (valence-)shell and we are actually
          //     looping over the individual contracted GTOs
          for (shell=0; shell < maxshell; shell++) {
            float contracted_gto = 0.0f;

            // Loop over the Gaussian primitives of this contracted 
            // basis function to build the atomic orbital
            // 
            // XXX there's a significant opportunity here for further
            //     speedup if we replace the entire set of primitives
            //     with the single gaussian that they are attempting 
            //     to model.  This could give us another 6x speedup in 
            //     some of the common/simple cases.
            int maxprim = num_prim_per_shell[shell_counter];
            int shelltype = shell_types[shell_counter];
            for (prim=0; prim<maxprim; prim++) {
              float exponent       = basis_array[prim_counter    ];
              float contract_coeff = basis_array[prim_counter + 1];

              // XXX By premultiplying the stored exponent factors etc,
              //     we should be able to use exp2f() rather than exp(),
              //     saving several FLOPS per iteration of this loop
#if defined(__GNUC__) && !defined(__ICC)
              // Use David Hardy's fast spline approximation instead
              // Works well for GCC, but runs slower for Intel C.
              contracted_gto += contract_coeff * aexpfnx(-exponent*dist2);
#elif defined(__ICC)
              // When compiling with ICC, we'll use an inlined 
              // single-precision expf() implementation based on the
              // cephes math library found on Netlib.  This outruns the
              // standard glibc expf() by over 2x in this algorithm.
              contracted_gto += contract_coeff * cephesfastexpf(-exponent*dist2);
#else
              // XXX By far the most costly operation here is exp(),
              //     for gcc builds, exp() accounts for 90% of the runtime
              contracted_gto += contract_coeff * expf(-exponent*dist2);
#endif
              prim_counter += 2;
            }

            /* multiply with the appropriate wavefunction coefficient */
            float tmpshell=0;
            switch (shelltype) {
              case S_SHELL:
                value += wave_f[ifunc++] * contracted_gto;
                break;

              case P_SHELL:
                tmpshell += wave_f[ifunc++] * xdist;
                tmpshell += wave_f[ifunc++] * ydist;
                tmpshell += wave_f[ifunc++] * zdist;
                value += tmpshell * contracted_gto;
                break;

              case D_SHELL:
                tmpshell += wave_f[ifunc++] * xdist2;
                tmpshell += wave_f[ifunc++] * xdist * ydist;
                tmpshell += wave_f[ifunc++] * ydist2;
                tmpshell += wave_f[ifunc++] * xdist * zdist;
                tmpshell += wave_f[ifunc++] * ydist * zdist;
                tmpshell += wave_f[ifunc++] * zdist2;
                value += tmpshell * contracted_gto;
                break;

              case F_SHELL:
                tmpshell += wave_f[ifunc++] * xdist3;         // xxx
                tmpshell += wave_f[ifunc++] * xdist2 * ydist; // xxy
                tmpshell += wave_f[ifunc++] * ydist2 * xdist; // xyy
                tmpshell += wave_f[ifunc++] * ydist3;         // yyy
                tmpshell += wave_f[ifunc++] * xdist2 * zdist; // xxz
                tmpshell += wave_f[ifunc++] * xdist * ydist * zdist; // xyz
                tmpshell += wave_f[ifunc++] * ydist2 * zdist; // yyz
                tmpshell += wave_f[ifunc++] * zdist2 * xdist; // xzz
                tmpshell += wave_f[ifunc++] * zdist2 * ydist; // yzz
                tmpshell += wave_f[ifunc++] * zdist3;         // zzz
                value += tmpshell * contracted_gto;
                break;
 
              case G_SHELL:
                tmpshell += wave_f[ifunc++] * xdist2 * xdist2; // xxxx
                tmpshell += wave_f[ifunc++] * xdist3 * ydist;  // xxxy
                tmpshell += wave_f[ifunc++] * xdist2 * ydist2; // xxyy
                tmpshell += wave_f[ifunc++] * ydist3 * xdist;  // xyyy
                tmpshell += wave_f[ifunc++] * ydist2 * ydist2; // yyyy
                tmpshell += wave_f[ifunc++] * xdist3 * zdist;  // xxxz
                tmpshell += wave_f[ifunc++] * xdist2 * ydist * zdist; // xxyz
                tmpshell += wave_f[ifunc++] * ydist2 * xdist * zdist; // xyyz
                tmpshell += wave_f[ifunc++] * ydist3 * zdist;  // yyyz
                tmpshell += wave_f[ifunc++] * xdist2 * zdist2; // xxzz
                tmpshell += wave_f[ifunc++] * zdist2 * xdist * ydist; // xyzz
                tmpshell += wave_f[ifunc++] * ydist2 * zdist2; // yyzz
                tmpshell += wave_f[ifunc++] * zdist3 * xdist;  // zzzx
                tmpshell += wave_f[ifunc++] * zdist3 * ydist;  // zzzy
                tmpshell += wave_f[ifunc++] * zdist2 * zdist2; // zzzz
                value += tmpshell * contracted_gto;
                break;

              default:
#if 1
                // avoid unnecessary branching and minimize use of pow()
                int i, j; 
                float xdp, ydp, zdp;
                float xdiv = 1.0f / xdist;
                for (j=0, zdp=1.0f; j<=shelltype; j++, zdp*=zdist) {
                  int imax = shelltype - j; 
                  for (i=0, ydp=1.0f, xdp=powf(xdist, float(imax)); i<=imax; i++, ydp*=ydist, xdp*=xdiv) {
                    tmpshell += wave_f[ifunc++] * xdp * ydp * zdp;
                  }
                }
                value += tmpshell * contracted_gto;
#else
                int i, j, k;
                for (k=0; k<=shelltype; k++) {
                  for (j=0; j<=shelltype; j++) {
                    for (i=0; i<=shelltype; i++) {
                      if (i+j+k==shelltype) {
                        value += wave_f[ifunc++] * contracted_gto
                          * pow(xdist,i) * pow(ydist,j) * pow(zdist,k);
                      }
                    }
                  }
                }
#endif
            } // end switch

            shell_counter++;
          } // end shell
        } // end atom

        // return either orbital density or orbital wavefunction amplitude
        if (density) {
          float orbdensity = value * value;
          if (value < 0.0)
            orbdensity = -orbdensity;
          orbitalgrid[gaddrzy + nx] = orbdensity;
        } else {
          orbitalgrid[gaddrzy + nx] = value;
        }
      }
    }
  }

  return 0;
}



#if defined(VMDORBUSESSE) && defined(__SSE2__)

//
// Adaptation of the Cephes exp() to an SSE-ized exp_ps() routine 
// originally by Julien Pommier
//   Copyright (C) 2007  Julien Pommier, ZLIB license
//   http://gruntthepeon.free.fr/ssemath/
// 
#ifdef _MSC_VER /* visual c++ */
# define ALIGN16_BEG __declspec(align(16))
# define ALIGN16_END
#else /* gcc or icc */
# define ALIGN16_BEG
# define ALIGN16_END __attribute__((aligned(16)))
#endif

#define _PS_CONST(Name, Val)                                            \
  static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

_PS_CONST(exp_hi,       88.3762626647949f);
_PS_CONST(exp_lo,       -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS_CONST(cephes_exp_C1, 0.693359375);
_PS_CONST(cephes_exp_C2, -2.12194440e-4);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1);
_PS_CONST(one, 1.0);
_PS_CONST(half, 0.5);

_PI32_CONST(0x7f, 0x7f);

typedef union xmm_mm_union {
  __m128 xmm;
  __m64 mm[2];
} xmm_mm_union;

#define COPY_XMM_TO_MM(xmm_, mm0_, mm1_) {          \
    xmm_mm_union u; u.xmm = xmm_;                   \
    mm0_ = u.mm[0];                                 \
    mm1_ = u.mm[1];                                 \
}

#define COPY_MM_TO_XMM(mm0_, mm1_, xmm_) {                         \
    xmm_mm_union u; u.mm[0]=mm0_; u.mm[1]=mm1_; xmm_ = u.xmm;      \
  }

__m128 exp_ps(__m128 x) {
  __m128 tmp = _mm_setzero_ps(), fx;
  __m64 mm0, mm1;

  x = _mm_min_ps(x, *(__m128*)_ps_exp_hi);
  x = _mm_max_ps(x, *(__m128*)_ps_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm_mul_ps(x, *(__m128*)_ps_cephes_LOG2EF);
  fx = _mm_add_ps(fx,*(__m128*)_ps_half);

  /* how to perform a floorf with SSE: just below */
  /* step 1 : cast to int */
  tmp = _mm_movehl_ps(tmp, fx);
  mm0 = _mm_cvttps_pi32(fx);
  mm1 = _mm_cvttps_pi32(tmp);
  /* step 2 : cast back to float */
  tmp = _mm_cvtpi32x2_ps(mm0, mm1);
  /* if greater, substract 1 */
  __m128 mask = _mm_cmpgt_ps(tmp, fx);
  mask = _mm_and_ps(mask, *(__m128*)_ps_one);
  fx = _mm_sub_ps(tmp, mask);

  tmp = _mm_mul_ps(fx, *(__m128*)_ps_cephes_exp_C1);
  __m128 z = _mm_mul_ps(fx, *(__m128*)_ps_cephes_exp_C2);
  x = _mm_sub_ps(x, tmp);
  x = _mm_sub_ps(x, z);

  z = _mm_mul_ps(x,x);

  __m128 y = *(__m128*)_ps_cephes_exp_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p5);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_one);

  /* build 2^n */
  z = _mm_movehl_ps(z, fx);
  mm0 = _mm_cvttps_pi32(fx);
  mm1 = _mm_cvttps_pi32(z);
  mm0 = _mm_add_pi32(mm0, *(__m64*)_pi32_0x7f);
  mm1 = _mm_add_pi32(mm1, *(__m64*)_pi32_0x7f);
  mm0 = _mm_slli_pi32(mm0, 23);
  mm1 = _mm_slli_pi32(mm1, 23);

  __m128 pow2n;
  COPY_MM_TO_XMM(mm0, mm1, pow2n);

  y = _mm_mul_ps(y, pow2n);
  _mm_empty();
  return y;
}


//
// David J. Hardy
// 12 Dec 2008
//
// aexpfnxsse() - SSE2 version of aexpfnx().
//
//
#if defined(__GNUC__) && ! defined(__INTEL_COMPILER)
#define __align(X)  __attribute__((aligned(X) ))
#if (__GNUC__ < 4)
#define MISSING_mm_cvtsd_f64
#endif
#else
#define __align(X) __declspec(align(X) )
#endif

typedef union SSEreg_t {
  __m128  f;  // 4x float (SSE)
  __m128i i;  // 4x 32-bit int (SSE2)
  struct {
    int r0, r1, r2, r3;  // get the individual registers
  } reg;
} SSEreg;

#define MLOG2EF    -1.44269504088896f

/*
 * Interpolating coefficients for linear blending of the
 * 3rd degree Taylor expansion of 2^x about 0 and -1.
 */
#define SCEXP0     1.0000000000000000f
#define SCEXP1     0.6987082824680118f
#define SCEXP2     0.2633174272827404f
#define SCEXP3     0.0923611991471395f
#define SCEXP4     0.0277520543324108f

/* for single precision float */
#define EXPOBIAS   127
#define EXPOSHIFT   23

/* cutoff is optional, but can help avoid unnecessary work */
#define ACUTOFF    -10

__m128 aexpfnxsse(__m128 x) {
  __align(16) SSEreg scal;
  __align(16) SSEreg n;
  __align(16) SSEreg y;

  scal.f = _mm_cmpge_ps(x, _mm_set_ps1(ACUTOFF));  /* Is x within cutoff? */

  /* If all x are outside of cutoff, return 0s. */
  if (_mm_movemask_ps(scal.f) == 0) {
    return _mm_setzero_ps();
  }
  /* Otherwise, scal.f contains mask to be ANDed with the scale factor */

  /*
   * Convert base:  exp(x) = 2^(N-d) where N is integer and 0 <= d < 1.
   *
   * Below we calculate n=N and x=-d, with "y" for temp storage,
   * calculate floor of x*log2(e) and subtract to get -d.
   */
  y.f = _mm_mul_ps(x, _mm_set_ps1(MLOG2EF));
  n.i = _mm_cvttps_epi32(y.f);
  x = _mm_cvtepi32_ps(n.i);
  x = _mm_sub_ps(x, y.f);

  /*
   * Approximate 2^{-d}, 0 <= d < 1, by interpolation.
   * Perform Horner's method to evaluate interpolating polynomial.
   */
  y.f = _mm_mul_ps(x, _mm_set_ps1(SCEXP4));      /* for x^4 term */
  y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP3));    /* for x^3 term */
  y.f = _mm_mul_ps(y.f, x);
  y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP2));    /* for x^2 term */
  y.f = _mm_mul_ps(y.f, x);
  y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP1));    /* for x^1 term */
  y.f = _mm_mul_ps(y.f, x);
  y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP0));    /* for x^0 term */

  /*
   * Calculate 2^N exactly by directly manipulating floating point exponent.
   * Bitwise AND the result with scal.f mask to create the scale factor,
   * then use it to scale y for the final result.
   */
  n.i = _mm_sub_epi32(_mm_set1_epi32(EXPOBIAS), n.i);
  n.i = _mm_slli_epi32(n.i, EXPOSHIFT);
  scal.f = _mm_and_ps(scal.f, n.f);
  y.f = _mm_mul_ps(y.f, scal.f);

  return y.f;
}


int evaluate_grid_sse(int numatoms,
                  const float *wave_f, const float *basis_array,
                  const float *atompos,
                  const int *atom_basis,
                  const int *num_shells_per_atom,
                  const int *num_prim_per_shell,
                  const int *shell_types,
                  const int *numvoxels,
                  float voxelsize,
                  const float *origin,
                  int density,
                  float * orbitalgrid) {
  if (!orbitalgrid)
    return -1;

  int nx, ny, nz;
  __attribute__((aligned(16))) float sxdelta[4]; // 16-byte aligned for SSE
  for (nx=0; nx<4; nx++) 
    sxdelta[nx] = ((float) nx) * voxelsize * ANGS_TO_BOHR;

  // Calculate the value of the orbital at each gridpoint and store in 
  // the current oribtalgrid array
  int numgridxy = numvoxels[0]*numvoxels[1];
  for (nz=0; nz<numvoxels[2]; nz++) {
    float grid_x, grid_y, grid_z;
    grid_z = origin[2] + nz * voxelsize;
    for (ny=0; ny<numvoxels[1]; ny++) {
      grid_y = origin[1] + ny * voxelsize;
      int gaddrzy = ny*numvoxels[0] + nz*numgridxy;
      for (nx=0; nx<numvoxels[0]; nx+=4) {
        grid_x = origin[0] + nx * voxelsize;

        // calculate the value of the wavefunction of the
        // selected orbital at the current grid point
        int at;
        int prim, shell;

        // initialize value of orbital at gridpoint
        __m128 value = _mm_setzero_ps();

        // initialize the wavefunction and shell counters
        int ifunc = 0; 
        int shell_counter = 0;

        // loop over all the QM atoms
        for (at=0; at<numatoms; at++) {
          int maxshell = num_shells_per_atom[at];
          int prim_counter = atom_basis[at];

          // calculate distance between grid point and center of atom
          float sxdist = (grid_x - atompos[3*at  ])*ANGS_TO_BOHR;
          float sydist = (grid_y - atompos[3*at+1])*ANGS_TO_BOHR;
          float szdist = (grid_z - atompos[3*at+2])*ANGS_TO_BOHR;

          float sydist2 = sydist*sydist;
          float szdist2 = szdist*szdist;
          float yzdist2 = sydist2 + szdist2;

          __m128 xdelta = _mm_load_ps(&sxdelta[0]); // aligned load
          __m128 xdist  = _mm_load_ps1(&sxdist);
          xdist = _mm_add_ps(xdist, xdelta);
          __m128 ydist  = _mm_load_ps1(&sydist);
          __m128 zdist  = _mm_load_ps1(&szdist);
          __m128 xdist2 = _mm_mul_ps(xdist, xdist);
          __m128 ydist2 = _mm_mul_ps(ydist, ydist);
          __m128 zdist2 = _mm_mul_ps(zdist, zdist);
          __m128 dist2  = _mm_load_ps1(&yzdist2); 
          dist2 = _mm_add_ps(dist2, xdist2);
 
          // loop over the shells belonging to this atom
          // XXX this is maybe a misnomer because in split valence
          //     basis sets like 6-31G we have more than one basis
          //     function per (valence-)shell and we are actually
          //     looping over the individual contracted GTOs
          for (shell=0; shell < maxshell; shell++) {
            __m128 contracted_gto = _mm_setzero_ps();

            // Loop over the Gaussian primitives of this contracted 
            // basis function to build the atomic orbital
            // 
            // XXX there's a significant opportunity here for further
            //     speedup if we replace the entire set of primitives
            //     with the single gaussian that they are attempting 
            //     to model.  This could give us another 6x speedup in 
            //     some of the common/simple cases.
            int maxprim = num_prim_per_shell[shell_counter];
            int shelltype = shell_types[shell_counter];
            for (prim=0; prim<maxprim; prim++) {
              // XXX pre-negate exponent value
              float exponent       = -basis_array[prim_counter    ];
              float contract_coeff =  basis_array[prim_counter + 1];

              // contracted_gto += contract_coeff * exp(-exponent*dist2);
              __m128 expval = _mm_mul_ps(_mm_load_ps1(&exponent), dist2);
              // SSE expf() required here
#if 1
              __m128 retval = aexpfnxsse(expval);
#else
              __m128 retval = exp_ps(expval);
#endif
              __m128 ctmp = _mm_mul_ps(_mm_load_ps1(&contract_coeff), retval);
              contracted_gto = _mm_add_ps(contracted_gto, ctmp);
              prim_counter += 2;
            }

            /* multiply with the appropriate wavefunction coefficient */
            __m128 tmpshell = _mm_setzero_ps();
            switch (shelltype) {
              case S_SHELL:
                value = _mm_add_ps(value, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), contracted_gto));
                break;

              case P_SHELL:
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), xdist));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), ydist));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), zdist));
                value = _mm_add_ps(value, _mm_mul_ps(tmpshell, contracted_gto));
                break;

              case D_SHELL:
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), xdist2));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(xdist, ydist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), ydist2));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(xdist, zdist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(ydist, zdist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), zdist2));
                value = _mm_add_ps(value, _mm_mul_ps(tmpshell, contracted_gto));
                break;

              case F_SHELL:
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(xdist2, xdist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(xdist2, ydist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(ydist2, xdist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(ydist2, ydist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(xdist2, zdist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(_mm_mul_ps(xdist, ydist), zdist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(ydist2, zdist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(zdist2, xdist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(zdist2, ydist)));
                tmpshell = _mm_add_ps(tmpshell, _mm_mul_ps(_mm_load_ps1(&wave_f[ifunc++]), _mm_mul_ps(zdist2, zdist)));
                value = _mm_add_ps(value, _mm_mul_ps(tmpshell, contracted_gto));
                break;
 
#if 0
              default:
                // avoid unnecessary branching and minimize use of pow()
                int i, j; 
                float xdp, ydp, zdp;
                float xdiv = 1.0f / xdist;
                for (j=0, zdp=1.0f; j<=shelltype; j++, zdp*=zdist) {
                  int imax = shelltype - j; 
                  for (i=0, ydp=1.0f, xdp=pow(xdist, imax); i<=imax; i++, ydp*=ydist, xdp*=xdiv) {
                    tmpshell += wave_f[ifunc++] * xdp * ydp * zdp;
                  }
                }
                value += tmpshell * contracted_gto;
#endif
            } // end switch

            shell_counter++;
          } // end shell
        } // end atom

        // return either orbital density or orbital wavefunction amplitude
        if (density) {
          __m128 mask = _mm_cmplt_ps(value, _mm_setzero_ps());
          __m128 sqdensity = _mm_mul_ps(value, value);
          __m128 orbdensity = sqdensity;
          __m128 nsqdensity = _mm_and_ps(sqdensity, mask);
          orbdensity = _mm_sub_ps(orbdensity, nsqdensity);
          orbdensity = _mm_sub_ps(orbdensity, nsqdensity);
          _mm_storeu_ps(&orbitalgrid[gaddrzy + nx], orbdensity);
        } else {
          _mm_storeu_ps(&orbitalgrid[gaddrzy + nx], value);
        }
      }
    }
  }

  return 0;
}

#endif


#if defined(VMDORBUSEAVX512) && defined(__AVX512F__) && defined(__AVX512ER__)

// On Xeon Phi, we will depend on the use of AVX512ER 
// reciprocal and exponential instructions rather than using
// our own hand-coded approximations for these functions.
// On future Xeon processors that lack the AVX512ER subset,
// we will again need the approximations, but we can wait to
// implement these until we determine what the performance 
// characteristics will likely be.
//

#if defined(__GNUC__) && ! defined(__INTEL_COMPILER)
#define __align(X)  __attribute__((aligned(X) ))
#else
#define __align(X) __declspec(align(X) )
#endif

#define MLOG2EF    -1.44269504088896f

#if 0
static void print_mm512_ps(__m512 v) {
  __attribute__((aligned(64))) float tmp[16]; // 64-byte aligned for AVX512
  _mm512_storeu_ps(&tmp[0], v);

  printf("mm512: ");
  int i;
  for (i=0; i<16; i++) 
    printf("%g ", tmp[i]);
  printf("\n");
}
#endif


//
// John Stone, March 2017
//
// aexpfnxavx512f() - AVX-512F version of aexpfnx().
//

/*
 * Interpolating coefficients for linear blending of the
 * 3rd degree Taylor expansion of 2^x about 0 and -1.
 */
#define SCEXP0     1.0000000000000000f
#define SCEXP1     0.6987082824680118f
#define SCEXP2     0.2633174272827404f
#define SCEXP3     0.0923611991471395f
#define SCEXP4     0.0277520543324108f

/* for single precision float */
#define EXPOBIAS   127
#define EXPOSHIFT   23

/* cutoff is optional, but can help avoid unnecessary work */
#define ACUTOFF    -10

typedef union AVX512reg_t {
  __m512  f;  // 16x float (AVX-512F)
  __m512i i;  // 16x 32-bit int (AVX-512F)
} AVX512reg;

__m512 aexpfnxavx512f(__m512 x) {
  __mmask16 mask;
  mask = _mm512_cmple_ps_mask(_mm512_set1_ps(ACUTOFF), x); // Is x within cutoff?
#if 0
  // If all x are outside of cutoff, return 0s.
  if (_mm512_movemask_ps(scal.f) == 0) {
    return _mm512_set1_ps(0.0f);
  }
  // Otherwise, scal.f contains mask to be ANDed with the scale factor
#endif

  /*
   * Convert base:  exp(x) = 2^(N-d) where N is integer and 0 <= d < 1.
   *
   * Below we calculate n=N and x=-d, with "y" for temp storage,
   * calculate floor of x*log2(e) and subtract to get -d.
   */
  __m512 mb = _mm512_mul_ps(x, _mm512_set1_ps(MLOG2EF));
  __m512 mbflr = _mm512_floor_ps(mb);
  __m512 d = _mm512_sub_ps(mbflr, mb);
  __m512 y;

  // Approximate 2^{-d}, 0 <= d < 1, by interpolation.
  // Perform Horner's method to evaluate interpolating polynomial.
  y = _mm512_fmadd_ps(d, _mm512_set1_ps(SCEXP4), _mm512_set1_ps(SCEXP3));
  y = _mm512_fmadd_ps(y, d, _mm512_set1_ps(SCEXP2));
  y = _mm512_fmadd_ps(y, d, _mm512_set1_ps(SCEXP1));
  y = _mm512_fmadd_ps(y, d, _mm512_set1_ps(SCEXP0));

  // Calculate 2^N exactly by directly manipulating floating point exponent,
  // then use it to scale y for the final result.
  __align(64) AVX512reg n;
  n.i = _mm512_sub_epi32(_mm512_set1_epi32(EXPOBIAS), n.i);
  n.i = _mm512_slli_epi32(n.i, EXPOSHIFT);
  n.f = _mm512_mask_mul_ps(n.f, mask, _mm512_set1_ps(0.0f), n.f);
  y = _mm512_mul_ps(y, n.f);
  return y;
}


//
// AVX-512F implementation for Xeons that don't have special fctn units
//
int evaluate_grid_avx512f(int numatoms,
                          const float *wave_f, const float *basis_array,
                          const float *atompos,
                          const int *atom_basis,
                          const int *num_shells_per_atom,
                          const int *num_prim_per_shell,
                          const int *shell_types,
                          const int *numvoxels,
                          float voxelsize,
                          const float *origin,
                          int density,
                          float * orbitalgrid) {
  if (!orbitalgrid)
    return -1;

  int nx, ny, nz;
  __attribute__((aligned(64))) float sxdelta[16]; // 64-byte aligned for AVX512
  for (nx=0; nx<16; nx++) 
    sxdelta[nx] = ((float) nx) * voxelsize * ANGS_TO_BOHR;

  // Calculate the value of the orbital at each gridpoint and store in 
  // the current oribtalgrid array
  int numgridxy = numvoxels[0]*numvoxels[1];
  for (nz=0; nz<numvoxels[2]; nz++) {
    float grid_x, grid_y, grid_z;
    grid_z = origin[2] + nz * voxelsize;
    for (ny=0; ny<numvoxels[1]; ny++) {
      grid_y = origin[1] + ny * voxelsize;
      int gaddrzy = ny*numvoxels[0] + nz*numgridxy;
      for (nx=0; nx<numvoxels[0]; nx+=16) {
        grid_x = origin[0] + nx * voxelsize;

        // calculate the value of the wavefunction of the
        // selected orbital at the current grid point
        int at;
        int prim, shell;

        // initialize value of orbital at gridpoint
        __m512 value = _mm512_set1_ps(0.0f);

        // initialize the wavefunction and shell counters
        int ifunc = 0; 
        int shell_counter = 0;

        // loop over all the QM atoms
        for (at=0; at<numatoms; at++) {
          int maxshell = num_shells_per_atom[at];
          int prim_counter = atom_basis[at];

          // calculate distance between grid point and center of atom
          float sxdist = (grid_x - atompos[3*at  ])*ANGS_TO_BOHR;
          float sydist = (grid_y - atompos[3*at+1])*ANGS_TO_BOHR;
          float szdist = (grid_z - atompos[3*at+2])*ANGS_TO_BOHR;

          float sydist2 = sydist*sydist;
          float szdist2 = szdist*szdist;
          float yzdist2 = sydist2 + szdist2;

          __m512 xdelta = _mm512_load_ps(&sxdelta[0]); // aligned load
          __m512 xdist  = _mm512_set1_ps(sxdist);
          xdist = _mm512_add_ps(xdist, xdelta);
          __m512 ydist  = _mm512_set1_ps(sydist);
          __m512 zdist  = _mm512_set1_ps(szdist);
          __m512 xdist2 = _mm512_mul_ps(xdist, xdist);
          __m512 ydist2 = _mm512_mul_ps(ydist, ydist);
          __m512 zdist2 = _mm512_mul_ps(zdist, zdist);
          __m512 dist2  = _mm512_set1_ps(yzdist2); 
          dist2 = _mm512_add_ps(dist2, xdist2);
 
          // loop over the shells belonging to this atom
          // XXX this is maybe a misnomer because in split valence
          //     basis sets like 6-31G we have more than one basis
          //     function per (valence-)shell and we are actually
          //     looping over the individual contracted GTOs
          for (shell=0; shell < maxshell; shell++) {
            __m512 contracted_gto = _mm512_set1_ps(0.0f);

            // Loop over the Gaussian primitives of this contracted 
            // basis function to build the atomic orbital
            // 
            // XXX there's a significant opportunity here for further
            //     speedup if we replace the entire set of primitives
            //     with the single gaussian that they are attempting 
            //     to model.  This could give us another 6x speedup in 
            //     some of the common/simple cases.
            int maxprim = num_prim_per_shell[shell_counter];
            int shelltype = shell_types[shell_counter];
            for (prim=0; prim<maxprim; prim++) {
              // XXX pre-negate exponent value
              float exponent       = -basis_array[prim_counter    ];
              float contract_coeff =  basis_array[prim_counter + 1];

              // contracted_gto += contract_coeff * exp(-exponent*dist2);
              __m512 expval = _mm512_mul_ps(_mm512_set1_ps(-exponent * MLOG2EF), dist2);
              // exp2f() equivalent required, use base-2 approximation
              __m512 retval = aexpfnxavx512f(expval);
              contracted_gto = _mm512_fmadd_ps(_mm512_set1_ps(contract_coeff), retval, contracted_gto);

              prim_counter += 2;
            }

            /* multiply with the appropriate wavefunction coefficient */
            __m512 tmpshell = _mm512_set1_ps(0.0f);
            switch (shelltype) {
              // use FMADD instructions
              case S_SHELL:
                value = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), contracted_gto, value);
                break;

              case P_SHELL:
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), xdist, tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), ydist, tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), zdist, tmpshell);
                value = _mm512_fmadd_ps(tmpshell, contracted_gto, value);
                break;

              case D_SHELL:
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), xdist2, tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist, ydist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), ydist2, tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist, zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(ydist, zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), zdist2, tmpshell);
                value = _mm512_fmadd_ps(tmpshell, contracted_gto, value);
                break;

              case F_SHELL:
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist2, xdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist2, ydist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(ydist2, xdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(ydist2, ydist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist2, zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(_mm512_mul_ps(xdist, ydist), zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(ydist2, zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(zdist2, xdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(zdist2, ydist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(zdist2, zdist), tmpshell);
                value = _mm512_fmadd_ps(tmpshell, contracted_gto, value);
                break;
 
#if 0
              default:
                // avoid unnecessary branching and minimize use of pow()
                int i, j; 
                float xdp, ydp, zdp;
                float xdiv = 1.0f / xdist;
                for (j=0, zdp=1.0f; j<=shelltype; j++, zdp*=zdist) {
                  int imax = shelltype - j; 
                  for (i=0, ydp=1.0f, xdp=pow(xdist, imax); i<=imax; i++, ydp*=ydist, xdp*=xdiv) {
                    tmpshell += wave_f[ifunc++] * xdp * ydp * zdp;
                  }
                }
                value += tmpshell * contracted_gto;
#endif
            } // end switch

            shell_counter++;
          } // end shell
        } // end atom

        // return either orbital density or orbital wavefunction amplitude
        if (density) {
          __mmask16 mask = _mm512_cmplt_ps_mask(value, _mm512_set1_ps(0.0f));
          __m512 sqdensity = _mm512_mul_ps(value, value);
          __m512 orbdensity = _mm512_mask_mul_ps(sqdensity, mask, sqdensity,
                                                 _mm512_set1_ps(-1.0f));
          _mm512_storeu_ps(&orbitalgrid[gaddrzy + nx], orbdensity);
        } else {
          _mm512_storeu_ps(&orbitalgrid[gaddrzy + nx], value);
        }
      }
    }
  }

  return 0;
}


//
// AVX-512ER implementation for Xeon Phi w/ special fctn units
//
int evaluate_grid_avx512er(int numatoms,
                           const float *wave_f, const float *basis_array,
                           const float *atompos,
                           const int *atom_basis,
                           const int *num_shells_per_atom,
                           const int *num_prim_per_shell,
                           const int *shell_types,
                           const int *numvoxels,
                           float voxelsize,
                           const float *origin,
                           int density,
                           float * orbitalgrid) {
  if (!orbitalgrid)
    return -1;

  int nx, ny, nz;
  __attribute__((aligned(64))) float sxdelta[16]; // 64-byte aligned for AVX512
  for (nx=0; nx<16; nx++) 
    sxdelta[nx] = ((float) nx) * voxelsize * ANGS_TO_BOHR;

  // Calculate the value of the orbital at each gridpoint and store in 
  // the current oribtalgrid array
  int numgridxy = numvoxels[0]*numvoxels[1];
  for (nz=0; nz<numvoxels[2]; nz++) {
    float grid_x, grid_y, grid_z;
    grid_z = origin[2] + nz * voxelsize;
    for (ny=0; ny<numvoxels[1]; ny++) {
      grid_y = origin[1] + ny * voxelsize;
      int gaddrzy = ny*numvoxels[0] + nz*numgridxy;
      for (nx=0; nx<numvoxels[0]; nx+=16) {
        grid_x = origin[0] + nx * voxelsize;

        // calculate the value of the wavefunction of the
        // selected orbital at the current grid point
        int at;
        int prim, shell;

        // initialize value of orbital at gridpoint
        __m512 value = _mm512_set1_ps(0.0f);

        // initialize the wavefunction and shell counters
        int ifunc = 0; 
        int shell_counter = 0;

        // loop over all the QM atoms
        for (at=0; at<numatoms; at++) {
          int maxshell = num_shells_per_atom[at];
          int prim_counter = atom_basis[at];

          // calculate distance between grid point and center of atom
          float sxdist = (grid_x - atompos[3*at  ])*ANGS_TO_BOHR;
          float sydist = (grid_y - atompos[3*at+1])*ANGS_TO_BOHR;
          float szdist = (grid_z - atompos[3*at+2])*ANGS_TO_BOHR;

          float sydist2 = sydist*sydist;
          float szdist2 = szdist*szdist;
          float yzdist2 = sydist2 + szdist2;

          __m512 xdelta = _mm512_load_ps(&sxdelta[0]); // aligned load
          __m512 xdist  = _mm512_set1_ps(sxdist);
          xdist = _mm512_add_ps(xdist, xdelta);
          __m512 ydist  = _mm512_set1_ps(sydist);
          __m512 zdist  = _mm512_set1_ps(szdist);
          __m512 xdist2 = _mm512_mul_ps(xdist, xdist);
          __m512 ydist2 = _mm512_mul_ps(ydist, ydist);
          __m512 zdist2 = _mm512_mul_ps(zdist, zdist);
          __m512 dist2  = _mm512_set1_ps(yzdist2); 
          dist2 = _mm512_add_ps(dist2, xdist2);
 
          // loop over the shells belonging to this atom
          // XXX this is maybe a misnomer because in split valence
          //     basis sets like 6-31G we have more than one basis
          //     function per (valence-)shell and we are actually
          //     looping over the individual contracted GTOs
          for (shell=0; shell < maxshell; shell++) {
            __m512 contracted_gto = _mm512_set1_ps(0.0f);

            // Loop over the Gaussian primitives of this contracted 
            // basis function to build the atomic orbital
            // 
            // XXX there's a significant opportunity here for further
            //     speedup if we replace the entire set of primitives
            //     with the single gaussian that they are attempting 
            //     to model.  This could give us another 6x speedup in 
            //     some of the common/simple cases.
            int maxprim = num_prim_per_shell[shell_counter];
            int shelltype = shell_types[shell_counter];
            for (prim=0; prim<maxprim; prim++) {
              // XXX pre-negate exponent value
              float exponent       = -basis_array[prim_counter    ];
              float contract_coeff =  basis_array[prim_counter + 1];

              // contracted_gto += contract_coeff * exp(-exponent*dist2);
#if 1
              __m512 expval = _mm512_mul_ps(_mm512_set1_ps(-exponent * MLOG2EF), dist2);
              // expf() equivalent required, use base-2 AVX-512ER instructions
              __m512 retval = _mm512_exp2a23_ps(expval);
              contracted_gto = _mm512_fmadd_ps(_mm512_set1_ps(contract_coeff), retval, contracted_gto);
#else
              __m512 expval = _mm512_mul_ps(_mm512_set1_ps(-exponent), dist2);
              // expf() equivalent required, use base-2 AVX-512ER instructions
              expval = _mm512_mul_ps(expval, _mm512_set1_ps(MLOG2EF));
              __m512 retval = _mm512_exp2a23_ps(expval);
              __m512 ctmp = _mm512_mul_ps(_mm512_set1_ps(contract_coeff), retval);
              contracted_gto = _mm512_add_ps(contracted_gto, ctmp);
#endif

              prim_counter += 2;
            }

            /* multiply with the appropriate wavefunction coefficient */
            __m512 tmpshell = _mm512_set1_ps(0.0f);
            switch (shelltype) {
              // use FMADD instructions
              case S_SHELL:
                value = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), contracted_gto, value);
                break;

              case P_SHELL:
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), xdist, tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), ydist, tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), zdist, tmpshell);
                value = _mm512_fmadd_ps(tmpshell, contracted_gto, value);
                break;

              case D_SHELL:
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), xdist2, tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist, ydist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), ydist2, tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist, zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(ydist, zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), zdist2, tmpshell);
                value = _mm512_fmadd_ps(tmpshell, contracted_gto, value);
                break;

              case F_SHELL:
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist2, xdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist2, ydist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(ydist2, xdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(ydist2, ydist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(xdist2, zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(_mm512_mul_ps(xdist, ydist), zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(ydist2, zdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(zdist2, xdist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(zdist2, ydist), tmpshell);
                tmpshell = _mm512_fmadd_ps(_mm512_set1_ps(wave_f[ifunc++]), _mm512_mul_ps(zdist2, zdist), tmpshell);
                value = _mm512_fmadd_ps(tmpshell, contracted_gto, value);
                break;

 
#if 0
              default:
                // avoid unnecessary branching and minimize use of pow()
                int i, j; 
                float xdp, ydp, zdp;
                float xdiv = 1.0f / xdist;
                for (j=0, zdp=1.0f; j<=shelltype; j++, zdp*=zdist) {
                  int imax = shelltype - j; 
                  for (i=0, ydp=1.0f, xdp=pow(xdist, imax); i<=imax; i++, ydp*=ydist, xdp*=xdiv) {
                    tmpshell += wave_f[ifunc++] * xdp * ydp * zdp;
                  }
                }
                value += tmpshell * contracted_gto;
#endif
            } // end switch

            shell_counter++;
          } // end shell
        } // end atom

        // return either orbital density or orbital wavefunction amplitude
        if (density) {
          __mmask16 mask = _mm512_cmplt_ps_mask(value, _mm512_set1_ps(0.0f));
          __m512 sqdensity = _mm512_mul_ps(value, value);
          __m512 orbdensity = _mm512_mask_mul_ps(sqdensity, mask, sqdensity,
                                                 _mm512_set1_ps(-1.0f));
          _mm512_storeu_ps(&orbitalgrid[gaddrzy + nx], orbdensity);
        } else {
          _mm512_storeu_ps(&orbitalgrid[gaddrzy + nx], value);
        }
      }
    }
  }

  return 0;
}

#endif




#if defined(VMDORBUSEVSX) && defined(__VSX__)
//
// John Stone, June 2016
//
// aexpfnxsse() - VSX version of aexpfnx().
//
#if defined(__GNUC__) && ! defined(__INTEL_COMPILER)
#define __align(X)  __attribute__((aligned(X) ))
#else
#define __align(X) __declspec(align(X) )
#endif

#define MLOG2EF    -1.44269504088896f

/*
 * Interpolating coefficients for linear blending of the
 * 3rd degree Taylor expansion of 2^x about 0 and -1.
 */
#define SCEXP0     1.0000000000000000f
#define SCEXP1     0.6987082824680118f
#define SCEXP2     0.2633174272827404f
#define SCEXP3     0.0923611991471395f
#define SCEXP4     0.0277520543324108f

/* for single precision float */
#define EXPOBIAS   127
#define EXPOSHIFT   23

/* cutoff is optional, but can help avoid unnecessary work */
#define ACUTOFF    -10

#if 0
vector float ref_expf(vector float x) {
  vector float result;

  int i;
  for (i=0; i<4; i++) {
    result[i] = expf(x[i]);
  }

  return result;
}
#endif

vector float aexpfnxvsx(vector float x) {
  // scal.f = _mm_cmpge_ps(x, _mm_set_ps1(ACUTOFF));  /* Is x within cutoff? */
  // 
  // If all x are outside of cutoff, return 0s.
  // if (_mm_movemask_ps(scal.f) == 0) {
  //   return _mm_setzero_ps();
  // }
  // Otherwise, scal.f contains mask to be ANDed with the scale factor

  /*
   * Convert base:  exp(x) = 2^(N-d) where N is integer and 0 <= d < 1.
   *
   * Below we calculate n=N and x=-d, with "y" for temp storage,
   * calculate floor of x*log2(e) and subtract to get -d.
   */
  vector float mb = vec_mul(x, vec_splats(MLOG2EF));
  vector float mbflr = vec_floor(mb);
  vector float d = vec_sub(mbflr, mb);
  vector float y;

  // Approximate 2^{-d}, 0 <= d < 1, by interpolation.
  // Perform Horner's method to evaluate interpolating polynomial.
  y = vec_madd(d, vec_splats(SCEXP4), vec_splats(SCEXP3));
  y = vec_madd(y, d, vec_splats(SCEXP2));
  y = vec_madd(y, d, vec_splats(SCEXP1));
  y = vec_madd(y, d, vec_splats(SCEXP0));

  return vec_mul(y, vec_expte(-mbflr));
}


int evaluate_grid_vsx(int numatoms,
                  const float *wave_f, const float *basis_array,
                  const float *atompos,
                  const int *atom_basis,
                  const int *num_shells_per_atom,
                  const int *num_prim_per_shell,
                  const int *shell_types,
                  const int *numvoxels,
                  float voxelsize,
                  const float *origin,
                  int density,
                  float * orbitalgrid) {
  if (!orbitalgrid)
    return -1;

  int nx, ny, nz;
  __attribute__((aligned(16))) float sxdelta[4]; // 16-byte aligned for VSX
  for (nx=0; nx<4; nx++) 
    sxdelta[nx] = ((float) nx) * voxelsize * ANGS_TO_BOHR;

  // Calculate the value of the orbital at each gridpoint and store in 
  // the current oribtalgrid array
  int numgridxy = numvoxels[0]*numvoxels[1];
  for (nz=0; nz<numvoxels[2]; nz++) {
    float grid_x, grid_y, grid_z;
    grid_z = origin[2] + nz * voxelsize;
    for (ny=0; ny<numvoxels[1]; ny++) {
      grid_y = origin[1] + ny * voxelsize;
      int gaddrzy = ny*numvoxels[0] + nz*numgridxy;
      for (nx=0; nx<numvoxels[0]; nx+=4) {
        grid_x = origin[0] + nx * voxelsize;

        // calculate the value of the wavefunction of the
        // selected orbital at the current grid point
        int at;
        int prim, shell;

        // initialize value of orbital at gridpoint
        vector float value = vec_splats(0.0f); 

        // initialize the wavefunction and shell counters
        int ifunc = 0; 
        int shell_counter = 0;

        // loop over all the QM atoms
        for (at=0; at<numatoms; at++) {
          int maxshell = num_shells_per_atom[at];
          int prim_counter = atom_basis[at];

          // calculate distance between grid point and center of atom
          float sxdist = (grid_x - atompos[3*at  ])*ANGS_TO_BOHR;
          float sydist = (grid_y - atompos[3*at+1])*ANGS_TO_BOHR;
          float szdist = (grid_z - atompos[3*at+2])*ANGS_TO_BOHR;

          float sydist2 = sydist*sydist;
          float szdist2 = szdist*szdist;
          float yzdist2 = sydist2 + szdist2;

          vector float xdelta =  *((__vector float *) &sxdelta[0]); // aligned load
          vector float xdist  = vec_splats(sxdist);
          xdist = vec_add(xdist, xdelta);
          vector float ydist  = vec_splats(sydist);
          vector float zdist  = vec_splats(szdist);
          vector float xdist2 = vec_mul(xdist, xdist);
          vector float ydist2 = vec_mul(ydist, ydist);
          vector float zdist2 = vec_mul(zdist, zdist);
          vector float dist2  = vec_splats(yzdist2); 
          dist2 = vec_add(dist2, xdist2);
 
          // loop over the shells belonging to this atom
          // XXX this is maybe a misnomer because in split valence
          //     basis sets like 6-31G we have more than one basis
          //     function per (valence-)shell and we are actually
          //     looping over the individual contracted GTOs
          for (shell=0; shell < maxshell; shell++) {
            vector float contracted_gto = vec_splats(0.0f);

            // Loop over the Gaussian primitives of this contracted 
            // basis function to build the atomic orbital
            // 
            // XXX there's a significant opportunity here for further
            //     speedup if we replace the entire set of primitives
            //     with the single gaussian that they are attempting 
            //     to model.  This could give us another 6x speedup in 
            //     some of the common/simple cases.
            int maxprim = num_prim_per_shell[shell_counter];
            int shelltype = shell_types[shell_counter];
            for (prim=0; prim<maxprim; prim++) {
              // XXX pre-negate exponent value
              float exponent       = -basis_array[prim_counter    ];
              float contract_coeff =  basis_array[prim_counter + 1];

              // contracted_gto += contract_coeff * exp(-exponent*dist2);
              vector float expval = vec_mul(vec_splats(exponent), dist2);

              // VSX expf() required here
              vector float retval = aexpfnxvsx(expval);

              vector float ctmp = vec_mul(vec_splats(contract_coeff), retval);
              contracted_gto = vec_add(contracted_gto, ctmp);
              prim_counter += 2;
            }

            /* multiply with the appropriate wavefunction coefficient */
            vector float tmpshell = vec_splats(0.0f);
            switch (shelltype) {
              case S_SHELL:
                value = vec_add(value, vec_mul(vec_splats(wave_f[ifunc++]), contracted_gto));
                break;

              case P_SHELL:
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), xdist));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), ydist));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), zdist));
                value = vec_add(value, vec_mul(tmpshell, contracted_gto));
                break;

              case D_SHELL:
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), xdist2));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(xdist, ydist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), ydist2));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(xdist, zdist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(ydist, zdist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), zdist2));
                value = vec_add(value, vec_mul(tmpshell, contracted_gto));
                break;

              case F_SHELL:
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(xdist2, xdist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(xdist2, ydist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(ydist2, xdist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(ydist2, ydist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(xdist2, zdist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(vec_mul(xdist, ydist), zdist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(ydist2, zdist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(zdist2, xdist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(zdist2, ydist)));
                tmpshell = vec_add(tmpshell, vec_mul(vec_splats(wave_f[ifunc++]), vec_mul(zdist2, zdist)));
                value = vec_add(value, vec_mul(tmpshell, contracted_gto));
                break;
 
#if 0
              default:
                // avoid unnecessary branching and minimize use of pow()
                int i, j; 
                float xdp, ydp, zdp;
                float xdiv = 1.0f / xdist;
                for (j=0, zdp=1.0f; j<=shelltype; j++, zdp*=zdist) {
                  int imax = shelltype - j; 
                  for (i=0, ydp=1.0f, xdp=pow(xdist, imax); i<=imax; i++, ydp*=ydist, xdp*=xdiv) {
                    tmpshell += wave_f[ifunc++] * xdp * ydp * zdp;
                  }
                }
                value += tmpshell * contracted_gto;
#endif
            } // end switch

            shell_counter++;
          } // end shell
        } // end atom

        // return either orbital density or orbital wavefunction amplitude
        if (density) {
          value = vec_cpsgn(value, vec_mul(value, value));

          float *ufptr = &orbitalgrid[gaddrzy + nx];
          ufptr[0] = value[0];
          ufptr[1] = value[1];
          ufptr[2] = value[2];
          ufptr[3] = value[3];
        } else {
          float *ufptr = &orbitalgrid[gaddrzy + nx];
          ufptr[0] = value[0];
          ufptr[1] = value[1];
          ufptr[2] = value[2];
          ufptr[3] = value[3];
        }
      }
    }
  }

  return 0;
}

#endif



//
// Multithreaded molecular orbital computation engine
//

typedef struct {
  int numatoms;
  const float *wave_f;
  const float *basis_array;
  const float *atompos;
  const int *atom_basis;
  const int *num_shells_per_atom;
  const int *num_prim_per_shell;
  const int *shell_types;
  const int *numvoxels;
  float voxelsize;
  int density;
  const float *origin;
  float *orbitalgrid;
} orbthrparms;


extern "C" void * orbitalthread(void *voidparms) {
  int numvoxels[3];
  float origin[3];
  orbthrparms *parms = NULL;
#if defined(VMDORBUSETHRPOOL)
  wkf_threadpool_worker_getdata(voidparms, (void **) &parms);
#else
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);
#endif

  numvoxels[0] = parms->numvoxels[0];
  numvoxels[1] = parms->numvoxels[1];
  numvoxels[2] = 1; // we compute only a single plane

  origin[0] = parms->origin[0];
  origin[1] = parms->origin[1];

  // loop over orbital planes
  int planesize = numvoxels[0] * numvoxels[1];
  wkf_tasktile_t tile;
#if defined(VMDORBUSETHRPOOL)
  while (wkf_threadpool_next_tile(voidparms, 1, &tile) != WKF_SCHED_DONE) {
#else
  while (wkf_threadlaunch_next_tile(voidparms, 1, &tile) != WKF_SCHED_DONE) {
#endif
    int k;
    for (k=tile.start; k<tile.end; k++) {
      origin[2] = parms->origin[2] + parms->voxelsize * k;
#if defined(VMDORBUSEAVX512) && defined(__AVX512F__) && defined(__AVX512ER__)

      if (1) {
        evaluate_grid_avx512er(parms->numatoms,
                      parms->wave_f, parms->basis_array,
                      parms->atompos,
                      parms->atom_basis,
                      parms->num_shells_per_atom,
                      parms->num_prim_per_shell,
                      parms->shell_types,
                      numvoxels,
                      parms->voxelsize,
                      origin,
                      parms->density,
                      parms->orbitalgrid + planesize*k);
      } else {
        evaluate_grid_avx512f(parms->numatoms,
                      parms->wave_f, parms->basis_array,
                      parms->atompos,
                      parms->atom_basis,
                      parms->num_shells_per_atom,
                      parms->num_prim_per_shell,
                      parms->shell_types,
                      numvoxels,
                      parms->voxelsize,
                      origin,
                      parms->density,
                      parms->orbitalgrid + planesize*k);
      }
#elif defined(VMDORBUSESSE) && defined(__SSE2__)
      evaluate_grid_sse(parms->numatoms,
                    parms->wave_f, parms->basis_array,
                    parms->atompos,
                    parms->atom_basis,
                    parms->num_shells_per_atom,
                    parms->num_prim_per_shell,
                    parms->shell_types,
                    numvoxels,
                    parms->voxelsize,
                    origin,
                    parms->density,
                    parms->orbitalgrid + planesize*k);
#elif defined(VMDORBUSEVSX) && defined(__VSX__)
      evaluate_grid_vsx(parms->numatoms,
                    parms->wave_f, parms->basis_array,
                    parms->atompos,
                    parms->atom_basis,
                    parms->num_shells_per_atom,
                    parms->num_prim_per_shell,
                    parms->shell_types,
                    numvoxels,
                    parms->voxelsize,
                    origin,
                    parms->density,
                    parms->orbitalgrid + planesize*k);
#else
      evaluate_grid(parms->numatoms,
                    parms->wave_f, parms->basis_array,
                    parms->atompos,
                    parms->atom_basis,
                    parms->num_shells_per_atom,
                    parms->num_prim_per_shell,
                    parms->shell_types,
                    numvoxels,
                    parms->voxelsize,
                    origin,
                    parms->density,
                    parms->orbitalgrid + planesize*k);
#endif
    }
  }

  return NULL;
}


int evaluate_grid_fast(
#if defined(VMDORBUSETHRPOOL) 
                       wkf_threadpool_t *thrpool, 
#else
                       int numcputhreads,
#endif
                       int numatoms,
                       const float *wave_f,
                       const float *basis_array,
                       const float *atompos,
                       const int *atom_basis,
                       const int *num_shells_per_atom,
                       const int *num_prim_per_shell,
                       const int *shell_types,
                       const int *numvoxels,
                       float voxelsize,
                       const float *origin,
                       int density,
                       float * orbitalgrid) {
  int rc=0;
  orbthrparms parms;

  parms.numatoms = numatoms;
  parms.wave_f = wave_f;
  parms.basis_array = basis_array;
  parms.atompos = atompos;
  parms.atom_basis = atom_basis;
  parms.num_shells_per_atom = num_shells_per_atom;
  parms.num_prim_per_shell = num_prim_per_shell;
  parms.shell_types = shell_types;
  parms.numvoxels = numvoxels;
  parms.voxelsize = voxelsize;
  parms.origin = origin;
  parms.density = density;
  parms.orbitalgrid = orbitalgrid;

  /* spawn child threads to do the work */
  wkf_tasktile_t tile;
  tile.start = 0;
  tile.end = numvoxels[2];

#if defined(VMDORBUSETHRPOOL) 
  wkf_threadpool_sched_dynamic(thrpool, &tile);
  rc = wkf_threadpool_launch(thrpool, orbitalthread, &parms, 1);
#else
  rc = wkf_threadlaunch(numcputhreads, &parms, orbitalthread, &tile);
#endif

  return rc;
}


void Orbital::print_wavefunction() {
  // XXX Android, IRIX, and Windows don't provide log2f(), nor log2() ?!?!?!?!
  // for now we'll just avoid compiling this debugging code
#if !(defined(_MSC_VER) || defined(ARCH_IRIX6) || defined(ARCH_IRIX6_64) || defined(ARCH_ANDROIDARMV7A))
  char shellname[6] = {'S', 'P', 'D', 'F', 'G', 'H'};
  int ifunc = 0;
  int at;
  int shell;
  for (at=0; at<numatoms; at++) {
    for (shell=0; shell < num_shells_per_atom[at]; shell++) {
      int shelltype = basis_set[at].shell[shell].type;

      // avoid unnecessary branching and minimize use of pow()
      int i, j, iang=0; 
      float xdist=2.0;
      float ydist=2.0;
      float zdist=2.0;
      float xdp, ydp, zdp;
      float xdiv = 1.0f / xdist;
      for (j=0, zdp=1.0f; j<=shelltype; j++, zdp*=zdist) {
        int imax = shelltype - j; 
        for (i=0, ydp=1.0f, xdp=pow(xdist, imax); i<=imax; i++, ydp*=ydist, xdp*=xdiv) {
          printf("%3i %c", at, shellname[shelltype]);
          int k, m=0;
          char buf[20]; buf[0] = '\0';
          for (k=0; k<(int)log2f(xdp); k++, m++) sprintf(buf+m, "x");
          for (k=0; k<(int)log2f(ydp); k++, m++) sprintf(buf+m, "y");
          for (k=0; k<(int)log2f(zdp); k++, m++) sprintf(buf+m, "z");
          //char *ang = qmdata->get_angular_momentum(at, shell, iang);
          printf("%-5s (%1.0f%1.0f%1.0f) wave_f[%3i] = % 11.6f\n", buf,
                 log2f(xdp), log2f(ydp), log2f(zdp), ifunc, wave_f[ifunc]);
          //delete [] ang;
          iang++;
          ifunc++;
        }
      }
    }
  }
#endif

}
