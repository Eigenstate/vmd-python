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
 *      $RCSfile: ScaleSpaceFilter.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.21 $        $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Perform scale-space filtering on 3-D volumes, e.g., cryo-EM density maps,
 *   for use in subsequent image segmentation algorithms.
 ***************************************************************************/

#include <string.h>
#include <stdio.h>
#include <map>
#include "ScaleSpaceFilter.h"
#include "Watershed.h"
#if defined(VMDCUDA)
#include "CUDAGaussianBlur.h"
#include "CUDASegmentation.h"
#endif
#include "ProfileHooks.h"

#define px_offset    (1)
#define py_offset    (width)
#define pz_offset    (heightWidth)
#define nx_offset    (-1)
#define ny_offset    (-width)
#define nz_offset    (-heightWidth)

#define nx_ny_offset (-1 - width)
#define nx_py_offset (-1 + width)
#define px_py_offset (1 + width)
#define px_ny_offset (1 - width)

#define px_pz_offset (1 + heightWidth)
#define nx_nz_offset (-1 - heightWidth)
#define nx_pz_offset (-1 + heightWidth)
#define px_nz_offset (1 - heightWidth)

#define py_pz_offset (width + heightWidth)
#define ny_nz_offset (-width - heightWidth)
#define ny_pz_offset (-width + heightWidth)
#define py_nz_offset (width - heightWidth)


template <typename GROUP_T, typename IMAGE_T>
ScaleSpaceFilter<GROUP_T, IMAGE_T>::ScaleSpaceFilter(int w, int h, int d,
                                                     long nGrps,
                                                     float initial_blur_sigma,
                                                     float new_blur_multiple,
                                                     bool cuda) {
  width = w;
  height = h;
  depth = d;
  heightWidth = long(height) * long(width);
  nVoxels = heightWidth * long(depth);
  nGroups = nGrps;
  current_blur = initial_blur_sigma;
  blur_multiple = new_blur_multiple;
  use_cuda = cuda;
  memset(&gpu_seq_tmp, 0, sizeof(gpu_seq_tmp));
  memset(&gpu_scanwork_tmp, 0, sizeof(gpu_scanwork_tmp));
  memset(&gpu_grpmaxidx_tmp, 0, sizeof(gpu_grpmaxidx_tmp));

  max_idx = allocate_array<unsigned long>(nVoxels);
  group_map = allocate_array<GROUP_T>(nVoxels);
}


template <typename GROUP_T, typename IMAGE_T>
ScaleSpaceFilter<GROUP_T, IMAGE_T>::~ScaleSpaceFilter() {
  free_array<unsigned long>(max_idx);
  free_array<GROUP_T>(group_map);

#if defined(VMDCUDA)
  free_gpu_temp_storage(&gpu_seq_tmp);
  free_gpu_temp_storage(&gpu_scanwork_tmp);
  free_gpu_temp_storage(&gpu_grpmaxidx_tmp);
#endif
}


template <typename GROUP_T, typename IMAGE_T>
template <typename ARRAY_T>
ARRAY_T* ScaleSpaceFilter<GROUP_T, IMAGE_T>::allocate_array(long num_elements) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::allocate_array()", 1);

  ARRAY_T* arr;
#if defined(VMDCUDA)
  if (use_cuda) {
    arr = (ARRAY_T*) alloc_cuda_array(num_elements * sizeof(ARRAY_T));
  } else 
#endif
  {
    arr = new ARRAY_T[num_elements];
  }

  PROFILE_POP_RANGE();

  return arr;
}


template <typename GROUP_T, typename IMAGE_T>
template <typename ARRAY_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::free_array(ARRAY_T*& arr) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::free_array()", 2);

#if defined(VMDCUDA)
  if (use_cuda) {
    free_cuda_array(arr);
  } else 
#endif
  {
    delete [] arr;
  }

  PROFILE_POP_RANGE();

  arr = NULL;
}


template <typename GROUP_T, typename IMAGE_T>
long ScaleSpaceFilter<GROUP_T, IMAGE_T>::merge(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian, MERGE_POLICY policy) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::merge()", 3);

  find_groups_max_idx(segments, gaussian);

  gaussian->blur(current_blur);

  if (policy == MERGE_HILL_CLIMB) {
    hill_climb_merge(segments, gaussian);
  } else if (policy == MERGE_WATERSHED_HILL_CLIMB) {
    watershed_hill_climb_merge(segments, gaussian);
  } else if (policy == MERGE_WATERSHED_OVERLAP) {
    watershed_overlap_merge(segments, gaussian);
  } else {
    fprintf(stderr, "ERROR: invalid merge policy.\n");
    return 0;
  }

  current_blur *= blur_multiple;

  PROFILE_POP_RANGE();

  return nGroups;
}


template <typename GROUP_T, typename IMAGE_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::watershed_hill_climb_merge(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian) {

  GROUP_T* new_segments = allocate_array<GROUP_T>(nVoxels);
  int imageongpu=1;
  IMAGE_T *imageptr = gaussian->get_image_d();
  if (!imageptr) {
    imageptr = gaussian->get_image();
    imageongpu=0;
  }
  Watershed<GROUP_T, IMAGE_T> watershed(height, width, depth, use_cuda);
  watershed.watershed(imageptr, imageongpu, new_segments, false);

  long n_new_groups = sequentialize_group_nums(new_segments, nVoxels);

#ifdef VMDCUDA
  if (use_cuda) {
    watershed_hill_climb_merge_cuda<GROUP_T>(segments, new_segments, imageptr, group_map, max_idx, height, width, depth, nGroups);
  } else
#endif
  {
    watershed_hill_climb_merge_cpu(segments, new_segments, imageptr);
  }

  nGroups = sequentialize_group_nums(segments, n_new_groups);
  
  free_array(new_segments);
}


template <typename GROUP_T, typename IMAGE_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::watershed_overlap_merge(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian) {

  GROUP_T* new_segments = allocate_array<GROUP_T>(nVoxels);


  int imageongpu=1;
  IMAGE_T *imageptr = gaussian->get_image_d();
  if (!imageptr) {
    imageptr = gaussian->get_image();
    imageongpu=0;
  }
  Watershed<GROUP_T, IMAGE_T> watershed(height, width, depth, use_cuda);
  watershed.watershed(imageptr, imageongpu, new_segments, false);
  long max_group_num = sequentialize_group_nums(segments, nVoxels);

#ifdef VMDCUDA
  if (use_cuda) {
    printf("CUDA watershed overlap merge not implemented.\n");
    //watershed_overlap_merge_cuda<GROUP_T>(segments, new_segments, group_map);
  } else
#endif
  {
    watershed_overlap_merge_cpu(segments, new_segments);
  }

  nGroups = sequentialize_group_nums(segments, max_group_num);
  
  free_array(new_segments);
}


template <typename GROUP_T, typename IMAGE_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::hill_climb_merge(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::hill_climb_merge()", 4);

#if defined(VMDCUDA)
  if (use_cuda) {
      hill_climb_merge_cuda<GROUP_T, IMAGE_T>(segments, gaussian->get_image_d(), max_idx,
                                       group_map, height, width, depth, nGroups);

  } else 
#endif
  {
    hill_climb_merge_cpu(segments, gaussian->get_image());
  }
  nGroups = sequentialize_group_nums(segments, nGroups);

  PROFILE_POP_RANGE();
}


template <typename GROUP_T, typename IMAGE_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::watershed_hill_climb_merge_cpu(GROUP_T* segments, GROUP_T* new_segments, IMAGE_T* image) {
  for (long g = 0; g < nGroups; g++) {
    long idx = find_local_maxima(max_idx[g], image);
    group_map[g] = new_segments[idx];
  }
  for (long v = 0; v < nVoxels; v++) {
    segments[v] = group_map[segments[v]];
  }
}


template <typename GROUP_T, typename IMAGE_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::watershed_overlap_merge_cpu(GROUP_T* segments, GROUP_T* new_segments) {
  std::map<GROUP_T, long>* group_counts = new std::map<GROUP_T, long>[nGroups];
  //std::map<GROUP_T, long> group_counts[nGroups];
  long v;
 
  for (v = 0; v < nVoxels; v++) {
    GROUP_T curr_group = segments[v];
    GROUP_T overlapping_group = new_segments[v];
    std::map<GROUP_T, long>& counts = group_counts[curr_group];
    counts[overlapping_group]++;

    //group_counts[segments[v]][new_segments[v]]++;  
  }
  //test = true;

  for (long g = 0; g < nGroups; g++) {
    long max_count = 0;
    long max_g = -1;
    for (typename std::map<GROUP_T, long>::iterator it = group_counts[g].begin(); it != group_counts[g].end(); it++) {
      if (it->second >= max_count) {
        max_count = it->second;
        max_g = it->first;
      }
    }
    group_map[g] = max_g;
  }

  for (v = 0; v < nVoxels; v++) {
    segments[v] = group_map[segments[v]];
  }

  delete [] group_counts;
}


template <typename GROUP_T, typename IMAGE_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::hill_climb_merge_cpu(GROUP_T* segments, IMAGE_T* image) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::hill_climb_merge_cpu()", 5);

#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (long g = 0; g < nGroups; ++g) {
    long x = max_idx[g];
    long new_group_idx = find_local_maxima(x, image);
    group_map[g] = segments[new_group_idx];

  }

#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (long idx = 0; idx < nVoxels; ++idx) {
    long curr_group = segments[idx];
    segments[idx] = group_map[curr_group];
  }

  PROFILE_POP_RANGE();
}


template <typename GROUP_T, typename IMAGE_T>
long ScaleSpaceFilter<GROUP_T, IMAGE_T>::sequentialize_group_nums_cpu(GROUP_T* segments, long max_group_num) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::sequentialize_group_nums_cpu()", 6);

  GROUP_T max_val = GROUP_T(0) - 1;
  long numGroups = 0;

  memset(group_map, max_val, max_group_num * sizeof(GROUP_T));

  for (long i=0; i<nVoxels; ++i) {
    const GROUP_T curr_group = segments[i];
    if (group_map[curr_group] == max_val) {
      group_map[curr_group] = numGroups++;
    }
    segments[i] = group_map[curr_group];
  }

  PROFILE_POP_RANGE();

  return numGroups;
}


template <typename GROUP_T, typename IMAGE_T>
long ScaleSpaceFilter<GROUP_T, IMAGE_T>::sequentialize_group_nums(GROUP_T* segments, long max_group_num) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::sequentialize_group_nums()", 7);
  long numGroups = 0;

#if defined(VMDCUDA)
  if (use_cuda) {
    numGroups = sequentialize_groups_cuda<GROUP_T>(segments, group_map, nVoxels, max_group_num, &gpu_seq_tmp, &gpu_scanwork_tmp);
  } else 
#endif
  {
    numGroups = sequentialize_group_nums_cpu(segments, max_group_num);
  }

  PROFILE_POP_RANGE();

  return numGroups;
}


template <typename GROUP_T, typename IMAGE_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::find_groups_max_idx(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::find_groups_max_idx()", 8);

#if defined(VMDCUDA)
  if (use_cuda) {
    find_groups_max_idx_cuda<GROUP_T, IMAGE_T>(segments, gaussian->get_image_d(),
                            max_idx, nVoxels, nGroups, &gpu_grpmaxidx_tmp);
  } else 
#endif
  {
    find_groups_max_idx_cpu(segments, gaussian->get_image());
  }

  PROFILE_POP_RANGE();
}


template <typename GROUP_T, typename IMAGE_T>
void ScaleSpaceFilter<GROUP_T, IMAGE_T>::find_groups_max_idx_cpu(GROUP_T* segments, IMAGE_T* voxel_values) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::find_groups_max_idx_cpu()", 9);

  memset(max_idx, -1, nGroups * sizeof(long)); //TODO change -1 use for unsigned types
  for (long idx = 0; idx < nVoxels; ++idx) {
    const GROUP_T group_number = segments[idx];
    const long group_max_idx = max_idx[group_number];
    if (group_max_idx == -1) {
      max_idx[group_number] = idx;
    } else if (voxel_values[group_max_idx] < voxel_values[idx]) {
      max_idx[group_number] = idx;
    }
  }

  PROFILE_POP_RANGE();
}


#define FIND_STEEPEST(offset, _x, _y, _z) {\
  if (image[curr_idx + offset] > current_value) {\
    current_value = image[curr_idx + offset];\
    new_x = _x;\
    new_y = _y;\
    new_z = _z;\
  }\
}

template <typename GROUP_T, typename IMAGE_T>
long ScaleSpaceFilter<GROUP_T, IMAGE_T>::find_local_maxima(long curr_idx, IMAGE_T* image) {
  PROFILE_PUSH_RANGE("ScaleSpaceFilter::find_local_maxima()", 10);

  long new_idx = curr_idx;
  int z = new_idx / heightWidth;
  long r = new_idx % heightWidth;
  int y = r / width;
  int x = r % width;
  do {
    // need x,y,z coords for bounds checking
    int new_x = x;
    int new_y = y;
    int new_z = z;
    curr_idx = new_idx;
    IMAGE_T current_value = image[curr_idx];

#if 1
    if (x < width-1) {
      FIND_STEEPEST(px_offset, x+1, y, z);

      if (y < height-1) {
        FIND_STEEPEST(px_offset + py_offset, x+1, y+1, z);
      }
      if (y > 0) {
        FIND_STEEPEST(px_offset + ny_offset, x+1, y-1, z);
      }

      if (z < depth-1) {
        FIND_STEEPEST(px_offset + pz_offset, x+1, y, z+1);
      }
      if (z > 0) {
        FIND_STEEPEST(px_offset + nz_offset, x+1, y, z-1);
      }
    }
    if (x > 0) {
      FIND_STEEPEST(nx_offset, x-1, y, z);

      if (y < height-1) {
        FIND_STEEPEST(nx_offset + py_offset, x-1, y+1, z);
      }
      if (y > 0) {
        FIND_STEEPEST(nx_offset + ny_offset, x-1, y-1, z);
      }

      if (z < depth-1) {
        FIND_STEEPEST(nx_offset + pz_offset, x-1, y, z+1);
      }
      if (z > 0) {
        FIND_STEEPEST(nx_offset + nz_offset, x-1, y, z-1);
      }
    }

    if (y < height-1) {
      FIND_STEEPEST(py_offset, x, y+1, z);
      
      if (z < depth-1) {
        FIND_STEEPEST(py_offset + pz_offset, x, y+1, z+1);
      }
      if (z > 0) {
        FIND_STEEPEST(py_offset + nz_offset, x, y+1, z-1);
      }
    }
    if (y > 0) {
      FIND_STEEPEST(ny_offset, x, y-1, z);

      if (z < depth-1) {
        FIND_STEEPEST(ny_offset + pz_offset, x, y-1, z+1);
      }
      if (z > 0) {
        FIND_STEEPEST(ny_offset + nz_offset, x, y-1, z-1);
      }
    }

    if (z < depth-1) {
      FIND_STEEPEST(pz_offset, x, y, z+1);
    }
    if (z > 0) {
      FIND_STEEPEST(nz_offset, x, y, z-1);
    }
#else
    if (x < width-1 && image[curr_idx + px_offset] > current_value) {
      current_value = image[curr_idx + px_offset];
      new_x = x + 1;
      new_y = y;
      new_z = z;
    }
    if (x > 0 && image[curr_idx + nx_offset] > current_value) {
      current_value = image[curr_idx + nx_offset];
      new_x = x - 1;
      new_y = y;
      new_z = z;
    }

    if (y < height-1 && image[curr_idx + py_offset] > current_value) {
      current_value = image[curr_idx + py_offset];
      new_x = x;
      new_y = y + 1;
      new_z = z;
    }
    if (y > 0 && image[curr_idx + ny_offset] > current_value) {
      current_value = image[curr_idx + ny_offset];
      new_x = x;
      new_y = y - 1;
      new_z = z;
    }

    if (z < depth-1 && image[curr_idx + pz_offset] > current_value) {
      current_value = image[curr_idx + pz_offset];
      new_x = x;
      new_y = y;
      new_z = z + 1;
    }
    if (z > 0 && image[curr_idx + nz_offset] > current_value) {
      current_value = image[curr_idx + nz_offset];
      new_x = x;
      new_y = y;
      new_z = z - 1;
    }
#endif

    x = new_x;
    y = new_y;
    z = new_z;
    new_idx = z*heightWidth + y*long(width) + x;
    // Keep going until none of our neighbors have a higher value 
  } while (new_idx != curr_idx);

  PROFILE_POP_RANGE();

  return new_idx;
}


// template instantiation convenience macros
#define INST_SCALE_SPACE(G_T) \
template class ScaleSpaceFilter<G_T, float>;\
template class ScaleSpaceFilter<G_T, unsigned short>;\
template class ScaleSpaceFilter<G_T, unsigned char>;

INST_SCALE_SPACE(unsigned long)
INST_SCALE_SPACE(unsigned int)
INST_SCALE_SPACE(unsigned short)




