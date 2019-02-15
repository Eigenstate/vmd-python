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
 *      $RCSfile: Watershed.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $        $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA-accelerated Watershed image segmentation
 ***************************************************************************/

#ifndef WATERSHED_H
#define WATERSHED_H


#if defined(WATERSHED_INTERNAL) || defined(ARCH_SOLARISX86_64)

/* Enables verbose printing throughout the watershed algorithm */

/*
 * Uses 18-connected 3D neighbors instead of 6
 * They produce very similar but not identical results
 * It may acutally be slightly faster to use 18 neighbors
 * but uses more memory per voxel.
 * Using 18 neighbors results in 99.9% similarity between CUDA and
 * CPU version instead of 100. This is not an error in the CUDA implementation.
 * It occurs because currently we only mark 6-connected neighbors to update
 * if a voxel changes, if the updating scheme is turned off
 * then this issue goes away.
 */
//#define CONNECTED_18

/* 
 * Keeps track of blocks that need to be updated and
 * only performs calulations for those regions
 */
#define BLOCK_UPDATES

/* 
 * Allocates extra memory so that we can skip some bounds
 * checking when doing block update bookkeeping
 */
#define FAST_UPDATES

/* Tracks how often the voxels are actually updated */
//#define STATS

/* Times how long the algorithm runs */
#define TIMER

#define FLOAT_DIFF      0.0001f
#define UPDATE_SIZE     4 // Update "blocks" size
#define UPDATE_SHIFT    2 // log2 of UPDATE_SIZE

#define EQ_SHIFT 6
#define EQ_MASK  0x3F
#define LOWER_MASK 0xFFFFFFC0

#define GET_N_LOWER(equal_and_lower) ((equal_and_lower) & EQ_MASK)

#define GET_N_EQ(equal_and_lower) ((equal_and_lower) >> EQ_SHIFT)

#define MAKE_EQUAL_AND_LOWER(equal, lower) (((equal) << EQ_SHIFT) | (lower))

#define REPLACE_LOWER(equal_and_lower, lower) (((equal_and_lower) & LOWER_MASK) | (lower))

// offsets from starting voxel to each neighbor
// px_ny represents the neighbor at (x+1, y-1, z) position where
// x, y, and z are the coordinates of the starting voxel
// Given a direction, such as px_ny, px_ny_offset gives the index offset
// such that starting_index + px_ny_offset = the index of the px_ny neighbor.
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

#define SWAP(a, b, type) {\
  type t = a;\
  a = b;\
  b = t;\
}


/// enum with bit flags for all of the neighbors that are "equal"
/// px_ny represents the neighbor at (x+1, y-1, z) position where
/// x, y, and z are the coordinates of the starting voxel.
enum direction {
  px = 0x00001,
  py = 0x00002,
  pz = 0x00004,
  nx = 0x00008,
  ny = 0x00010,
  nz = 0x00020,

  nx_ny = 0x00040,
  nx_py = 0x00080,
  px_py = 0x00100,
  px_ny = 0x00200,

  px_pz = 0x00400,
  nx_nz = 0x00800,
  nx_pz = 0x01000,
  px_nz = 0x02000,

  py_pz = 0x04000,
  ny_nz = 0x08000,
  ny_pz = 0x10000,
  py_nz = 0x20000
};

/// enum used to store the "lowest neighbor" in a char
/// px_ny represents the neighbor at (x+1, y-1, z) position where
/// x, y, and z are the coordinates of the starting voxel
enum neighbor {
  px_idx = 1,
  py_idx,
  pz_idx,
  nx_idx,
  ny_idx,
  nz_idx,

  nx_ny_idx,
  nx_py_idx,
  px_py_idx,
  px_ny_idx,

  px_pz_idx,
  nx_nz_idx,
  nx_pz_idx,
  px_nz_idx,

  py_pz_idx,
  ny_nz_idx,
  ny_pz_idx,
  py_nz_idx
};

#endif // WATERSHED_INTERNAL


template <typename GROUP_T, typename IMAGE_T>
struct state_t {
  GROUP_T* group;
  IMAGE_T* value;
};

template <typename IMAGE_T>
struct group_t {
  IMAGE_T* max_value;
  int* max_idx;
  bool init;
};

template<typename GROUP_T, typename IMAGE_T>
struct watershed_gpu_state_t {
  int* eq_and_lower_d;
  IMAGE_T* current_value_d;
  IMAGE_T* next_value_d;
  unsigned char* current_update_d;
  unsigned char* next_update_d;
  GROUP_T* segments_d;
  int height;
  int width;
  int depth;
  bool init;
};

/// Class that performs watershed segmentation and filtering
template<typename GROUP_T, typename IMAGE_T>
class Watershed {
  public:

    /// Creates a watershed object and allocates internal arrays
    Watershed<GROUP_T, IMAGE_T>(unsigned int h, unsigned int w, unsigned int d, bool cuda=true);

    /// copy segmented group info to given volume
    /// XXX this API needs to evolve to a volume of integer types
    void getSegmentedVoxels(IMAGE_T* voxels);
    
    IMAGE_T* getRawVoxels(); ///< Returns pointer to the internal intensity array

    ~Watershed<GROUP_T, IMAGE_T>();

    /// Runs the watershed algorithm
    void watershed(IMAGE_T* image, int imageongpu, GROUP_T* segments, bool verbose=true);

  private:
    bool use_cuda; ///< tracks whether watershed object should use CUDA or not
    watershed_gpu_state_t<GROUP_T, IMAGE_T> gpu_state;

    int* equal_and_lower; ///< array that holds the equal neighbor bitmap
                          ///< and lower neighbor number packed into
                          ///< one int for each voxel

    unsigned char* current_update; ///< blocks needing update in current iteration
    unsigned char* next_update;  ///< blocks needing update in next iteration

    int height;                  ///< height of 3D density map
    int width;                   ///< width of 3D density map
    int depth;                   ///< depth of 3D density map
    int heightWidth;             ///< height * width
    long nVoxels;                   ///< number of voxels

    unsigned long update_width;  ///< width of the current/next_update array
    unsigned int update_offset;  ///< padding bytes at start/end of update array

    group_t<IMAGE_T> group_data;          ///< group data
    long nGroups;                 ///< number of groups
    int neighbor_offsets[19];    ///< convert an "offset number" to index offset

    state_t<GROUP_T, IMAGE_T> current_state;    ///< state for current iteration of watershed
    state_t<GROUP_T, IMAGE_T> next_state;       ///< state for next iteration of watershed

    void init(IMAGE_T* image);     ///< watershed initialization step

    void init_neighbor_offsets();///< Builds neigbor_offsets array


    /// Performs 3D gaussian blur on the intensity array with
    /// given sigma $(reps) number of times
    /// XXX we should probably factor out the Gaussian blur from
    /// XXX the Watershed class, and instead build up a new
    /// XXX "scale space" segmentation class that calls a separate
    /// XXX Gaussian blur implementation (itself based on either
    /// XXX convolution for small kernel sizes, or FFT for large ones.
    void gaussian3D(const float sigma, const int reps);

#if defined(VMDCUDA)
    void watershed_gpu(GROUP_T* segments_d); ///< Runs watershed algorithm with CUDA
#endif

    void watershed_cpu(GROUP_T* segments); ///< Runs watershed algorithm on CPU

    unsigned int update(); ///< Performs one step of watershed

    /// Performs one step of watershed with block updates
    unsigned int update_blocks();

    /// Performs one step of watershed while keeping track of various stats
    unsigned int update_stats(unsigned int& numUpdates);


    /// Performs one step of watershed algorithm with block updates 
    /// while tracking various stats
    unsigned int update_blocks_stats(unsigned int& numUpdates,
                                     unsigned int& numBlockUpdates); 

    /// returns index of local maxima found
    /// by steepest ascent from given coordinates
    long find_local_maxima(long x, long y, long z);
};


#endif
