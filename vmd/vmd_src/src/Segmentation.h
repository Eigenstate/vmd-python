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
 *      $RCSfile: Segmentation.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.6 $        $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Scale-space variant of Watershed image segmentation intended for use
 *   on 3-D cryo-EM density maps.
 ***************************************************************************/
#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "molfile_plugin.h"
#include "GaussianBlur.h"
#include "ScaleSpaceFilter.h"
#include "VolumetricData.h"

#define TIMER


/// Top-level class that given an image will
/// segement the image using watershed and then
/// postprocess the segmentation using a
/// space scale filter algorithm
class Segmentation {

  public:
    /// Creates a segmentation object given a 3D float array and
    /// volumetric metadata. Optional argument to disable CUDA.
    Segmentation(const VolumetricData *map, bool cuda=true);

    /// Creates a segmentation object given a 3D float array and
    /// volumetric metadata. Optional argument to disable CUDA.
    Segmentation(float* data,
                 molfile_volumetric_t* metatdata,
                 bool cuda=true);

    Segmentation(unsigned short* data,
                 molfile_volumetric_t* metatdata,
                 bool cuda=true);

    Segmentation(unsigned char* data,
                 molfile_volumetric_t* metatdata,
                 bool cuda=true);

    ~Segmentation();

    /// Runs the segmentation algorithm
    /// until there are <= num_final_groups
    double segment(int   num_final_groups,
                   float watershed_blur_sigma,
                   float blur_initial_sigma,
                   float blur_multiple,
                   MERGE_POLICY policy,
                   const bool verbose=true);

    /// Copies the segmentation into the provided results
    /// array. Can be of type int, short, or (legacy) float
    template <typename OUT_T>
    void get_results(OUT_T* results);

    /// Returns the number of groups in the segmentation
    unsigned long get_num_groups();

  private:

    // The object keeps track of the current smallest datatype
    // we can use based on the number of groups in the segmentation.
    enum Datatype {
      DT_LONG,
      DT_INT,
      DT_SHORT,
      DT_CHAR,
      DT_ULONG,
      DT_UINT,
      DT_USHORT,
      DT_UCHAR,
      DT_FLOAT,
      DT_DOUBLE,
      DT_ERROR
    };

    unsigned long height;
    unsigned long width;
    unsigned long depth;
    unsigned long nVoxels;
    unsigned long nGroups;
    Datatype group_type;
    Datatype image_type;
    GaussianBlur<float>* gaussian_f;
    GaussianBlur<unsigned short>* gaussian_us;
    GaussianBlur<unsigned char>* gaussian_uc;
    unsigned long* segments_ul;
    unsigned int* segments_ui;
    unsigned short* segments_us;
    bool verbose;


    // The object will always try to use cuda unless explicitly told not to.
    // If an error occurs at any point in the segmentation process we should be 
    // able to fall back to the CPU seamlessly (TODO fallback not implemented yet)
    bool use_cuda;

    /// Performs initialization work shared by all constructors
    void init(unsigned long w,
              unsigned long h,
              unsigned long d,
              bool cuda);

    /// Checks if we can use a smaller datatype to represent the groups,
    /// and if so will transition to that type
    void update_type();

    void switch_to_cpu();

    /// Returns the smallest type we can use given the number of groups
    Datatype get_new_group_type();

    /// Copies an array of IN_T to an array of OUT_T. If copy_from_gpu
    /// it implies that IN_T is in device mem and OUT_T is not. Otherwise
    /// it will automatically infer the memory location based on internal object state
    template <typename IN_T, typename OUT_T>
    void copy_and_convert_type(IN_T* in, OUT_T* out, unsigned long num_elems,
                               bool copy_from_gpu=false);

    /// Performs a blur on the input and then runs the watershed algorithm,
    /// copying the output to segments
    template <typename GROUP_T>
    void get_watershed_segmentation(float watershed_blur_sigma,
                                    GROUP_T* segments);

    /// Runs the space-scale filtering algorithm until there are
    /// num_final_groups
    template <typename GROUP_T>
    void merge_groups(unsigned long num_final_groups,
                      float blur_initial_sigma,
                      float blur_multiple, 
                      MERGE_POLICY policy,
                      GROUP_T* segments);

    template <typename GROUP_T, typename IMAGE_T>
    void merge_groups_helper(unsigned long num_final_groups,
                      float blur_initial_sigma,
                      float blur_multiple, 
                      GROUP_T* segments,
                      MERGE_POLICY policy,
                      GaussianBlur<IMAGE_T>* gaussian);

    template <typename GROUP_T>
    void sequentialize_groups(GROUP_T* segments,
                              GROUP_T* group_map);

    template <typename GROUP_T>
    unsigned long sequentialize_groups_cpu(GROUP_T* segments,
                                           GROUP_T* group_map);

    /// Level of indirection that returns a pointer to
    /// newly allocated array either in main mem or device mem
    /// depending on the state of the object
    template <typename GROUP_T>
    GROUP_T* allocate_array(unsigned long num_elements);

    template <typename GROUP_T>
    void free_array(GROUP_T*& arr);

};

#endif
