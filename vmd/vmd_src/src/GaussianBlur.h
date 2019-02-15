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
 *      $RCSfile: GaussianBlur.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.9 $        $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Perform Gaussian blur filtering on 3-D volumes. Primarily for use in
 *   scale-space filtering for 3-D image segmentation of Cryo-EM density maps.
 ***************************************************************************/

#ifndef GAUSSIAN_H
#define GAUSSIAN_H

template <typename IMAGE_T>
class GaussianBlur {
  public:

    /// Creates GaussianBlur object from image with specified height,
    /// width, and depth. This allocates internal memory
    /// and copies the image.
    GaussianBlur<IMAGE_T>(IMAGE_T* image, int w, int h, int d, bool cuda=false);

    ~GaussianBlur<IMAGE_T>();

    /// Performs a GaussianBlur blur with the specified sigma.
    void blur(float sigma);

    /// Returns a pointer the current blurred image.
    /// This pointer become invalid if blur is called again.
    IMAGE_T* get_image();
    IMAGE_T* get_image_d();

  private:
    IMAGE_T* image;
    IMAGE_T* image_d;
    IMAGE_T* scratch;
    IMAGE_T* scratch_d;
    int height;
    int width;
    int depth;
    long heightWidth;
    long nVoxels;
    bool cuda;

    int getKernelSizeForSigma(float sigma);
    void fillGaussianBlurKernel3D(float sigma, int size, float* kernel);
    void fillGaussianBlurKernel1D(float sigma, int size, float* kernel);

    void blur_cpu(float sigma);
    void blur_cuda(float sigma);
    int host_image_needs_update;
};

#endif //GAUSSIAN
