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
 *	$RCSfile: FileRenderList.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.29 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * The FileRenderList class maintains a database of avbailable FileRenderer
 * objects
 *
 ***************************************************************************/
#ifndef FILERENDERLIST_H
#define FILERENDERLIST_H

#include "NameList.h"

class FileRenderer;
class VMDApp;

/// Manage a list of FileRenderer objects that can be used to render a scene
class FileRenderList {
private:
  /// list of FileRenderer objects, with their name as a lookup key
  NameList<FileRenderer *> renderList;

  VMDApp *app;

#if defined(VMDLIBOPTIX)
  // check for user-supplied remote VCA rendering cluster URL, user, passwd,
  // if all three are set, we attach to the remote cluster and if successful,
  // the remote connection is used during OptiX context initialization.
  void *cluster_dev;
#endif

public:
  FileRenderList(VMDApp *);
  virtual ~FileRenderList(void);
  
  /// add a new render class and its corresponding name
  void add(FileRenderer *);
  
  /// figure out how many render classes are installed
  int num(void);
  
  /// return the short name (used in scripts) for the ith class
  const char * name(int);

  /// return the "pretty" name (used in GUIs) for the ith class
  const char * pretty_name(int);
  
  /// given a render name, return the corresponding class
  FileRenderer *find(const char *);

  /// given a "pretty" render name, return the corresponding class
  FileRenderer *find_pretty_name(const char *);

  /// find the short name that corresponds to a "pretty" GUI name
  const char *find_short_name_from_pretty_name(const char *pretty);

  /// do the rendering
  int render(const char *filename, const char *method, const char *extcmd);

  /// set the command string to execute after producing the scene file.
  int set_render_option(const char *, const char *);

  /// does renderer support antialiasing
  int has_antialiasing(const char *method);

  /// Set the AA sample count; return the new value
  int aasamples(const char *method, int aasamples);

  /// Set the AO sample count; return the new value
  int aosamples(const char *method, int aosamples);

  /// Get/set the image size
  int imagesize(const char *method, int *width, int *height);

  /// Does the renderer support arbitrary image size?
  int has_imagesize(const char *method);

  /// Get/set the aspect ratio.  A negative value will be ignored.  Return
  /// success and place the new value in the passed-in pointer.
  int aspectratio(const char *method, float *aspect);

  /// Number of file formats supported by the given renderer
  int numformats(const char *method);

  /// Name of the ith format; by default, returns current format
  const char *format(const char *method, int i = -1);

  /// Set the output format for the given renderer.  Return success.
  int set_format(const char *method, const char *format);
};
  
#endif

