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
 *      $RCSfile: PluginMgr.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Plugin Manager: Scans a specified set of directories looking for shared
 *   libraries that implement the vmdplugin interface.  Stores a copy of 
 *   the plugins it finds so that they can be passed off to other routines that
 *   know the specific interface for the plugin.
 * 
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/
#ifndef PLUGIN_MGR_H__
#define PLUGIN_MGR_H__

#include "ResizeArray.h"
#include "vmdplugin.h"

typedef ResizeArray<vmdplugin_t *> PluginList;

/// Scans a specified set of directories looking for shared
/// libraries that implement the vmdplugin interface.
/// Stores a copy of the plugins it finds so that they can be passed to 
/// routines that know the specific interface for the plugin.
class PluginMgr {
public:
  PluginMgr();
  virtual ~PluginMgr();

  /// Load any plugins are that statically linked into the application.  
  /// Return true if successful, or false if no static plugins are available.
  int load_static_plugins();

  /// Load the specified shared library and access all plugins found therein.
  /// Return the number of plugins found in the library, or -1 if an error
  /// occurred.
  int load_sharedlibrary_plugins(const char *path);
  
  /// Return plugins for the specified type and/or name; omitting both returns
  /// all plugins.  Stores plugins in the passed-in array and returns the
  /// number of plugins added to the list.
  int plugins(PluginList &, const char * = 0, const char * = 0);
  
protected:
  int add_plugin(const char *path, const char *file);
  PluginList pluginlist;
  ResizeArray<void *>handlelist;

  static int register_cb(void *, vmdplugin_t *);
  int num_in_library;
  const char *curpath;  ///< path to plugin file currently being processed
};    

#endif
