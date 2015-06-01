/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: PluginMgr.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.36 $       $Date: 2011/03/07 21:19:18 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Plugin manager code, loads and registers static and dynamic 
 *  plugins, does version checks, etc.
 * 
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 * 
 ***************************************************************************/

#include "PluginMgr.h"
#include "vmddlopen.h"
#include "Inform.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef VMDSTATICPLUGINS
#include "libmolfile_plugin.h"
#endif

extern "C" {
  typedef int (*initfunc)(void);
  typedef int (*regfunc)(void *, vmdplugin_register_cb);
  typedef int (*finifunc)(void);
}

// return true of a plugin is the same or older than ones we already have
static int plugincmp(const vmdplugin_t *newp, const vmdplugin_t *oldp) {
  int issameorolder = 
         !strcmp(newp->type, oldp->type) && 
         !strcmp(newp->name, oldp->name) &&
         (newp->majorv <= oldp->majorv) &&           // only load new ones
         ((newp->majorv < oldp->majorv) ||           // only load new ones
          (newp->minorv <= oldp->minorv)) &&
         !strcmp(newp->author, oldp->author) &&
         (newp->is_reentrant == oldp->is_reentrant); // reentrant comes first

  return issameorolder;
}

PluginMgr::PluginMgr() {
#ifdef VMDSTATICPLUGINS
  MOLFILE_INIT_ALL
#endif
}

PluginMgr::~PluginMgr() {
  int i;
  for (i=0; i<handlelist.num(); i++) {
    void *ffunc = vmddlsym(handlelist[i], "vmdplugin_fini");
    if (ffunc) {
      ((finifunc)(ffunc))();
    }
    vmddlclose(handlelist[i]);
  }
#ifdef VMDSTATICPLUGINS
  MOLFILE_FINI_ALL
#endif
}

int PluginMgr::load_static_plugins() {
#ifdef VMDSTATICPLUGINS
  num_in_library = 0;
  curpath = "statically linked.";
  MOLFILE_REGISTER_ALL(this, register_cb)
  return 1;
#else
  return 0;
#endif
}


int PluginMgr::register_cb(void *v, vmdplugin_t *plugin) {
  PluginMgr *self = (PluginMgr *)v;

  // check that the ABI version matches
  if (plugin->abiversion != vmdplugin_ABIVERSION) {
    msgWarn << "Rejecting plugin with incorrect ABI version: "
            << self->curpath << sendmsg;
    return -1;
  }

  // check that there are no null terms in the plugin
  if (!plugin->type || !plugin->name || !plugin->author) {
    msgWarn << "Rejecting plugin with NULL header values."
            << self->curpath << sendmsg;
    return -1;
  }

  // check new plugin against already-loaded plugins
  for (int i=0; i<self->pluginlist.num(); i++) {
    if (plugincmp(plugin, self->pluginlist[i])) {
      // don't add new plugin if we already have an identical or older version
      return 0;
    }
  }

  self->num_in_library++;
  self->pluginlist.append(plugin);
  return 0;
}
 
int PluginMgr::load_sharedlibrary_plugins(const char *fullpath) {
  // Open the dll; try to execute the init function.
  void *handle = vmddlopen(fullpath);
  if (!handle) {
    msgWarn << "Unable to open dynamic library '" << fullpath << "'." << sendmsg;
    msgWarn << vmddlerror() << sendmsg; 
    return -1;
  }

  if (handlelist.find(handle) >= 0) {
    msgWarn << "Already have a handle to the shared library " 
            << fullpath << "." << sendmsg;
    return 0;
  }

  void *ifunc = vmddlsym(handle, "vmdplugin_init");
  if (ifunc && ((initfunc)(ifunc))()) {
    msgWarn << "vmdplugin_init() for " << fullpath 
            << " returned an error; plugin(s) not loaded." << sendmsg;
    vmddlclose(handle);
    return 0;
  }
  handlelist.append(handle);
   
  void *registerfunc = vmddlsym(handle, "vmdplugin_register");
  num_in_library = 0;
  curpath = fullpath;
  if (!registerfunc) {
    msgWarn << "Didn't find the register function in" << fullpath 
            << "; plugin(s) not loaded." << sendmsg;
  } else {
    // Load plugins from the library.
    ((regfunc)registerfunc)(this, register_cb);
  } 
  return num_in_library;
}

int PluginMgr::plugins(PluginList &pl, const char *type, const char *name) {
  int nfound = 0;
  for (int i=0; i<pluginlist.num(); i++) {
    vmdplugin_t *p = pluginlist[i]; 
    if (type && strcmp(p->type, type)) continue;
    if (name && strcmp(p->name, name)) continue;
    pl.append(p);
    nfound++;
  }
  return nfound;
}

#ifdef TEST_PLUGIN_MGR

int main(int argc, char *argv[]) {
  PluginMgr *mgr = new PluginMgr;
  for (int i=1; i<argc; i++) {
    int rc = mgr->load_sharedlibrary_plugins(argv[i]);
    printf("Scanning plugin %s returned %d\n", argv[i], rc);
  }
  PluginList *pl = mgr->plugins();
  printf("found %d plugins\n", pl->num());
  for (int i=0; i<pl->num(); i++) {
    Plugin *p = (*pl)[i];
    printf("type: %s\nname: %s\nauthor: %s\n",
      p->type(), p->name(), p->author());
    printf("version %d%d\n", p->majorv(), p->minorv());
    printf("is%sreentrant\n", p->is_reentrant() ? " " : " not ");
  }
  delete pl;
  delete mgr;
  return 0;
}
  
#endif
