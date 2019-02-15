/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef CMD_PLUGIN_H__
#define CMD_PLUGIN_H__

#include "Command.h"

/// Update the GUI plugin list with any newly loaded plugins
class CmdPluginUpdate: public Command {
protected:
  void create_text() {
    *cmdText << "plugin update" << ends;
  }
public:
  CmdPluginUpdate() : Command(PLUGIN_UPDATE) { }
};

#endif
