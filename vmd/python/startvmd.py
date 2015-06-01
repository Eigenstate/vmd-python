############################################################################
#cr
#cr            (C) Copyright 1995-2007 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: startvmd.py,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.4 $        $Date: 2007/01/12 20:19:10 $
#
############################################################################

import vmd, select, threading, sys, os

def vmdloop(app):
  vmd.VMDApp_activate_menus(app)
  vmd.VMDApp_menu_show(app, "main", 1)
  while 1:
    if not vmd.VMDApp_VMDupdate(app, 1):
      break
    select.select([], [], [], 0.001)
  vmd.delete_VMDApp(app)
  del app
  print "Exiting VMD event loop"

vmd.init_allocators()
v = vmd.new_VMDApp()
vmd.VMDApp_VMDinit(v,[""], "WIN", (50, 50), (400, 400))
vmd.VMDApp_deactivate_uitext_stdin(v)
thread = threading.Thread(target=vmdloop, args=(v,))
thread.setDaemon(1)
thread.start()

