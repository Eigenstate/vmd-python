#!/bin/csh -f
## In case the installation didn't add these (which means you didn't
# do the 'configure', here's the default settings:
#set defaultvmddir=/usr/local/lib/vmd
#set vmdbasename=vmd

############################################################################
#cr                                                                       
#cr            (C) Copyright 1995-2009 The Board of Trustees of the          
#cr                        University of Illinois                         
#cr                         All Rights Reserved                           
#cr                                                                       
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: vmd.csh,v $
#       $Author: johns $        $Locker:  $                $State: Exp $
#       $Revision: 1.106 $      $Date: 2016/10/21 23:21:59 $
#
############################################################################
# DESCRIPTION:
#
# csh script to start up vmd, using an xterm-like window for the console
#
############################################################################

#
# User configurable/overridable default settings.
#

# find where the vmd executable is located
if ($?VMDDIR == "0") then
  setenv VMDDIR "$defaultvmddir"
endif

# Location for initial launch of the executable 
# This is added for correct operation on Scyld/Clustermatic clusters
# The regular VMDDIR must be set to the slave node directory location of the
# binary and libraries, which is needed after Bproc spawns the slave process
if ($?MASTERVMDDIR == "0") then
  setenv MASTERVMDDIR "$defaultvmddir"
endif

# set default display device to be windowed
if ($?VMDDISPLAYDEVICE == "0") then
  setenv VMDDISPLAYDEVICE win
endif

# set serial port device to which a Spaceball 6DOF input device
# is connected.  This allows VMD to use the Spaceball as an additional
# input device.  Remember to check permissions on the serial port device
# before you tell VMD to use it.
# Common names for serial ports on Unix are:
#   Solaris: /dev/ttya    /dev/ttyb    /dev/term/a  /dev/term/b
#     Linux: /dev/ttyS0   /dev/ttyS1
#      IRIX: /dev/ttyd1   /dev/ttyd2   /dev/ttyd3   /dev/ttyd4
#     HP-UX: /dev/tty0p0  /dev/tty1p0 
# setenv VMDSPACEBALLPORT /dev/ttyS1

# set a default window position, where x is 0 at the left side of the screen
# and y is 0 at the bottom of the screen.
if ($?VMDSCRPOS == "0") then
  setenv VMDSCRPOS "596 190"
endif

# set a default window size.
if ($?VMDSCRSIZE == "0") then
  setenv VMDSCRSIZE "669 834"
endif

# set a default screen height
if ($?VMDSCRHEIGHT == "0") then
  setenv VMDSCRHEIGHT 6.0
endif

# set a default screen distance
if ($?VMDSCRDIST == "0") then
  setenv VMDSCRDIST -2.0
endif

# set the default behavior for enable/disable of the VMD title screen
if ($?VMDTITLE == "0") then
  setenv VMDTITLE on
endif

# set the default geometry (size/position) used for the VMD command window
if ($?VMD_WINGEOM == "0") then
  setenv VMD_WINGEOM "-geometry 80x11-0-0"
endif

#
# Don't edit items below unless you know what you are doing.
#

# Define the various script library locations
setenv TCL_LIBRARY "$VMDDIR/scripts/tcl"

if ($?PYTHONPATH == "0") then
  setenv PYTHONPATH "$VMDDIR/scripts/python"
else
  setenv PYTHONPATH "$PYTHONPATH/:$VMDDIR/scripts/python"
endif

## if the location of the babel binary isn't already defined
##  1st see if it is on the path
#if ($?VMDBABELBIN == "0") then
#  foreach i ($path)
#    if (-x $i/babel) then
#        setenv VMDBABELBIN "$i/babel"
#        break
#    endif
#  end
#endif

## if not, and if the BABEL_DIR is set, see if the binary is
## in that directory
#if ($?VMDBABELBIN == "0") then
#  if ($?BABEL_DIR) then
#    if (-x "$BABEL_DIR/babel") then
#      setenv VMDBABELBIN "$BABEL_DIR/babel"
#    endif
#  endif
#endif

#  otherwise, outta luck

# check if we're requesting to run without any graphics, and disable
# Spaceball input and X-Windows DISPLAY if no graphics.
# check if VMD is being run as a web helper application, if so, then
# when we spawn the VMD binary, we don't run it as a background process.
set VMDWEBHELPER=0
set VMDRUNDEBUGGER=0
@ parmcount = 0
foreach i ( $* )
  @ parmcount++
  if ( "$argv[$parmcount]" == "-webhelper") then
    set VMDWEBHELPER=1
  endif

  if ( "$argv[$parmcount]" == "-debug") then
    set VMDRUNDEBUGGER=1
  endif

  if ( "$argv[$parmcount]" == "-node") then
    @ parm = $parmcount + 1
    if ( $parm <= $#argv ) then
      setenv VMDSLAVENODE "$argv[$parm]"
    endif
    if ($?DISPLAY) then
      unsetenv DISPLAY
    endif
  endif

  if ( "$argv[$parmcount]" == "-h" || "$argv[$parmcount]" == "--help" ) then
    if ($?DISPLAY) then
      unsetenv DISPLAY
    endif
  endif

  if ( "$argv[$parmcount]" == "-dispdev") then
    @ parm = $parmcount + 1
    if ( $parm <= $#argv ) then
      if ( "$argv[$parm]" == "none" || "$argv[$parm]" == "text" ) then
        if ($?VMDSPACEBALLPORT) then
          unsetenv VMDSPACEBALLPORT
        endif
        if ($?DISPLAY) then
          unsetenv DISPLAY
        endif
      endif
    endif
  endif
end


# determine type of machine, and run appropriate executable
set MACHARCH=`uname -s`
set VMDDEBUGGER="dbx"
switch ($MACHARCH)
case *IRIX*:
  set MACHVER=`uname -r | cut -f1 -d.`
  set VMD_WINTERM=/usr/bin/X11/xterm
  set VMD_WINOPTS="-sb -sl 1000 -e"
  if ("$MACHVER" == "6") then
    set ARCH=IRIX6
  else
    echo "Error: Unknown or unsupported IRIX version $MACHVER"
    exit 1
  endif
breaksw

case *HP-UX*:
  set MACHVER=`uname -r | cut -f2 -d.`
  set VMD_WINTERM=xterm
  set VMD_WINOPTS='-sb -sl 1000 -e'
  if ( "$MACHVER" == "09" ) then
    set ARCH=HPUX9
  else if ( "$MACHVER" == "10" ) then
    set ARCH=HPUX10
  else if ( "$MACHVER" == "11" ) then
    set ARCH=HPUX11
  else
    echo "Error: Unknown or unsupported HP-UX version $MACHVER"
    exit 1
  endif
breaksw

case *AIX*:
  set VERSION=`uname -v`
  set ARCH=AIX${VERSION}
  set VMD_WINTERM=/usr/lpp/X11/bin/aixterm
  set VMD_WINOPTS='-sb -sl 1000 -e'
breaksw

case *FreeBSD*:
  # The standard options
  if (`uname -m` == "i386") then
    set ARCH=FREEBSD
  else if (`uname -m` == "amd64") then
    set ARCH=FREEBSDAMD64
  else
    echo "Error: unsupported FreeBSD version $MACHVER"
    exit 1
  endif
  set VMD_WINTERM=xterm
  set VMD_WINOPTS='-sb -sl 1000 -e'
  set VMDDEBUGGER="gdb"
breaksw

case *Linux*:
  # The standard options
  if (`uname --machine` == "alpha") then
    set ARCH=LINUXALPHA
  else if (`uname --machine` == "aarch64") then
    # AppliedMicro X-Gene, NVIDIA Jetson TX1
    set ARCH=LINUXCARMA
  else if (`uname --machine` == "armv7l") then
    # NVIDIA CARMA, KAYLA, and Jetson TK1
    set ARCH=LINUXCARMA
  else if (`uname --machine` == "x86_64") then
    # Test to see if a 64-bit version of VMD exists
    # in the installation directory, and use the 64-bit
    # version if it is there.
    if ( -e "${VMDDIR}/${vmdbasename}_BLUEWATERS" ) then
      set ARCH=BLUEWATERS
    else if ( -e "${VMDDIR}/${vmdbasename}_CRAY_XC" ) then
      set ARCH=CRAY_XC
    else if ( -e "${VMDDIR}/${vmdbasename}_CRAY_XK" ) then
      set ARCH=CRAY_XK
    else if ( -e "${VMDDIR}/${vmdbasename}_LINUXAMD64" ) then
      set ARCH=LINUXAMD64
    else
      set ARCH=LINUX
    endif
  else if (`uname --machine` == "ia64") then
    set ARCH=LINUXIA64
  else if (`uname --machine` == "ppc") then
    set ARCH=LINUXPPC
  else if (`uname --machine` == "ppc64") then
    if ( -e "${VMDDIR}/${vmdbasename}_LINUXPPC64" ) then
      set ARCH=LINUXPPC64
    else
      set ARCH=LINUXPPC
    endif
  else if (`uname --machine` == "ppc64le") then
    if ( -e "${VMDDIR}/${vmdbasename}_LINUXPPC64LE" ) then
      set ARCH=LINUXPPC64LE
    else if ( -e "${VMDDIR}/${vmdbasename}_OPENPOWER" ) then
      set ARCH=OPENPOWER
    else
      set ARCH=SUMMIT
    endif
  else
    set ARCH=LINUX
  endif
  set VMD_WINTERM=xterm
  set VMD_WINOPTS='-sb -sl 1000 -e'
  set VMDDEBUGGER="gdb"
breaksw

case *SunOS*:
  # The standard options
  if (`uname -p` == "sparc") then
    set ARCH=SOLARIS2
  else
    set ARCH=SOLARISX86
  endif

  set VMD_WINTERM=/usr/openwin/bin/xterm
  set VMD_WINOPTS='-sb -sl 1000 -e'
breaksw

case *OSF*:
  set ARCH=TRU64
  set VMD_WINTERM=xterm
  set VMD_WINOPTS='-sb -sl 1000 -e'
breaksw

case *Rhapsody*:
case *Darwin*:
  if ($?TMPDIR == "0" ) then
    setenv TMPDIR /tmp
  endif
  set ARCH=MACOSX
  set VMD_WINTERM=xterm
  set VMD_WINOPTS='-sb -sl 1000 -e'
  ## Override default window size and position 
  setenv VMDSCRPOS "400 200"
  setenv VMDSCRSIZE "512 512"
  set VMDDEBUGGER="gdb"
breaksw

case *:
  echo "Unsupported architechture."
  echo "Must be AIX, HP-UX, IRIX, Linux, SunOS, or TRU64."
breaksw

endsw

# Test to see if a 64-bit version of VMD exists
# in the installation directory, and use the 64-bit
# version if it is there.
if ( -e "${MASTERVMDDIR}/${vmdbasename}_${ARCH}_64" ) then
  set ARCH=${ARCH}_64
endif

# set VMD executable name based on architecture
set execname="$vmdbasename"_$ARCH

# update shared library search path so we find redistributable
# shared libraries needed for compiler runtimes, CUDA, etc.
if ($?LD_LIBRARY_PATH == 0) then
  setenv LD_LIBRARY_PATH ${MASTERVMDDIR}
else
  setenv LD_LIBRARY_PATH ${MASTERVMDDIR}:${LD_LIBRARY_PATH}
endif

# detect if we have rlwrap available to have commandline editing
set vmdprefixcmd=""
if (("${ARCH}" == "LINUX") || ("${ARCH}" == "LINUXAMD64")) then
  set rlwrap=`which rlwrap`
  if ( -x "$rlwrap" ) then
    if ( -f ${MASTERVMDDIR}/vmd_completion.dat ) then 
      set vmdprefixcmd="rlwrap -C vmd -c -b(){}[],&^%#;|\\ -f ${MASTERVMDDIR}/vmd_completion.dat "
    else 
      set vmdprefixcmd="rlwrap -C vmd -c -b(){}[],&^%#;|\\ "
    endif
  endif
endif

# set the path to a few external programs
# Stride -- used to generate cartoon representations etc.
if ($?STRIDE_BIN == "0") then
  if (-x "$MASTERVMDDIR/stride_$ARCH") then
    setenv STRIDE_BIN "$VMDDIR/stride_$ARCH"
  endif
endif

# Surf -- used to generate molecular surfaces
if ($?SURF_BIN == "0") then
  if (-x "$MASTERVMDDIR/surf_$ARCH") then
    setenv SURF_BIN "$VMDDIR/surf_$ARCH"
  endif
endif

# Tachyon -- used to generate ray traced graphics
if ($?TACHYON_BIN == "0") then
  if (-x "$MASTERVMDDIR/tachyon_$ARCH") then
    setenv TACHYON_BIN "$VMDDIR/tachyon_$ARCH"
  endif
endif

if ($?VMDCUSTOMIZESTARTUP) then
  source $VMDCUSTOMIZESTARTUP
endif

# if VMDRUNDEBUGGER is set, spawn VMD in a debugger
if ("$VMDRUNDEBUGGER" == "1") then
  echo "***"
  echo "*** Running VMD in debugger, type 'run' at debugger prompt"
  echo "***"
  "$VMDDEBUGGER" "$MASTERVMDDIR/$execname"

else if ($ARCH == "BLUEWATERS" || $ARCH == "CRAY_XC" || $ARCH == "CRAY_XK") then
  ##
  ## NCSA Blue Waters, ORNL Titan Cray XK7 
  ##
  ## On Blue Waters, jobs w/ OpenGL Pbuffer supprt are launched with 
  ## the "xk,gres=viz" feature, e.g. an interactive 512-node job:
  ##   qsub -I -V -l nodes=512:ppn=16:xk,gres=viz
  ## 
  ## if we are running outside of the queueing system, then we
  ## run the VMD the normal way
  if ($?PBS_NUM_PPN == "0") then
    # tell VMD not to initialize MPI
    setenv VMDNOMPI 1
    $vmdprefixcmd "$MASTERVMDDIR/$execname" $*
    exit 0
  endif

  setenv MAXTHR           $PBS_NUM_PPN
  setenv WKFORCECPUCOUNT  $MAXTHR
  setenv VMDFORCECPUCOUNT $MAXTHR
  setenv RTFORCECPUCOUNT  $MAXTHR
  
  # total num $PBS_NP
  @ mppwidth = $PBS_NUM_NODES
  @ mppdepth = $PBS_NUM_PPN
  @ mppnppn = 1
  
  # enable OpenGL Pbuffer off-screen rendering support on Cray XK7 nodes...
  setenv DISPLAY :0
  module add opengl

  echo "Launching VMD w/ mppwidth=$mppwidth, mppdepth=$mppdepth, mppnppn=$mppnppn"
  aprun -n $mppwidth -d $mppdepth -N $mppnppn "$MASTERVMDDIR/$execname" $*

else if ($?VMDSLAVENODE) then
  ##
  ## If VMDSLAVENODE is set, then run on a Scyld/Clustermatic slave node
  ##
  ##
  ## When running VMD on a cluster, you may have to add dynamic link libraries
  ## on the local disks on the slave nodes for it to run.  If so, you may also
  ## need to update the LD_LIBRARY_PATH variable for the location of these
  ## libraries on the slave node filesystem
  ##
  ## setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/scr1/johns/cluster/lib/shared
  ##
  echo "***" 
  echo "*** Running VMD on cluster node $VMDSLAVENODE..."
  echo "***"
  bpsh $VMDSLAVENODE "$MASTERVMDDIR/$execname" $*
else
#  # if DISPLAY is set, spawn off a terminal, else use current terminal
#  if ($?DISPLAY) then
#    if ("$VMDWEBHELPER" == "1") then
##      exec $VMD_WINTERM -T "vmd console" $VMD_WINGEOM $VMD_WINOPTS \
##  		"$MASTERVMDDIR/$execname" $*
#      exec $VMD_WINTERM -T "vmd console" $VMD_WINGEOM $VMD_WINOPTS \
#	"env LD_LIBRARY_PATH=$LD_LIBRARY_PATH $MASTERVMDDIR/$execname" $*
#    else
##      exec $VMD_WINTERM -T "vmd console" $VMD_WINGEOM $VMD_WINOPTS \
##  		"$MASTERVMDDIR/$execname" $* &
#      exec $VMD_WINTERM -T "vmd console" $VMD_WINGEOM $VMD_WINOPTS \
#	"env LD_LIBRARY_PATH=$LD_LIBRARY_PATH $MASTERVMDDIR/$execname" $* &
#    endif
#  else
##
##    
##  MPI-based VMD runs usually need a bit of help to determine appropriate
##  node count to use when calling mpirun, here are a couple of examples
##  that work for the machines at NCSA based on the PBS queuing system:
##
##  NCSA "AC" GPU cluster:
##    set nodecount = `qstat -f $PBS_JOBID |grep nodect | awk '{print $3}'`
##    echo "Starting VMD on $nodecount nodes..."
##    mpirun -machinefile $PBS_NODEFILE -n $nodecount "$MASTERVMDDIR/$execname.mpi" $*
##
##    
## Normal startup on a typical desktop machine...
##
    $vmdprefixcmd "$MASTERVMDDIR/$execname" $*
#  endif
endif

