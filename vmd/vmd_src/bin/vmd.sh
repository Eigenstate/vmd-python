#!/bin/sh 
## In case the installation didn't add these (which means you didn't
# do the 'configure', here's the default settings:
#defaultvmddir=/usr/local/lib/vmd
#vmdbasename=vmd

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
#       $RCSfile: vmd.sh,v $
#       $Author: johns $        $Locker:  $                $State: Exp $
#       $Revision: 1.18 $      $Date: 2016/11/15 14:49:55 $
#
############################################################################
# DESCRIPTION:
#
# bourne shell script to start up vmd, using an xterm-like window for the console
#
############################################################################

#
# User configurable/overridable default settings.
#

# find where the vmd executable is located
VMDDIR=${VMDDIR-${defaultvmddir}}
export VMDDIR

# Location for initial launch of the executable 
# This is added for correct operation on Scyld/Clustermatic clusters
# The regular VMDDIR must be set to the slave node directory location of the
# binary and libraries, which is needed after Bproc spawns the slave process
MASTERVMDDIR=${MASTERVMDDIR-${defaultvmddir}}
export MASTERVMDDIR 

# set default display device to be windowed
VMDDISPLAYDEVICE=${VMDDISPLAYDEVICE-win}
export VMDDISPLAYDEVICE

# set serial port device to which a Spaceball 6DOF input device
# is connected.  This allows VMD to use the Spaceball as an additional
# input device.  Remember to check permissions on the serial port device
# before you tell VMD to use it.
# Common names for serial ports on Unix are:
#   Solaris: /dev/ttya    /dev/ttyb    /dev/term/a  /dev/term/b
#     Linux: /dev/ttyS0   /dev/ttyS1
#      IRIX: /dev/ttyd1   /dev/ttyd2   /dev/ttyd3   /dev/ttyd4
#     HP-UX: /dev/tty0p0  /dev/tty1p0 
# VMDSPACEBALLPORT=/dev/ttyS1
# export VMDSPACEBALLPORT

# set a default window position, where x is 0 at the left side of the screen
# and y is 0 at the bottom of the screen.
VMDSCRPOS=${VMDSCRPOS-596 190}
export VMDSCRPOS

# set a default window size.
VMDSCRSIZE=${VMDSCRSIZE-669 834}
export VMDSCRSIZE

# set a default screen height
VMDSCRHEIGHT=${VMDSCRHEIGHT-6.0}
export VMDSCRHEIGHT

# set a default screen distance
VMDSCRDIST=${VMDSCRDIST--2.0}
export VMDSCRDIST

# set the default behavior for enable/disable of the VMD title screen
VMDTITLE=${VMDTITLE-on}
export VMDTITLE

# set the default geometry (size/position) used for the VMD command window
VMD_WINGEOM=${VMD_WINGEOM--geometry 80x11-0-0}
export VMD_WINGEOM

#
# Don't edit items below unless you know what you are doing.
#

# Define the various script library locations
TCL_LIBRARY="${VMDDIR}/scripts/tcl"
export TCL_LIBRARY

if [ -z "${PYTHONPATH}" ]
then
  PYTHONPATH="${VMDDIR}/scripts/python"
else
  PYTHONPATH="${PYTHONPATH}/:${VMDDIR}/scripts/python"
fi
export PYTHONPATH

## if the location of the babel binary isn't already defined
##  1st see if it is on the path
#if [ -z "${VMDBABELBIN}" ]
#then
#  oldifs="${IFS}"
#  IFS=':'
#  for i in ${PATH} 
#  do
#    if [ -x "${i}/babel" ]
#    then
#        VMDBABELBIN="${i}/babel"
#        break
#    fi
#  done
#  IFS="${oldifs}"
#fi

## if not, and if the BABEL_DIR is set, see if the binary is
## in that directory
#if [ -z "${VMDBABELBIN}" ] && [ -n "${BABEL_DIR}" ]
#then
#    if [ -x "$BABEL_DIR/babel" ]
#    then
#      VMDBABELBIN="$BABEL_DIR/babel"
#    fi
#fi

#if [ -n "${VMDBABELBIN}" ]
#then 
#   export VMDBABELBIN
#fi
#  otherwise, outta luck

# check if we're requesting to run without any graphics, and disable
# Spaceball input and X-Windows DISPLAY if no graphics.
# check if VMD is being run as a web helper application, if so, then
# when we spawn the VMD binary, we don't run it as a background process.
VMDWEBHELPER=0
VMDRUNDEBUGGER=0

needarg=""
for i in "$@"
do
  if [ -z "$needarg" ]
  then
    case "$i" in

      -webhelper)
        VMDWEBHELPER=1
        export VMDWEBHELPER
        ;;

      -debug)
        VMDRUNDEBUGGER=1
        ;;

      -node)
        needarg="$i"
        if [ -n "$DISPLAY" ]
        then
          unset DISPLAY
        fi
        ;;
        
      -h | --help)
        if [ -n "$DISPLAY" ]
        then
          unset DISPLAY
        fi
        ;;

      -dispdev)
        needarg="$i"
        if [ -n "$DISPLAY" ]
        then
          unset DISPLAY
        fi
        ;;

    esac

  else

    case "$needarg" in

      -node)
        VMDSLAVENODE="$i"
        export VMDSLAVENODE
        ;;

      -dispdev)
        if [ "$i" = "none" ] || [ "$i" = "text" ]
        then
          if [ -n "$VMDSPACEBALLPORT" ]
          then
            unset VMDSPACEBALLPORT
          fi
          if [ -n "$DISPLAY" ]
          then
            unset DISPLAY
          fi
        fi
        ;;
    esac
    needarg=""
  fi
done

# determine type of machine, and run appropriate executable
MACHARCH=`uname -s`
VMDDEBUGGER="dbx"

case $MACHARCH in

  *IRIX*)
    MACHVER=`uname -r | cut -f1 -d.`
    VMD_WINTERM=/usr/bin/X11/xterm
    VMD_WINOPTS="-sb -sl 1000 -e"
    if [ "$MACHVER" = "6" ]
    then
      ARCH=IRIX6
    else
      echo "Error: Unknown or unsupported IRIX version $MACHVER"
      exit 1
    fi
    ;;

  *HP-UX*)
    MACHVER=`uname -r | cut -f2 -d.`
    VMD_WINTERM=xterm
    VMD_WINOPTS='-sb -sl 1000 -e'
    if [ "$MACHVER" = "09" ]
    then
      ARCH=HPUX9
    elif [ "$MACHVER" = "10" ]
    then
      ARCH=HPUX10
    elif [ "$MACHVER" = "11" ]
    then
      ARCH=HPUX11
    else
      echo "Error: Unknown or unsupported HP-UX version $MACHVER"
      exit 1
    fi
    ;;

  *AIX*)
    VERSION=`uname -v`
    ARCH=AIX${VERSION}
    VMD_WINTERM=/usr/lpp/X11/bin/aixterm
    VMD_WINOPTS='-sb -sl 1000 -e'
    ;;

  *FreeBSD*)
    # The standard options
    if [ `uname -m` = "i386" ]
    then
      ARCH=FREEBSD
    elif [ `uname -m` = "amd64" ]
    then
      ARCH=FREEBSDAMD64
    else
      echo "Error: unsupported FreeBSD version $MACHVER"
      exit 1
    fi
    VMD_WINTERM=xterm
    VMD_WINOPTS='-sb -sl 1000 -e'
    VMDDEBUGGER="gdb"
    ;;

  *Linux*)
    # The standard options
    if [ `uname --machine` = "alpha" ]
    then
      ARCH=LINUXALPHA
    elif [ `uname --machine` = "aarch64" ]
    then
      # AppliedMicro X-Gene, NVIDIA Jetson TX1
      ARCH=LINUXCARMA
    elif [ `uname --machine` = "armv7l" ]
    then
      # NVIDIA CARMA, KAYLA, and Jetson TK1
      ARCH=LINUXCARMA
    elif [ `uname --machine` = "x86_64" ]
    then
      # Test to see if a 64-bit version of VMD exists
      # in the installation directory, and use the 64-bit
      # version if it is there.
      if [ -x "${VMDDIR}/${vmdbasename}_BLUEWATERS" ]
      then
        ARCH=BLUEWATERS
      elif [ -x "${VMDDIR}/${vmdbasename}_CRAY_XC" ]
      then
        ARCH=CRAY_XC
      elif [ -x "${VMDDIR}/${vmdbasename}_CRAY_XK" ]
      then
        ARCH=CRAY_XK
      elif [ -x "${VMDDIR}/${vmdbasename}_LINUXAMD64" ]
      then
        ARCH=LINUXAMD64
      else
        ARCH=LINUX
      fi
    elif [ `uname --machine` = "ia64" ]
    then
      ARCH=LINUXIA64
    elif [ `uname --machine` = "ppc64" ]
    then
      # Test to see if a 64-bit version of VMD exists
      # in the installation directory, and use the 64-bit
      # version if it is there.
      if [ -x "${VMDDIR}/${vmdbasename}_LINUXPPC64" ]
      then
        ARCH=LINUXPPC64
      else
        ARCH=LINUXPPC
      fi
    elif [ `uname --machine` = "ppc64le" ]
    then
      # Test to see if a 64-bit version of VMD exists
      # in the installation directory, and use the 64-bit
      # version if it is there.
      if [ -x "${VMDDIR}/${vmdbasename}_LINUXPPC64LE" ]
      then
        ARCH=LINUXPPC64LE
      elif [ -x "${VMDDIR}/${vmdbasename}_OPENPOWER" ]
      then
        ARCH=OPENPOWER
      else
        ARCH=SUMMIT
      fi
    elif [ `uname --machine` = "ppc" ]
    then
      ARCH=LINUXPPC
    else
      ARCH=LINUX
    fi
    VMD_WINTERM=xterm
    VMD_WINOPTS='-sb -sl 1000 -e'
    VMDDEBUGGER="gdb"
    ;;

  *SunOS*)
    # The standard options
    if [ `uname -p` = "sparc" ]
    then
      ARCH=SOLARIS2
    else
      ARCH=SOLARISX86
    fi

    VMD_WINTERM=/usr/openwin/bin/xterm
    VMD_WINOPTS='-sb -sl 1000 -e'
    ;;

  *OSF*)
    ARCH=TRU64
    VMD_WINTERM=xterm
    VMD_WINOPTS='-sb -sl 1000 -e'
    ;;

  *Rhapsody* | *Darwin*)
     if [ -z "$TMPDIR" ]
     then
       TMPDIR=/tmp
       export TMPDIR
     fi
    ARCH=MACOSX
    VMD_WINTERM=xterm
    VMD_WINOPTS='-sb -sl 1000 -e'
    ## Override default window size and position 
    VMDSCRPOS="400 200"
    VMDSCRSIZE="512 512"
    export VMDSCRPOS VMDSCRSIZE
    VMDDEBUGGER="gdb"
    ;;

   *)
  echo "Unsupported architechture."
  echo "Must be AIX, HP-UX, IRIX, Linux, SunOS, or TRU64."
    ;;

esac

# Test to see if a 64-bit version of VMD exists
# in the installation directory, and use the 64-bit
# version if it is there.
if [ -x "${MASTERVMDDIR}/${vmdbasename}_${ARCH}_64" ]
then
  ARCH=${ARCH}_64
fi

# set VMD executable name based on architecture
execname="$vmdbasename"_$ARCH

# update shared library search path so we find redistributable
# shared libraries needed for compiler runtimes, CUDA, etc.
if [ -z "${LD_LIBRARY_PATH}" ]
then
  LD_LIBRARY_PATH="${MASTERVMDDIR}"
else
  LD_LIBRARY_PATH="${MASTERVMDDIR}:${LD_LIBRARY_PATH}"
fi
export LD_LIBRARY_PATH


# detect if we have rlwrap available to have commandline editing
case $MACHARCH in
  *Linux*)
    if hash rlwrap 2>/dev/null
    then
      if [ -f "${MASTERVMDDIR}/vmd_completion.dat" ]
      then
        vmdprefixcmd="rlwrap -C vmd -c -b(){}[],&^%#;|\\ -f ${MASTERVMDDIR}/vmd_completion.dat "
      else
        vmdprefixcmd="rlwrap -C vmd -c -b(){}[],&^%#;|\\ "
      fi
    else
      vmdprefixcmd=""
    fi
    ;;

   *)
    vmdprefixcmd=""
    ;;
esac


# set the path to a few external programs
# Stride -- used to generate cartoon representations etc.
if [ -z "$STRIDE_BIN" ]
then
  if [ -x "$MASTERVMDDIR/stride_$ARCH" ]
  then
    STRIDE_BIN="$VMDDIR/stride_$ARCH"
    export STRIDE_BIN
  fi
fi

# Surf -- used to generate molecular surfaces
if [ -z "$SURF_BIN" ]
then
  if [ -x "$MASTERVMDDIR/surf_$ARCH" ]
  then
    SURF_BIN="$VMDDIR/surf_$ARCH"
    export SURF_BIN
  fi
fi

# Tachyon -- used to generate ray traced graphics
if [ -z "$TACHYON_BIN" ]
then
  if [ -x "$MASTERVMDDIR/tachyon_$ARCH" ]
  then
    TACHYON_BIN="$VMDDIR/tachyon_$ARCH"
    export TACHYON_BIN
  fi
fi

if [ -n "$VMDCUSTOMIZESTARTUP" ]
then
  . $VMDCUSTOMIZESTARTUP
fi

# if VMDRUNDEBUGGER is set, spawn VMD in a debugger
if [ "$VMDRUNDEBUGGER" = "1" ]
then
  echo "***"
  echo "*** Running VMD in debugger, type 'run' at debugger prompt"
  echo "***"
  "$VMDDEBUGGER" "$MASTERVMDDIR/$execname"
elif [ -n "$VMDSLAVENODE" ]
then
  ##
  ##if VMDSLAVENODE is set, then run on a Scyld/Clustermatic slave node
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
  bpsh $VMDSLAVENODE "$MASTERVMDDIR/$execname" "$@"
else
#  # if DISPLAY is set, spawn off a terminal, else use current terminal
#  if [ -n "$DISPLAY" ]
#    then
#    if [ "$VMDWEBHELPER" = "1" ]
#      then
#      exec $VMD_WINTERM -T "vmd console" $VMD_WINGEOM $VMD_WINOPTS \
#  		"$MASTERVMDDIR/$execname" "$@"
#    else
#      exec $VMD_WINTERM -T "vmd console" $VMD_WINGEOM $VMD_WINOPTS \
#  		"$MASTERVMDDIR/$execname" "$@" &
#    fi
#  else
    exec $vmdprefixcmd "$MASTERVMDDIR/$execname" "$@"
#  fi
fi

