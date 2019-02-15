#!/bin/csh

##
## Script for building plugins on all supported platforms
##
#setenv DATE `date +%Y-%m-%d-%T`
setenv DATE `date +%m%d-%H%M%S`

##
## BioCoRE logging (eventually, need other changes first)
##
#setenv BUILDNUM `cat /Projects/vmd/vmd/vmdbuild.number`;
#setenv LOGRUN  'biolog -f -p vmd -k "VMD plugins build $BUILDNUM, BUILD SUMMARY" -s "VMD build $BUILDNUM, BUILD SUMMARY"'
#setenv LOGGING 'biolog -f -p vmd -k "VMD plugins build $BUILDNUM, Platform: $1"  -s "VMD build $BUILDNUM, Platform: $1"'

setenv unixdir `pwd` 

##
## Check for builds on remote hosted supercomputers, etc.
##
switch ( `hostname` )
 ## Amazon EC2
 case ip-*-*-*-*:
    echo "Using build settings for Amazon EC2"
    setenv TCLINC -I/home/ec2-user/vmd/lib/tcl/include
    setenv TCLLIB -L/home/ec2-user/vmd/lib/tcl
    cd $unixdir; gmake LINUXAMD64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXAMD64 >& log.LINUXAMD64.$DATE < /dev/null &
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

 ## IBM Poughkeepsie center P8 "Minsky" and P9 "Newell" test machines
 case p10login4: 
 case p10login3: 
    echo "Using build settings for POWER8/9+P/V100 test box"
    setenv TCLINC -I/gpfs/gpfs_gl4_16mb/b8p148/b8p148aa/vmd/lib/tcl/include
    setenv TCLLIB -L/gpfs/gpfs_gl4_16mb/b8p148/b8p148aa/vmd/lib/tcl
    cd $unixdir; make OPENPOWER TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_OPENPOWER >& log.OPENPOWER.$DATE < /dev/null &
#    setenv NETCDFINC -I/gpfs/gpfs_gl4_16mb/b8p148/b8p148aa/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/gpfs/gpfs_gl4_16mb/b8p148/b8p148aa/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

 ## NVIDIA P8 "Minsky" test machines
 case pwr02: 
 case pwr03: 
    echo "Using build settings for POWER8+P100 test box"
    setenv TCLINC -I/home/jstone/vmd/lib/tcl/include
    setenv TCLLIB -L/home/jstone/vmd/lib/tcl
    cd $unixdir; make OPENPOWER TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SUMMIT >& log.OPENPOWER.$DATE < /dev/null &
#    setenv NETCDFINC -I/autofs/na3_home1/stonej1/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/autofs/na3_home1/stonej1/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

 ## psgcluster.nvidia.com
 case psgcluster*:
    echo "Using build settings for PSG cluster"
    setenv TCLINC -I/home/jstone/vmd/lib/tcl/include
    setenv TCLLIB -L/home/jstone/vmd/lib/tcl
    cd $unixdir; make LINUXAMD64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXAMD64 >& log.LINUXAMD64.$DATE < /dev/null &
#    setenv NETCDFINC -I/autofs/na3_home1/stonej1/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/autofs/na3_home1/stonej1/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

 # KTH PDC
 case beskow-login*:
    echo "Using build settings for KTH PDC Beskow Cray XC40"
    setenv TCLINC -I/cfs/klemming/nobackup/j/johnst/vmd/lib/tcl/include
    setenv TCLLIB -L/cfs/klemming/nobackup/j/johnst/vmd/lib/tcl
    cd $unixdir; gmake CRAY_XC TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_CRAY_XC >& log.CRAY_XC.$DATE < /dev/null &
#    setenv NETCDFINC -I/autofs/na3_home1/stonej1/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/autofs/na3_home1/stonej1/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;


 ## ORNL "Summit" P9+Volta
 ## ORNL Crest, Summit precursor 
 case login1:
 case crest-login1*:
    echo "Using build settings for ORNL Crest IBM POWER8"
    setenv TCLINC -I/autofs/nccs-svm1_home1/stonej1/vmd/lib/tcl/include
    setenv TCLLIB -L/autofs/nccs-svm1_home1/stonej1/vmd/lib/tcl
    cd $unixdir; make SUMMIT TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SUMMIT >& log.SUMMIT.$DATE < /dev/null &
#    setenv NETCDFINC -I/autofs/na3_home1/stonej1/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/autofs/na3_home1/stonej1/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;


 ## ORNL "Titan" Cray XK7
 case titan-ext1:
 case titan-ext2:
 case titan-ext3:
 case titan-ext4:
 case titan-ext5:
 case titan-ext6:
 case titan-ext7:
    echo "Using build settings for ORNL Titan Cray XK7"
    setenv TCLINC -I/autofs/na3_home1/stonej1/vmd/lib/tcl/include
    setenv TCLLIB -L/autofs/na3_home1/stonej1/vmd/lib/tcl
    cd $unixdir; gmake CRAY_XK TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_CRAY_XK >& log.CRAY_XK.$DATE < /dev/null &
#    setenv NETCDFINC -I/autofs/na3_home1/stonej1/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/autofs/na3_home1/stonej1/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;


 ## IU Big Red II Cray XE6/XK7
 case login2:
 case login3:
    echo "Using build settings for IU Cray Big Red II"
    setenv TCLINC -I/N/u/johstone/BigRed2/vmd/lib/tcl/include
    setenv TCLLIB -L/N/u/johstone/BigRed2/vmd/lib/tcl
    cd $unixdir; gmake CRAY_XK TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_CRAY_XK >& log.CRAY_XK.$DATE < /dev/null &
#    setenv NETCDFINC -I/N/u/johstone/BigRed2/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/N/u/johstone/BigRed2/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

 ## NCSA "Blue Waters" Cray XE6/XK7
 case h2ologin1:
 case h2ologin2:
 case h2ologin3:
 case h2ologin4:
    echo "Using build settings for NCSA Cray Blue Waters"
    setenv TCLINC -I/u/sciteam/stonej/vmd/lib/tcl/include
    setenv TCLLIB -L/u/sciteam/stonej/vmd/lib/tcl
    cd $unixdir; gmake BLUEWATERS TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_BLUEWATERS >& log.BLUEWATERS.$DATE < /dev/null &
#    setenv NETCDFINC -I/home/stonej/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/home/stonej/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

 ## CSCS Piz Daint Cray XC50
 case daint*:
    echo "Using build settings for CSCS Cray XC50 Piz Daint"
    setenv TCLINC -I/users/stonej/vmd/lib/tcl/include
    setenv TCLLIB -L/users/stonej/vmd/lib/tcl
    cd $unixdir; gmake CRAY_XC TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_CRAY_XC >& log.CRAY_XC.$DATE < /dev/null &
#    setenv NETCDFINC -I/home/stonej/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/home/stonej/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

 ## NCSA "Blue Drop"
 case bd-login:
    echo "Using build settings for NCSA IBM Blue Drop..."
    setenv TCLINC -I/home/stonej/vmd/lib/tcl/include
    setenv TCLLIB -L/home/stonej/vmd/lib/tcl
    cd $unixdir; gmake BLUEWATERS TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_BLUEWATERS >& log.BLUEWATERS.$DATE < /dev/null &
#    setenv NETCDFINC -I/home/stonej/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/home/stonej/vmd/lib/netcdf
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

 ## NCSA 'qp' or 'ac' GPU cluster
 case acfs*:
 case ac.ncsa*:
 case qp.ncsa*:
   echo "Using build settings for NCSA GPU cluster..."
   setenv NETCDFINC -I/home/ac/stonej/vmd/lib/netcdf/include
   setenv NETCDFLIB -L/home/ac/stonej/vmd/lib/netcdf
   setenv TCLINC -I/home/ac/stonej/vmd/lib/tcl/include
   setenv TCLLIB -L/home/ac/stonej/vmd/lib/tcl
   cd $unixdir; gmake LINUXAMD64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUXAMD64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXAMD64 >& log.LINUXAMD64.$DATE  < /dev/null & 
   echo "Waiting for all plugin make jobs to complete..."
   wait;
   echo ""
   echo "Plugin builds done..."
   breaksw;

  ## NCSA "Blue Print", "Copper"
  case bp-login1:
  case cu12:
    echo "Using build settings for NCSA IBM Regatta..."
    setenv TCLINC -I/u/home/ac/stonej/vmd/lib/tcl/include
    setenv TCLLIB -L/u/home/ac/stonej/vmd/lib/tcl
    cd $unixdir; gmake AIX6_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_AIX6_64 >& log.AIX6_64.$DATE < /dev/null &
#    setenv NETCDFINC -I/u/ac/stonej/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/u/ac/stonej/vmd/lib/netcdf
#    setenv TCLINC -I/u/ac/stonej/vmd/lib/tcl/include
#    setenv TCLLIB -L/u/ac/stonej/vmd/lib/tcl
#    cd $unixdir; gmake AIX6 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_AIX6 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_AIX6 >& log.AIX6.$DATE < /dev/null &
#    cd $unixdir; gmake AIX6_64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_AIX6_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_AIX6_64 >& log.AIX6_64.$DATE  < /dev/null &
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo ""
    echo "Plugin builds done..."
    breaksw;


   ## Indiana BigRed
  case s10c2b6:
    echo "Using build settings for IU BigRed PowerPC Linux..."
#    setenv NETCDFINC -I/N/hd03/tg-johns/BigRed/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/N/hd03/tg-johns/BigRed/vmd/lib/netcdf
    setenv TCLINC -I/N/hd03/tg-johns/BigRed/vmd/lib/tcl/include
    setenv TCLLIB -L/N/hd03/tg-johns/BigRed/vmd/lib/tcl
#    cd $unixdir; gmake LINUXPPC64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUXPPC64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXPPC64 >& log.LINUXPPC64.$DATE < /dev/null & 
    cd $unixdir; gmake LINUXPPC64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXPPC64 >& log.LINUXPPC64.$DATE < /dev/null &
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

  ## TCBG development machines 
  case tegra-ubuntu*:
    echo "Using build settings for TB CARMA dev board..."
    setenv TCLINC -I/home/johns/vmd/lib/tcl/include
    setenv TCLLIB -L/home/johns/vmd/lib/tcl
    ssh -x cupertino "cd $unixdir; gmake LINUXCARMA TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXCARMA >& log.CARMA.$DATE " < /dev/null &
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

  ## TCBG development machines 
  case seco-gpu-devkit*:
    echo "Using build settings for TB KAYLA dev board..."
    setenv TCLINC -I/home/johns/vmd/lib/tcl/include
    setenv TCLLIB -L/home/johns/vmd/lib/tcl
    ssh -x localhost "cd $unixdir; gmake LINUXCARMA TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXCARMA >& log.KAYLA.$DATE " < /dev/null &
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;

  case dallas*:
  case casablanca*:
  case moline*:
    echo "Using build settings for TB network..."
    setenv HDFINC "-I/Projects/vmd/vmd/lib/hdf5/hdf5-1.10.2/src -I/Projects/vmd/vmd/lib/hdf5/hdf5-1.10.2/hl/src"
    setenv HDFLIB -L/Projects/vmd/vmd/lib/hdf5
    setenv HDFLDFLAGS "-lhdf5 -lhdf5_hl"

    setenv NETCDFINC -I/Projects/vmd/vmd/lib/netcdf/include
    setenv NETCDFLIB -L/Projects/vmd/vmd/lib/netcdf

    setenv SQLITEINC -I/Projects/vmd/vmd/lib/sqlite/sqlite
    setenv SQLITELIB -L/Projects/vmd/vmd/lib/sqlite

    setenv TCLINC -I/Projects/vmd/vmd/lib/tcl/include
    ## MacOS X framework paths
    setenv TCLLIB -F/Projects/vmd/vmd/lib/tcl

# Use our own custom Tcl framework
#    ssh -x sydney "cd $unixdir; gmake MACOSX NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSX TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_MACOSX >& log.MACOSX.$DATE " < /dev/null &
# Use Apple-Provided Tcl framework
#    ssh -x sydney "cd $unixdir; gmake MACOSX NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSX TCLINC=-F/System/Library/Frameworks TCLLIB=-F/System/Library/Frameworks >& log.MACOSX.$DATE " < /dev/null &
# Use our own custom Tcl framework
#    ssh -x juneau "cd $unixdir; gmake MACOSXX86 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSXX86 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_MACOSXX86 >& log.MACOSXX86.$DATE " < /dev/null &

## enable libexpat for the HOOMD plugin, which also requires MacOS X 10.5,
## enable sqlite for dmsplugin.
#    ssh -x bogota "cd $unixdir; gmake MACOSXX86 EXPATDYNAMIC=1 EXPATINC=-I/usr/include EXPATLIB=-L/usr/lib64 EXPATLDFLAGS=-lexpat NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSXX86 SQLITEDYNAMIC=1 SQLITEINC=$SQLITEINC SQLITELIB=$SQLITELIB/lib_MACOSXX86 SQLITELDFLAGS=-lsqlite3 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_MACOSXX86 >& log.MACOSXX86.$DATE " < /dev/null &
    ssh -x malaga "cd $unixdir; gmake MACOSXX86 EXPATDYNAMIC=1 EXPATINC=-I/usr/include EXPATLIB=-L/usr/lib64 EXPATLDFLAGS=-lexpat NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSXX86 SQLITEDYNAMIC=1 SQLITEINC=$SQLITEINC SQLITELIB=$SQLITELIB/lib_MACOSXX86 SQLITELDFLAGS=-lsqlite3 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_MACOSXX86 >& log.MACOSXX86.$DATE " < /dev/null &

# Use Apple-Provided Tcl framework
#    ssh -x juneau "cd $unixdir; gmake MACOSXX86 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSXX86 TCLINC=-F/System/Library/Frameworks TCLLIB=-F/System/Library/Frameworks >& log.MACOSXX86.$DATE " < /dev/null &

## 64-bit MacOS X builds
## XXX uses hacked header path for now until the other platforms are 
##     also compiled against Tcl/Tk 8.5.9
#    ssh -x melbourne "cd $unixdir; gmake MACOSXX86_64 EXPATDYNAMIC=1 EXPATINC=-I/usr/include EXPATLIB=-L/usr/lib64 EXPATLDFLAGS=-lexpat SQLITEDYNAMIC=1 SQLITEINC=$SQLITEINC SQLITELIB=$SQLITELIB/lib_MACOSXX86_64 SQLITELDFLAGS=-lsqlite3 TCLINC=-I/Projects/vmd/vmd/lib/tcl/lib_MACOSXX86_64/Tcl.framework/Headers TCLLIB=$TCLLIB/lib_MACOSXX86_64 >& log.MACOSXX86_64.$DATE " < /dev/null &
    ssh -x malaga "cd $unixdir; gmake MACOSXX86_64 EXPATDYNAMIC=1 EXPATINC=-I/usr/include EXPATLIB=-L/usr/lib EXPATLDFLAGS=-lexpat SQLITEDYNAMIC=1 SQLITEINC=$SQLITEINC SQLITELIB=$SQLITELIB/lib_MACOSXX86_64 SQLITELDFLAGS=-lsqlite3 TCLINC=-I/Projects/vmd/vmd/lib/tcl/lib_MACOSXX86_64/Tcl.framework/Headers TCLLIB=$TCLLIB/lib_MACOSXX86_64 >& log.MACOSXX86_64.$DATE " < /dev/null &

    ##
    ## link paths for rest of the unix platforms
    ##
    setenv TCLLIB -L/Projects/vmd/vmd/lib/tcl
    setenv SQLITELIB -L/Projects/vmd/vmd/lib/sqlite
    setenv HDFLIB -L/Projects/vmd/vmd/lib/hdf5
    setenv HDFLDFLAGS "-lhdf5 -lhdf5_hl"

# Android builds for ARM V7A hardware, using Android NDK cross compilers
    ssh -x asuncion "cd $unixdir; gmake ANDROIDARMV7A TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_ANDROIDARMV7A >& log.ANDROIDARMV7A.$DATE " < /dev/null &

# build X11/Unix style 64-bit VMD for MacOS X since Tcl/Tk use Carbon otherwise
#    ssh -x bogota "cd $unixdir; gmake MACOSXX86_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_MACOSXX86_64 >& log.MACOSXX86_64.$DATE " < /dev/null &

#    ssh -x dallas "cd $unixdir; gmake LINUX NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUX TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUX >& log.LINUX.$DATE " < /dev/null &

#    ssh -x dallas "cd $unixdir; gmake LINUXAMD64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUXAMD64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXAMD64 >& log.LINUXAMD64.$DATE " < /dev/null &

## HOOMD plugin requires libexpat 
#    ssh -x dallas "cd $unixdir; gmake LINUX EXPATDYNAMIC=1 EXPATINC=-I/usr/include EXPATLIB=-L/usr/lib EXPATLDFLAGS=-lexpat NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUX SQLITEDYNAMIC=1 SQLITEINC=$SQLITEINC SQLITELIB=$SQLITELIB/lib_LINUX SQLITELDFLAGS=-lsqlite3 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUX >& log.LINUX.$DATE " < /dev/null &

    ssh -x asuncion "cd $unixdir; gmake LINUXAMD64 EXPATDYNAMIC=1 EXPATINC=-I/usr/include EXPATLIB=-L/usr/lib64 EXPATLDFLAGS=-lexpat HDFDYNAMIC=1 HDFINC='$HDFINC' HDFLIB=$HDFLIB/lib_LINUXAMD64 HDFLDFLAGS='$HDFLDFLAGS' NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUXAMD64 SQLITEDYNAMIC=1 SQLITEINC=$SQLITEINC SQLITELIB=$SQLITELIB/lib_LINUXAMD64 SQLITELDFLAGS=-lsqlite3 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXAMD64 >& log.LINUXAMD64.$DATE " < /dev/null &

    ssh -x cancun "cd $unixdir; gmake SOLARISX86_64 SOLARISX86_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARISX86_64 >& log.SOLARISX86_64.$DATE" < /dev/null &

    wait;
    echo ""
exit

    ## Win32 include/link paths
    setenv windir /cygdrive/j/plugins
    setenv TCLINC -IJ:/vmd/lib/tcl/include
    setenv TCLLIB /LIBPATH:J:/vmd/lib/tcl
    ssh -1 -x administrator@malta "cd $windir; make WIN32 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_WIN32 >& log.WIN32.$DATE" < /dev/null &

    ## Win64 include/link paths
    setenv windir /cygdrive/j/plugins
    setenv TCLINC -IJ:/vmd/lib/tcl/include
    setenv TCLLIB /LIBPATH:J:/vmd/lib/tcl
#    ssh -1 -x Administrator@honolulu "cd $windir; make WIN64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_WIN64 >& log.WIN64.$DATE" < /dev/null &
#    ssh -1 -x Administrator@honolulu "cd $windir; make WIN64 >& log.WIN64.$DATE" < /dev/null &

    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo ""
    echo "Plugin builds done..."
    breaksw;

  ## proteus/toledo CUDA test boxes
  case proteus*:
  case photon*:
    echo "Using build settings for TB network..."
    setenv NETCDFINC -I/Projects/vmd/vmd/lib/netcdf/include
    setenv NETCDFLIB -L/Projects/vmd/vmd/lib/netcdf
    setenv TCLINC -I/Projects/vmd/vmd/lib/tcl/include
    setenv TCLLIB -L/Projects/vmd/vmd/lib/tcl
    gmake LINUX NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUX TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUX
    echo "Plugin builds done..."
    breaksw;
 
  ## NCSA 'Cobalt' SGI Altix 
  case co-login*:
    echo "Using build settings for NCSA SGI Altix..."
    setenv NETCDFINC -I/home/ac/stonej/vmd/lib/netcdf/include
    setenv NETCDFLIB -L/home/ac/stonej/vmd/lib/netcdf
    setenv TCLINC -I/home/ac/stonej/vmd/lib/tcl/include
    setenv TCLLIB -L/home/ac/stonej/vmd/lib/tcl
    cd $unixdir; gmake LINUXIA64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUXIA64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXIA64 >& log.LINUXIA64.$DATE  < /dev/null & 
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo ""
    echo "Plugin builds done..."
    breaksw;

 
  ## Photon (John's E4500)
  case photon*:
    echo "Using build settings for Photon..."
    setenv NETCDFINC -I/home/johns/vmd/lib/netcdf/include
    setenv NETCDFLIB -L/home/johns/vmd/lib/netcdf
    setenv TCLINC -I/home/johns/vmd/lib/tcl/include
    setenv TCLLIB -L/home/johns/vmd/lib/tcl
#    ssh -x photon "cd $unixdir; gmake SOLARIS2_64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_SOLARIS2_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARIS2_64 >& log.SOLARIS2_64.$DATE" < /dev/null &
    cd $unixdir; gmake SOLARIS2_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARIS2_64
 >& log.SOLARIS2_64.$DATE  < /dev/null &                                        
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;


  ###
  ### XXXNEWPLATFORM
  ###
  default:
    echo "Unrecognized host system, add your own switch statement to customize"
    echo "for your build environment.  Edit build.csh and change the variables"
    echo "in the section marked XXXNEWPLATFORM."
    # setenv TCLINC -I/your/tcl/include/directory
    # setenv TCLLIB -L/your/tcl/library/directory
    # cd $unixdir; gmake LINUX TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUX >& log.LINUX.$DATE  < /dev/null &
    # echo "Waiting for all plugin make jobs to complete..."
    # wait;
    # echo ""
    # echo "Plugin builds done..."
    breaksw;
endsw



