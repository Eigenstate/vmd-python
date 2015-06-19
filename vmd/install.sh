#!/bin/bash
set -e

TARGET="$1"
VMDDIR="$2"
ANACONDIR="$3"
INSTDIR="$4"
vmd_src="$PWD"

#TODO: Auto-detect this
export LIBRARY_PATH="$ANACONDIR/lib:$LIBRARY_PATH"

# Set needed variables
echo "Setting environment variables"
export VMDINSTALLNAME="vmd-1.9.2-python"
export VMDINSTALLBINDIR="$VMDDIR"
export VMDINSTALLLIBRARYDIR="$VMDDIR"
export PLUGINDIR="$VMDDIR/plugins"

export NETCDFLIB="-L$ANACONDDIR/lib"
export NETCDFINC="-I$ANACONDIR/include"
export NETCDFLDFLAGS="-lnetcdf"

export TCLLIB="-L$ANACONDIR/lib"
export TCLINC="-I$ANACONDIR/include"
export TCLLDFLAGS="-ltcl"

export SQLITELIB="-L$ANACONDIR/lib"
export SQLITEINC="-I$ANACONDIR/include"
export SQLITELDFLAGS="-lsqlite3"

export EXPATLIB="-L$ANACONDIR/lib"
export EXPATINC="-I$ANACONDIR/include"
export EXPATLDFLAGS="-lexpat"

export NUMPY_LIBRARY_DIR="$ANACONDIR/lib/python2.7/site-packages/numpy/core/lib"
export NUMPY_INCLUDE_DIR="$ANACONDIR/lib/python2.7/site-packages/numpy/core/include"

export TCL_LIBRARY_DIR="$ANACONDIR/lib"
export TCL_INCLUDE_DIR="$ANACONDIR/include"

export PYTHON_LIBRARY_DIR="$ANACONDIR/lib/python2.7"
export PYTHON_INCLUDE_DIR="$ANACONDIR/include/python2.7"

export VMDEXTRALIBS="$SQLITELDFLAGS $EXPATLDFLAGS"

# Use python 2.7, in case this is an old configure
#sed -i 's/-lpython2.5/-lpython2.7/' "$vmd_src/vmd/configure"

# Compile the plugins
echo "Compiling plugins to $PLUGINDIR"
mkdir -p $PLUGINDIR
cd $vmd_src/plugins
gmake $TARGET
gmake distrib

echo "Linking $PLUGINDIR -> $vmd_src/vmd/plugins"
rm -rf $vmd_src/vmd/plugins
ln -s $PLUGINDIR $vmd_src/vmd/plugins

# Set the configure options
echo "$TARGET PTHREADS COLVARS NETCDF TCL PYTHON NUMPY SHARED NOSILENT" > "$vmd_src/vmd/configure.options"

# Change the source code to include our VMDDIR
sed "s@DEFAULT@$INSTDIR@" < $vmd_src/vmd/src/TclTextInterp.default > $vmd_src/vmd/src/TclTextInterp.C

# Compile the main library
cd $vmd_src/vmd
$vmd_src/vmd/configure
cd $vmd_src/vmd/src
make veryclean
make vmd.so
make install

