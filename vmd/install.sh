#!/bin/bash
set -e

TARGET="$1"
VMDDIR="$2"
ANACONDIR="$3"
vmd_src="$PWD"

#TODO: Auto-detect this
export LD_LIBRARY_PATH="$ANACONDIR/lib:$LD_LIBRARY_PATH"

# Set needed variables
echo "Setting environment variables"
export VMDINSTALLNAME="vmd-1.9.2-python"
export VMDINSTALLBINDIR="$VMDDIR"
export VMDINSTALLLIBRARYDIR="$VMDDIR"
export PLUGINDIR="$VMDDIR/plugins"

export NETCDFLIB="-L$ANACONDIR/lib"
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

# Clean up previous installation
if [[ -d $PLUGINDIR ]]; then
    echo "Deleting previous plugin directory $PLUGINDIR"
    rm -r $PLUGINDIR
fi

# Compile the plugins
echo "Compiling plugins to $PLUGINDIR"
mkdir -p $PLUGINDIR
cd $vmd_src/plugins
make $TARGET
make distrib

echo "Linking $PLUGINDIR -> $vmd_src/vmd_src/plugins"
ln -s $PLUGINDIR $vmd_src/vmd_src/plugins

# Set the configure options
echo "$TARGET PTHREADS COLVARS NETCDF TCL PYTHON NUMPY SHARED NOSILENT" > "$vmd_src/vmd_src/configure.options"

# Compile the main library
cd $vmd_src/vmd_src
$vmd_src/vmd_src/configure
cd $vmd_src/vmd_src/src
make veryclean
make vmd.so
make install

# Remove symlink so install doesn't freak out
rm $vmd_src/vmd_src/plugins

# Copy init.py file into build directory
#cp "$vmd_src/__init__.py" "$VMDDIR"

