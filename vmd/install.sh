#!/bin/bash
set -e

TARGET="$1"
VMDDIR="$2"
ANACONDIR="$3"
vmd_src="$(dirname $0)"

# Set needed variables
echo "Setting environment variables"
export VMDINSTALLNAME="vmd-1.9.3-python"
export VMDINSTALLBINDIR="$VMDDIR"
export VMDINSTALLLIBRARYDIR="$VMDDIR"
export PLUGINDIR="$VMDDIR/plugins"

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
rm -f $vmd_src/vmd_src/plugins
ln -s $PLUGINDIR $vmd_src/vmd_src/plugins

# Set the configure options
echo "$TARGET PTHREADS COLVARS NETCDF TCL IMD PYTHON NUMPY SHARED OPENGLPBUFFER EGLPBUFFER NOSILENT $DEBUG" > "$vmd_src/vmd_src/configure.options"
if [[ "$TARGET" == *"64"* ]]; then
    echo " LP64" >> "$vmd_src/vmd_src/configure.options"
fi

# Build with gcc on osx
if [[ $TARGET == *"MACOSX"* ]]; then
    echo "Building with GCC on OSX"
    echo " GCC" >> "$vmd_src/vmd_src/configure.options"
fi

# Compile the main library
cd $vmd_src/vmd_src
$vmd_src/vmd_src/configure
cd $vmd_src/vmd_src/src
make veryclean
make libvmd.so
make install

# Clean up built files in src dir
cd $vmd_src/vmd_src/src
make veryclean
cd $vmd_src/plugins
make clean

# Remove symlink so install doesn't freak out
rm $vmd_src/vmd_src/plugins

# Copy init.py file into build directory
cp "$vmd_src/vmd_src/__init__.py" "$VMDDIR"

# Copy tests into build directory so they're accessible
cp -r "$vmd_src/../test" "$VMDDIR"

