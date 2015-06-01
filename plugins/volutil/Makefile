# Makefile for volutil

.SILENT:

.SUFFIXES:

PKGNAME = volutil
PKGVERSION = 1.3

AR = ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/${PKGNAME}${PKGVERSION}
SRCDIR=src
INCDIR=-Isrc -I../include -I${COMPILEDIR}/lib_${ARCH}/molfile

# GNU make 3.81 apparently doesn't like recursive variable definitions
# CXXFLAGS=${CXXFLAGS} -DDEBUG -g -Wall -pedantic
CXXFLAGS+= -DDEBUG -g -Wall -pedantic

VPATH = ${ARCHDIR}

#
# Rules
#

bins: ${ARCHDIR} ${ARCHDIR}/volutil
# enable builds of volutil on Windows.
win32bins: ${ARCHDIR} ${ARCHDIR}/volutil.exe
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	echo "Copying ${PKGNAME} ${PKGVERSION} files to"
	for localname in `find ../compile -name volutil -o -name volutil.exe` ; do \
                pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
                dir=`dirname $(PLUGINDIR)/$$pluginname`; \
                mkdir -p $$dir; \
		echo "    $$dir"; \
                cp $$localname $$dir; \
        cp volutil.tcl pkgIndex.tcl $$dir; \
        done

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

VOLUTILOBJS = ${ARCHDIR}/volmap.o \
	      ${ARCHDIR}/binary_ops.o \
	      ${ARCHDIR}/convolutions.o \
	      ${ARCHDIR}/unary_ops.o \
	      ${ARCHDIR}/voxels.o \
	      ${ARCHDIR}/ops.o \
	      ${ARCHDIR}/vec.o \
	      ${ARCHDIR}/hash.o \
	      ${ARCHDIR}/getplugins.o \
	      ${ARCHDIR}/molfile.o \
	      ${ARCHDIR}/params.o \
	      ${ARCHDIR}/main.o

# Win32/Win64 compiles
${ARCHDIR}/volutil.exe : ${VOLUTILOBJS} ${COMPILEDIR}/lib_${ARCH}/molfile/libmolfile_plugin.lib
	LINK ${VOLUTILOBJS} /OUT:${ARCHDIR}/volutil.exe /LIBPATH:${COMPILEDIR}/lib_${ARCH}/molfile libmolfile_plugin.lib ${TCLLIB} ${TCLLDFLAGS} ${NETCDFLIB} ${NETCDFLDFLAGS} ${EXPATLIB} ${EXPATLDFLAGS} ${SQLITELIB} ${SQLITELDFLAGS} wsock32.lib

# all other platforms
  
${ARCHDIR}/volutil: ${VOLUTILOBJS} ${COMPILEDIR}/lib_${ARCH}/molfile/libmolfile_plugin.a
	${CXX} ${CXXFLAGS} ${VOLUTILOBJS} -o ${ARCHDIR}/volutil -L${COMPILEDIR}/lib_${ARCH}/molfile -lmolfile_plugin ${TCLLIB} ${TCLLDFLAGS} ${NETCDFLIB} ${NETCDFLDFLAGS} ${EXPATLIB} ${EXPATLDFLAGS} ${SQLITELIB} ${SQLITELDFLAGS} -lm
     
${ARCHDIR}/volmap.o: ${SRCDIR}/volmap.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/volmap.C ${COPTO}${ARCHDIR}/volmap.o

${ARCHDIR}/binary_ops.o: ${SRCDIR}/binary_ops.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/binary_ops.C ${COPTO}${ARCHDIR}/binary_ops.o

${ARCHDIR}/convolutions.o: ${SRCDIR}/convolutions.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/convolutions.C ${COPTO}${ARCHDIR}/convolutions.o

${ARCHDIR}/unary_ops.o: ${SRCDIR}/unary_ops.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/unary_ops.C ${COPTO}${ARCHDIR}/unary_ops.o

${ARCHDIR}/voxels.o: ${SRCDIR}/voxels.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/voxels.C ${COPTO}${ARCHDIR}/voxels.o

${ARCHDIR}/ops.o: ${SRCDIR}/ops.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/ops.C ${COPTO}${ARCHDIR}/ops.o


${ARCHDIR}/vec.o: ${SRCDIR}/vec.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/vec.C ${COPTO}${ARCHDIR}/vec.o

${ARCHDIR}/hash.o: ${SRCDIR}/hash.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/hash.C ${COPTO}${ARCHDIR}/hash.o

${ARCHDIR}/getplugins.o: ${SRCDIR}/getplugins.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/getplugins.C ${COPTO}${ARCHDIR}/getplugins.o

${ARCHDIR}/molfile.o: ${SRCDIR}/molfile.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/molfile.C ${COPTO}${ARCHDIR}/molfile.o

${ARCHDIR}/params.o: ${SRCDIR}/params.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/params.C ${COPTO}${ARCHDIR}/params.o

${ARCHDIR}/main.o: ${SRCDIR}/main.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/main.C ${COPTO}${ARCHDIR}/main.o

