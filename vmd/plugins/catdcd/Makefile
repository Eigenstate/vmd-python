# Makefile for catdcd

#.SILENT:

.SUFFIXES: 

AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/bin/catdcd5.2
SRCDIR=.
INCDIR=-I. -I../include -I${COMPILEDIR}/lib_${ARCH}/molfile

VPATH = ${ARCHDIR}

#
# Rules
#

bins: ${ARCHDIR} ${ARCHDIR}/catdcd
win32bins:
# enable builds of catdcd on Windows.
# win32bins: ${ARCHDIR} ${ARCHDIR}/catdcd.exe
dynlibs: 
staticlibs: 
win32staticlibs: 

distrib:
	for localname in `find ../compile -name catdcd -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
	done

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

CATDCDOBJS = ${ARCHDIR}/hash.o ${ARCHDIR}/catdcd.o

# Win32/Win64 compiles
${ARCHDIR}/catdcd.exe : ${CATDCDOBJS} ${COMPILEDIR}/lib_${ARCH}/molfile/libmolfile_plugin.lib
	LINK ${CATDCDOBJS} /OUT:${ARCHDIR}/catdcd.exe /LIBPATH:${COMPILEDIR}/lib_${ARCH}/molfile libmolfile_plugin.lib ${TCLLIB} ${TCLLDFLAGS} ${NETCDFLIB} ${NETCDFLDFLAGS} ${EXPATLIB} ${EXPATLDFLAGS} ${SQLITELIB} ${SQLITELDFLAGS} ${LDFLAGS}

# all other platforms
${ARCHDIR}/catdcd : ${CATDCDOBJS} ${COMPILEDIR}/lib_${ARCH}/molfile/libmolfile_plugin.a
	${CXX} ${CXXFLAGS} ${CATDCDOBJS} -o ${ARCHDIR}/catdcd -L${COMPILEDIR}/lib_${ARCH}/molfile -lmolfile_plugin ${TCLLIB} ${TCLLDFLAGS} ${NETCDFLIB} ${NETCDFLDFLAGS} ${EXPATLIB} ${EXPATLDFLAGS} ${SQLITELIB} ${SQLITELDFLAGS} ${LDFLAGS} -lm

${ARCHDIR}/hash.o: hash.h hash.c
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/hash.c $(COPTO)${ARCHDIR}/hash.o

${ARCHDIR}/catdcd.o: hash.h catdcd.c
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/catdcd.c $(COPTO)${ARCHDIR}/catdcd.o


