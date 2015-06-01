# Makefile for cionize

#.SILENT:

.SUFFIXES: 

AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/bin/cionize1.0
TCLDIR=${COMPILEDIR}/lib_${ARCH}/tcl/cionize1.0
SRCDIR=.
INCDIR=-I. -I../include -I${COMPILEDIR}/lib_${ARCH}/molfile

VPATH = ${ARCHDIR}

#
# Rules
#

bins: ${ARCHDIR} ${ARCHDIR}/cionize
win32bins:
# enable builds of cionize on Windows.
# win32bins: ${ARCHDIR} ${ARCHDIR}/cionize.exe
dynlibs: 
staticlibs: 
win32staticlibs: 

distrib:
	for localname in `find ../compile -name cionize -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		tclplugname=`echo $$pluginname | sed s/\\\/bin\\\//\\\/tcl\\\// `; \
		tcldir=`dirname $(PLUGINDIR)/$$tclplugname`; \
		mkdir -p $$tcldir; \
		cp cionize.tcl pkgIndex.tcl $$tcldir; \
	done

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

IONIZEOBJS = ${ARCHDIR}/hash.o ${ARCHDIR}/threads.o ${ARCHDIR}/util.o ${ARCHDIR}/getplugins.o ${ARCHDIR}/energythr.o ${ARCHDIR}/cionize_gridio.o ${ARCHDIR}/cionize_molfileio.o ${ARCHDIR}/cionize_internals.o ${ARCHDIR}/cionize_userio.o
MGPOTOBJS = ${ARCHDIR}/mgpot.o ${ARCHDIR}/mgpot_error.o ${ARCHDIR}/mgpot_lattice.o ${ARCHDIR}/mgpot_setup.o ${ARCHDIR}/mgpot_shortrng.o ${ARCHDIR}/mgpot_longrng.o ${ARCHDIR}/mgpot_cubic.o ${ARCHDIR}/mgpot_quintic1.o
#MGPOTFLAGS = -DDEBUG

# Win32/Win64 compiles
${ARCHDIR}/cionize.exe : ${IONIZEOBJS} ${MGPOTOBJS} ${ARCHDIR}/cionize.o ${COMPILEDIR}/lib_${ARCH}/molfile/libmolfile_plugin.lib
	LINK ${IONIZEOBJS} ${MGPOTOBJS} /OUT:${ARCHDIR}/cionize.exe /LIBPATH:${COMPILEDIR}/lib_${ARCH}/molfile libmolfile_plugin.lib ${TCLLIB} ${TCLLDFLAGS} ${NETCDFLIB} ${NETCDFLDFLAGS} ${EXPATLIB} ${EXPATLDFLAGS} ${SQLITELIB} ${SQLITELDFLAGS}

# all other platforms
${ARCHDIR}/cionize: ${IONIZEOBJS} ${MGPOTOBJS} ${ARCHDIR}/cionize.o ${COMPILEDIR}/lib_${ARCH}/molfile/libmolfile_plugin.a
	${CXX} ${CXXFLAGS} ${ARCHDIR}/cionize.o ${IONIZEOBJS} ${MGPOTOBJS} -o ${ARCHDIR}/cionize -L${COMPILEDIR}/lib_${ARCH}/molfile -lmolfile_plugin ${TCLLIB} ${TCLLDFLAGS} ${NETCDFLIB} ${NETCDFLDFLAGS} ${EXPATLIB} ${EXPATLDFLAGS} ${SQLITELIB} ${SQLITELDFLAGS} -lm

${ARCHDIR}/hash.o: hash.h hash.c
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/hash.c $(COPTO)${ARCHDIR}/hash.o

${ARCHDIR}/threads.o: threads.h threads.c
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/threads.c $(COPTO)${ARCHDIR}/threads.o

${ARCHDIR}/util.o: util.h util.c
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/util.c $(COPTO)${ARCHDIR}/util.o

${ARCHDIR}/getplugins.o: getplugins.h getplugins.c
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/getplugins.c $(COPTO)${ARCHDIR}/getplugins.o
	
${ARCHDIR}/energythr.o: energythr.h energythr.c ${ARCHDIR}/util.o ${ARCHDIR}/threads.o
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/energythr.c $(COPTO)${ARCHDIR}/energythr.o

${ARCHDIR}/cionize.o: hash.h cionize.c ${ARCHDIR}/energythr.o ${ARCHDIR}/threads.o ${ARCHDIR}/util.o mgpot.h ${ARCHDIR}/mgpot.o
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/cionize.c $(COPTO)${ARCHDIR}/cionize.o

${ARCHDIR}/mgpot.o: mgpot.h mgpot.c mgpot_defn.h mgpot_lattice.h mgpot_split.h mgpot_error.h util.h
	${CC}  ${CCFLAGS} ${MGPOTFLAGS} ${INCDIR} -c ${SRCDIR}/mgpot.c $(COPTO)${ARCHDIR}/mgpot.o

${ARCHDIR}/mgpot_error.o: mgpot_error.c mgpot_error.h
	${CC}  ${CCFLAGS} ${MGPOTFLAGS} ${INCDIR} -c ${SRCDIR}/mgpot_error.c $(COPTO)${ARCHDIR}/mgpot_error.o

${ARCHDIR}/mgpot_lattice.o: mgpot_lattice.c mgpot_lattice.h mgpot_error.h
	${CC}  ${CCFLAGS} ${MGPOTFLAGS} ${INCDIR} -c ${SRCDIR}/mgpot_lattice.c $(COPTO)${ARCHDIR}/mgpot_lattice.o

${ARCHDIR}/cionize_gridio.o: cionize_gridio.h cionize_gridio.c
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/cionize_gridio.c $(COPTO)${ARCHDIR}/cionize_gridio.o

${ARCHDIR}/cionize_molfileio.o: cionize_molfileio.h cionize_molfileio.c
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/cionize_molfileio.c $(COPTO)${ARCHDIR}/cionize_molfileio.o

${ARCHDIR}/cionize_userio.o: cionize_userio.h cionize_userio.c
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/cionize_userio.c $(COPTO)${ARCHDIR}/cionize_userio.o

${ARCHDIR}/cionize_internals.o: cionize_internals.h cionize_internals.c
	${CC}  ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/cionize_internals.c $(COPTO)${ARCHDIR}/cionize_internals.o

${ARCHDIR}/mgpot_setup.o: mgpot_setup.c mgpot_defn.h mgpot_lattice.h mgpot_split.h mgpot_error.h
	${CC}  ${CCFLAGS} ${MGPOTFLAGS} ${INCDIR} -c ${SRCDIR}/mgpot_setup.c $(COPTO)${ARCHDIR}/mgpot_setup.o

${ARCHDIR}/mgpot_shortrng.o: mgpot_shortrng.c mgpot_defn.h mgpot_lattice.h mgpot_split.h mgpot_error.h
	${CC}  ${CCFLAGS} ${MGPOTFLAGS} ${INCDIR} -c ${SRCDIR}/mgpot_shortrng.c $(COPTO)${ARCHDIR}/mgpot_shortrng.o

${ARCHDIR}/mgpot_longrng.o: mgpot_longrng.c mgpot_defn.h mgpot_lattice.h mgpot_split.h mgpot_error.h
	${CC}  ${CCFLAGS} ${MGPOTFLAGS} ${INCDIR} -c ${SRCDIR}/mgpot_longrng.c $(COPTO)${ARCHDIR}/mgpot_longrng.o

${ARCHDIR}/mgpot_cubic.o: mgpot_cubic.c mgpot_defn.h mgpot_lattice.h mgpot_split.h mgpot_error.h
	${CC}  ${CCFLAGS} ${MGPOTFLAGS} ${INCDIR} -c ${SRCDIR}/mgpot_cubic.c $(COPTO)${ARCHDIR}/mgpot_cubic.o

${ARCHDIR}/mgpot_quintic1.o: mgpot_quintic1.c mgpot_defn.h mgpot_lattice.h mgpot_split.h mgpot_error.h
	${CC}  ${CCFLAGS} ${MGPOTFLAGS} ${INCDIR} -c ${SRCDIR}/mgpot_quintic1.c $(COPTO)${ARCHDIR}/mgpot_quintic1.o
