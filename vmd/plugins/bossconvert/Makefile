# Makefile for bossconvert
TCLFILES = bossconvert.tcl pkgIndex.tcl

PKGNAME = bossconvert
PKGVERSION = 1.0

AR = ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/${PKGNAME}${PKGVERSION}
TCLDIR=${COMPILEDIR}/lib_${ARCH}/tcl/${PKGNAME}${PKGVERSION}
SRCDIR=src
INCDIR=-Isrc

VPATH = src ${ARCHDIR}

#
# Rules
# 

bins: ${ARCHDIR} ${ARCHDIR}/Topology
win32bins: ${ARCHDIR} ${ARCHDIR}/Topology.exe
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	echo "Copying ${PKGNAME} $(PKGVERSION) files to"
	for localname in `find ../compile -name Topology -o -name Topology.exe` ; do \
		  pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		  dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		  mkdir -p $$dir; \
		  echo "    $$dir"; \
		  cp $$localname $$dir; \
		  cp $(TCLFILES) $$dir; \
		  cp imprlist oplsaa.par oplsaa.sb $$dir; \
	done


${ARCHDIR}:
	 mkdir -p ${ARCHDIR}

OBJS = ${ARCHDIR}/Main.o \
	${ARCHDIR}/Parser.o \
	${ARCHDIR}/Worker.o \
	${ARCHDIR}/Atom.o \

# BossConvert Topology
${ARCHDIR}/Topology.exe: ${OBJS}
	LINK ${OBJS} /OUT:${ARCHDIR}/Topology.exe

${ARCHDIR}/Topology: ${OBJS}
	${CXX} ${CXXFLAGS} ${OBJS} -o ${ARCHDIR}/Topology -lm

${ARCHDIR}/Parser.o: ${SRCDIR}/Parser.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/Parser.cpp $(COPTO)${ARCHDIR}/Parser.o

${ARCHDIR}/Worker.o: ${SRCDIR}/Worker.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/Worker.cpp $(COPTO)${ARCHDIR}/Worker.o

${ARCHDIR}/Atom.o: ${SRCDIR}/Atom.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/Atom.cpp $(COPTO)${ARCHDIR}/Atom.o

${ARCHDIR}/Main.o: ${SRCDIR}/Main.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/Main.cpp $(COPTO)${ARCHDIR}/Main.o
