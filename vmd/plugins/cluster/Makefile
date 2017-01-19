# Makefile for psfgen

.SILENT:

.SUFFIXES: 

TCLFILES = cluster.tcl pkgIndex.tcl
PKGVERSION = 1.2

AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/cluster${PKGVERSION}
SRCDIR=src
INCDIR=-Isrc
VPATH = src ${ARCHDIR}

#
# Rules
#
bins: ${ARCHDIR} ${ARCHDIR}/cluster
win32bins: ${ARCHDIR} ${ARCHDIR}/cluster.exe
dynlibs:
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying cluster $(PKGVERSION) files to"
	for localname in `find ../compile -name cluster -o -name cluster.exe` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		echo "    $$dir"; \
		cp $$localname $$dir; \
        cp $(TCLFILES) $$dir; \
	done
    
${ARCHDIR}:
	mkdir -p ${ARCHDIR}

OBJS = ${ARCHDIR}/cluster.o

# Win32/Win64 compiles
${ARCHDIR}/cluster.exe : ${OBJS}
	LINK ${OBJS} /OUT:${ARCHDIR}/cluster.exe

# all other platforms
${ARCHDIR}/cluster : ${OBJS}
	${CXX} ${CXXFLAGS} ${OBJS} -o ${ARCHDIR}/cluster -lm

${ARCHDIR}/cluster.o: ${SRCDIR}/cluster.c
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/cluster.c $(COPTO)${ARCHDIR}/cluster.o

