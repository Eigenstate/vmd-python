# Makefile for PME potential library

.SILENT:

.SUFFIXES: 

AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/pmepot1.0
SRCDIR=src
INCDIR=-Isrc
VPATH = src ${ARCHDIR}

##
## Only build pmepot if we have a Tcl library 
##
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
TARGETS = ${ARCHDIR} ${ARCHDIR}/libpmepot.so
endif
endif
endif


#
# Rules
#
bins:
win32bins:
dynlibs: ${TARGETS}
staticlibs: 
win32staticlibs: 

distrib:
	for localname in `find ../compile -name libpmepot.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		cp src/pkgIndex.tcl $$dir ; \
		cp src/pmepot.tcl $$dir ; \
		cp src/pmepot_gui.tcl $$dir ; \
	done

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

PMEPOTOBJS = ${ARCHDIR}/pmepot.o \
	${ARCHDIR}/gridcharges.o \
	${ARCHDIR}/recipspace.o \
	${ARCHDIR}/dxfile.o \
	${ARCHDIR}/pub3dfft.o

${ARCHDIR}/libpmepot.so : ${PMEPOTOBJS} ${ARCHDIR}/tcl_pmepot.o
	if [ -n "${TCLSHLD}" ]; \
	then ${TCLSHLD} $(LOPTO)$@ ${PMEPOTOBJS} ${ARCHDIR}/tcl_pmepot.o ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	else ${SHLD} $(LOPTO)$@ ${PMEPOTOBJS} ${ARCHDIR}/tcl_pmepot.o ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	fi

${ARCHDIR}/tcl_pmepot.o: tcl_pmepot.c pmepot.h
	${CC} ${CCFLAGS} ${TCLINC} ${INCDIR} -D_${ARCH} -DPMEPOTTCLDLL_EXPORTS -c $< $(COPTO)$@

${ARCHDIR}/pmepot.o: pmepot.c pmepot.h pub3dfft.h
	${CC} ${CCFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/gridcharges.o: gridcharges.c 
	${CC} ${CCFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/recipspace.o: recipspace.c
	${CC} ${CCFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/dxfile.o: dxfile.c 
	${CC} ${CCFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/pub3dfft.o: pub3dfft.c pub3dfft.h
	${CC} ${CCFLAGS} ${INCDIR} -c $< $(COPTO)$@

#${ARCHDIR}/%.o: %.c 
#	${CC} ${CCFLAGS} ${INCDIR} -c $< $(COPTO)$@

