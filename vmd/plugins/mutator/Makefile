##
## Mutator plugin (and alchemify shared library)
##
SRCDIR=src
INCDIR=-Isrc 
COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/mutator1.5
VPATH = src ${ARCHDIR}

##
## Only build alchemify if we have a Tcl library 
##
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
TARGETS = ${ARCHDIR} ${ARCHDIR}/libalchemify.so
endif
endif
endif

bins:
win32bins:
dynlibs: ${TARGETS}
staticlibs:
win32staticlibs:

distrib:
	for localname in `find ../compile -name libalchemify.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		cp mutator.tcl mutator_gui.tcl pkgIndex.tcl $$dir ; \
	done
	
${ARCHDIR}:
	mkdir -p ${ARCHDIR}

ALCHEMIFYOBJS=${ARCHDIR}/libalchemify.o  \
	${ARCHDIR}/alchemify_tcl_wrap.o \
	${ARCHDIR}/alchemify_tcl.o

${ARCHDIR}/libalchemify.so : ${ALCHEMIFYOBJS}
	if [ -n "${TCLSHLD}" ]; \
	then ${TCLSHLD} $(LOPTO)$@ ${ALCHEMIFYOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	else ${SHLD} $(LOPTO)$@ ${ALCHEMIFYOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	fi


${ARCHDIR}/alchemify_tcl.o : alchemify_tcl.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/alchemify_tcl.c $(COPTO)${ARCHDIR}/alchemify_tcl.o

${ARCHDIR}/libalchemify.o : libalchemify.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/libalchemify.c $(COPTO)${ARCHDIR}/libalchemify.o

${ARCHDIR}/alchemify_tcl_wrap.c : alchemify_tcl.i
	swig -tcl8 -o alchemify_tcl_wrap.c alchemify_tcl.i

${ARCHDIR}/alchemify_tcl_wrap.o : alchemify_tcl_wrap.c
	${CC} ${CCFLAGS} ${TCLINC} ${INCDIR} -c ${SRCDIR}/alchemify_tcl_wrap.c $(COPTO)${ARCHDIR}/alchemify_tcl_wrap.o


