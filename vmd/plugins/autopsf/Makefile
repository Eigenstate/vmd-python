AR= ar
ARFLAGS = cr
RANLIB = ranlib

SRCDIR=src
INCDIR=-Isrc -I../molfile_plugin/src
COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/autopsf1.7
VPATH = src ${ARCHDIR}

##
## Only build psfcheck if we have a Tcl library 
##
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
TARGETS = ${ARCHDIR} ${ARCHDIR}/psfcheck.so
endif
endif
endif

bins:
win32bins:
dynlibs: ${TARGETS}
staticlibs:
win32staticlibs:

distrib:
	for localname in `find ../compile -name psfcheck.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		cp autopsf.tcl pkgIndex.tcl $$dir ; \
	done
	
${ARCHDIR}:
	mkdir -p ${ARCHDIR}

AUTOPSFOBJS=${ARCHDIR}/psfatom.o  \
	${ARCHDIR}/psfcheck.o  \
	${ARCHDIR}/psfreslist.o  \
	${ARCHDIR}/psfres.o \
	${ARCHDIR}/psftype.o \
	${ARCHDIR}/psftypelist.o \
	${ARCHDIR}/psfbond.o \
	${ARCHDIR}/psfatomlist.o \
	${ARCHDIR}/psfcheck_wrap.o

${ARCHDIR}/psfcheck.so : ${AUTOPSFOBJS}
	if [ -n "${TCLSHLD}" ]; \
	then ${TCLSHLD} $(LOPTO)$@ ${AUTOPSFOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	else ${SHLD} $(LOPTO)$@ ${AUTOPSFOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	fi

${ARCHDIR}/psfatom.o : psfatom.h psfatom.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psfatom.C $(COPTO)${ARCHDIR}/psfatom.o

${ARCHDIR}/psfres.o : psfres.h psfres.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psfres.C $(COPTO)${ARCHDIR}/psfres.o

${ARCHDIR}/psftype.o : psftype.h psftype.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psftype.C $(COPTO)${ARCHDIR}/psftype.o

${ARCHDIR}/psfreslist.o : psfreslist.h psfreslist.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psfreslist.C $(COPTO)${ARCHDIR}/psfreslist.o

${ARCHDIR}/psftypelist.o : psftypelist.h psftypelist.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psftypelist.C $(COPTO)${ARCHDIR}/psftypelist.o

${ARCHDIR}/psfbond.o : psfbond.h psfbond.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psfbond.C $(COPTO)${ARCHDIR}/psfbond.o

${ARCHDIR}/psfbondlist.o : psfbondlist.h psfbondlist.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psfbondlist.C $(COPTO)${ARCHDIR}/psfbondlist.o

${ARCHDIR}/psfatomlist.o : psfatomlist.h psfatomlist.C
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psfatomlist.C $(COPTO)${ARCHDIR}/psfatomlist.o

${ARCHDIR}/psfcheck.o : psfcheck.C psfcheck.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psfcheck.C $(COPTO)${ARCHDIR}/psfcheck.o

${ARCHDIR}/psfcheck_wrap.C : psfcheck.i
	swig -tcl8 -c++ -o psfcheck_wrap.C psfcheck.i

${ARCHDIR}/psfcheck_wrap.o : psfcheck_wrap.C
	${CXX} ${CXXFLAGS} ${TCLINC} ${INCDIR} -c ${SRCDIR}/psfcheck_wrap.C $(COPTO)${ARCHDIR}/psfcheck_wrap.o

