# Makefile for hesstrans dso library

.SUFFIXES: 

AR= ar
ARFLAGS = cr
RANLIB = ranlib

VERSION=1.0

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/hesstrans$(VERSION)
SRCDIR=src
INCDIR=-Isrc
#CXXFLAGS = -g $(CXXFLAGS)
CXXFLAGS += -g 
VPATH = src ${ARCHDIR}

##
## Disable building hesstrans plugin on platforms that it can't be ported to
## XXX ideally we'll fix the code, but until then...
##
ifneq ($(ARCH),BLUEWATERS)
ifneq ($(ARCH),LINUXIA64)
ifneq ($(ARCH),WIN64)
##
## Only build hesstrans if we have a Tcl library 
##
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
TARGETS = ${ARCHDIR} ${ARCHDIR}/hesstrans.so
endif
endif
endif

endif
endif
endif

bins:
win32bins:
dynlibs: ${TARGETS}
staticlibs:
win32staticlibs:

distrib:
	for localname in `find ../compile -name hesstrans.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		cp pkgIndex.tcl $$dir ; \
	done

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

HESSTRANSOBJS=${ARCHDIR}/newmat1.o \
	${ARCHDIR}/newmat2.o \
	${ARCHDIR}/newmat3.o \
	${ARCHDIR}/newmat4.o \
	${ARCHDIR}/newmat5.o \
	${ARCHDIR}/newmat6.o \
	${ARCHDIR}/newmat7.o \
	${ARCHDIR}/newmat8.o \
	${ARCHDIR}/newmat9.o \
	${ARCHDIR}/newmatex.o \
	${ARCHDIR}/bandmat.o \
	${ARCHDIR}/submat.o \
	${ARCHDIR}/myexcept.o \
	${ARCHDIR}/cholesky.o \
	${ARCHDIR}/evalue.o \
	${ARCHDIR}/fft.o \
	${ARCHDIR}/hholder.o \
	${ARCHDIR}/jacobi.o \
	${ARCHDIR}/newfft.o \
	${ARCHDIR}/sort.o \
	${ARCHDIR}/svd.o \
	${ARCHDIR}/newmatrm.o \
	${ARCHDIR}/hesstrans.o \
	${ARCHDIR}/tcl_hesstrans.o


${ARCHDIR}/hesstrans.so : ${HESSTRANSOBJS}
	if [ -n "${TCLSHLD}" ]; \
	then ${TCLSHLD} $(LOPTO)$@ ${HESSTRANSOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	else ${SHLD} $(LOPTO)$@ ${HESSTRANSOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	fi


${ARCHDIR}/tcl_hesstrans.o: tcl_hesstrans.C hesstrans.h
	${CXX} ${CXXFLAGS} -g ${TCLINC} ${INCDIR} -D_${ARCH} -DHESSTRANSTCLDLL_EXPORTS -c $< $(COPTO)$@

${ARCHDIR}/hesstrans.o:	hesstrans.C hesstrans.h newmatap.h newmatio.h
	${CXX} ${CXXFLAGS} -g ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat1.o: newmat1.C newmat.h include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat2.o: newmat2.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat3.o: newmat3.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat4.o: newmat4.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat5.o: newmat5.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat6.o: newmat6.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat7.o: newmat7.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat8.o: newmat8.C include.h newmat.h newmatrc.h precisio.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmat9.o: newmat9.C include.h newmat.h newmatio.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmatex.o: newmatex.C include.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/bandmat.o: bandmat.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/submat.o: submat.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/myexcept.o: myexcept.C include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/cholesky.o: cholesky.C include.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/evalue.o: evalue.C include.h newmatap.h newmatrm.h precisio.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/fft.o: fft.C include.h newmatap.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/hholder.o: hholder.C include.h newmatap.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/jacobi.o: jacobi.C include.h newmatap.h precisio.h newmatrm.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newfft.o: newfft.C newmatap.h newmat.h include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/sort.o: sort.C include.h newmatap.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/svd.o: svd.C include.h newmatap.h newmatrm.h precisio.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmatrm.o: newmatrm.C newmat.h newmatrm.h include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/newmatnl.o: newmatnl.C newmatap.h newmatnl.h newmat.h include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@

${ARCHDIR}/solution.o: solution.C include.h boolean.h myexcept.h solution.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c $< $(COPTO)$@


