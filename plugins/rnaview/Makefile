# Makefile for rnaview

.SILENT:

.SUFFIXES:

TCLFILES = rnaview.tcl pkgIndex.tcl

PKGNAME = rnaview
PKGVERSION = 1.1

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

bins: ${ARCHDIR} ${ARCHDIR}/rnaview
win32bins:
# enable builds of rnaview on Windows.
win32bins: ${ARCHDIR} ${ARCHDIR}/rnaview.exe
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	echo "Copying ${PKGNAME} $(PKGVERSION) files to"
	for localname in `find ../compile -name rnaview -o -name rnaview.exe` ; do \
	  	pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
                dir=`dirname $(PLUGINDIR)/$$pluginname`; \
                mkdir -p $$dir; \
                echo "    $$dir"; \
                cp $$localname $$dir; \
		cp $(TCLFILES) $$dir; \
		cp -r BASEPARS $$dir; \
        done

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

OBJS = ${ARCHDIR}/analyze.o \
       ${ARCHDIR}/fpair.o \
       ${ARCHDIR}/fpair_sub.o \
       ${ARCHDIR}/multiple.o \
       ${ARCHDIR}/nrutil.o \
       ${ARCHDIR}/pair_type.o \
       ${ARCHDIR}/pattern.o \
       ${ARCHDIR}/ps-xy.o \
       ${ARCHDIR}/ps-xy-sub.o \
       ${ARCHDIR}/rnaview.o \
       ${ARCHDIR}/rnaxml-new.o \
       ${ARCHDIR}/statistics.o \
       ${ARCHDIR}/vrml.o \
       ${ARCHDIR}/xml2ps.o 

# rnaview
${ARCHDIR}/rnaview.exe: ${OBJS}
	LINK ${OBJS} /OUT:${ARCHDIR}/rnaview.exe
${ARCHDIR}/rnaview: ${OBJS}
	${CC} ${CCFLAGS} ${OBJS} -o ${ARCHDIR}/rnaview -lm

${ARCHDIR}/analyze.o: ${SRCDIR}/analyze.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/analyze.c $(COPTO)${ARCHDIR}/analyze.o
${ARCHDIR}/fpair.o: ${SRCDIR}/fpair.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/fpair.c $(COPTO)${ARCHDIR}/fpair.o
${ARCHDIR}/fpair_sub.o: ${SRCDIR}/fpair_sub.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/fpair_sub.c $(COPTO)${ARCHDIR}/fpair_sub.o
${ARCHDIR}/multiple.o: ${SRCDIR}/multiple.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/multiple.c $(COPTO)${ARCHDIR}/multiple.o
${ARCHDIR}/nrutil.o: ${SRCDIR}/nrutil.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/nrutil.c $(COPTO)${ARCHDIR}/nrutil.o
${ARCHDIR}/pair_type.o: ${SRCDIR}/pair_type.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/pair_type.c $(COPTO)${ARCHDIR}/pair_type.o
${ARCHDIR}/pattern.o: ${SRCDIR}/pattern.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/pattern.c $(COPTO)${ARCHDIR}/pattern.o
${ARCHDIR}/ps-xy.o: ${SRCDIR}/ps-xy.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/ps-xy.c $(COPTO)${ARCHDIR}/ps-xy.o
${ARCHDIR}/ps-xy-sub.o: ${SRCDIR}/ps-xy-sub.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/ps-xy-sub.c $(COPTO)${ARCHDIR}/ps-xy-sub.o
${ARCHDIR}/rnaview.o: ${SRCDIR}/rnaview.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/rnaview.c $(COPTO)${ARCHDIR}/rnaview.o
${ARCHDIR}/rnaxml-new.o: ${SRCDIR}/rnaxml-new.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/rnaxml-new.c $(COPTO)${ARCHDIR}/rnaxml-new.o
${ARCHDIR}/statistics.o: ${SRCDIR}/statistics.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/statistics.c $(COPTO)${ARCHDIR}/statistics.o
${ARCHDIR}/vrml.o: ${SRCDIR}/vrml.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/vrml.c $(COPTO)${ARCHDIR}/vrml.o
${ARCHDIR}/xml2ps.o: ${SRCDIR}/xml2ps.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/xml2ps.c $(COPTO)${ARCHDIR}/xml2ps.o

