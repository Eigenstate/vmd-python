# Makefile for clustalw

.SILENT:

.SUFFIXES: 

TCLFILES = clustalw.tcl clustalw_help pkgIndex.tcl
PKGVERSION = 1.1
PKGNAME = clustalw


AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/${PKGNAME}${PKGVERSION}
SRCDIR=src
INCDIR=-Isrc
VPATH = src ${ARCHDIR}

#
# Rules
#
bins: ${ARCHDIR} ${ARCHDIR}/clustalw
win32bins: ${ARCHDIR} ${ARCHDIR}/clustalw.exe
dynlibs:
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying ${PKGNAME} $(PKGVERSION) files to"
	for localname in `find ../compile -name clustalw -o -name clustalw.exe` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		echo "    $$dir"; \
		cp $$localname $$dir; \
        cp $(TCLFILES) $$dir; \
	done
    
${ARCHDIR}:
	mkdir -p ${ARCHDIR}

OBJS = ${ARCHDIR}/alnscore.o \
	${ARCHDIR}/amenu.o \
	${ARCHDIR}/calcgapcoeff.o \
	${ARCHDIR}/calcprf1.o \
	${ARCHDIR}/calcprf2.o \
	${ARCHDIR}/calctree.o \
	${ARCHDIR}/clustalw.o \
	${ARCHDIR}/gcgcheck.o \
	${ARCHDIR}/interface.o \
	${ARCHDIR}/malign.o \
	${ARCHDIR}/pairalign.o \
	${ARCHDIR}/prfalign.o \
	${ARCHDIR}/random.o \
	${ARCHDIR}/readmat.o \
	${ARCHDIR}/sequence.o \
	${ARCHDIR}/showpair.o \
	${ARCHDIR}/trees.o \
	${ARCHDIR}/util.o


# clustalw
${ARCHDIR}/clustalw.exe: ${OBJS}
	LINK ${OBJS} /OUT:${ARCHDIR}/clustalw.exe
${ARCHDIR}/clustalw: ${OBJS}
	${CC} ${CCFLAGS} ${OBJS} -o ${ARCHDIR}/clustalw -lm
    
    
# Objects.
${ARCHDIR}/alnscore.o: ${SRCDIR}/alnscore.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/alnscore.c $(COPTO)${ARCHDIR}/alnscore.o
${ARCHDIR}/amenu.o: ${SRCDIR}/amenu.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/amenu.c $(COPTO)${ARCHDIR}/amenu.o
${ARCHDIR}/calcgapcoeff.o: ${SRCDIR}/calcgapcoeff.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/calcgapcoeff.c $(COPTO)${ARCHDIR}/calcgapcoeff.o
${ARCHDIR}/calcprf1.o: ${SRCDIR}/calcprf1.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/calcprf1.c $(COPTO)${ARCHDIR}/calcprf1.o
${ARCHDIR}/calcprf2.o: ${SRCDIR}/calcprf2.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/calcprf2.c $(COPTO)${ARCHDIR}/calcprf2.o
${ARCHDIR}/calctree.o: ${SRCDIR}/calctree.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/calctree.c $(COPTO)${ARCHDIR}/calctree.o
${ARCHDIR}/clustalw.o: ${SRCDIR}/clustalw.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/clustalw.c $(COPTO)${ARCHDIR}/clustalw.o
${ARCHDIR}/gcgcheck.o: ${SRCDIR}/gcgcheck.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/gcgcheck.c $(COPTO)${ARCHDIR}/gcgcheck.o
${ARCHDIR}/interface.o: ${SRCDIR}/interface.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/interface.c $(COPTO)${ARCHDIR}/interface.o
${ARCHDIR}/malign.o: ${SRCDIR}/malign.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/malign.c $(COPTO)${ARCHDIR}/malign.o
${ARCHDIR}/pairalign.o: ${SRCDIR}/pairalign.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/pairalign.c $(COPTO)${ARCHDIR}/pairalign.o
${ARCHDIR}/prfalign.o: ${SRCDIR}/prfalign.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/prfalign.c $(COPTO)${ARCHDIR}/prfalign.o
${ARCHDIR}/random.o: ${SRCDIR}/random.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/random.c $(COPTO)${ARCHDIR}/random.o
${ARCHDIR}/readmat.o: ${SRCDIR}/readmat.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/readmat.c $(COPTO)${ARCHDIR}/readmat.o
${ARCHDIR}/sequence.o: ${SRCDIR}/sequence.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/sequence.c $(COPTO)${ARCHDIR}/sequence.o
${ARCHDIR}/showpair.o: ${SRCDIR}/showpair.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/showpair.c $(COPTO)${ARCHDIR}/showpair.o
${ARCHDIR}/trees.o: ${SRCDIR}/trees.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/trees.c $(COPTO)${ARCHDIR}/trees.o
${ARCHDIR}/util.o: ${SRCDIR}/util.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/util.c $(COPTO)${ARCHDIR}/util.o

