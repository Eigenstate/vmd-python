# Makefile for stamp

.SILENT:

.SUFFIXES: 

TCLFILES = stamp.tcl pkgIndex.tcl dssp.directories pdb.directories stamp.defaults
PKGVERSION = 1.2
PKGNAME = stamp


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
bins: ${ARCHDIR} ${ARCHDIR}/stamp
win32bins: ${ARCHDIR} ${ARCHDIR}/stamp.exe
dynlibs:
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying ${PKGNAME} $(PKGVERSION) files to"
	for localname in `find ../compile -name stamp -o -name stamp.exe` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		echo "    $$dir"; \
		cp $$localname $$dir; \
        cp $(TCLFILES) $$dir; \
	done
    
${ARCHDIR}:
	mkdir -p ${ARCHDIR}

OBJS = ${ARCHDIR}/a3to1.o \
	${ARCHDIR}/aliseq.o \
	${ARCHDIR}/ccprobcalc.o \
	${ARCHDIR}/closefile.o \
	${ARCHDIR}/count_domain.o \
	${ARCHDIR}/display_align.o \
	${ARCHDIR}/dosort.o \
	${ARCHDIR}/get_clust.o \
	${ARCHDIR}/getdomain.o \
	${ARCHDIR}/getdsspsum.o \
	${ARCHDIR}/getfile.o \
	${ARCHDIR}/getks.o \
	${ARCHDIR}/getsec.o \
	${ARCHDIR}/gjnoc.o \
	${ARCHDIR}/gjutil.o \
	${ARCHDIR}/idist.o \
	${ARCHDIR}/igetca.o \
	${ARCHDIR}/igetcadssp.o \
	${ARCHDIR}/ltou.o \
	${ARCHDIR}/makeafile.o \
	${ARCHDIR}/matfit.o \
	${ARCHDIR}/matinv.o \
	${ARCHDIR}/matmult.o \
	${ARCHDIR}/murzin_P.o \
	${ARCHDIR}/newoutput.o \
	${ARCHDIR}/openfile.o \
	${ARCHDIR}/pairfit.o \
	${ARCHDIR}/pairpath.o \
	${ARCHDIR}/pairwise.o \
	${ARCHDIR}/printmat.o \
	${ARCHDIR}/printmat2.o \
	${ARCHDIR}/probcalc.o \
	${ARCHDIR}/probplot.o \
	${ARCHDIR}/qkfit.o \
	${ARCHDIR}/readtree.o \
	${ARCHDIR}/reval.o \
	${ARCHDIR}/revmatmult.o \
	${ARCHDIR}/rossmann.o \
	${ARCHDIR}/roughfit.o \
	${ARCHDIR}/scan.o \
	${ARCHDIR}/sec_content.o \
	${ARCHDIR}/slow_scan.o \
	${ARCHDIR}/stamp.o \
	${ARCHDIR}/stampgetpars.o \
	${ARCHDIR}/sw7ccs.o \
	${ARCHDIR}/swstruc.o \
	${ARCHDIR}/testfile.o \
	${ARCHDIR}/treefit.o \
	${ARCHDIR}/treepath.o \
	${ARCHDIR}/treewise.o \
	${ARCHDIR}/update.o \
	${ARCHDIR}/utol.o



# stamp
${ARCHDIR}/stamp.exe: ${OBJS}
	LINK ${OBJS} /OUT:${ARCHDIR}/stamp.exe
${ARCHDIR}/stamp: ${OBJS}
	${CC} ${CCFLAGS} ${OBJS} -o ${ARCHDIR}/stamp -lm
    
    
# Objects.
${ARCHDIR}/a3to1.o: ${SRCDIR}/a3to1.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/a3to1.c $(COPTO)${ARCHDIR}/a3to1.o
${ARCHDIR}/aliseq.o: ${SRCDIR}/aliseq.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/aliseq.c $(COPTO)${ARCHDIR}/aliseq.o
${ARCHDIR}/ccprobcalc.o: ${SRCDIR}/ccprobcalc.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/ccprobcalc.c $(COPTO)${ARCHDIR}/ccprobcalc.o
${ARCHDIR}/closefile.o: ${SRCDIR}/closefile.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/closefile.c $(COPTO)${ARCHDIR}/closefile.o
${ARCHDIR}/count_domain.o: ${SRCDIR}/count_domain.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/count_domain.c $(COPTO)${ARCHDIR}/count_domain.o
${ARCHDIR}/display_align.o: ${SRCDIR}/display_align.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/display_align.c $(COPTO)${ARCHDIR}/display_align.o
${ARCHDIR}/dosort.o: ${SRCDIR}/dosort.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/dosort.c $(COPTO)${ARCHDIR}/dosort.o
${ARCHDIR}/get_clust.o: ${SRCDIR}/get_clust.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/get_clust.c $(COPTO)${ARCHDIR}/get_clust.o
${ARCHDIR}/getdomain.o: ${SRCDIR}/getdomain.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/getdomain.c $(COPTO)${ARCHDIR}/getdomain.o
${ARCHDIR}/getdsspsum.o: ${SRCDIR}/getdsspsum.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/getdsspsum.c $(COPTO)${ARCHDIR}/getdsspsum.o
${ARCHDIR}/getfile.o: ${SRCDIR}/getfile.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/getfile.c $(COPTO)${ARCHDIR}/getfile.o
${ARCHDIR}/getks.o: ${SRCDIR}/getks.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/getks.c $(COPTO)${ARCHDIR}/getks.o
${ARCHDIR}/getsec.o: ${SRCDIR}/getsec.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/getsec.c $(COPTO)${ARCHDIR}/getsec.o
${ARCHDIR}/gjnoc.o: ${SRCDIR}/gjnoc.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/gjnoc.c $(COPTO)${ARCHDIR}/gjnoc.o
${ARCHDIR}/gjutil.o: ${SRCDIR}/gjutil.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/gjutil.c $(COPTO)${ARCHDIR}/gjutil.o
${ARCHDIR}/idist.o: ${SRCDIR}/idist.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/idist.c $(COPTO)${ARCHDIR}/idist.o
${ARCHDIR}/igetca.o: ${SRCDIR}/igetca.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/igetca.c $(COPTO)${ARCHDIR}/igetca.o
${ARCHDIR}/igetcadssp.o: ${SRCDIR}/igetcadssp.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/igetcadssp.c $(COPTO)${ARCHDIR}/igetcadssp.o
${ARCHDIR}/ltou.o: ${SRCDIR}/ltou.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/ltou.c $(COPTO)${ARCHDIR}/ltou.o
${ARCHDIR}/makeafile.o: ${SRCDIR}/makeafile.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/makeafile.c $(COPTO)${ARCHDIR}/makeafile.o
${ARCHDIR}/matfit.o: ${SRCDIR}/matfit.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/matfit.c $(COPTO)${ARCHDIR}/matfit.o
${ARCHDIR}/matinv.o: ${SRCDIR}/matinv.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/matinv.c $(COPTO)${ARCHDIR}/matinv.o
${ARCHDIR}/matmult.o: ${SRCDIR}/matmult.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/matmult.c $(COPTO)${ARCHDIR}/matmult.o
${ARCHDIR}/murzin_P.o: ${SRCDIR}/murzin_P.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/murzin_P.c $(COPTO)${ARCHDIR}/murzin_P.o
${ARCHDIR}/newoutput.o: ${SRCDIR}/newoutput.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/newoutput.c $(COPTO)${ARCHDIR}/newoutput.o
${ARCHDIR}/openfile.o: ${SRCDIR}/openfile.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/openfile.c $(COPTO)${ARCHDIR}/openfile.o
${ARCHDIR}/pairfit.o: ${SRCDIR}/pairfit.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/pairfit.c $(COPTO)${ARCHDIR}/pairfit.o
${ARCHDIR}/pairpath.o: ${SRCDIR}/pairpath.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/pairpath.c $(COPTO)${ARCHDIR}/pairpath.o
${ARCHDIR}/pairwise.o: ${SRCDIR}/pairwise.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/pairwise.c $(COPTO)${ARCHDIR}/pairwise.o
${ARCHDIR}/printmat.o: ${SRCDIR}/printmat.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/printmat.c $(COPTO)${ARCHDIR}/printmat.o
${ARCHDIR}/printmat2.o: ${SRCDIR}/printmat2.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/printmat2.c $(COPTO)${ARCHDIR}/printmat2.o
${ARCHDIR}/probcalc.o: ${SRCDIR}/probcalc.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/probcalc.c $(COPTO)${ARCHDIR}/probcalc.o
${ARCHDIR}/probplot.o: ${SRCDIR}/probplot.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/probplot.c $(COPTO)${ARCHDIR}/probplot.o
${ARCHDIR}/qkfit.o: ${SRCDIR}/qkfit.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/qkfit.c $(COPTO)${ARCHDIR}/qkfit.o
${ARCHDIR}/readtree.o: ${SRCDIR}/readtree.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/readtree.c $(COPTO)${ARCHDIR}/readtree.o
${ARCHDIR}/reval.o: ${SRCDIR}/reval.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/reval.c $(COPTO)${ARCHDIR}/reval.o
${ARCHDIR}/revmatmult.o: ${SRCDIR}/revmatmult.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/revmatmult.c $(COPTO)${ARCHDIR}/revmatmult.o
${ARCHDIR}/rossmann.o: ${SRCDIR}/rossmann.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/rossmann.c $(COPTO)${ARCHDIR}/rossmann.o
${ARCHDIR}/roughfit.o: ${SRCDIR}/roughfit.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/roughfit.c $(COPTO)${ARCHDIR}/roughfit.o
${ARCHDIR}/scan.o: ${SRCDIR}/scan.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/scan.c $(COPTO)${ARCHDIR}/scan.o
${ARCHDIR}/sec_content.o: ${SRCDIR}/sec_content.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/sec_content.c $(COPTO)${ARCHDIR}/sec_content.o
${ARCHDIR}/slow_scan.o: ${SRCDIR}/slow_scan.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/slow_scan.c $(COPTO)${ARCHDIR}/slow_scan.o
${ARCHDIR}/stamp.o: ${SRCDIR}/stamp.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/stamp.c $(COPTO)${ARCHDIR}/stamp.o
${ARCHDIR}/stampgetpars.o: ${SRCDIR}/stampgetpars.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/stampgetpars.c $(COPTO)${ARCHDIR}/stampgetpars.o
${ARCHDIR}/sw7ccs.o: ${SRCDIR}/sw7ccs.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/sw7ccs.c $(COPTO)${ARCHDIR}/sw7ccs.o
${ARCHDIR}/swstruc.o: ${SRCDIR}/swstruc.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/swstruc.c $(COPTO)${ARCHDIR}/swstruc.o
${ARCHDIR}/testfile.o: ${SRCDIR}/testfile.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/testfile.c $(COPTO)${ARCHDIR}/testfile.o
${ARCHDIR}/treefit.o: ${SRCDIR}/treefit.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/treefit.c $(COPTO)${ARCHDIR}/treefit.o
${ARCHDIR}/treepath.o: ${SRCDIR}/treepath.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/treepath.c $(COPTO)${ARCHDIR}/treepath.o
${ARCHDIR}/treewise.o: ${SRCDIR}/treewise.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/treewise.c $(COPTO)${ARCHDIR}/treewise.o
${ARCHDIR}/update.o: ${SRCDIR}/update.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/update.c $(COPTO)${ARCHDIR}/update.o
${ARCHDIR}/utol.o: ${SRCDIR}/utol.c
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/utol.c $(COPTO)${ARCHDIR}/utol.o

