# Makefile for psfgen

.SILENT:

.SUFFIXES: 

AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/psfgen1.6
SRCDIR=src
INCDIR=-Isrc -I../include -I${COMPILEDIR}/lib_${ARCH}/molfile
VPATH = src ${ARCHDIR}


##
## Only build psfgen if we have a Tcl library
##
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
TARGETS = ${ARCHDIR} ${ARCHDIR}/libpsfgen.so
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

distrib: doc/ug.pdf
	for localname in `find ../compile -name libpsfgen.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		cp doc/ug.pdf $$dir ; \
		cp src/pkgIndex.tcl $$dir ; \
	done

doc/ug.pdf: doc/ug_macros.tex doc/ug_psfgen.tex doc/ug.tex doc/ug_title.tex
	cd doc; pdflatex ug; pdflatex ug

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

PSFGENOBJS = ${ARCHDIR}/charmm_file.o \
	${ARCHDIR}/charmm_parse_topo_defs.o \
	${ARCHDIR}/extract_alias.o \
	${ARCHDIR}/hash.o \
	${ARCHDIR}/hasharray.o \
	${ARCHDIR}/memarena.o \
	${ARCHDIR}/pdb_file.o \
	${ARCHDIR}/pdb_file_extract.o \
	${ARCHDIR}/psf_file.o \
	${ARCHDIR}/psf_file_extract.o \
	${ARCHDIR}/stringhash.o \
	${ARCHDIR}/topo_defs.o \
	${ARCHDIR}/topo_mol.o \
	${ARCHDIR}/topo_mol_output.o \
	${ARCHDIR}/topo_mol_pluginio.o

${ARCHDIR}/libpsfgen.so : ${PSFGENOBJS} ${ARCHDIR}/tcl_psfgen.o
	if [ -n "${TCLSHLD}" ]; \
	then ${TCLSHLD} $(LOPTO)$@ ${PSFGENOBJS} ${ARCHDIR}/tcl_psfgen.o ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	else ${SHLD} $(LOPTO)$@ ${PSFGENOBJS} ${ARCHDIR}/tcl_psfgen.o ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	fi

${ARCHDIR}/charmm_file.o: charmm_file.c charmm_file.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/charmm_file.c $(COPTO)${ARCHDIR}/charmm_file.o

${ARCHDIR}/charmm_parse_topo_defs.o: charmm_parse_topo_defs.c charmm_file.h topo_defs.h charmm_parse_topo_defs.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/charmm_parse_topo_defs.c $(COPTO)${ARCHDIR}/charmm_parse_topo_defs.o
	
${ARCHDIR}/extract_alias.o: extract_alias.c stringhash.h extract_alias.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/extract_alias.c $(COPTO)${ARCHDIR}/extract_alias.o

${ARCHDIR}/hasharray.o: hasharray.c hash.h memarena.h hasharray.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/hasharray.c $(COPTO)${ARCHDIR}/hasharray.o

${ARCHDIR}/hash.o: hash.c hash.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/hash.c $(COPTO)${ARCHDIR}/hash.o

${ARCHDIR}/memarena.o: memarena.c memarena.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/memarena.c $(COPTO)${ARCHDIR}/memarena.o

${ARCHDIR}/pdb_file.o: pdb_file.c pdb_file.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/pdb_file.c $(COPTO)${ARCHDIR}/pdb_file.o

${ARCHDIR}/pdb_file_extract.o: pdb_file_extract.c pdb_file_extract.h stringhash.h topo_mol.h topo_defs.h pdb_file.h extract_alias.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/pdb_file_extract.c $(COPTO)${ARCHDIR}/pdb_file_extract.o

${ARCHDIR}/psf_file.o: psf_file.c psf_file.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/psf_file.c $(COPTO)${ARCHDIR}/psf_file.o

${ARCHDIR}/psf_file_extract.o: psf_file_extract.c psf_file.h psf_file_extract.h topo_mol.h topo_defs.h topo_mol_struct.h hasharray.h memarena.h topo_defs_struct.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/psf_file_extract.c $(COPTO)${ARCHDIR}/psf_file_extract.o

${ARCHDIR}/stringhash.o: stringhash.c memarena.h hasharray.h stringhash.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/stringhash.c $(COPTO)${ARCHDIR}/stringhash.o

${ARCHDIR}/tcl_main.o: tcl_main.c
	${CC} ${CCFLAGS} ${TCLINC} ${INCDIR} -c ${SRCDIR}/tcl_main.c $(COPTO)${ARCHDIR}/tcl_main.o

${ARCHDIR}/tcl_psfgen.o: tcl_psfgen.c psfgen.h topo_defs.h topo_mol.h stringhash.h charmm_parse_topo_defs.h topo_mol_output.h topo_mol_pluginio.h pdb_file_extract.h psf_file_extract.h
	${CC} ${CCFLAGS} ${TCLINC} ${INCDIR} -D_${ARCH} -DPSFGENTCLDLL_EXPORTS -c ${SRCDIR}/tcl_psfgen.c $(COPTO)${ARCHDIR}/tcl_psfgen.o

${ARCHDIR}/topo_defs.o: topo_defs.c topo_defs_struct.h hasharray.h topo_defs.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/topo_defs.c $(COPTO)${ARCHDIR}/topo_defs.o

${ARCHDIR}/topo_mol.o: topo_mol.c topo_defs_struct.h hasharray.h topo_defs.h topo_mol_struct.h memarena.h topo_mol.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/topo_mol.c $(COPTO)${ARCHDIR}/topo_mol.o

${ARCHDIR}/topo_mol_output.o: topo_mol_output.c topo_mol_output.h topo_mol.h topo_defs.h topo_mol_struct.h hasharray.h memarena.h topo_defs_struct.h pdb_file.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/topo_mol_output.c $(COPTO)${ARCHDIR}/topo_mol_output.o

${ARCHDIR}/topo_mol_pluginio.o: topo_mol_pluginio.c topo_mol_pluginio.h topo_mol.h topo_defs.h topo_mol_struct.h hasharray.h memarena.h topo_defs_struct.h pdb_file.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/topo_mol_pluginio.c $(COPTO)${ARCHDIR}/topo_mol_pluginio.o
