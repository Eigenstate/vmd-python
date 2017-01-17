# Makefile for nlenergy

.SILENT:

.SUFFIXES: 

AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/nlenergy1.0
SRCDIR=src
INCDIR=-Isrc -I../include -I${COMPILEDIR}/lib_${ARCH}/molfile
VPATH = src ${ARCHDIR}


##
## Only build nlenergy if we have a Tcl library
##
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
TARGETS = ${ARCHDIR} ${ARCHDIR}/libnlenergy.so
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
	for localname in `find ../compile -name libnlenergy.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		cp src/pkgIndex.tcl $$dir ; \
	done

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

NLENERGYOBJS = ${ARCHDIR}/charmm_file.o \
	${ARCHDIR}/topo_mol_output.o \
	${ARCHDIR}/topo_mol_pluginio.o

${ARCHDIR}/libnlenergy.so : ${NLENERGYOBJS} ${ARCHDIR}/tcl_psfgen.o
	if [ -n "${TCLSHLD}" ]; \
	then ${TCLSHLD} $(LOPTO)$@ ${NLENERGYOBJS} ${ARCHDIR}/tcl_psfgen.o ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	else ${SHLD} $(LOPTO)$@ ${NLENERGYOBJS} ${ARCHDIR}/tcl_psfgen.o ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	fi



${ARCHDIR}/topo_mol_output.o: topo_mol_output.c topo_mol_output.h topo_mol.h topo_defs.h topo_mol_struct.h hasharray.h memarena.h topo_defs_struct.h pdb_file.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/topo_mol_output.c $(COPTO)${ARCHDIR}/topo_mol_output.o

${ARCHDIR}/topo_mol_pluginio.o: topo_mol_pluginio.c topo_mol_pluginio.h topo_mol.h topo_defs.h topo_mol_struct.h hasharray.h memarena.h topo_defs_struct.h pdb_file.h
	${CC} ${CCFLAGS} ${INCDIR} -c ${SRCDIR}/topo_mol_pluginio.c $(COPTO)${ARCHDIR}/topo_mol_pluginio.o



force/angle.c
force/bond.c
force/dihedral.c
force/fbonded.c
force/fnbcut.c
force/fnbcut_eval.c
force/improper.c
force/nbpair.c
molfiles/bincoord.c
molfiles/charmm.c
molfiles/config.c
molfiles/pdbatom.c
molfiles/psf.c
molfiles/xplor.c
moltypes/coord.c
moltypes/domain.c
moltypes/energy.c
moltypes/exclude.c
moltypes/forceprm.c
moltypes/forceprm_init.c
moltypes/forceprm_nonb.c
moltypes/simprm.c
moltypes/topology.c
moltypes/topology_init.c
nlbase/array.c
nlbase/arrmap.c
nlbase/error.c
nlbase/idlist.c
nlbase/io.c
nlbase/mem.c
nlbase/numeric.c
nlbase/random.c
nlbase/sort.c
nlbase/string.c
nlbase/strmap.c
nlbase/test_array.c
nlbase/test_arrmap.c
nlbase/test_idlist.c
nlbase/test_io.c
nlbase/test_list.c
nlbase/test_mem.c
nlbase/test_sort.c
nlbase/test_strmap2.c
nlbase/test_strmap.c
nlenergy/add.c
nlenergy/eval.c
nlenergy/fileio.c
nlenergy/get.c
nlenergy/nlenergy.c
nlenergy/parse.c
nlenergy/remove.c
nlenergy/set.c
nlenergy/tcl_nlenergy.c
nlenergy/tclwrap.c



