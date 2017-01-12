.SILENT:

.SUFFIXES: 


VMFILES = phylotree.tcl phylotree_analysis.tcl phylotree_data.tcl \
phylotree_export.tcl phylotree_je.tcl phylotree_matrixviewer.tcl \
phylotree_newick.tcl phylotree_nexus.tcl phylotree_nodeprops.tcl \
phylotree_widget.tcl pkgIndex.tcl cluster.tcl
    
VMVERSION = 1.2
DIR = $(PLUGINDIR)/noarch/tcl/phylotree$(VMVERSION)


AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/phylotree${VMVERSION}
SRCDIR=src
INCDIR=-Isrc
VPATH = src ${ARCHDIR}

#
# Rules
#
bins: ${ARCHDIR} ${ARCHDIR}/upgmaclu
win32bins: ${ARCHDIR} ${ARCHDIR}/upgmaclu.exe
dynlibs:
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying phylotree $(VMVERSION) files to"
	for localname in `find ../compile -name upgmaclu -o -name upgmaclu.exe` ;do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		echo "    $$dir"; \
		cp $$localname $$dir; \
	   cp $(VMFILES) $$dir; \
	done
    
    
${ARCHDIR}:
	mkdir -p ${ARCHDIR}

OBJS = ${ARCHDIR}/upgmaclu.o

# Win32/Win64 compiles
${ARCHDIR}/upgmaclu.exe : ${OBJS}
	LINK ${OBJS} /OUT:${ARCHDIR}/upgmaclu.exe

# all other platforms
${ARCHDIR}/upgmaclu : ${OBJS}
	${CXX} ${CXXFLAGS} ${OBJS} -o ${ARCHDIR}/upgmaclu -lm

${ARCHDIR}/upgmaclu.o: ${SRCDIR}/upgmaclu.c
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/upgmaclu.c $(COPTO)${ARCHDIR}/upgmaclu.o

