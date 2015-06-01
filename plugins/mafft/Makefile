.SILENT:

.SUFFIXES: 

TCLFILES = mafft.tcl pkgIndex.tcl
PKGVERSION = 1.1
PKGNAME = mafft


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
# bins: ${ARCHDIR} ${ARCHDIR}/mafft.exe
bins: ${ARCHDIR}
# win32bins: ${ARCHDIR} ${ARCHDIR}/clustalw.exe
win32bins: ${ARCHDIR}
dynlibs:
staticlibs: 
win32staticlibs: 

distrib: 
	echo "Copying ${PKGNAME} $(PKGVERSION) files to"
	for localname in `find ../compile -name mafft.tcl -o -name pkgIndex.tcl` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $$dir; \
	done
    
${ARCHDIR}:
	mkdir -p ${ARCHDIR}
	cp $(TCLFILES) ${ARCHDIR}

OBJS = 


