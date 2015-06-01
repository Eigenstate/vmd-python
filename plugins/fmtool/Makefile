# Makefile for fmtool

.SILENT:

.SUFFIXES: 

VERSION=0.1
COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/bin/fmtool${VERSION}
SRCDIR=.
INCDIR=-I. 
CXXFLAGS=-Wall

VPATH = ${ARCHDIR}

#
# Rules
#

bins: ${ARCHDIR} ${ARCHDIR}/fmtool
win32bins:
# enable builds of fmtool on Windows.
# win32bins: ${ARCHDIR} ${ARCHDIR}/fmtool.exe
dynlibs: 
staticlibs: 
win32staticlibs: 

distrib:
	for localname in `find ../compile -name fmtool -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
	done

${ARCHDIR}:
	mkdir -p ${ARCHDIR}

fmtoolOBJS = ${ARCHDIR}/make_psf.o ${ARCHDIR}/psf_from_file_xy_to_r.o \
             ${ARCHDIR}/psf_from_file.o \
             ${ARCHDIR}/read_conf.o ${ARCHDIR}/write_psf.o \
             ${ARCHDIR}/time_steps_xy_to_r.o ${ARCHDIR}/frap.o 

# Win32/Win64 compiles
${ARCHDIR}/fmtool.exe : ${fmtoolOBJS} 
	LINK ${fmtoolOBJS} /OUT:${ARCHDIR}/fmtool.exe ${TCLLIB} ${TCLLDFLAGS} ${NETCDFLIB} ${NETCDFLDFLAGS}

# all other platforms
${ARCHDIR}/fmtool : ${fmtoolOBJS} 
	${CXX} ${CXXFLAGS} ${fmtoolOBJS} -o ${ARCHDIR}/fmtool ${TCLLIB} ${TCLLDFLAGS} ${NETCDFLIB} ${NETCDFLDFLAGS} -lm

${ARCHDIR}/make_psf.o : make_psf.C psf_from_file_xy_to_r.h psf_from_file.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/make_psf.C -o $(COPTO)${ARCHDIR}/make_psf.o

${ARCHDIR}/psf_from_file_xy_to_r.o: psf_from_file_xy_to_r.C psf_from_file_xy_to_r.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psf_from_file_xy_to_r.C -o $(COPTO)${ARCHDIR}/psf_from_file_xy_to_r.o
	
${ARCHDIR}/psf_from_file.o: psf_from_file.C psf_from_file.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/psf_from_file.C -o $(COPTO)${ARCHDIR}/psf_from_file.o

${ARCHDIR}/read_conf.o: read_conf.C read_conf.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/read_conf.C -o $(COPTO)${ARCHDIR}/read_conf.o

${ARCHDIR}/write_psf.o: write_psf.C write_psf.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/write_psf.C -o $(COPTO)${ARCHDIR}/write_psf.o

${ARCHDIR}/time_steps_xy_to_r.o: time_steps_xy_to_r.C time_steps_xy_to_r.h 
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/time_steps_xy_to_r.C -o $(COPTO)${ARCHDIR}/time_steps_xy_to_r.o

${ARCHDIR}/frap.o: frap.C make_psf.h write_psf.h read_conf.h time_steps_xy_to_r.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/frap.C -o $(COPTO)${ARCHDIR}/frap.o

clean:
	rm -f ${ARCHDIR}/fmtool.exe ${fmtoolOBJS} ${ARCHDIR}/fmtool

testconf: ${ARCHDIR}/fmtool
	${ARCHDIR}/fmtool test/1.conf


