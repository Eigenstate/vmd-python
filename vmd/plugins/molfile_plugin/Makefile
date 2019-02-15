# Makefile for molecule file readers
# $Id: Makefile,v 1.132 2019/02/07 21:22:05 johns Exp $

.SILENT:

.SUFFIXES:

PLUGINAPI = molfile_plugin.h vmdplugin.h vmdconio.h
COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/molfile

SRCDIR = ./src
INCDIR = -I../include -I${SRCDIR}

VPATH = ${SRCDIR}:../include:${ARCHDIR}

SCCFLAGS = $(CCFLAGS) $(DEF)"STATIC_PLUGIN"
SCXXFLAGS = $(CCFLAGS) $(DEF)"STATIC_PLUGIN"

#
# Rules
#

STATICPLUGINS = abinitplugin amiraplugin avsplugin babelplugin basissetplugin bgfplugin binposplugin biomoccaplugin brixplugin carplugin ccp4plugin corplugin cpmdplugin crdplugin cubeplugin dcdplugin dlpolyplugin dsn6plugin dxplugin edmplugin fs4plugin gamessplugin graspplugin grdplugin gridplugin gromacsplugin jsplugin lammpsplugin mapplugin mdfplugin mol2plugin moldenplugin molemeshplugin msmsplugin namdbinplugin offplugin parm7plugin parmplugin pbeqplugin pdbplugin pdbxplugin phiplugin pltplugin plyplugin pqrplugin psfplugin raster3dplugin rst7plugin situsplugin spiderplugin stlplugin tinkerplugin uhbdplugin vaspchgcarplugin vaspoutcarplugin vaspparchgplugin vaspposcarplugin vasp5xdatcarplugin vaspxdatcarplugin vaspxmlplugin vtkplugin xbgfplugin xsfplugin xyzplugin

PLUGINS = abinitplugin.so amiraplugin.so avsplugin.so babelplugin.so basissetplugin.so bgfplugin.so binposplugin.so biomoccaplugin.so brixplugin.so carplugin.so ccp4plugin.so corplugin.so cpmdplugin.so crdplugin.so cubeplugin.so dcdplugin.so dlpolyplugin.so dsn6plugin.so dxplugin.so edmplugin.so fs4plugin.so gamessplugin.so graspplugin.so grdplugin.so gridplugin.so gromacsplugin.so jsplugin.so lammpsplugin.so mapplugin.so mdfplugin.so mol2plugin.so moldenplugin.so molemeshplugin.so msmsplugin.so namdbinplugin.so offplugin.so parm7plugin.so parmplugin.so pbeqplugin.so pdbplugin.so pdbxplugin.so phiplugin.so pltplugin.so plyplugin.so pqrplugin.so psfplugin.so raster3dplugin.so rst7plugin.so situsplugin.so spiderplugin.so stlplugin.so tinkerplugin.so uhbdplugin.so vaspchgcarplugin.so vaspoutcarplugin.so vaspparchgplugin.so vaspposcarplugin.so vasp5xdatcarplugin.so vaspxdatcarplugin.so vaspxmlplugin.so vtkplugin.so xbgfplugin.so xsfplugin.so xyzplugin.so


#
# Check to see if we're building on Android or not. If not, we
# include some plugins that don't currently compile cleanly using
# the Android NDK cross-development toolchain for Linux.
# XXX non-portable GNU make syntax used here...
#
ifneq ($(ARCH),ANDROIDARMV7A)
STATICPLUGINS += dtrplugin maeffplugin
PLUGINS       += dtrplugin.so maeffplugin.so
endif

#
# Check to see if we're building on platforms where the current
# ORCA plugin works.  The code is still not yet C++03 friendly yet,
# so it can't be compiled on some targets presently.
#
ifeq ($(ARCH),LINUXAMD64)
STATICPLUGINS += orcaplugin
PLUGINS       += orcaplugin.so
endif
ifeq ($(ARCH),MACOSXX86)
STATICPLUGINS += orcaplugin
PLUGINS       += orcaplugin.so
endif


#
# Check to see if we should build the optional expat-based plugins
# XXX non-portable GNU make syntax used here...
#
ifdef EXPATLIB
ifdef EXPATINC
ifdef EXPATLDFLAGS
ifndef EXPATDYNAMIC
# Only add the HOOMD plugin as a static plugin if we compiled against
# a static libexpat.a, as VMD itself will also have to be linked against
# the same static library for this to work.  In the case of a dynamic
# library, we can get by linking only the plugin itself, and either
# assuming it is installed already on the target OS, or included in the
# VMD redistributables.
STATICPLUGINS += hoomdplugin
endif
PLUGINS       += hoomdplugin.so
endif
endif
endif


#
# Check to see if we should build the optional NetCDF-based plugins
# XXX non-portable GNU make syntax used here...
#
ifdef NETCDFLIB
ifdef NETCDFINC
ifdef NETCDFLDFLAGS
ifndef NETCDFDYNAMIC
# Only add the NetCDF plugin as a static plugin if we compiled against
# a static libnetcdf.a, as VMD itself will also have to be linked against
# the same static library for this to work.  In the case of a dynamic
# library, we can get by linking only the plugin itself, and either
# assuming it is installed already on the target OS, or included in the
# VMD redistributables.
STATICPLUGINS += netcdfplugin
endif
PLUGINS       += netcdfplugin.so
endif
endif
endif


#
# Check to see if we should build the optional NetCDF-based plugins
# XXX non-portable GNU make syntax used here...
#
ifdef HDFLIB
ifdef HDFINC
ifdef HDFLDFLAGS
ifndef HDFDYNAMIC
# Only add plugin that use HDF as static plugins if we compiled against
# a static libnetcdf.a, as VMD itself will also have to be linked against
# the same static library for this to work.  In the case of a dynamic
# library, we can get by linking only the plugin itself, and either
# assuming it is installed already on the target OS, or included in the
# VMD redistributables.
STATICPLUGINS += lmplugin
endif
PLUGINS       += lmplugin.so
endif
endif
endif


#
# Check to see if we should build the optional sqlite-based plugins
# XXX non-portable GNU make syntax used here...
#
ifdef SQLITELIB
ifdef SQLITEINC
ifdef SQLITELDFLAGS
ifndef SQLITEDYNAMIC
# Only add the dmsplugin as a static plugin if we compiled against
# a static sqlite.a, as VMD itself will also have to be linked against
# the same static library for this to work.  In the case of a dynamic
# library, we can get by linking only the plugin itself, and either
# assuming it is installed already on the target OS, or included in the
# VMD redistributables.
STATICPLUGINS += dmsplugin
endif
PLUGINS       += dmsplugin.so
endif
endif
endif


#
# Check to see if we should build the Tcl-based plugins
# XXX non-portable GNU make syntax used here...
# XXX We can safetly assume that VMD is compiled against Tcl, so there is
#     no need to check if the Tcl libs are static or dynamic here.
#
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
STATICPLUGINS += vtfplugin webpdbplugin
PLUGINS       += vtfplugin.so webpdbplugin.so
endif
endif
endif


#
# Check to see if we should build plugins that use the Gromacs TNG library
# XXX non-portable GNU make syntax used here...
#
ifdef TNGLIB
ifdef TNGINC
ifdef TNGLDFLAGS
ifndef TNGDYNAMIC
STATICPLUGINS += tngplugin
endif
PLUGINS       += tngplugin.so
endif
endif
endif


# list of all optional plugins for use by distrib target
OPTPLUGINS = dmsplugin.so hoomdplugin.so lmplugin.so netcdfplugin.so orcaplugin.so tngplugin.so vtfplugin.so webpdbplugin.so

STATICS = libmolfile_plugin.a libmolfile_plugin.h
WIN32STATICS = libmolfile_plugin.lib libmolfile_plugin.h
DISTFILES = $(PLUGINS) $(OPTPLUGINS) $(STATICS) $(WIN32STATICS)

bins:
win32bins:
dynlibs: ${ARCHDIR} mesg $(PLUGINS)
staticlibs: ${ARCHDIR} $(STATICS)
win32staticlibs: ${ARCHDIR} $(WIN32STATICS)

distrib:
	@echo "Copying molfile plugins to $(PLUGINDIR) destination area"
	for file in $(DISTFILES) ; do \
		echo "    $$file ..."; \
		for localname in `find ../compile -name $$file -print`; do \
			pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
			dir=`dirname $(PLUGINDIR)/$$pluginname`; \
			mkdir -p $$dir; \
			cp -p $$localname $(PLUGINDIR)/$$pluginname; \
		done; \
	done;

mesg:
	@echo "Building Molecule File Reader plugins"


#
# plugin library rules
#

abinitplugin.so: ${ARCHDIR}/abinitplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

amiraplugin.so: ${ARCHDIR}/amiraplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

avsplugin.so: ${ARCHDIR}/avsplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

babelplugin.so: ${ARCHDIR}/babelplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

basissetplugin.so: ${ARCHDIR}/basissetplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

bgfplugin.so: ${ARCHDIR}/bgfplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

binposplugin.so: ${ARCHDIR}/binposplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

biomoccaplugin.so: ${ARCHDIR}/biomoccaplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

brixplugin.so: ${ARCHDIR}/brixplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

carplugin.so: ${ARCHDIR}/carplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

ccp4plugin.so: ${ARCHDIR}/ccp4plugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

corplugin.so: ${ARCHDIR}/corplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

cpmdplugin.so: ${ARCHDIR}/cpmdplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

crdplugin.so: ${ARCHDIR}/crdplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

cubeplugin.so: ${ARCHDIR}/cubeplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

dcdplugin.so: ${ARCHDIR}/dcdplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

dlpolyplugin.so: ${ARCHDIR}/dlpolyplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

dsn6plugin.so: ${ARCHDIR}/dsn6plugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

dxplugin.so: ${ARCHDIR}/dxplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

edmplugin.so: ${ARCHDIR}/edmplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

fs4plugin.so: ${ARCHDIR}/fs4plugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

gamessplugin.so: ${ARCHDIR}/gamessplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

graspplugin.so: ${ARCHDIR}/graspplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

grdplugin.so: ${ARCHDIR}/grdplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

gridplugin.so: ${ARCHDIR}/gridplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

gromacsplugin.so: ${ARCHDIR}/gromacsplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

jsplugin.so: ${ARCHDIR}/jsplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

lammpsplugin.so: ${ARCHDIR}/lammpsplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

mapplugin.so: ${ARCHDIR}/mapplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

mdfplugin.so: ${ARCHDIR}/mdfplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

mol2plugin.so: ${ARCHDIR}/mol2plugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

moldenplugin.so: ${ARCHDIR}/moldenplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

molemeshplugin.so: ${ARCHDIR}/molemeshplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

msmsplugin.so: ${ARCHDIR}/msmsplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

namdbinplugin.so: ${ARCHDIR}/namdbinplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

offplugin.so: ${ARCHDIR}/offplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

orcaplugin.so: ${ARCHDIR}/orcaplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

parm7plugin.so: ${ARCHDIR}/parm7plugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

parmplugin.so: ${ARCHDIR}/parmplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

pbeqplugin.so: ${ARCHDIR}/pbeqplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

pdbplugin.so: ${ARCHDIR}/pdbplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

pdbxplugin.so: ${ARCHDIR}/pdbxplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

phiplugin.so: ${ARCHDIR}/phiplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

pltplugin.so: ${ARCHDIR}/pltplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

plyplugin.so: ${ARCHDIR}/plyplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

pqrplugin.so: ${ARCHDIR}/pqrplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

psfplugin.so: ${ARCHDIR}/psfplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

raster3dplugin.so: ${ARCHDIR}/raster3dplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

rst7plugin.so: ${ARCHDIR}/rst7plugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

situsplugin.so: ${ARCHDIR}/situsplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

spiderplugin.so: ${ARCHDIR}/spiderplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

stlplugin.so: ${ARCHDIR}/stlplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

tinkerplugin.so: ${ARCHDIR}/tinkerplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

uhbdplugin.so: ${ARCHDIR}/uhbdplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

vaspchgcarplugin.so: ${ARCHDIR}/vaspchgcarplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

vaspoutcarplugin.so: ${ARCHDIR}/vaspoutcarplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

vaspparchgplugin.so: ${ARCHDIR}/vaspparchgplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

vaspposcarplugin.so: ${ARCHDIR}/vaspposcarplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

vasp5xdatcarplugin.so: ${ARCHDIR}/vasp5xdatcarplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

vaspxdatcarplugin.so: ${ARCHDIR}/vaspxdatcarplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

vaspxmlplugin.so: ${ARCHDIR}/vaspxmlplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

vtkplugin.so: ${ARCHDIR}/vtkplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

xbgfplugin.so: ${ARCHDIR}/xbgfplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

xsfplugin.so: ${ARCHDIR}/xsfplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

xyzplugin.so: ${ARCHDIR}/xyzplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)


##
## In-development plugins that aren't part of the build quite yet...
##
cpmdlogplugin.so: ${ARCHDIR}/cpmdlogplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

gaussianplugin.so: ${ARCHDIR}/gaussianplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)


##
## Optionally compiled plugins that have library or platform-specific
## dependencies of some kind
##
dmsplugin.so: ${ARCHDIR}/dmsplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(SQLITELIB) $(SQLITELDFLAGS) $(LDFLAGS)

dtrplugin.so: ${ARCHDIR}/dtrplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

hoomdplugin.so: ${ARCHDIR}/hoomdplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(EXPATLIB) $(EXPATLDFLAGS) $(LDFLAGS)

lmplugin.so: ${ARCHDIR}/lmplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(HDFLIB) $(HDFLDFLAGS) $(LDFLAGS)

maeffplugin.so: ${ARCHDIR}/maeffplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(LDFLAGS)

netcdfplugin.so: ${ARCHDIR}/netcdfplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(NETCDFLIB) $(NETCDFLDFLAGS) $(LDFLAGS)

tngplugin.so: ${ARCHDIR}/tngplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(TNGLIB) $(TNGLDFLAGS) $(LDFLAGS)

vtfplugin.so: ${ARCHDIR}/vtfplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(TCLLIB) $(TCLLDFLAGS) $(LDFLAGS)

webpdbplugin.so: ${ARCHDIR}/webpdbplugin.o
	$(SHLD) $(LOPTO)${ARCHDIR}/$@ $? $(TCLLIB) $(TCLLDFLAGS) $(LDFLAGS)


#
# object files
#
${ARCHDIR}/abinitplugin.o: abinitplugin.c ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/amiraplugin.o: amiraplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/avsplugin.o: avsplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/babelplugin.o: babelplugin.c readpdb.h vmddir.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/basissetplugin.o: basissetplugin.c ${PLUGINAPI} qmplugin.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/bgfplugin.o: bgfplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/binposplugin.o: binposplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/biomoccaplugin.o: biomoccaplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/brixplugin.o: brixplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/carplugin.o: carplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/ccp4plugin.o: ccp4plugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/corplugin.o: corplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/cpmdplugin.o: cpmdplugin.c ${PLUGINAPI} unit_conversion.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/crdplugin.o: crdplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/cubeplugin.o: cubeplugin.C ${PLUGINAPI} periodic_table.h unit_conversion.h
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/dcdplugin.o: dcdplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/dlpolyplugin.o: dlpolyplugin.c ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/dsn6plugin.o: dsn6plugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/dxplugin.o: dxplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/edmplugin.o: edmplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/fs4plugin.o: fs4plugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/gamessplugin.o: gamessplugin.c ${PLUGINAPI} qmplugin.h periodic_table.h unit_conversion.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/graspplugin.o: graspplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/grdplugin.o: grdplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/gridplugin.o: gridplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/gromacsplugin.o: gromacsplugin.C ${PLUGINAPI} Gromacs.h
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/jsplugin.o: jsplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/lammpsplugin.o: lammpsplugin.c ${PLUGINAPI} hash.c hash.h inthash.c inthash.h periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/mapplugin.o: mapplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/mdfplugin.o: mdfplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/mol2plugin.o: mol2plugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/moldenplugin.o: moldenplugin.c ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/molemeshplugin.o: molemeshplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/msmsplugin.o: msmsplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/namdbinplugin.o: namdbinplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/offplugin.o: offplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/orcaplugin.o: orcaplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/parm7plugin.o: parm7plugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/parmplugin.o: parmplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/pbeqplugin.o: pbeqplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/pdbplugin.o: pdbplugin.c readpdb.h ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/pdbxplugin.o: pdbxplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/phiplugin.o: phiplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/pltplugin.o: pltplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/plyplugin.o: plyplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/pqrplugin.o: pqrplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/psfplugin.o: psfplugin.c fortread.h ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/raster3dplugin.o: raster3dplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/rst7plugin.o: rst7plugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/situsplugin.o: situsplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/spiderplugin.o: spiderplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/stlplugin.o: stlplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/tinkerplugin.o: tinkerplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/uhbdplugin.o: uhbdplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vaspchgcarplugin.o: vaspchgcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vaspoutcarplugin.o: vaspoutcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vaspparchgplugin.o: vaspparchgplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vaspposcarplugin.o: vaspposcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vasp5xdatcarplugin.o: vasp5xdatcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vaspxdatcarplugin.o: vaspxdatcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vaspxmlplugin.o: vaspxmlplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vtkplugin.o: vtkplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/xbgfplugin.o: xbgfplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/xsfplugin.o: xsfplugin.C ${PLUGINAPI} periodic_table.h
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/xyzplugin.o: xyzplugin.c ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@


##
## In-development plugins that aren't part of the build quite yet...
##
${ARCHDIR}/cpmdlogplugin.o: cpmdlogplugin.c ${PLUGINAPI} gaussianplugin.h periodic_table.h unit_conversion.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/gaussianplugin.o: gaussianplugin.c ${PLUGINAPI} gaussianplugin.h periodic_table.h unit_conversion.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@


##
## Optionally compiled plugins that have library or platform-specific
## dependencies of some kind
##
${ARCHDIR}/dmsplugin.o: dmsplugin.cxx ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(SQLITEINC) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/dtrplugin.o: dtrplugin.cxx ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/hoomdplugin.o: hoomdplugin.c ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(EXPATINC) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/lmplugin.o: lmplugin.C ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(HDFINC) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/maeffplugin.o: maeffplugin.cxx ${PLUGINAPI}
	$(CXX) $(CXXFLAGS) $(SHLDFLAGS) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/netcdfplugin.o: netcdfplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(NETCDFINC) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/tngplugin.o: tngplugin.C ${PLUGINAPI}
	${CXX} ${CXXFLAGS} $(SHLDFLAGS) $(TNGINC) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/vtfplugin.o: vtfplugin.c ${PLUGINAPI}
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(TCLINC) $(INCDIR) -c $< $(COPTO)$@

${ARCHDIR}/webpdbplugin.o: webpdbplugin.c readpdb.h ${PLUGINAPI} periodic_table.h
	$(CC) $(CCFLAGS) $(SHLDFLAGS) $(TCLINC) $(INCDIR) -c $< $(COPTO)$@



#
# archive rules
#
ARCHIVEOBJS = ${ARCHDIR}/abinitplugin-s.o ${ARCHDIR}/amiraplugin-s.o ${ARCHDIR}/avsplugin-s.o ${ARCHDIR}/babelplugin-s.o ${ARCHDIR}/basissetplugin-s.o ${ARCHDIR}/bgfplugin-s.o ${ARCHDIR}/binposplugin-s.o ${ARCHDIR}/biomoccaplugin-s.o ${ARCHDIR}/brixplugin-s.o ${ARCHDIR}/carplugin-s.o ${ARCHDIR}/ccp4plugin-s.o ${ARCHDIR}/corplugin-s.o ${ARCHDIR}/cpmdplugin-s.o ${ARCHDIR}/crdplugin-s.o ${ARCHDIR}/cubeplugin-s.o ${ARCHDIR}/dcdplugin-s.o ${ARCHDIR}/dlpolyplugin-s.o ${ARCHDIR}/dsn6plugin-s.o ${ARCHDIR}/dxplugin-s.o ${ARCHDIR}/edmplugin-s.o ${ARCHDIR}/fs4plugin-s.o ${ARCHDIR}/gamessplugin-s.o ${ARCHDIR}/graspplugin-s.o ${ARCHDIR}/grdplugin-s.o ${ARCHDIR}/gridplugin-s.o ${ARCHDIR}/gromacsplugin-s.o ${ARCHDIR}/jsplugin-s.o ${ARCHDIR}/lammpsplugin-s.o ${ARCHDIR}/mapplugin-s.o ${ARCHDIR}/mdfplugin-s.o ${ARCHDIR}/mol2plugin-s.o ${ARCHDIR}/moldenplugin-s.o ${ARCHDIR}/molemeshplugin-s.o ${ARCHDIR}/msmsplugin-s.o ${ARCHDIR}/namdbinplugin-s.o ${ARCHDIR}/offplugin-s.o ${ARCHDIR}/parm7plugin-s.o ${ARCHDIR}/parmplugin-s.o ${ARCHDIR}/pbeqplugin-s.o ${ARCHDIR}/pdbplugin-s.o ${ARCHDIR}/pdbxplugin-s.o ${ARCHDIR}/phiplugin-s.o ${ARCHDIR}/pltplugin-s.o ${ARCHDIR}/plyplugin-s.o ${ARCHDIR}/pqrplugin-s.o ${ARCHDIR}/psfplugin-s.o ${ARCHDIR}/raster3dplugin-s.o ${ARCHDIR}/rst7plugin-s.o ${ARCHDIR}/situsplugin-s.o ${ARCHDIR}/spiderplugin-s.o ${ARCHDIR}/stlplugin-s.o ${ARCHDIR}/tinkerplugin-s.o ${ARCHDIR}/uhbdplugin-s.o ${ARCHDIR}/vaspchgcarplugin-s.o ${ARCHDIR}/vaspoutcarplugin-s.o ${ARCHDIR}/vaspparchgplugin-s.o ${ARCHDIR}/vaspposcarplugin-s.o ${ARCHDIR}/vasp5xdatcarplugin-s.o ${ARCHDIR}/vaspxdatcarplugin-s.o ${ARCHDIR}/vaspxmlplugin-s.o ${ARCHDIR}/vtkplugin-s.o ${ARCHDIR}/xbgfplugin-s.o ${ARCHDIR}/xsfplugin-s.o ${ARCHDIR}/xyzplugin-s.o


#
# Check to see if we're building on Android or not. If not, we
# include some plugins that don't currently compile cleanly using
# the Android NDK cross-development toolchain for Linux.
# XXX non-portable GNU make syntax used here...
#
ifneq ($(ARCH),ANDROIDARMV7A)
ARCHIVEOBJS += ${ARCHDIR}/dtrplugin-s.o ${ARCHDIR}/maeffplugin-s.o
endif

#
# Check to see if we're building on platforms where the current
# ORCA plugin works.  The code is still not yet C++03 friendly yet,
# so it can't be compiled on some targets presently.
#
ifeq ($(ARCH),LINUXAMD64)
ARCHIVEOBJS += ${ARCHDIR}/orcaplugin-s.o
endif
ifeq ($(ARCH),MACOSXX86)
ARCHIVEOBJS += ${ARCHDIR}/orcaplugin-s.o
endif

#
# Check to see if we should build the optional expat-based plugins
# XXX non-portable GNU make syntax used here...
#
ifdef EXPATLIB
ifdef EXPATINC
ifdef EXPATLDFLAGS
ifndef EXPATDYNAMIC
ARCHIVEOBJS += ${ARCHDIR}/hoomdplugin-s.o
endif
endif
endif
endif


#
# Check to see if we should build the optional HDF5-based plugins
# XXX non-portable GNU make syntax used here...
#
ifdef HDFLIB
ifdef HDFINC
ifdef HDFLDFLAGS
ifndef HDFDYNAMIC
ARCHIVEOBJS += ${ARCHDIR}/lmplugin-s.o
endif
endif
endif
endif


#
# Check to see if we should build the optional NetCDF-based plugins
# XXX non-portable GNU make syntax used here...
#
ifdef NETCDFLIB
ifdef NETCDFINC
ifdef NETCDFLDFLAGS
ifndef NETCDFDYNAMIC
ARCHIVEOBJS += ${ARCHDIR}/netcdfplugin-s.o
endif
endif
endif
endif


#
# Check to see if we should build the Tcl-based plugins
# XXX non-portable GNU make syntax used here...
#
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
ARCHIVEOBJS += ${ARCHDIR}/vtfplugin-s.o ${ARCHDIR}/webpdbplugin-s.o
endif
endif
endif


#
# Check to see if we should build plugins that use the Gromacs TNG library
# XXX non-portable GNU make syntax used here...
#
ifdef TNGLIB
ifdef TNGINC
ifdef TNGLDFLAGS
ARCHIVEOBJS += ${ARCHDIR}/tngplugin-s.o
endif
endif
endif

#
# Check to see if we should build plugins that use the Desmond thing
# XXX non-portable GNU make syntax used here...
#
ifdef SQLITELIB
ifdef SQLITEINC
ifdef SQLITELDFLAGS
ARCHIVEOBJS += ${ARCHDIR}/dmsplugin-s.o
endif
endif
endif

libmolfile_plugin.a: ${ARCHIVEOBJS}
	rm -f ${ARCHDIR}/$@
	$(AR) cr ${ARCHDIR}/$@ ${ARCHIVEOBJS}
	$(RANLIB) ${ARCHDIR}/$@

libmolfile_plugin.lib: ${ARCHIVEOBJS}
	rm -f ${ARCHDIR}/$@
	lib /OUT:${ARCHDIR}/$@ ${ARCHIVEOBJS}

libmolfile_plugin.h: ${ARCHIVEOBJS}
	rm -f ${ARCHDIR}/$@
	touch ${ARCHDIR}/$@
	../create_static_header.sh MOLFILE molfile ${ARCHDIR}/$@ ${STATICPLUGINS}

#
# object files suitable for static linking
#
${ARCHDIR}/abinitplugin-s.o: abinitplugin.c ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_abinitplugin" -c $< $(COPTO)$@

${ARCHDIR}/amiraplugin-s.o: amiraplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_amiraplugin" -c $< $(COPTO)$@

${ARCHDIR}/avsplugin-s.o: avsplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_avsplugin" -c $< $(COPTO)$@

${ARCHDIR}/babelplugin-s.o: babelplugin.c readpdb.h vmddir.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_babelplugin" -c $< $(COPTO)$@

${ARCHDIR}/basissetplugin-s.o: basissetplugin.c ${PLUGINAPI} qmplugin.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_basissetplugin" -c $< $(COPTO)$@

${ARCHDIR}/bgfplugin-s.o: bgfplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_bgfplugin" -c $< $(COPTO)$@

${ARCHDIR}/binposplugin-s.o: binposplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_binposplugin" -c $< $(COPTO)$@

${ARCHDIR}/biomoccaplugin-s.o: biomoccaplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_biomoccaplugin" -c $< $(COPTO)$@

${ARCHDIR}/brixplugin-s.o: brixplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_brixplugin" -c $< $(COPTO)$@

${ARCHDIR}/carplugin-s.o: carplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_carplugin" -c $< $(COPTO)$@

${ARCHDIR}/ccp4plugin-s.o: ccp4plugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_ccp4plugin" -c $< $(COPTO)$@

${ARCHDIR}/corplugin-s.o: corplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_corplugin" -c $< $(COPTO)$@

${ARCHDIR}/cpmdplugin-s.o: cpmdplugin.c ${PLUGINAPI} unit_conversion.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_cpmdplugin" -c $< $(COPTO)$@

${ARCHDIR}/crdplugin-s.o: crdplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_crdplugin" -c $< $(COPTO)$@

${ARCHDIR}/cubeplugin-s.o: cubeplugin.C ${PLUGINAPI} periodic_table.h unit_conversion.h
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_cubeplugin" -c $< $(COPTO)$@

${ARCHDIR}/dcdplugin-s.o: dcdplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_dcdplugin" -c $< $(COPTO)$@

${ARCHDIR}/dlpolyplugin-s.o: dlpolyplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_dlpolyplugin" -c $< $(COPTO)$@

${ARCHDIR}/dsn6plugin-s.o: dsn6plugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_dsn6plugin" -c $< $(COPTO)$@

${ARCHDIR}/dxplugin-s.o: dxplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_dxplugin" -c $< $(COPTO)$@

${ARCHDIR}/edmplugin-s.o: edmplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_edmplugin" -c $< $(COPTO)$@

${ARCHDIR}/fs4plugin-s.o: fs4plugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_fs4plugin" -c $< $(COPTO)$@

${ARCHDIR}/gamessplugin-s.o: gamessplugin.c ${PLUGINAPI} qmplugin.h periodic_table.h unit_conversion.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_gamessplugin" -c $< $(COPTO)$@

${ARCHDIR}/graspplugin-s.o: graspplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_graspplugin" -c $< $(COPTO)$@

${ARCHDIR}/grdplugin-s.o: grdplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_grdplugin" -c $< $(COPTO)$@

${ARCHDIR}/gridplugin-s.o: gridplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_gridplugin" -c $< $(COPTO)$@

${ARCHDIR}/gromacsplugin-s.o: gromacsplugin.C ${PLUGINAPI} Gromacs.h
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_gromacsplugin" -c $< $(COPTO)$@

${ARCHDIR}/jsplugin-s.o: jsplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_jsplugin" -c $< $(COPTO)$@

${ARCHDIR}/lammpsplugin-s.o: lammpsplugin.c ${PLUGINAPI} hash.c hash.h inthash.c inthash.h periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_lammpsplugin" -c $< $(COPTO)$@

${ARCHDIR}/mapplugin-s.o: mapplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_mapplugin" -c $< $(COPTO)$@

${ARCHDIR}/mdfplugin-s.o: mdfplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_mdfplugin" -c $< $(COPTO)$@

${ARCHDIR}/mol2plugin-s.o: mol2plugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_mol2plugin" -c $< $(COPTO)$@

${ARCHDIR}/moldenplugin-s.o: moldenplugin.c ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_moldenplugin" -c $< $(COPTO)$@

${ARCHDIR}/molemeshplugin-s.o: molemeshplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_molemeshplugin" -c $< $(COPTO)$@

${ARCHDIR}/msmsplugin-s.o: msmsplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_msmsplugin" -c $< $(COPTO)$@

${ARCHDIR}/namdbinplugin-s.o: namdbinplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_namdbinplugin" -c $< $(COPTO)$@

${ARCHDIR}/offplugin-s.o: offplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_offplugin" -c $< $(COPTO)$@

${ARCHDIR}/orcaplugin-s.o: orcaplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_orcaplugin" -c $< $(COPTO)$@

${ARCHDIR}/parm7plugin-s.o: parm7plugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_parm7plugin" -c $< $(COPTO)$@

${ARCHDIR}/parmplugin-s.o: parmplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_parmplugin" -c $< $(COPTO)$@

${ARCHDIR}/pbeqplugin-s.o: pbeqplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_pbeqplugin" -c $< $(COPTO)$@

${ARCHDIR}/pdbplugin-s.o: pdbplugin.c readpdb.h ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_pdbplugin" -c $< $(COPTO)$@

${ARCHDIR}/pdbxplugin-s.o: pdbxplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_pdbxplugin" -c $< $(COPTO)$@

${ARCHDIR}/phiplugin-s.o: phiplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_phiplugin" -c $< $(COPTO)$@

${ARCHDIR}/pltplugin-s.o: pltplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_pltplugin" -c $< $(COPTO)$@

${ARCHDIR}/plyplugin-s.o: plyplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_plyplugin" -c $< $(COPTO)$@

${ARCHDIR}/pqrplugin-s.o: pqrplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_pqrplugin" -c $< $(COPTO)$@

${ARCHDIR}/psfplugin-s.o: psfplugin.c fortread.h ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_psfplugin" -c $< $(COPTO)$@

${ARCHDIR}/raster3dplugin-s.o: raster3dplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_raster3dplugin" -c $< $(COPTO)$@

${ARCHDIR}/rst7plugin-s.o: rst7plugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_rst7plugin" -c $< $(COPTO)$@

${ARCHDIR}/situsplugin-s.o: situsplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_situsplugin" -c $< $(COPTO)$@

${ARCHDIR}/spiderplugin-s.o: spiderplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_spiderplugin" -c $< $(COPTO)$@

${ARCHDIR}/stlplugin-s.o: stlplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_stlplugin" -c $< $(COPTO)$@

${ARCHDIR}/tinkerplugin-s.o: tinkerplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_tinkerplugin" -c $< $(COPTO)$@

${ARCHDIR}/uhbdplugin-s.o: uhbdplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_uhbdplugin" -c $< $(COPTO)$@

${ARCHDIR}/vaspchgcarplugin-s.o: vaspchgcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_vaspchgcarplugin" -c $< $(COPTO)$@

${ARCHDIR}/vaspoutcarplugin-s.o: vaspoutcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_vaspoutcarplugin" -c $< $(COPTO)$@

${ARCHDIR}/vaspparchgplugin-s.o: vaspparchgplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_vaspparchgplugin" -c $< $(COPTO)$@

${ARCHDIR}/vaspposcarplugin-s.o: vaspposcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_vaspposcarplugin" -c $< $(COPTO)$@

${ARCHDIR}/vasp5xdatcarplugin-s.o: vasp5xdatcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_vasp5xdatcarplugin" -c $< $(COPTO)$@

${ARCHDIR}/vaspxdatcarplugin-s.o: vaspxdatcarplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_vaspxdatcarplugin" -c $< $(COPTO)$@

${ARCHDIR}/vaspxmlplugin-s.o: vaspxmlplugin.c vaspplugin.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_vaspxmlplugin" -c $< $(COPTO)$@

${ARCHDIR}/vtkplugin-s.o: vtkplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_vtkplugin" -c $< $(COPTO)$@

${ARCHDIR}/xbgfplugin-s.o: xbgfplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_xbgfplugin" -c $< $(COPTO)$@

${ARCHDIR}/xsfplugin-s.o: xsfplugin.C ${PLUGINAPI} periodic_table.h
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_xsfplugin" -c $< $(COPTO)$@

${ARCHDIR}/xyzplugin-s.o: xyzplugin.c ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_xyzplugin" -c $< $(COPTO)$@


##
## In-development plugins that aren't part of the build quite yet...
##
${ARCHDIR}/cpmdlogplugin-s.o: cpmdlogplugin.c ${PLUGINAPI} gaussianplugin.h periodic_table.h unit_conversion.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_cpmdlogplugin" -c $< $(COPTO)$@

${ARCHDIR}/gaussianplugin-s.o: gaussianplugin.c ${PLUGINAPI}  gaussianplugin.h periodic_table.h unit_conversion.h
	${CC} ${SCCFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_gaussianplugin" -c $< $(COPTO)$@


##
## Optionally compiled plugins that have library or platform-specific
## dependencies of some kind
##
${ARCHDIR}/dmsplugin-s.o: dmsplugin.cxx ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(SQLITEINC) $(INCDIR) $(DEF)"VMDPLUGIN=molfile_dmsplugin" -c $< $(COPTO)$@

${ARCHDIR}/dtrplugin-s.o: dtrplugin.cxx ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_dtrplugin" -c $< $(COPTO)$@

${ARCHDIR}/hoomdplugin-s.o: hoomdplugin.c ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(EXPATINC) $(INCDIR) $(DEF)"VMDPLUGIN=molfile_hoomdplugin" -c $< $(COPTO)$@

${ARCHDIR}/lmplugin-s.o: lmplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(HDFINC) $(INCDIR) $(DEF)"VMDPLUGIN=molfile_lmplugin" -c $< $(COPTO)$@

${ARCHDIR}/maeffplugin-s.o: maeffplugin.cxx ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(INCDIR) $(DEF)"VMDPLUGIN=molfile_maeffplugin" -c $< $(COPTO)$@

${ARCHDIR}/netcdfplugin-s.o: netcdfplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(NETCDFINC) $(INCDIR) $(DEF)"VMDPLUGIN=molfile_netcdfplugin" -c $< $(COPTO)$@

${ARCHDIR}/tngplugin-s.o: tngplugin.C ${PLUGINAPI}
	${CXX} ${SCXXFLAGS} $(TNGINC) $(INCDIR) $(DEF)"VMDPLUGIN=molfile_tngplugin" -c $< $(COPTO)$@

${ARCHDIR}/vtfplugin-s.o: vtfplugin.c ${PLUGINAPI}
	${CC} ${SCCFLAGS} $(INCDIR) $(TCLINC) $(DEF)"VMDPLUGIN=molfile_vtfplugin" -c $< $(COPTO)$@

${ARCHDIR}/webpdbplugin-s.o: webpdbplugin.c readpdb.h ${PLUGINAPI} periodic_table.h
	${CC} ${SCCFLAGS} $(INCDIR) $(TCLINC) $(DEF)"VMDPLUGIN=molfile_webpdbplugin" -c $< $(COPTO)$@


${ARCHDIR} :
	mkdir -p ${ARCHDIR}

clean:
	find ${COMPILEDIR} \( -name *.o -o -name *.a -o -name *.so -o -name *.dll \) -print | xargs rm -f

