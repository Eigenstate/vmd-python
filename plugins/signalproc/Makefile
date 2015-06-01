# Makefile for signalproc library

.SUFFIXES: 

AR= ar
ARFLAGS = cr
RANLIB = ranlib

PLUGINNAME=signalproc
VERSION=1.1

COMPILEDIR = ../compile
ARCHDIR=$(COMPILEDIR)/lib_$(ARCH)/tcl/$(PLUGINNAME)$(VERSION)
BINFILES=$(ARCHDIR)/fftcmds.so $(ARCHDIR)/sgsmooth.so $(ARCHDIR)/specden.so
TCLFILES=pkgIndex.tcl data_io_lib.tcl fftpack.tcl signalproc.tcl

SRCDIR=src
INCDIR=-Isrc

VPATH = src $(ARCHDIR)

##
## Only build fftcmds if we have a Tcl library 
##
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
TARGETS = $(ARCHDIR) $(BINFILES)
endif
endif
endif

bins:
win32bins:
dynlibs: $(TARGETS)
staticlibs:
win32staticlibs:

distrib:
	for localname in `find ../compile -name specden.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		cp $(TCLFILES) $$dir ; \
	done
	for localname in `find ../compile -name fftcmds.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
	done
	for localname in `find ../compile -name sgsmooth.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
	done


$(ARCHDIR):
	mkdir -p $(ARCHDIR)


FFTCMDSOBJS = $(ARCHDIR)/kiss_fft.o $(ARCHDIR)/tcl_fftcmds.o

$(ARCHDIR)/fftcmds.so : $(FFTCMDSOBJS)
	if [ -n "$(TCLSHLD)" ]; \
	then $(TCLSHLD) $(LOPTO)$@ $(FFTCMDSOBJS) $(TCLLIB) $(TCLLDFLAGS) $(LDFLAGS); \
	else $(SHLD) $(LOPTO)$@ $(FFTCMDSOBJS) $(TCLLIB) $(TCLLDFLAGS) $(LDFLAGS); \
	fi

$(ARCHDIR)/tcl_fftcmds.o: tcl_fftcmds.c kiss_fft.h openmp-util.h
	$(CC) $(CCFLAGS) $(TCLINC) $(INCDIR) -D_$(ARCH) -DFFTCMDSTCLDLL_EXPORTS -c $< $(COPTO)$@

$(ARCHDIR)/kiss_fft.o: kiss_fft.c kiss_fft.h
	$(CC) $(CCFLAGS) $(INCDIR) -c $< $(COPTO)$@



SPECDENOBJS = $(ARCHDIR)/specden.o $(ARCHDIR)/tcl_specden.o

$(ARCHDIR)/specden.so : $(SPECDENOBJS)
	if [ -n "$(TCLSHLD)" ]; \
	then $(TCLSHLD) $(LOPTO)$@ $(SPECDENOBJS) $(TCLLIB) $(TCLLDFLAGS) $(LDFLAGS) $(CXXLDFLAGS) ; \
	else $(SHLD) $(LOPTO)$@ $(SPECDENOBJS) $(TCLLIB) $(TCLLDFLAGS) $(CXXLDFLAGS) $(LDFLAGS); \
	fi

$(ARCHDIR)/tcl_specden.o: tcl_specden.c specden.h openmp-util.h
	$(CC) $(CCFLAGS) $(TCLINC) $(INCDIR) -D_$(ARCH) -DSPECDENTCLDLL_EXPORTS -c $< $(COPTO)$@

$(ARCHDIR)/specden.o: specden.c specden.h
	$(CC) $(CCFLAGS) $(INCDIR) -c $< $(COPTO)$@



SGSMOOTHOBJS = $(ARCHDIR)/sgsmooth.o $(ARCHDIR)/tcl_sgsmooth.o

$(ARCHDIR)/sgsmooth.so : $(SGSMOOTHOBJS)
	if [ -n "$(TCLSHLD)" ]; \
	then $(TCLSHLD) $(LOPTO)$@ $(SGSMOOTHOBJS) $(TCLLIB) $(TCLLDFLAGS) $(LDFLAGS) $(CXXLDFLAGS) ; \
	else $(SHLD) $(LOPTO)$@ $(SGSMOOTHOBJS) $(TCLLIB) $(TCLLDFLAGS) $(CXXLDFLAGS) $(LDFLAGS); \
	fi

$(ARCHDIR)/tcl_sgsmooth.o: tcl_sgsmooth.c sgsmooth.h openmp-util.h
	$(CC) $(CCFLAGS) $(TCLINC) $(INCDIR) -D_$(ARCH) -DSGSMOOTHTCLDLL_EXPORTS -c $< $(COPTO)$@

$(ARCHDIR)/sgsmooth.o: sgsmooth.C sgsmooth.h
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< $(COPTO)$@
