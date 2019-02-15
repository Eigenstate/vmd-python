##
## Plugin compilation and distribution rules
## Copyright 2002-2011, Board of Trustees of the University of Illinois
## Theoretical and Computational Biophysics Group
##
## $Id: Makefile,v 1.142 2018/02/22 22:21:55 johns Exp $
## 
## Parallel builds break because of molfile_plugin

.NOTPARALLEL:

.SILENT:

default: make-arch-help

world:
	@echo "Building all supported targets..."
	csh -f build.csh

# These plugins were, at one time, kept in a separate multiseq CVS tree and 
# need to be periodically synced up with the main VMD plugin CVS when updated.
# They are listed separately here to make this distinction more transparent.
MSEQBUILDDIRS = blast clustalw libbiokit \
                multiseq multiseqdialog phylotree \
                psipred seqdata seqedit stamp

# When building from the independent multiseq tree, we'll use an empty list
# since the dependency will be dealt with manually.
#MSEQBUILDDIRS = 

# The Intersurf plugin is compiled manually and should only be
# listed if the binaries are up-to-date for the target platform(s)
#INTERSURF = intersurf
INTERSURF = 

# List of plugins to build
#   Note: molfile plugins must be first since some of the 
#         other plugins directly depend on them
#
# Removed these since they don't exist?
#bfeestimator
#  modelmaker
# structurecheck
BUILDDIRS = \
  apbsrun \
  atomedit \
  autoimd \
  autoionize \
  autopsf \
  bdtk \
  bendix \
  bignum \
  biocore \
  bossconvert \
  catdcd \
  cgtools \
  chirality \
  cionize \
  cispeptide \
  cliptool \
  clonerep \
  colorscalebar \
  contactmap \
  dataimport \
  demomaster \
  dipwatch \
  dowser \
  exectool \
  extendedpdb \
  fftk \
  gofrgui \
  heatmapper \
  hbonds \
  hesstrans \
  idatm \
  ilstools \
  imdmenu \
  inorganicbuilder \
  irspecgui \
  mafft \
  mdff \
  membrane \
  mergestructs \
  molefacture \
  moltoptools \
  multimolanim \
  multiplot \
  multitext \
  mutator \
  namdenergy \
  namdgui \
  namdplot \
  namdserver \
  nanotube \
  navfly \
  navigate \
  networkview \
  nmwiz \
  optimization \
  palettetool \
  paratool \
  parsefep \
  pbctools \
  pdbtool \
  plumed \
  pmepot \
  propka \
  psfgen \
  qmtool \
  ramaplot \
  readcharmmpar \
  readcharmmtop \
  remote \
  resptool \
  rnaview \
  rmsd \
  rmsdtt \
  rmsdvt \
  ruler \
  runante \
  runsqm \
  saltbr \
  signalproc \
  solvate \
  ssrestraints \
  stingtool \
  symmetrytool \
  timeline \
  topotools \
  torsionplot \
  trunctraj \
  utilities \
  vdna \
  viewchangerender \
  viewmaster \
  volmapgui \
  volutil \
  vmddebug \
  vmdlite \
  vmdmovie \
  vmdprefs \
  vmdtkcon \
  zoomseq \
  $(MSEQBUILDDIRS) \
  $(INTERSURF)

INCLUDEDIR = include
PLUGINHEADERS = vmdplugin.h molfile_plugin.h vmdconio.h

distrib: 
	@echo "Populating distribution directory with compiled plugins"
	-mkdir -p $(PLUGINDIR)/include
	for header in $(PLUGINHEADERS); do cp -p $(INCLUDEDIR)/$$header $(PLUGINDIR)/include || exit 1 ; done
	for dir in molfile_plugin $(BUILDDIRS); do cd $$dir && $(MAKE) distrib && cd .. || exit 1 ; done

include Make-arch

molfilelibs:
	cd molfile_plugin; $(MAKE) dynlibs && cd .. || exit 1
	cd molfile_plugin; $(MAKE) staticlibs && cd .. || exit 1

win32molfilelibs:
	cd molfile_plugin; $(MAKE) dynlibs && cd .. || exit 1
	cd molfile_plugin; $(MAKE) win32staticlibs && cd .. || exit 1

dynlibs: molfilelibs
	for dir in $(BUILDDIRS); do cd $$dir && $(MAKE) dynlibs && cd .. || exit 1 ; done

staticlibs: molfilelibs
	for dir in $(BUILDDIRS); do cd $$dir && $(MAKE) staticlibs && cd .. || exit 1 ; done

win32staticlibs: win32molfilelibs
	for dir in $(BUILDDIRS); do cd $$dir && $(MAKE) win32staticlibs && cd .. || exit 1 ; done

bins:
	for dir in molfile_plugin $(BUILDDIRS); do cd $$dir && $(MAKE) bins && cd .. || exit 1 ; done

win32bins:
	for dir in molfile_plugin $(BUILDDIRS); do cd $$dir && $(MAKE) win32bins && cd .. || exit 1 ; done

clean:
	find compile \( -name \*.o -o -name \*.a -o -name \*.so -o -name \*.exp -o -name \*.lib -o -name \*.h \) -print | xargs rm -f
	find compile \( -name lib_\* \) -print | xargs rm -rf
	rm -f log.*
	rm -f psfgen/so_locations
	rm -f molfile_plugin/so_locations

checkperms:
	@echo "Looking for bad file permissions..."
	find . ! -perm +004

