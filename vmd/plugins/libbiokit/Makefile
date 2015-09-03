# Makefile for libbiokit

.SILENT:

.SUFFIXES: 

TCLFILES = libbiokit.tcl libbiokit_entropy.tcl libbiokit_profcombine.tcl metric.tcl pkgIndex.tcl
PKGVERSION = 1.1
PKGNAME = libbiokit


AR= ar
ARFLAGS = cr
RANLIB = ranlib

COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/${PKGNAME}${PKGVERSION}
SRCDIR=src/libbiokit
UTILSRCDIR=src/utils
INTERFACESRCDIR=src/interface
INCDIR=-Isrc/libbiokit
CXXFLAGS= -g -O0 
VPATH = src ${ARCHDIR}

# Only define tagets for dynlib if the TCL libraries are available.
ifdef TCLLIB
ifdef TCLINC
ifdef TCLLDFLAGS
DYNLIBTARGETS = ${ARCHDIR} ${ARCHDIR}/tcl_libbiokit.so
endif
endif
endif

#
# Rules
#
#${ARCHDIR}/percid ${ARCHDIR}/qpair ${ARCHDIR}/rmsd ${ARCHDIR}/structqr
bins: ${ARCHDIR} ${ARCHDIR}/q ${ARCHDIR}/seqqr ${ARCHDIR}/structqr \
${ARCHDIR}/qpair ${ARCHDIR}/rmsd ${ARCHDIR}/percid
win32bins: ${ARCHDIR} ${ARCHDIR}/percid.exe ${ARCHDIR}/q.exe ${ARCHDIR}/qpair.exe ${ARCHDIR}/rmsd.exe ${ARCHDIR}/seqqr.exe ${ARCHDIR}/structqr.exe
dynlibs: ${DYNLIBTARGETS}
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying ${PKGNAME} $(PKGVERSION) files."
	for localname in `find ../compile -name tcl_libbiokit.so -o -name percid -o -name percid.exe -o -name q -o -name q.exe -o -name qpair -o -name qpair.exe -o -name rmsd -o -name rmsd.exe -o -name seqqr -o -name seqqr.exe -o -name structqr -o -name structqr.exe` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		echo "  $$localname -> $$dir"; \
		cp $$localname $$dir; \
        cp $(TCLFILES) $$dir; \
	done
    
${ARCHDIR}:
	mkdir -p ${ARCHDIR}

DYNLIBOBJS = \
    ${ARCHDIR}/alignedSequence.o \
	${ARCHDIR}/alignedStructure.o \
    ${ARCHDIR}/alphabet.o \
	${ARCHDIR}/alphabetBuilder.o \
	${ARCHDIR}/aminoAcidQR.o \
    ${ARCHDIR}/Atom.o \
    ${ARCHDIR}/AtomList.o \
	${ARCHDIR}/binaryQR.o \
    ${ARCHDIR}/Contact.o \
    ${ARCHDIR}/ContactList.o \
    ${ARCHDIR}/ContactTools.o \
	${ARCHDIR}/coordinate3D.o \
	${ARCHDIR}/fastaReader.o \
	${ARCHDIR}/fastaWriter.o \
	${ARCHDIR}/PointerList.o \
	${ARCHDIR}/PIDTools.o \
	${ARCHDIR}/qTools.o \
	${ARCHDIR}/Residue.o \
   ${ARCHDIR}/rnaQR.o \
	${ARCHDIR}/sequence.o \
	${ARCHDIR}/sequenceAlignment.o \
	${ARCHDIR}/structureAlignment.o \
	${ARCHDIR}/sequenceQR.o \
	${ARCHDIR}/ShortIntList.o \
	${ARCHDIR}/structure.o \
	${ARCHDIR}/symbol.o \
	${ARCHDIR}/symbolList.o \
	${ARCHDIR}/typeConvert.o
    
OBJS = ${DYNLIBOBJS} \
	${ARCHDIR}/pdbReader.o \
	${ARCHDIR}/structureAlignmentReader.o \
	${ARCHDIR}/structureQR.o \
	${ARCHDIR}/rmsdTools.o 

#	${ARCHDIR}/alnReader.o \

# tcl_libbiokit.so
${ARCHDIR}/tcl_libbiokit.so : ${ARCHDIR}/tcl_libbiokit_wrap.o ${ARCHDIR}/tcl_libbiokit.o ${DYNLIBOBJS}
	if [ -n "${TCLSHLD}" ]; \
	then ${TCLSHLD} $(LOPTO)$@ ${ARCHDIR}/tcl_libbiokit_wrap.o ${ARCHDIR}/tcl_libbiokit.o ${DYNLIBOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	else ${SHLD} $(LOPTO)$@ ${ARCHDIR}/tcl_libbiokit_wrap.o ${ARCHDIR}/tcl_libbiokit.o ${DYNLIBOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	fi
${ARCHDIR}/tcl_libbiokit_wrap.o : ${INTERFACESRCDIR}/tcl_libbiokit_wrap.cpp
	${CXX} ${CXXFLAGS} ${TCLINC} ${INCDIR} -c ${INTERFACESRCDIR}/tcl_libbiokit_wrap.cpp $(COPTO)${ARCHDIR}/tcl_libbiokit_wrap.o
${ARCHDIR}/tcl_libbiokit.o: ${INTERFACESRCDIR}/tcl_libbiokit.cpp ${INTERFACESRCDIR}/tcl_libbiokit.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${INTERFACESRCDIR}/tcl_libbiokit.cpp $(COPTO)${ARCHDIR}/tcl_libbiokit.o
${INTERFACESRCDIR}/tcl_libbiokit_wrap.cpp : ${INTERFACESRCDIR}/tcl_libbiokit.i
	swig -tcl -c++ -o ${INTERFACESRCDIR}/tcl_libbiokit_wrap.cpp ${INTERFACESRCDIR}/tcl_libbiokit.i

# percid
${ARCHDIR}/percid.exe: ${ARCHDIR}/percid.o ${OBJS}
	LINK ${ARCHDIR}/percid.o ${OBJS} /OUT:${ARCHDIR}/percid.exe
${ARCHDIR}/percid: ${ARCHDIR}/percid.o ${OBJS}
	${CXX} ${CXXFLAGS} ${ARCHDIR}/percid.o ${OBJS} -o ${ARCHDIR}/percid -lm
${ARCHDIR}/percid.o: ${UTILSRCDIR}/percid.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${UTILSRCDIR}/percid.cpp $(COPTO)${ARCHDIR}/percid.o

# qpair
${ARCHDIR}/qpair.exe: ${ARCHDIR}/qPair.o ${OBJS}
	LINK ${ARCHDIR}/qPair.o ${OBJS} /OUT:${ARCHDIR}/qpair.exe
${ARCHDIR}/qpair: ${ARCHDIR}/qPair.o ${OBJS}
	${CXX} ${CXXFLAGS} ${ARCHDIR}/qPair.o ${OBJS} -o ${ARCHDIR}/qpair -lm
${ARCHDIR}/qPair.o: ${UTILSRCDIR}/qPair.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${UTILSRCDIR}/qPair.cpp $(COPTO)${ARCHDIR}/qPair.o

# rmsd
${ARCHDIR}/rmsd.exe: ${ARCHDIR}/rmsd.o ${OBJS}
	LINK ${ARCHDIR}/rmsd.o ${OBJS} /OUT:${ARCHDIR}/rmsd.exe
${ARCHDIR}/rmsd: ${ARCHDIR}/rmsd.o ${OBJS}
	${CXX} ${CXXFLAGS} ${ARCHDIR}/rmsd.o ${OBJS} -o ${ARCHDIR}/rmsd -lm
${ARCHDIR}/rmsd.o: ${UTILSRCDIR}/rmsd.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${UTILSRCDIR}/rmsd.cpp $(COPTO)${ARCHDIR}/rmsd.o

# q
${ARCHDIR}/q.exe: ${ARCHDIR}/Q.o ${OBJS}
	LINK ${ARCHDIR}/Q.o ${OBJS} /OUT:${ARCHDIR}/q.exe
${ARCHDIR}/q: ${ARCHDIR}/Q.o ${OBJS}
	${CXX} ${CXXFLAGS} ${ARCHDIR}/Q.o ${OBJS} -o ${ARCHDIR}/q -lm
${ARCHDIR}/Q.o: ${UTILSRCDIR}/Q.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${UTILSRCDIR}/Q.cpp $(COPTO)${ARCHDIR}/Q.o

# seqqr
${ARCHDIR}/seqqr.exe: ${ARCHDIR}/seqqr.o ${OBJS}
	LINK ${ARCHDIR}/seqqr.o ${OBJS} /OUT:${ARCHDIR}/seqqr.exe
${ARCHDIR}/seqqr: ${ARCHDIR}/seqqr.o ${OBJS}
	${CXX} ${CXXFLAGS} ${ARCHDIR}/seqqr.o ${OBJS} -o ${ARCHDIR}/seqqr -lm
${ARCHDIR}/seqqr.o: ${UTILSRCDIR}/seqqr.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${UTILSRCDIR}/seqqr.cpp $(COPTO)${ARCHDIR}/seqqr.o

# structqr
${ARCHDIR}/structqr.exe : ${ARCHDIR}/structqr.o ${OBJS}
	LINK ${ARCHDIR}/structqr.o ${OBJS} /OUT:${ARCHDIR}/structqr.exe
${ARCHDIR}/structqr: ${ARCHDIR}/structqr.o ${OBJS}
	${CXX} ${CXXFLAGS} ${ARCHDIR}/structqr.o ${OBJS} -o ${ARCHDIR}/structqr -lm
${ARCHDIR}/structqr.o: ${UTILSRCDIR}/structqr.cpp
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${UTILSRCDIR}/structqr.cpp $(COPTO)${ARCHDIR}/structqr.o
    
# Library objects.
${ARCHDIR}/alignedSequence.o: ${SRCDIR}/alignedSequence.cpp ${SRCDIR}/alignedSequence.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/alignedSequence.cpp $(COPTO)${ARCHDIR}/alignedSequence.o
${ARCHDIR}/alignedStructure.o: ${SRCDIR}/alignedStructure.cpp ${SRCDIR}/alignedStructure.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/alignedStructure.cpp $(COPTO)${ARCHDIR}/alignedStructure.o
${ARCHDIR}/alnReader.o: ${SRCDIR}/alnReader.cpp ${SRCDIR}/alnReader.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/alnReader.cpp $(COPTO)${ARCHDIR}/alnReader.o
${ARCHDIR}/alphabet.o: ${SRCDIR}/alphabet.cpp ${SRCDIR}/alphabet.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/alphabet.cpp $(COPTO)${ARCHDIR}/alphabet.o
${ARCHDIR}/alphabetBuilder.o: ${SRCDIR}/alphabetBuilder.cpp ${SRCDIR}/alphabetBuilder.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/alphabetBuilder.cpp $(COPTO)${ARCHDIR}/alphabetBuilder.o
${ARCHDIR}/aminoAcidQR.o: ${SRCDIR}/aminoAcidQR.cpp ${SRCDIR}/aminoAcidQR.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/aminoAcidQR.cpp $(COPTO)${ARCHDIR}/aminoAcidQR.o
${ARCHDIR}/Atom.o: ${SRCDIR}/Atom.cpp ${SRCDIR}/Atom.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/Atom.cpp $(COPTO)${ARCHDIR}/Atom.o
${ARCHDIR}/AtomList.o: ${SRCDIR}/AtomList.cpp ${SRCDIR}/AtomList.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/AtomList.cpp $(COPTO)${ARCHDIR}/AtomList.o
${ARCHDIR}/binaryQR.o: ${SRCDIR}/binaryQR.cpp ${SRCDIR}/binaryQR.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/binaryQR.cpp $(COPTO)${ARCHDIR}/binaryQR.o
${ARCHDIR}/Contact.o: ${SRCDIR}/Contact.cpp ${SRCDIR}/Contact.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/Contact.cpp $(COPTO)${ARCHDIR}/Contact.o
${ARCHDIR}/ContactList.o: ${SRCDIR}/ContactList.cpp ${SRCDIR}/ContactList.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/ContactList.cpp $(COPTO)${ARCHDIR}/ContactList.o
${ARCHDIR}/ContactTools.o: ${SRCDIR}/ContactTools.cpp ${SRCDIR}/ContactTools.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/ContactTools.cpp $(COPTO)${ARCHDIR}/ContactTools.o
${ARCHDIR}/coordinate3D.o: ${SRCDIR}/coordinate3D.cpp ${SRCDIR}/coordinate3D.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/coordinate3D.cpp $(COPTO)${ARCHDIR}/coordinate3D.o
${ARCHDIR}/fastaReader.o: ${SRCDIR}/fastaReader.cpp ${SRCDIR}/fastaReader.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/fastaReader.cpp $(COPTO)${ARCHDIR}/fastaReader.o
${ARCHDIR}/fastaWriter.o: ${SRCDIR}/fastaWriter.cpp ${SRCDIR}/fastaWriter.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/fastaWriter.cpp $(COPTO)${ARCHDIR}/fastaWriter.o
${ARCHDIR}/pdbReader.o: ${SRCDIR}/pdbReader.cpp ${SRCDIR}/pdbReader.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/pdbReader.cpp $(COPTO)${ARCHDIR}/pdbReader.o
${ARCHDIR}/PIDTools.o: ${SRCDIR}/PIDTools.cpp ${SRCDIR}/PIDTools.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/PIDTools.cpp $(COPTO)${ARCHDIR}/PIDTools.o
${ARCHDIR}/PointerList.o: ${SRCDIR}/PointerList.cpp ${SRCDIR}/PointerList.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/PointerList.cpp $(COPTO)${ARCHDIR}/PointerList.o
${ARCHDIR}/qTools.o: ${SRCDIR}/qTools.cpp ${SRCDIR}/qTools.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/qTools.cpp $(COPTO)${ARCHDIR}/qTools.o
${ARCHDIR}/Residue.o: ${SRCDIR}/Residue.cpp ${SRCDIR}/Residue.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/Residue.cpp $(COPTO)${ARCHDIR}/Residue.o
${ARCHDIR}/rmsdTools.o: ${SRCDIR}/rmsdTools.cpp ${SRCDIR}/rmsdTools.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/rmsdTools.cpp $(COPTO)${ARCHDIR}/rmsdTools.o
${ARCHDIR}/rnaQR.o: ${SRCDIR}/rnaQR.cpp ${SRCDIR}/rnaQR.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/rnaQR.cpp $(COPTO)${ARCHDIR}/rnaQR.o
${ARCHDIR}/ShortIntList.o: ${SRCDIR}/ShortIntList.cpp ${SRCDIR}/ShortIntList.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/ShortIntList.cpp $(COPTO)${ARCHDIR}/ShortIntList.o
${ARCHDIR}/sequence.o: ${SRCDIR}/sequence.cpp ${SRCDIR}/sequence.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/sequence.cpp $(COPTO)${ARCHDIR}/sequence.o
${ARCHDIR}/sequenceAlignment.o: ${SRCDIR}/sequenceAlignment.cpp ${SRCDIR}/sequenceAlignment.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/sequenceAlignment.cpp $(COPTO)${ARCHDIR}/sequenceAlignment.o
${ARCHDIR}/sequenceQR.o: ${SRCDIR}/sequenceQR.cpp ${SRCDIR}/sequenceQR.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/sequenceQR.cpp $(COPTO)${ARCHDIR}/sequenceQR.o
${ARCHDIR}/structure.o: ${SRCDIR}/structure.cpp ${SRCDIR}/structure.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/structure.cpp $(COPTO)${ARCHDIR}/structure.o
${ARCHDIR}/structureAlignment.o: ${SRCDIR}/structureAlignment.cpp ${SRCDIR}/structureAlignment.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/structureAlignment.cpp $(COPTO)${ARCHDIR}/structureAlignment.o
${ARCHDIR}/structureAlignmentReader.o: ${SRCDIR}/structureAlignmentReader.cpp ${SRCDIR}/structureAlignmentReader.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/structureAlignmentReader.cpp $(COPTO)${ARCHDIR}/structureAlignmentReader.o
${ARCHDIR}/structureQR.o: ${SRCDIR}/structureQR.cpp ${SRCDIR}/structureQR.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/structureQR.cpp $(COPTO)${ARCHDIR}/structureQR.o
${ARCHDIR}/symbol.o: ${SRCDIR}/symbol.cpp ${SRCDIR}/symbol.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/symbol.cpp $(COPTO)${ARCHDIR}/symbol.o
${ARCHDIR}/symbolList.o: ${SRCDIR}/symbolList.cpp ${SRCDIR}/symbolList.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/symbolList.cpp $(COPTO)${ARCHDIR}/symbolList.o
${ARCHDIR}/typeConvert.o: ${SRCDIR}/typeConvert.cpp ${SRCDIR}/typeConvert.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/typeConvert.cpp $(COPTO)${ARCHDIR}/typeConvert.o

