# hello emacs this is a -*- makefile -*-
# $Id: Makefile,v 1.1 2006/03/10 22:48:47 johns Exp $
########################################################################
ARCH           = LINUX
ARCHDIR        = ../../compile/lib_$(ARCH)/molfile
#FC            = gfortran
FC             = g77 #-fno-second-underscore
CC             = gcc
LD             = $(FC)
OPT            = -O2 -Wall
CPPFLAGS       = -I../../include -I$(ARCHDIR) -D_F77_F2C_UNDERSCORE
CFLAGS         = -c $(CPPFLAGS) $(OPT)
FFLAGS         = -c $(OPT)
NETCDFLIB      = -L../../../vmd/lib/netcdf/lib_LINUX
NETCDFLDFLAGS  = -lnetcdf
TCLLIB         = -L../../../vmd/lib/tcl/lib_LINUX
TCLLDFLAGS     = -ltcl8.4
LDFLAGS        = -L$(ARCHDIR) $(TCLLIB) $(NETCDFLIB)
LDLIBS         =  -lmolfile_plugin $(NETCDFLDFLAGS) $(TCLLDFLAGS) -lstdc++ -ldl
########################################################################
OBJECTS  = tester.o f77_molfile.o 

default: tester

tester: $(OBJECTS)
	$(FC) -o $@ $(LDFLAGS) $^ $(LDLIBS)

clean:
	rm -f tester $(OBJECTS)
########################################################################
# pattern rules
.SUFFIXES:
.SUFFIXES: .c .f .o

.c.o:
	$(CC) $(CFLAGS) $< -o $@

.f.o:
	$(FC) $(FFLAGS) $< -o $@

# dependencies
tester.o     : tester.f
f77_molfile.o: f77_molfile.c
