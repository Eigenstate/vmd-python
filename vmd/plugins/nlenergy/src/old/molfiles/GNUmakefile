# GNUmakefile
# definitions for building "molfiles"

TOP = ../..
NAME = molfiles
TYPE = lib
OBJ = xplor charmm psf pdbatom bincoord config
LIB = moltypes nlbase m
LINKCPP = no
HEADER = molfiles xplor charmm psf pdbatom bincoord config
DEMO =
DEMOOBJ =
DEMOLIB = molfiles moltypes nlbase
DEMOLINKCPP =

ifeq (0,$(MAKELEVEL))
.PHONY : src clean
src clean : ; $(MAKE) -C $(TOP) SRC="$(NAME)" $@
endif
