# GNUmakefile
# definitions for building "force"

TOP = ../..
NAME = force
TYPE = lib
OBJ = fbonded angle bond dihedral improper fnbcut fnbcut_eval nbpair
LIB = moltypes nlbase m
LINKCPP = no
HEADER = force fbonded fnbcut nbpair
DEMO =
DEMOOBJ =
DEMOLIB =
DEMOLINKCPP =

ifeq (0,$(MAKELEVEL))
.PHONY : src clean
src clean : ; $(MAKE) -C $(TOP) SRC="$(NAME)" $@
endif
