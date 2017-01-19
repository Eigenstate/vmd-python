# GNUmakefile
# definitions for building "moltypes"

TOP = ../..
NAME = moltypes
TYPE = lib
OBJ = forceprm forceprm_init forceprm_nonb topology topology_init \
      domain coord simprm energy exclude
LIB = nlbase m
LINKCPP = no
HEADER = moltypes forceprm topology domain coord simprm energy
DEMO =
DEMOOBJ =
DEMOLIB =
DEMOLINKCPP =

ifeq (0,$(MAKELEVEL))
.PHONY : src clean
src clean : ; $(MAKE) -C $(TOP) SRC="$(NAME)" $@
endif
