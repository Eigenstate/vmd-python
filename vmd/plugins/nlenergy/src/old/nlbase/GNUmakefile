# GNUmakefile
# definitions for building "nlbase"

TOP = ../..
NAME = nlbase
TYPE = lib
OBJ = error io mem array strmap arrmap idlist string random sort numeric
LIB = m
LINKCPP = no
HEADER = nlbase \
	 error io mem array strmap arrmap idlist string random sort numeric
DEMO = test_io test_mem test_array test_strmap test_strmap2 test_arrmap \
       test_idlist test_sort
DEMOOBJ =
DEMOLIB = nlbase
DEMOLINKCPP =

ifeq (0,$(MAKELEVEL))
.PHONY : src clean
src clean : ; $(MAKE) -C $(TOP) SRC="$(NAME)" $@
endif
