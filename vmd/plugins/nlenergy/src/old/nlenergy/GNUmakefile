# GNUmakefile
# definitions for building "nlenergy"

TOP = ../..
NAME = nlenergy
TYPE = plugin
OBJ = tcl_nlenergy nlenergy parse get set add remove fileio eval tclwrap
LIB = force molfiles moltypes nlbase $(darwin_tcl) m
LINKCPP =
HEADER =
DEMO =
DEMOOBJ =
DEMOLIB =
DEMOLINKCPP =

ifeq (0,$(MAKELEVEL))
.PHONY : src clean
src clean : ; $(MAKE) -C $(TOP) SRC="$(NAME)" $@
endif

CPPFLAGS := -I/Projects/vmd/vmd/lib/tcl/include $(CPPFLAGS)
