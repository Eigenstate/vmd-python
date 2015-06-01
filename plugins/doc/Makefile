#
# Plugin documentation Makefile
#
.SILENT:

ECHO            = echo
DELETEDIR       = rm -rf
DOXYGEN         = doxygen

default:
	$(ECHO) "Please choose one of the following forms of documentation:"
	$(ECHO) "----------------------------------------------------------"
	$(ECHO) "Doxygen Programmer Guides:"
	$(ECHO) "  all      -- Build all doxygen documentation   "

all:
	$(MAKE) doxygen;

#
# Auto-generated source code documentation using "doxygen"
#
doxygen:
	$(DELETEDIR) dox
	$(DOXYGEN) Doxyfile

