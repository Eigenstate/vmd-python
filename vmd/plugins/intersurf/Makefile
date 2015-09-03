
VMFILES = color_func.tcl intersurf_gui.tcl nicostuf.tcl pkgIndex.tcl license.txt
VMVERSION = 1.1
DIR = $(PLUGINDIR)/noarch/tcl/intersurf$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying intersurf $(VMVERSION) files to $(DIR)"
	for localname in `find bin -name "intersurf_*" -print` ; do \
		archname=`echo $$localname | sed s/bin\\\/intersurf_// | sed s/\\\.so// | sed s/\\\.dll// `; \
		dir=$(PLUGINDIR)/$$archname/tcl/intersurf$(VMVERSION); \
		mkdir -p $$dir; \
		mkdir -p $$dir/bin; \
		cp $(VMFILES) $$dir; \
		case $$localname in \
		  'bin/intersurf_WIN32.dll' ) cp $$localname $$dir/bin/intersurf.dll;; \
		  *) cp $$localname $$dir/bin/intersurf.so;; \
		esac; \
        done

