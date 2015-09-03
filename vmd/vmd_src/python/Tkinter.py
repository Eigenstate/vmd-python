############################################################################
#cr
#cr            (C) Copyright 1995-2007 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: Tkinter.py,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.4 $        $Date: 2007/01/12 20:19:10 $
#
############################################################################

"""Wrapper functions for Tcl/Tk.

Tkinter provides classes which allow the display, positioning and
control of widgets. Toplevel widgets are Tk and Toplevel. Other
widgets are Frame, Label, Entry, Text, Canvas, Button, Radiobutton,
Checkbutton, Scale, Listbox, Scrollbar, OptionMenu. Properties of the
widgets are specified with keyword arguments.  Keyword arguments have
the same name as the corresponding resource under Tk.

Widgets are positioned with one of the geometry managers Place, Pack
or Grid. These managers can be called with methods place, pack, grid
available in every Widget.

Actions are bound to events by resources (e.g. keyword argument
command) or with the method bind.

Example (Hello, World):
import Tkinter
from Tkconstants import *
tk = Tkinter.Tk()
frame = Tkinter.Frame(tk, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
label = Tkinter.Label(frame, text="Hello, World")
label.pack(fill=X, expand=1)
button = Tkinter.Button(frame,text="Exit",command=tk.destroy)
button.pack(side=BOTTOM)
tk.mainloop()
"""

"""
This is Tkinter.py from Python 2.2.3.  Older versions of Tkinter do not
work with Tk version 8.4.2.  In order to work with newer versions of Tk,
we need to provide our own working version of Tkinter.py rather than relying
on whatever version is installed on the system.
"""

__version__ = "$Revision: 1.4 $"

import sys
if sys.platform == "win32":
    import FixTk # Attempt to configure Tcl/Tk without requiring PATH
import _tkinter # If this fails your Python may not be configured for Tk
tkinter = _tkinter # b/w compat for export
TclError = _tkinter.TclError
from types import *
from Tkconstants import *
try:
    import MacOS; _MacOS = MacOS; del MacOS
except ImportError:
    _MacOS = None

TkVersion = float(_tkinter.TK_VERSION)
TclVersion = float(_tkinter.TCL_VERSION)

READABLE = _tkinter.READABLE
WRITABLE = _tkinter.WRITABLE
EXCEPTION = _tkinter.EXCEPTION

# These are not always defined, e.g. not on Win32 with Tk 8.0 :-(
try: _tkinter.createfilehandler
except AttributeError: _tkinter.createfilehandler = None
try: _tkinter.deletefilehandler
except AttributeError: _tkinter.deletefilehandler = None


def _flatten(tuple):
    """Internal function."""
    res = ()
    for item in tuple:
        if type(item) in (TupleType, ListType):
            res = res + _flatten(item)
        elif item is not None:
            res = res + (item,)
    return res

try: _flatten = _tkinter._flatten
except AttributeError: pass

def _cnfmerge(cnfs):
    """Internal function."""
    if type(cnfs) is DictionaryType:
        return cnfs
    elif type(cnfs) in (NoneType, StringType):
        return cnfs
    else:
        cnf = {}
        for c in _flatten(cnfs):
            try:
                cnf.update(c)
            except (AttributeError, TypeError), msg:
                print "_cnfmerge: fallback due to:", msg
                for k, v in c.items():
                    cnf[k] = v
        return cnf

try: _cnfmerge = _tkinter._cnfmerge
except AttributeError: pass

class Event:
    """Container for the properties of an event.

    Instances of this type are generated if one of the following events occurs:

    KeyPress, KeyRelease - for keyboard events
    ButtonPress, ButtonRelease, Motion, Enter, Leave, MouseWheel - for mouse events
    Visibility, Unmap, Map, Expose, FocusIn, FocusOut, Circulate,
    Colormap, Gravity, Reparent, Property, Destroy, Activate,
    Deactivate - for window events.

    If a callback function for one of these events is registered
    using bind, bind_all, bind_class, or tag_bind, the callback is
    called with an Event as first argument. It will have the
    following attributes (in braces are the event types for which
    the attribute is valid):

        serial - serial number of event
    num - mouse button pressed (ButtonPress, ButtonRelease)
    focus - whether the window has the focus (Enter, Leave)
    height - height of the exposed window (Configure, Expose)
    width - width of the exposed window (Configure, Expose)
    keycode - keycode of the pressed key (KeyPress, KeyRelease)
    state - state of the event as a number (ButtonPress, ButtonRelease,
                            Enter, KeyPress, KeyRelease,
                            Leave, Motion)
    state - state as a string (Visibility)
    time - when the event occurred
    x - x-position of the mouse
    y - y-position of the mouse
    x_root - x-position of the mouse on the screen
             (ButtonPress, ButtonRelease, KeyPress, KeyRelease, Motion)
    y_root - y-position of the mouse on the screen
             (ButtonPress, ButtonRelease, KeyPress, KeyRelease, Motion)
    char - pressed character (KeyPress, KeyRelease)
    send_event - see X/Windows documentation
    keysym - keysym of the the event as a string (KeyPress, KeyRelease)
    keysym_num - keysym of the event as a number (KeyPress, KeyRelease)
    type - type of the event as a number
    widget - widget in which the event occurred
    delta - delta of wheel movement (MouseWheel)
    """
    pass

_support_default_root = 1
_default_root = None

def NoDefaultRoot():
    """Inhibit setting of default root window.

    Call this function to inhibit that the first instance of
    Tk is used for windows without an explicit parent window.
    """
    global _support_default_root
    _support_default_root = 0
    global _default_root
    _default_root = None
    del _default_root

def _tkerror(err):
    """Internal function."""
    pass

def _exit(code='0'):
    """Internal function. Calling it will throw the exception SystemExit."""
    raise SystemExit, code

_varnum = 0
class Variable:
    """Internal class. Base class to define value holders for e.g. buttons."""
    _default = ""
    def __init__(self, master=None):
        """Construct a variable with an optional MASTER as master widget.
        The variable is named PY_VAR_number in Tcl.
        """
        global _varnum
        if not master:
            master = _default_root
        self._master = master
        self._tk = master.tk
        self._name = 'PY_VAR' + `_varnum`
        _varnum = _varnum + 1
        self.set(self._default)
    def __del__(self):
        """Unset the variable in Tcl."""
        self._tk.globalunsetvar(self._name)
    def __str__(self):
        """Return the name of the variable in Tcl."""
        return self._name
    def set(self, value):
        """Set the variable to VALUE."""
        return self._tk.globalsetvar(self._name, value)
    def trace_variable(self, mode, callback):
        """Define a trace callback for the variable.

        MODE is one of "r", "w", "u" for read, write, undefine.
        CALLBACK must be a function which is called when
        the variable is read, written or undefined.

        Return the name of the callback.
        """
        cbname = self._master._register(callback)
        self._tk.call("trace", "variable", self._name, mode, cbname)
        return cbname
    trace = trace_variable
    def trace_vdelete(self, mode, cbname):
        """Delete the trace callback for a variable.

        MODE is one of "r", "w", "u" for read, write, undefine.
        CBNAME is the name of the callback returned from trace_variable or trace.
        """
        self._tk.call("trace", "vdelete", self._name, mode, cbname)
        self._master.deletecommand(cbname)
    def trace_vinfo(self):
        """Return all trace callback information."""
        return map(self._tk.split, self._tk.splitlist(
            self._tk.call("trace", "vinfo", self._name)))

class StringVar(Variable):
    """Value holder for strings variables."""
    _default = ""
    def __init__(self, master=None):
        """Construct a string variable.

        MASTER can be given as master widget."""
        Variable.__init__(self, master)

    def get(self):
        """Return value of variable as string."""
        return self._tk.globalgetvar(self._name)

class IntVar(Variable):
    """Value holder for integer variables."""
    _default = 0
    def __init__(self, master=None):
        """Construct an integer variable.

        MASTER can be given as master widget."""
        Variable.__init__(self, master)

    def get(self):
        """Return the value of the variable as an integer."""
        return getint(self._tk.globalgetvar(self._name))

class DoubleVar(Variable):
    """Value holder for float variables."""
    _default = 0.0
    def __init__(self, master=None):
        """Construct a float variable.

        MASTER can be given as a master widget."""
        Variable.__init__(self, master)

    def get(self):
        """Return the value of the variable as a float."""
        return getdouble(self._tk.globalgetvar(self._name))

class BooleanVar(Variable):
    """Value holder for boolean variables."""
    _default = "false"
    def __init__(self, master=None):
        """Construct a boolean variable.

        MASTER can be given as a master widget."""
        Variable.__init__(self, master)

    def get(self):
        """Return the value of the variable as 0 or 1."""
        return self._tk.getboolean(self._tk.globalgetvar(self._name))

def mainloop(n=0):
    """Run the main loop of Tcl."""
    _default_root.tk.mainloop(n)

getint = int

getdouble = float

def getboolean(s):
    """Convert true and false to integer values 1 and 0."""
    return _default_root.tk.getboolean(s)

# Methods defined on both toplevel and interior widgets
class Misc:
    """Internal class.

    Base class which defines methods common for interior widgets."""

    # XXX font command?
    _tclCommands = None
    def destroy(self):
        """Internal function.

        Delete all Tcl commands created for
        this widget in the Tcl interpreter."""
        if self._tclCommands is not None:
            for name in self._tclCommands:
                #print '- Tkinter: deleted command', name
                self.tk.deletecommand(name)
            self._tclCommands = None
    def deletecommand(self, name):
        """Internal function.

        Delete the Tcl command provided in NAME."""
        #print '- Tkinter: deleted command', name
        self.tk.deletecommand(name)
        try:
            self._tclCommands.remove(name)
        except ValueError:
            pass
    def tk_strictMotif(self, boolean=None):
        """Set Tcl internal variable, whether the look and feel
        should adhere to Motif.

        A parameter of 1 means adhere to Motif (e.g. no color
        change if mouse passes over slider).
        Returns the set value."""
        return self.tk.getboolean(self.tk.call(
            'set', 'tk_strictMotif', boolean))
    def tk_bisque(self):
        """Change the color scheme to light brown as used in Tk 3.6 and before."""
        self.tk.call('tk_bisque')
    def tk_setPalette(self, *args, **kw):
        """Set a new color scheme for all widget elements.

        A single color as argument will cause that all colors of Tk
        widget elements are derived from this.
        Alternatively several keyword parameters and its associated
        colors can be given. The following keywords are valid:
        activeBackground, foreground, selectColor,
        activeForeground, highlightBackground, selectBackground,
        background, highlightColor, selectForeground,
        disabledForeground, insertBackground, troughColor."""
        self.tk.call(('tk_setPalette',)
              + _flatten(args) + _flatten(kw.items()))
    def tk_menuBar(self, *args):
        """Do not use. Needed in Tk 3.6 and earlier."""
        pass # obsolete since Tk 4.0
    def wait_variable(self, name='PY_VAR'):
        """Wait until the variable is modified.

        A parameter of type IntVar, StringVar, DoubleVar or
        BooleanVar must be given."""
        self.tk.call('tkwait', 'variable', name)
    waitvar = wait_variable # XXX b/w compat
    def wait_window(self, window=None):
        """Wait until a WIDGET is destroyed.

        If no parameter is given self is used."""
        if window is None:
            window = self
        self.tk.call('tkwait', 'window', window._w)
    def wait_visibility(self, window=None):
        """Wait until the visibility of a WIDGET changes
        (e.g. it appears).

        If no parameter is given self is used."""
        if window is None:
            window = self
        self.tk.call('tkwait', 'visibility', window._w)
    def setvar(self, name='PY_VAR', value='1'):
        """Set Tcl variable NAME to VALUE."""
        self.tk.setvar(name, value)
    def getvar(self, name='PY_VAR'):
        """Return value of Tcl variable NAME."""
        return self.tk.getvar(name)
    getint = int
    getdouble = float
    def getboolean(self, s):
        """Return 0 or 1 for Tcl boolean values true and false given as parameter."""
        return self.tk.getboolean(s)
    def focus_set(self):
        """Direct input focus to this widget.

        If the application currently does not have the focus
        this widget will get the focus if the application gets
        the focus through the window manager."""
        self.tk.call('focus', self._w)
    focus = focus_set # XXX b/w compat?
    def focus_force(self):
        """Direct input focus to this widget even if the
        application does not have the focus. Use with
        caution!"""
        self.tk.call('focus', '-force', self._w)
    def focus_get(self):
        """Return the widget which has currently the focus in the
        application.

        Use focus_displayof to allow working with several
        displays. Return None if application does not have
        the focus."""
        name = self.tk.call('focus')
        if name == 'none' or not name: return None
        return self._nametowidget(name)
    def focus_displayof(self):
        """Return the widget which has currently the focus on the
        display where this widget is located.

        Return None if the application does not have the focus."""
        name = self.tk.call('focus', '-displayof', self._w)
        if name == 'none' or not name: return None
        return self._nametowidget(name)
    def focus_lastfor(self):
        """Return the widget which would have the focus if top level
        for this widget gets the focus from the window manager."""
        name = self.tk.call('focus', '-lastfor', self._w)
        if name == 'none' or not name: return None
        return self._nametowidget(name)
    def tk_focusFollowsMouse(self):
        """The widget under mouse will get automatically focus. Can not
        be disabled easily."""
        self.tk.call('tk_focusFollowsMouse')
    def tk_focusNext(self):
        """Return the next widget in the focus order which follows
        widget which has currently the focus.

        The focus order first goes to the next child, then to
        the children of the child recursively and then to the
        next sibling which is higher in the stacking order.  A
        widget is omitted if it has the takefocus resource set
        to 0."""
        name = self.tk.call('tk_focusNext', self._w)
        if not name: return None
        return self._nametowidget(name)
    def tk_focusPrev(self):
        """Return previous widget in the focus order. See tk_focusNext for details."""
        name = self.tk.call('tk_focusPrev', self._w)
        if not name: return None
        return self._nametowidget(name)
    def after(self, ms, func=None, *args):
        """Call function once after given time.

        MS specifies the time in milliseconds. FUNC gives the
        function which shall be called. Additional parameters
        are given as parameters to the function call.  Return
        identifier to cancel scheduling with after_cancel."""
        if not func:
            # I'd rather use time.sleep(ms*0.001)
            self.tk.call('after', ms)
        else:
            # XXX Disgusting hack to clean up after calling func
            tmp = []
            def callit(func=func, args=args, self=self, tmp=tmp):
                try:
                    apply(func, args)
                finally:
                    try:
                        self.deletecommand(tmp[0])
                    except TclError:
                        pass
            name = self._register(callit)
            tmp.append(name)
            return self.tk.call('after', ms, name)
    def after_idle(self, func, *args):
        """Call FUNC once if the Tcl main loop has no event to
        process.

        Return an identifier to cancel the scheduling with
        after_cancel."""
        return apply(self.after, ('idle', func) + args)
    def after_cancel(self, id):
        """Cancel scheduling of function identified with ID.

        Identifier returned by after or after_idle must be
        given as first parameter."""
        self.tk.call('after', 'cancel', id)
    def bell(self, displayof=0):
        """Ring a display's bell."""
        self.tk.call(('bell',) + self._displayof(displayof))
    # Clipboard handling:
    def clipboard_clear(self, **kw):
        """Clear the data in the Tk clipboard.

        A widget specified for the optional displayof keyword
        argument specifies the target display."""
        if not kw.has_key('displayof'): kw['displayof'] = self._w
        self.tk.call(('clipboard', 'clear') + self._options(kw))
    def clipboard_append(self, string, **kw):
        """Append STRING to the Tk clipboard.

        A widget specified at the optional displayof keyword
        argument specifies the target display. The clipboard
        can be retrieved with selection_get."""
        if not kw.has_key('displayof'): kw['displayof'] = self._w
        self.tk.call(('clipboard', 'append') + self._options(kw)
              + ('--', string))
    # XXX grab current w/o window argument
    def grab_current(self):
        """Return widget which has currently the grab in this application
        or None."""
        name = self.tk.call('grab', 'current', self._w)
        if not name: return None
        return self._nametowidget(name)
    def grab_release(self):
        """Release grab for this widget if currently set."""
        self.tk.call('grab', 'release', self._w)
    def grab_set(self):
        """Set grab for this widget.

        A grab directs all events to this and descendant
        widgets in the application."""
        self.tk.call('grab', 'set', self._w)
    def grab_set_global(self):
        """Set global grab for this widget.

        A global grab directs all events to this and
        descendant widgets on the display. Use with caution -
        other applications do not get events anymore."""
        self.tk.call('grab', 'set', '-global', self._w)
    def grab_status(self):
        """Return None, "local" or "global" if this widget has
        no, a local or a global grab."""
        status = self.tk.call('grab', 'status', self._w)
        if status == 'none': status = None
        return status
    def lower(self, belowThis=None):
        """Lower this widget in the stacking order."""
        self.tk.call('lower', self._w, belowThis)
    def option_add(self, pattern, value, priority = None):
        """Set a VALUE (second parameter) for an option
        PATTERN (first parameter).

        An optional third parameter gives the numeric priority
        (defaults to 80)."""
        self.tk.call('option', 'add', pattern, value, priority)
    def option_clear(self):
        """Clear the option database.

        It will be reloaded if option_add is called."""
        self.tk.call('option', 'clear')
    def option_get(self, name, className):
        """Return the value for an option NAME for this widget
        with CLASSNAME.

        Values with higher priority override lower values."""
        return self.tk.call('option', 'get', self._w, name, className)
    def option_readfile(self, fileName, priority = None):
        """Read file FILENAME into the option database.

        An optional second parameter gives the numeric
        priority."""
        self.tk.call('option', 'readfile', fileName, priority)
    def selection_clear(self, **kw):
        """Clear the current X selection."""
        if not kw.has_key('displayof'): kw['displayof'] = self._w
        self.tk.call(('selection', 'clear') + self._options(kw))
    def selection_get(self, **kw):
        """Return the contents of the current X selection.

        A keyword parameter selection specifies the name of
        the selection and defaults to PRIMARY.  A keyword
        parameter displayof specifies a widget on the display
        to use."""
        if not kw.has_key('displayof'): kw['displayof'] = self._w
        return self.tk.call(('selection', 'get') + self._options(kw))
    def selection_handle(self, command, **kw):
        """Specify a function COMMAND to call if the X
        selection owned by this widget is queried by another
        application.

        This function must return the contents of the
        selection. The function will be called with the
        arguments OFFSET and LENGTH which allows the chunking
        of very long selections. The following keyword
        parameters can be provided:
        selection - name of the selection (default PRIMARY),
        type - type of the selection (e.g. STRING, FILE_NAME)."""
        name = self._register(command)
        self.tk.call(('selection', 'handle') + self._options(kw)
              + (self._w, name))
    def selection_own(self, **kw):
        """Become owner of X selection.

        A keyword parameter selection specifies the name of
        the selection (default PRIMARY)."""
        self.tk.call(('selection', 'own') +
                 self._options(kw) + (self._w,))
    def selection_own_get(self, **kw):
        """Return owner of X selection.

        The following keyword parameter can
        be provided:
        selection - name of the selection (default PRIMARY),
        type - type of the selection (e.g. STRING, FILE_NAME)."""
        if not kw.has_key('displayof'): kw['displayof'] = self._w
        name = self.tk.call(('selection', 'own') + self._options(kw))
        if not name: return None
        return self._nametowidget(name)
    def send(self, interp, cmd, *args):
        """Send Tcl command CMD to different interpreter INTERP to be executed."""
        return self.tk.call(('send', interp, cmd) + args)
    def lower(self, belowThis=None):
        """Lower this widget in the stacking order."""
        self.tk.call('lower', self._w, belowThis)
    def tkraise(self, aboveThis=None):
        """Raise this widget in the stacking order."""
        self.tk.call('raise', self._w, aboveThis)
    lift = tkraise
    def colormodel(self, value=None):
        """Useless. Not implemented in Tk."""
        return self.tk.call('tk', 'colormodel', self._w, value)
    def winfo_atom(self, name, displayof=0):
        """Return integer which represents atom NAME."""
        args = ('winfo', 'atom') + self._displayof(displayof) + (name,)
        return getint(self.tk.call(args))
    def winfo_atomname(self, id, displayof=0):
        """Return name of atom with identifier ID."""
        args = ('winfo', 'atomname') \
               + self._displayof(displayof) + (id,)
        return self.tk.call(args)
    def winfo_cells(self):
        """Return number of cells in the colormap for this widget."""
        return getint(
            self.tk.call('winfo', 'cells', self._w))
    def winfo_children(self):
        """Return a list of all widgets which are children of this widget."""
        result = []
        for child in self.tk.splitlist(
            self.tk.call('winfo', 'children', self._w)):
            try:
                # Tcl sometimes returns extra windows, e.g. for
                # menus; those need to be skipped
                result.append(self._nametowidget(child))
            except KeyError:
                pass
        return result

    def winfo_class(self):
        """Return window class name of this widget."""
        return self.tk.call('winfo', 'class', self._w)
    def winfo_colormapfull(self):
        """Return true if at the last color request the colormap was full."""
        return self.tk.getboolean(
            self.tk.call('winfo', 'colormapfull', self._w))
    def winfo_containing(self, rootX, rootY, displayof=0):
        """Return the widget which is at the root coordinates ROOTX, ROOTY."""
        args = ('winfo', 'containing') \
               + self._displayof(displayof) + (rootX, rootY)
        name = self.tk.call(args)
        if not name: return None
        return self._nametowidget(name)
    def winfo_depth(self):
        """Return the number of bits per pixel."""
        return getint(self.tk.call('winfo', 'depth', self._w))
    def winfo_exists(self):
        """Return true if this widget exists."""
        return getint(
            self.tk.call('winfo', 'exists', self._w))
    def winfo_fpixels(self, number):
        """Return the number of pixels for the given distance NUMBER
        (e.g. "3c") as float."""
        return getdouble(self.tk.call(
            'winfo', 'fpixels', self._w, number))
    def winfo_geometry(self):
        """Return geometry string for this widget in the form "widthxheight+X+Y"."""
        return self.tk.call('winfo', 'geometry', self._w)
    def winfo_height(self):
        """Return height of this widget."""
        return getint(
            self.tk.call('winfo', 'height', self._w))
    def winfo_id(self):
        """Return identifier ID for this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'id', self._w))
    def winfo_interps(self, displayof=0):
        """Return the name of all Tcl interpreters for this display."""
        args = ('winfo', 'interps') + self._displayof(displayof)
        return self.tk.splitlist(self.tk.call(args))
    def winfo_ismapped(self):
        """Return true if this widget is mapped."""
        return getint(
            self.tk.call('winfo', 'ismapped', self._w))
    def winfo_manager(self):
        """Return the window mananger name for this widget."""
        return self.tk.call('winfo', 'manager', self._w)
    def winfo_name(self):
        """Return the name of this widget."""
        return self.tk.call('winfo', 'name', self._w)
    def winfo_parent(self):
        """Return the name of the parent of this widget."""
        return self.tk.call('winfo', 'parent', self._w)
    def winfo_pathname(self, id, displayof=0):
        """Return the pathname of the widget given by ID."""
        args = ('winfo', 'pathname') \
               + self._displayof(displayof) + (id,)
        return self.tk.call(args)
    def winfo_pixels(self, number):
        """Rounded integer value of winfo_fpixels."""
        return getint(
            self.tk.call('winfo', 'pixels', self._w, number))
    def winfo_pointerx(self):
        """Return the x coordinate of the pointer on the root window."""
        return getint(
            self.tk.call('winfo', 'pointerx', self._w))
    def winfo_pointerxy(self):
        """Return a tuple of x and y coordinates of the pointer on the root window."""
        return self._getints(
            self.tk.call('winfo', 'pointerxy', self._w))
    def winfo_pointery(self):
        """Return the y coordinate of the pointer on the root window."""
        return getint(
            self.tk.call('winfo', 'pointery', self._w))
    def winfo_reqheight(self):
        """Return requested height of this widget."""
        return getint(
            self.tk.call('winfo', 'reqheight', self._w))
    def winfo_reqwidth(self):
        """Return requested width of this widget."""
        return getint(
            self.tk.call('winfo', 'reqwidth', self._w))
    def winfo_rgb(self, color):
        """Return tuple of decimal values for red, green, blue for
        COLOR in this widget."""
        return self._getints(
            self.tk.call('winfo', 'rgb', self._w, color))
    def winfo_rootx(self):
        """Return x coordinate of upper left corner of this widget on the
        root window."""
        return getint(
            self.tk.call('winfo', 'rootx', self._w))
    def winfo_rooty(self):
        """Return y coordinate of upper left corner of this widget on the
        root window."""
        return getint(
            self.tk.call('winfo', 'rooty', self._w))
    def winfo_screen(self):
        """Return the screen name of this widget."""
        return self.tk.call('winfo', 'screen', self._w)
    def winfo_screencells(self):
        """Return the number of the cells in the colormap of the screen
        of this widget."""
        return getint(
            self.tk.call('winfo', 'screencells', self._w))
    def winfo_screendepth(self):
        """Return the number of bits per pixel of the root window of the
        screen of this widget."""
        return getint(
            self.tk.call('winfo', 'screendepth', self._w))
    def winfo_screenheight(self):
        """Return the number of pixels of the height of the screen of this widget
        in pixel."""
        return getint(
            self.tk.call('winfo', 'screenheight', self._w))
    def winfo_screenmmheight(self):
        """Return the number of pixels of the height of the screen of
        this widget in mm."""
        return getint(
            self.tk.call('winfo', 'screenmmheight', self._w))
    def winfo_screenmmwidth(self):
        """Return the number of pixels of the width of the screen of
        this widget in mm."""
        return getint(
            self.tk.call('winfo', 'screenmmwidth', self._w))
    def winfo_screenvisual(self):
        """Return one of the strings directcolor, grayscale, pseudocolor,
        staticcolor, staticgray, or truecolor for the default
        colormodel of this screen."""
        return self.tk.call('winfo', 'screenvisual', self._w)
    def winfo_screenwidth(self):
        """Return the number of pixels of the width of the screen of
        this widget in pixel."""
        return getint(
            self.tk.call('winfo', 'screenwidth', self._w))
    def winfo_server(self):
        """Return information of the X-Server of the screen of this widget in
        the form "XmajorRminor vendor vendorVersion"."""
        return self.tk.call('winfo', 'server', self._w)
    def winfo_toplevel(self):
        """Return the toplevel widget of this widget."""
        return self._nametowidget(self.tk.call(
            'winfo', 'toplevel', self._w))
    def winfo_viewable(self):
        """Return true if the widget and all its higher ancestors are mapped."""
        return getint(
            self.tk.call('winfo', 'viewable', self._w))
    def winfo_visual(self):
        """Return one of the strings directcolor, grayscale, pseudocolor,
        staticcolor, staticgray, or truecolor for the
        colormodel of this widget."""
        return self.tk.call('winfo', 'visual', self._w)
    def winfo_visualid(self):
        """Return the X identifier for the visual for this widget."""
        return self.tk.call('winfo', 'visualid', self._w)
    def winfo_visualsavailable(self, includeids=0):
        """Return a list of all visuals available for the screen
        of this widget.

        Each item in the list consists of a visual name (see winfo_visual), a
        depth and if INCLUDEIDS=1 is given also the X identifier."""
        data = self.tk.split(
            self.tk.call('winfo', 'visualsavailable', self._w,
                     includeids and 'includeids' or None))
        if type(data) is StringType:
            data = [self.tk.split(data)]
        return map(self.__winfo_parseitem, data)
    def __winfo_parseitem(self, t):
        """Internal function."""
        return t[:1] + tuple(map(self.__winfo_getint, t[1:]))
    def __winfo_getint(self, x):
        """Internal function."""
        return int(x, 0)
    def winfo_vrootheight(self):
        """Return the height of the virtual root window associated with this
        widget in pixels. If there is no virtual root window return the
        height of the screen."""
        return getint(
            self.tk.call('winfo', 'vrootheight', self._w))
    def winfo_vrootwidth(self):
        """Return the width of the virtual root window associated with this
        widget in pixel. If there is no virtual root window return the
        width of the screen."""
        return getint(
            self.tk.call('winfo', 'vrootwidth', self._w))
    def winfo_vrootx(self):
        """Return the x offset of the virtual root relative to the root
        window of the screen of this widget."""
        return getint(
            self.tk.call('winfo', 'vrootx', self._w))
    def winfo_vrooty(self):
        """Return the y offset of the virtual root relative to the root
        window of the screen of this widget."""
        return getint(
            self.tk.call('winfo', 'vrooty', self._w))
    def winfo_width(self):
        """Return the width of this widget."""
        return getint(
            self.tk.call('winfo', 'width', self._w))
    def winfo_x(self):
        """Return the x coordinate of the upper left corner of this widget
        in the parent."""
        return getint(
            self.tk.call('winfo', 'x', self._w))
    def winfo_y(self):
        """Return the y coordinate of the upper left corner of this widget
        in the parent."""
        return getint(
            self.tk.call('winfo', 'y', self._w))
    def update(self):
        """Enter event loop until all pending events have been processed by Tcl."""
        self.tk.call('update')
    def update_idletasks(self):
        """Enter event loop until all idle callbacks have been called. This
        will update the display of windows but not process events caused by
        the user."""
        self.tk.call('update', 'idletasks')
    def bindtags(self, tagList=None):
        """Set or get the list of bindtags for this widget.

        With no argument return the list of all bindtags associated with
        this widget. With a list of strings as argument the bindtags are
        set to this list. The bindtags determine in which order events are
        processed (see bind)."""
        if tagList is None:
            return self.tk.splitlist(
                self.tk.call('bindtags', self._w))
        else:
            self.tk.call('bindtags', self._w, tagList)
    def _bind(self, what, sequence, func, add, needcleanup=1):
        """Internal function."""
        if type(func) is StringType:
            self.tk.call(what + (sequence, func))
        elif func:
            funcid = self._register(func, self._substitute,
                        needcleanup)
            cmd = ('%sif {"[%s %s]" == "break"} break\n'
                   %
                   (add and '+' or '',
                funcid, self._subst_format_str))
            self.tk.call(what + (sequence, cmd))
            return funcid
        elif sequence:
            return self.tk.call(what + (sequence,))
        else:
            return self.tk.splitlist(self.tk.call(what))
    def bind(self, sequence=None, func=None, add=None):
        """Bind to this widget at event SEQUENCE a call to function FUNC.

        SEQUENCE is a string of concatenated event
        patterns. An event pattern is of the form
        <MODIFIER-MODIFIER-TYPE-DETAIL> where MODIFIER is one
        of Control, Mod2, M2, Shift, Mod3, M3, Lock, Mod4, M4,
        Button1, B1, Mod5, M5 Button2, B2, Meta, M, Button3,
        B3, Alt, Button4, B4, Double, Button5, B5 Triple,
        Mod1, M1. TYPE is one of Activate, Enter, Map,
        ButtonPress, Button, Expose, Motion, ButtonRelease
        FocusIn, MouseWheel, Circulate, FocusOut, Property,
        Colormap, Gravity Reparent, Configure, KeyPress, Key,
        Unmap, Deactivate, KeyRelease Visibility, Destroy,
        Leave and DETAIL is the button number for ButtonPress,
        ButtonRelease and DETAIL is the Keysym for KeyPress and
        KeyRelease. Examples are
        <Control-Button-1> for pressing Control and mouse button 1 or
        <Alt-A> for pressing A and the Alt key (KeyPress can be omitted).
        An event pattern can also be a virtual event of the form
        <<AString>> where AString can be arbitrary. This
        event can be generated by event_generate.
        If events are concatenated they must appear shortly
        after each other.

        FUNC will be called if the event sequence occurs with an
        instance of Event as argument. If the return value of FUNC is
        "break" no further bound function is invoked.

        An additional boolean parameter ADD specifies whether FUNC will
        be called additionally to the other bound function or whether
        it will replace the previous function.

        Bind will return an identifier to allow deletion of the bound function with
        unbind without memory leak.

        If FUNC or SEQUENCE is omitted the bound function or list
        of bound events are returned."""

        return self._bind(('bind', self._w), sequence, func, add)
    def unbind(self, sequence, funcid=None):
        """Unbind for this widget for event SEQUENCE  the
        function identified with FUNCID."""
        self.tk.call('bind', self._w, sequence, '')
        if funcid:
            self.deletecommand(funcid)
    def bind_all(self, sequence=None, func=None, add=None):
        """Bind to all widgets at an event SEQUENCE a call to function FUNC.
        An additional boolean parameter ADD specifies whether FUNC will
        be called additionally to the other bound function or whether
        it will replace the previous function. See bind for the return value."""
        return self._bind(('bind', 'all'), sequence, func, add, 0)
    def unbind_all(self, sequence):
        """Unbind for all widgets for event SEQUENCE all functions."""
        self.tk.call('bind', 'all' , sequence, '')
    def bind_class(self, className, sequence=None, func=None, add=None):

        """Bind to widgets with bindtag CLASSNAME at event
        SEQUENCE a call of function FUNC. An additional
        boolean parameter ADD specifies whether FUNC will be
        called additionally to the other bound function or
        whether it will replace the previous function. See bind for
        the return value."""

        return self._bind(('bind', className), sequence, func, add, 0)
    def unbind_class(self, className, sequence):
        """Unbind for a all widgets with bindtag CLASSNAME for event SEQUENCE
        all functions."""
        self.tk.call('bind', className , sequence, '')
    def mainloop(self, n=0):
        """Call the mainloop of Tk."""
        self.tk.mainloop(n)
    def quit(self):
        """Quit the Tcl interpreter. All widgets will be destroyed."""
        self.tk.quit()
    def _getints(self, string):
        """Internal function."""
        if string:
            return tuple(map(getint, self.tk.splitlist(string)))
    def _getdoubles(self, string):
        """Internal function."""
        if string:
            return tuple(map(getdouble, self.tk.splitlist(string)))
    def _getboolean(self, string):
        """Internal function."""
        if string:
            return self.tk.getboolean(string)
    def _displayof(self, displayof):
        """Internal function."""
        if displayof:
            return ('-displayof', displayof)
        if displayof is None:
            return ('-displayof', self._w)
        return ()
    def _options(self, cnf, kw = None):
        """Internal function."""
        if kw:
            cnf = _cnfmerge((cnf, kw))
        else:
            cnf = _cnfmerge(cnf)
        res = ()
        for k, v in cnf.items():
            if v is not None:
                if k[-1] == '_': k = k[:-1]
                if callable(v):
                    v = self._register(v)
                res = res + ('-'+k, v)
        return res
    def nametowidget(self, name):
        """Return the Tkinter instance of a widget identified by
        its Tcl name NAME."""
        w = self
        if name[0] == '.':
            w = w._root()
            name = name[1:]
        while name:
            i = name.find('.')
            if i >= 0:
                name, tail = name[:i], name[i+1:]
            else:
                tail = ''
            w = w.children[name]
            name = tail
        return w
    _nametowidget = nametowidget
    def _register(self, func, subst=None, needcleanup=1):
        """Return a newly created Tcl function. If this
        function is called, the Python function FUNC will
        be executed. An optional function SUBST can
        be given which will be executed before FUNC."""
        f = CallWrapper(func, subst, self).__call__
        name = `id(f)`
        try:
            func = func.im_func
        except AttributeError:
            pass
        try:
            name = name + func.__name__
        except AttributeError:
            pass
        self.tk.createcommand(name, f)
        if needcleanup:
            if self._tclCommands is None:
                self._tclCommands = []
            self._tclCommands.append(name)
        #print '+ Tkinter created command', name
        return name
    register = _register
    def _root(self):
        """Internal function."""
        w = self
        while w.master: w = w.master
        return w
    _subst_format = ('%#', '%b', '%f', '%h', '%k',
             '%s', '%t', '%w', '%x', '%y',
             '%A', '%E', '%K', '%N', '%W', '%T', '%X', '%Y', '%D')
    _subst_format_str = " ".join(_subst_format)
    def _substitute(self, *args):
        """Internal function."""
        if len(args) != len(self._subst_format): return args
        getboolean = self.tk.getboolean

        getint = int
        def getint_event(s):
            """Tk changed behavior in 8.4.2, returning "??" rather more often."""
            try:
                return int(s)
            except ValueError:
                return s

        nsign, b, f, h, k, s, t, w, x, y, A, E, K, N, W, T, X, Y, D = args
        # Missing: (a, c, d, m, o, v, B, R)
        e = Event()
        # serial field: valid vor all events
        # number of button: ButtonPress and ButtonRelease events only
        # height field: Configure, ConfigureRequest, Create,
        # ResizeRequest, and Expose events only
        # keycode field: KeyPress and KeyRelease events only
        # time field: "valid for events that contain a time field"
        # width field: Configure, ConfigureRequest, Create, ResizeRequest,
        # and Expose events only
        # x field: "valid for events that contain a x field"
        # y field: "valid for events that contain a y field"
        # keysym as decimal: KeyPress and KeyRelease events only
        # x_root, y_root fields: ButtonPress, ButtonRelease, KeyPress,
        # KeyRelease,and Motion events
        e.serial = getint(nsign)
        e.num = getint_event(b)
        try: e.focus = getboolean(f)
        except TclError: pass
        e.height = getint_event(h)
        e.keycode = getint_event(k)
        e.state = getint_event(s)
        e.time = getint_event(t)
        e.width = getint_event(w)
        e.x = getint_event(x)
        e.y = getint_event(y)
        e.char = A
        try: e.send_event = getboolean(E)
        except TclError: pass
        e.keysym = K
        e.keysym_num = getint_event(N)
        e.type = T
        try:
            e.widget = self._nametowidget(W)
        except KeyError:
            e.widget = W
        e.x_root = getint_event(X)
        e.y_root = getint_event(Y)
        try:
            e.delta = getint(D)
        except ValueError:
            e.delta = 0
        return (e,)
    def _report_exception(self):
        """Internal function."""
        import sys
        exc, val, tb = sys.exc_type, sys.exc_value, sys.exc_traceback
        root = self._root()
        root.report_callback_exception(exc, val, tb)
    # These used to be defined in Widget:
    def configure(self, cnf=None, **kw):
        """Configure resources of a widget.

        The values for resources are specified as keyword
        arguments. To get an overview about
        the allowed keyword arguments call the method keys.
        """
        # XXX ought to generalize this so tag_config etc. can use it
        if kw:
            cnf = _cnfmerge((cnf, kw))
        elif cnf:
            cnf = _cnfmerge(cnf)
        if cnf is None:
            cnf = {}
            for x in self.tk.split(
                self.tk.call(self._w, 'configure')):
                cnf[x[0][1:]] = (x[0][1:],) + x[1:]
            return cnf
        if type(cnf) is StringType:
            x = self.tk.split(self.tk.call(
                self._w, 'configure', '-'+cnf))
            return (x[0][1:],) + x[1:]
        self.tk.call((self._w, 'configure')
              + self._options(cnf))
    config = configure
    def cget(self, key):
        """Return the resource value for a KEY given as string."""
        return self.tk.call(self._w, 'cget', '-' + key)
    __getitem__ = cget
    def __setitem__(self, key, value):
        self.configure({key: value})
    def keys(self):
        """Return a list of all resource names of this widget."""
        return map(lambda x: x[0][1:],
               self.tk.split(self.tk.call(self._w, 'configure')))
    def __str__(self):
        """Return the window path name of this widget."""
        return self._w
    # Pack methods that apply to the master
    _noarg_ = ['_noarg_']
    def pack_propagate(self, flag=_noarg_):
        """Set or get the status for propagation of geometry information.

        A boolean argument specifies whether the geometry information
        of the slaves will determine the size of this widget. If no argument
        is given the current setting will be returned.
        """
        if flag is Misc._noarg_:
            return self._getboolean(self.tk.call(
                'pack', 'propagate', self._w))
        else:
            self.tk.call('pack', 'propagate', self._w, flag)
    propagate = pack_propagate
    def pack_slaves(self):
        """Return a list of all slaves of this widget
        in its packing order."""
        return map(self._nametowidget,
               self.tk.splitlist(
                   self.tk.call('pack', 'slaves', self._w)))
    slaves = pack_slaves
    # Place method that applies to the master
    def place_slaves(self):
        """Return a list of all slaves of this widget
        in its packing order."""
        return map(self._nametowidget,
               self.tk.splitlist(
                   self.tk.call(
                       'place', 'slaves', self._w)))
    # Grid methods that apply to the master
    def grid_bbox(self, column=None, row=None, col2=None, row2=None):
        """Return a tuple of integer coordinates for the bounding
        box of this widget controlled by the geometry manager grid.

        If COLUMN, ROW is given the bounding box applies from
        the cell with row and column 0 to the specified
        cell. If COL2 and ROW2 are given the bounding box
        starts at that cell.

        The returned integers specify the offset of the upper left
        corner in the master widget and the width and height.
        """
        args = ('grid', 'bbox', self._w)
        if column is not None and row is not None:
            args = args + (column, row)
        if col2 is not None and row2 is not None:
            args = args + (col2, row2)
        return self._getints(apply(self.tk.call, args)) or None

    bbox = grid_bbox
    def _grid_configure(self, command, index, cnf, kw):
        """Internal function."""
        if type(cnf) is StringType and not kw:
            if cnf[-1:] == '_':
                cnf = cnf[:-1]
            if cnf[:1] != '-':
                cnf = '-'+cnf
            options = (cnf,)
        else:
            options = self._options(cnf, kw)
        if not options:
            res = self.tk.call('grid',
                       command, self._w, index)
            words = self.tk.splitlist(res)
            dict = {}
            for i in range(0, len(words), 2):
                key = words[i][1:]
                value = words[i+1]
                if not value:
                    value = None
                elif '.' in value:
                    value = getdouble(value)
                else:
                    value = getint(value)
                dict[key] = value
            return dict
        res = self.tk.call(
                  ('grid', command, self._w, index)
                  + options)
        if len(options) == 1:
            if not res: return None
            # In Tk 7.5, -width can be a float
            if '.' in res: return getdouble(res)
            return getint(res)
    def grid_columnconfigure(self, index, cnf={}, **kw):
        """Configure column INDEX of a grid.

        Valid resources are minsize (minimum size of the column),
        weight (how much does additional space propagate to this column)
        and pad (how much space to let additionally)."""
        return self._grid_configure('columnconfigure', index, cnf, kw)
    columnconfigure = grid_columnconfigure
    def grid_location(self, x, y):
        """Return a tuple of column and row which identify the cell
        at which the pixel at position X and Y inside the master
        widget is located."""
        return self._getints(
            self.tk.call(
                'grid', 'location', self._w, x, y)) or None
    def grid_propagate(self, flag=_noarg_):
        """Set or get the status for propagation of geometry information.

        A boolean argument specifies whether the geometry information
        of the slaves will determine the size of this widget. If no argument
        is given, the current setting will be returned.
        """
        if flag is Misc._noarg_:
            return self._getboolean(self.tk.call(
                'grid', 'propagate', self._w))
        else:
            self.tk.call('grid', 'propagate', self._w, flag)
    def grid_rowconfigure(self, index, cnf={}, **kw):
        """Configure row INDEX of a grid.

        Valid resources are minsize (minimum size of the row),
        weight (how much does additional space propagate to this row)
        and pad (how much space to let additionally)."""
        return self._grid_configure('rowconfigure', index, cnf, kw)
    rowconfigure = grid_rowconfigure
    def grid_size(self):
        """Return a tuple of the number of column and rows in the grid."""
        return self._getints(
            self.tk.call('grid', 'size', self._w)) or None
    size = grid_size
    def grid_slaves(self, row=None, column=None):
        """Return a list of all slaves of this widget
        in its packing order."""
        args = ()
        if row is not None:
            args = args + ('-row', row)
        if column is not None:
            args = args + ('-column', column)
        return map(self._nametowidget,
               self.tk.splitlist(self.tk.call(
                   ('grid', 'slaves', self._w) + args)))

    # Support for the "event" command, new in Tk 4.2.
    # By Case Roole.

    def event_add(self, virtual, *sequences):
        """Bind a virtual event VIRTUAL (of the form <<Name>>)
        to an event SEQUENCE such that the virtual event is triggered
        whenever SEQUENCE occurs."""
        args = ('event', 'add', virtual) + sequences
        self.tk.call(args)

    def event_delete(self, virtual, *sequences):
        """Unbind a virtual event VIRTUAL from SEQUENCE."""
        args = ('event', 'delete', virtual) + sequences
        self.tk.call(args)

    def event_generate(self, sequence, **kw):
        """Generate an event SEQUENCE. Additional
        keyword arguments specify parameter of the event
        (e.g. x, y, rootx, rooty)."""
        args = ('event', 'generate', self._w, sequence)
        for k, v in kw.items():
            args = args + ('-%s' % k, str(v))
        self.tk.call(args)

    def event_info(self, virtual=None):
        """Return a list of all virtual events or the information
        about the SEQUENCE bound to the virtual event VIRTUAL."""
        return self.tk.splitlist(
            self.tk.call('event', 'info', virtual))

    # Image related commands

    def image_names(self):
        """Return a list of all existing image names."""
        return self.tk.call('image', 'names')

    def image_types(self):
        """Return a list of all available image types (e.g. phote bitmap)."""
        return self.tk.call('image', 'types')


class CallWrapper:
    """Internal class. Stores function to call when some user
    defined Tcl function is called e.g. after an event occurred."""
    def __init__(self, func, subst, widget):
        """Store FUNC, SUBST and WIDGET as members."""
        self.func = func
        self.subst = subst
        self.widget = widget
    def __call__(self, *args):
        """Apply first function SUBST to arguments, than FUNC."""
        try:
            if self.subst:
                args = apply(self.subst, args)
            return apply(self.func, args)
        except SystemExit, msg:
            raise SystemExit, msg
        except:
            self.widget._report_exception()


class Wm:
    """Provides functions for the communication with the window manager."""
    def wm_aspect(self,
              minNumer=None, minDenom=None,
              maxNumer=None, maxDenom=None):
        """Instruct the window manager to set the aspect ratio (width/height)
        of this widget to be between MINNUMER/MINDENOM and MAXNUMER/MAXDENOM. Return a tuple
        of the actual values if no argument is given."""
        return self._getints(
            self.tk.call('wm', 'aspect', self._w,
                     minNumer, minDenom,
                     maxNumer, maxDenom))
    aspect = wm_aspect
    def wm_client(self, name=None):
        """Store NAME in WM_CLIENT_MACHINE property of this widget. Return
        current value."""
        return self.tk.call('wm', 'client', self._w, name)
    client = wm_client
    def wm_colormapwindows(self, *wlist):
        """Store list of window names (WLIST) into WM_COLORMAPWINDOWS property
        of this widget. This list contains windows whose colormaps differ from their
        parents. Return current list of widgets if WLIST is empty."""
        if len(wlist) > 1:
            wlist = (wlist,) # Tk needs a list of windows here
        args = ('wm', 'colormapwindows', self._w) + wlist
        return map(self._nametowidget, self.tk.call(args))
    colormapwindows = wm_colormapwindows
    def wm_command(self, value=None):
        """Store VALUE in WM_COMMAND property. It is the command
        which shall be used to invoke the application. Return current
        command if VALUE is None."""
        return self.tk.call('wm', 'command', self._w, value)
    command = wm_command
    def wm_deiconify(self):
        """Deiconify this widget. If it was never mapped it will not be mapped.
        On Windows it will raise this widget and give it the focus."""
        return self.tk.call('wm', 'deiconify', self._w)
    deiconify = wm_deiconify
    def wm_focusmodel(self, model=None):
        """Set focus model to MODEL. "active" means that this widget will claim
        the focus itself, "passive" means that the window manager shall give
        the focus. Return current focus model if MODEL is None."""
        return self.tk.call('wm', 'focusmodel', self._w, model)
    focusmodel = wm_focusmodel
    def wm_frame(self):
        """Return identifier for decorative frame of this widget if present."""
        return self.tk.call('wm', 'frame', self._w)
    frame = wm_frame
    def wm_geometry(self, newGeometry=None):
        """Set geometry to NEWGEOMETRY of the form =widthxheight+x+y. Return
        current value if None is given."""
        return self.tk.call('wm', 'geometry', self._w, newGeometry)
    geometry = wm_geometry
    def wm_grid(self,
         baseWidth=None, baseHeight=None,
         widthInc=None, heightInc=None):
        """Instruct the window manager that this widget shall only be
        resized on grid boundaries. WIDTHINC and HEIGHTINC are the width and
        height of a grid unit in pixels. BASEWIDTH and BASEHEIGHT are the
        number of grid units requested in Tk_GeometryRequest."""
        return self._getints(self.tk.call(
            'wm', 'grid', self._w,
            baseWidth, baseHeight, widthInc, heightInc))
    grid = wm_grid
    def wm_group(self, pathName=None):
        """Set the group leader widgets for related widgets to PATHNAME. Return
        the group leader of this widget if None is given."""
        return self.tk.call('wm', 'group', self._w, pathName)
    group = wm_group
    def wm_iconbitmap(self, bitmap=None):
        """Set bitmap for the iconified widget to BITMAP. Return
        the bitmap if None is given."""
        return self.tk.call('wm', 'iconbitmap', self._w, bitmap)
    iconbitmap = wm_iconbitmap
    def wm_iconify(self):
        """Display widget as icon."""
        return self.tk.call('wm', 'iconify', self._w)
    iconify = wm_iconify
    def wm_iconmask(self, bitmap=None):
        """Set mask for the icon bitmap of this widget. Return the
        mask if None is given."""
        return self.tk.call('wm', 'iconmask', self._w, bitmap)
    iconmask = wm_iconmask
    def wm_iconname(self, newName=None):
        """Set the name of the icon for this widget. Return the name if
        None is given."""
        return self.tk.call('wm', 'iconname', self._w, newName)
    iconname = wm_iconname
    def wm_iconposition(self, x=None, y=None):
        """Set the position of the icon of this widget to X and Y. Return
        a tuple of the current values of X and X if None is given."""
        return self._getints(self.tk.call(
            'wm', 'iconposition', self._w, x, y))
    iconposition = wm_iconposition
    def wm_iconwindow(self, pathName=None):
        """Set widget PATHNAME to be displayed instead of icon. Return the current
        value if None is given."""
        return self.tk.call('wm', 'iconwindow', self._w, pathName)
    iconwindow = wm_iconwindow
    def wm_maxsize(self, width=None, height=None):
        """Set max WIDTH and HEIGHT for this widget. If the window is gridded
        the values are given in grid units. Return the current values if None
        is given."""
        return self._getints(self.tk.call(
            'wm', 'maxsize', self._w, width, height))
    maxsize = wm_maxsize
    def wm_minsize(self, width=None, height=None):
        """Set min WIDTH and HEIGHT for this widget. If the window is gridded
        the values are given in grid units. Return the current values if None
        is given."""
        return self._getints(self.tk.call(
            'wm', 'minsize', self._w, width, height))
    minsize = wm_minsize
    def wm_overrideredirect(self, boolean=None):
        """Instruct the window manager to ignore this widget
        if BOOLEAN is given with 1. Return the current value if None
        is given."""
        return self._getboolean(self.tk.call(
            'wm', 'overrideredirect', self._w, boolean))
    overrideredirect = wm_overrideredirect
    def wm_positionfrom(self, who=None):
        """Instruct the window manager that the position of this widget shall
        be defined by the user if WHO is "user", and by its own policy if WHO is
        "program"."""
        return self.tk.call('wm', 'positionfrom', self._w, who)
    positionfrom = wm_positionfrom
    def wm_protocol(self, name=None, func=None):
        """Bind function FUNC to command NAME for this widget.
        Return the function bound to NAME if None is given. NAME could be
        e.g. "WM_SAVE_YOURSELF" or "WM_DELETE_WINDOW"."""
        if callable(func):
            command = self._register(func)
        else:
            command = func
        return self.tk.call(
            'wm', 'protocol', self._w, name, command)
    protocol = wm_protocol
    def wm_resizable(self, width=None, height=None):
        """Instruct the window manager whether this width can be resized
        in WIDTH or HEIGHT. Both values are boolean values."""
        return self.tk.call('wm', 'resizable', self._w, width, height)
    resizable = wm_resizable
    def wm_sizefrom(self, who=None):
        """Instruct the window manager that the size of this widget shall
        be defined by the user if WHO is "user", and by its own policy if WHO is
        "program"."""
        return self.tk.call('wm', 'sizefrom', self._w, who)
    sizefrom = wm_sizefrom
    def wm_state(self, newstate=None):
        """Query or set the state of this widget as one of normal, icon,
        iconic (see wm_iconwindow), withdrawn, or zoomed (Windows only)."""
        return self.tk.call('wm', 'state', self._w, newstate)
    state = wm_state
    def wm_title(self, string=None):
        """Set the title of this widget."""
        return self.tk.call('wm', 'title', self._w, string)
    title = wm_title
    def wm_transient(self, master=None):
        """Instruct the window manager that this widget is transient
        with regard to widget MASTER."""
        return self.tk.call('wm', 'transient', self._w, master)
    transient = wm_transient
    def wm_withdraw(self):
        """Withdraw this widget from the screen such that it is unmapped
        and forgotten by the window manager. Re-draw it with wm_deiconify."""
        return self.tk.call('wm', 'withdraw', self._w)
    withdraw = wm_withdraw


class Tk(Misc, Wm):
    """Toplevel widget of Tk which represents mostly the main window
    of an appliation. It has an associated Tcl interpreter."""
    _w = '.'
    def __init__(self, screenName=None, baseName=None, className='Tk'):
        """Return a new Toplevel widget on screen SCREENNAME. A new Tcl interpreter will
        be created. BASENAME will be used for the identification of the profile file (see
        readprofile).
        It is constructed from sys.argv[0] without extensions if None is given. CLASSNAME
        is the name of the widget class."""
        global _default_root
        self.master = None
        self.children = {}
        if baseName is None:
            import sys, os
            baseName = os.path.basename(sys.argv[0])
            baseName, ext = os.path.splitext(baseName)
            if ext not in ('.py', '.pyc', '.pyo'):
                baseName = baseName + ext
        self.tk = _tkinter.create(screenName, baseName, className)
        if _MacOS and hasattr(_MacOS, 'SchedParams'):
            # Disable event scanning except for Command-Period
            _MacOS.SchedParams(1, 0)
            # Work around nasty MacTk bug
            # XXX Is this one still needed?
            self.update()
        # Version sanity checks
        tk_version = self.tk.getvar('tk_version')
        if tk_version != _tkinter.TK_VERSION:
            raise RuntimeError, \
            "tk.h version (%s) doesn't match libtk.a version (%s)" \
            % (_tkinter.TK_VERSION, tk_version)
        tcl_version = self.tk.getvar('tcl_version')
        if tcl_version != _tkinter.TCL_VERSION:
            raise RuntimeError, \
            "tcl.h version (%s) doesn't match libtcl.a version (%s)" \
            % (_tkinter.TCL_VERSION, tcl_version)
        if TkVersion < 4.0:
            raise RuntimeError, \
            "Tk 4.0 or higher is required; found Tk %s" \
            % str(TkVersion)
        self.tk.createcommand('tkerror', _tkerror)
        self.tk.createcommand('exit', _exit)
        self.readprofile(baseName, className)
        if _support_default_root and not _default_root:
            _default_root = self
        self.protocol("WM_DELETE_WINDOW", self.destroy)
    def destroy(self):
        """Destroy this and all descendants widgets. This will
        end the application of this Tcl interpreter."""
        for c in self.children.values(): c.destroy()
        self.tk.call('destroy', self._w)
        Misc.destroy(self)
        global _default_root
        if _support_default_root and _default_root is self:
            _default_root = None
    def readprofile(self, baseName, className):
        """Internal function. It reads BASENAME.tcl and CLASSNAME.tcl into
        the Tcl Interpreter and calls execfile on BASENAME.py and CLASSNAME.py if
        such a file exists in the home directory."""
        import os
        if os.environ.has_key('HOME'): home = os.environ['HOME']
        else: home = os.curdir
        class_tcl = os.path.join(home, '.%s.tcl' % className)
        class_py = os.path.join(home, '.%s.py' % className)
        base_tcl = os.path.join(home, '.%s.tcl' % baseName)
        base_py = os.path.join(home, '.%s.py' % baseName)
        dir = {'self': self}
        exec 'from Tkinter import *' in dir
        if os.path.isfile(class_tcl):
            self.tk.call('source', class_tcl)
        if os.path.isfile(class_py):
            execfile(class_py, dir)
        if os.path.isfile(base_tcl):
            self.tk.call('source', base_tcl)
        if os.path.isfile(base_py):
            execfile(base_py, dir)
    def report_callback_exception(self, exc, val, tb):
        """Internal function. It reports exception on sys.stderr."""
        import traceback, sys
        sys.stderr.write("Exception in Tkinter callback\n")
        sys.last_type = exc
        sys.last_value = val
        sys.last_traceback = tb
        traceback.print_exception(exc, val, tb)

# Ideally, the classes Pack, Place and Grid disappear, the
# pack/place/grid methods are defined on the Widget class, and
# everybody uses w.pack_whatever(...) instead of Pack.whatever(w,
# ...), with pack(), place() and grid() being short for
# pack_configure(), place_configure() and grid_columnconfigure(), and
# forget() being short for pack_forget().  As a practical matter, I'm
# afraid that there is too much code out there that may be using the
# Pack, Place or Grid class, so I leave them intact -- but only as
# backwards compatibility features.  Also note that those methods that
# take a master as argument (e.g. pack_propagate) have been moved to
# the Misc class (which now incorporates all methods common between
# toplevel and interior widgets).  Again, for compatibility, these are
# copied into the Pack, Place or Grid class.

class Pack:
    """Geometry manager Pack.

    Base class to use the methods pack_* in every widget."""
    def pack_configure(self, cnf={}, **kw):
        """Pack a widget in the parent widget. Use as options:
        after=widget - pack it after you have packed widget
        anchor=NSEW (or subset) - position widget according to
                                  given direction
                before=widget - pack it before you will pack widget
        expand=1 or 0 - expand widget if parent size grows
        fill=NONE or X or Y or BOTH - fill widget if widget grows
        in=master - use master to contain this widget
        ipadx=amount - add internal padding in x direction
        ipady=amount - add internal padding in y direction
        padx=amount - add padding in x direction
        pady=amount - add padding in y direction
        side=TOP or BOTTOM or LEFT or RIGHT -  where to add this widget.
        """
        self.tk.call(
              ('pack', 'configure', self._w)
              + self._options(cnf, kw))
    pack = configure = config = pack_configure
    def pack_forget(self):
        """Unmap this widget and do not use it for the packing order."""
        self.tk.call('pack', 'forget', self._w)
    forget = pack_forget
    def pack_info(self):
        """Return information about the packing options
        for this widget."""
        words = self.tk.splitlist(
            self.tk.call('pack', 'info', self._w))
        dict = {}
        for i in range(0, len(words), 2):
            key = words[i][1:]
            value = words[i+1]
            if value[:1] == '.':
                value = self._nametowidget(value)
            dict[key] = value
        return dict
    info = pack_info
    propagate = pack_propagate = Misc.pack_propagate
    slaves = pack_slaves = Misc.pack_slaves

class Place:
    """Geometry manager Place.

    Base class to use the methods place_* in every widget."""
    def place_configure(self, cnf={}, **kw):
        """Place a widget in the parent widget. Use as options:
        in=master - master relative to which the widget is placed.
        x=amount - locate anchor of this widget at position x of master
        y=amount - locate anchor of this widget at position y of master
        relx=amount - locate anchor of this widget between 0.0 and 1.0
                      relative to width of master (1.0 is right edge)
            rely=amount - locate anchor of this widget between 0.0 and 1.0
                      relative to height of master (1.0 is bottom edge)
            anchor=NSEW (or subset) - position anchor according to given direction
        width=amount - width of this widget in pixel
        height=amount - height of this widget in pixel
        relwidth=amount - width of this widget between 0.0 and 1.0
                          relative to width of master (1.0 is the same width
                  as the master)
            relheight=amount - height of this widget between 0.0 and 1.0
                           relative to height of master (1.0 is the same
                   height as the master)
            bordermode="inside" or "outside" - whether to take border width of master widget
                                               into account
            """
        for k in ['in_']:
            if kw.has_key(k):
                kw[k[:-1]] = kw[k]
                del kw[k]
        self.tk.call(
              ('place', 'configure', self._w)
              + self._options(cnf, kw))
    place = configure = config = place_configure
    def place_forget(self):
        """Unmap this widget."""
        self.tk.call('place', 'forget', self._w)
    forget = place_forget
    def place_info(self):
        """Return information about the placing options
        for this widget."""
        words = self.tk.splitlist(
            self.tk.call('place', 'info', self._w))
        dict = {}
        for i in range(0, len(words), 2):
            key = words[i][1:]
            value = words[i+1]
            if value[:1] == '.':
                value = self._nametowidget(value)
            dict[key] = value
        return dict
    info = place_info
    slaves = place_slaves = Misc.place_slaves

class Grid:
    """Geometry manager Grid.

    Base class to use the methods grid_* in every widget."""
    # Thanks to Masazumi Yoshikawa (yosikawa@isi.edu)
    def grid_configure(self, cnf={}, **kw):
        """Position a widget in the parent widget in a grid. Use as options:
        column=number - use cell identified with given column (starting with 0)
        columnspan=number - this widget will span several columns
        in=master - use master to contain this widget
        ipadx=amount - add internal padding in x direction
        ipady=amount - add internal padding in y direction
        padx=amount - add padding in x direction
        pady=amount - add padding in y direction
        row=number - use cell identified with given row (starting with 0)
        rowspan=number - this widget will span several rows
        sticky=NSEW - if cell is larger on which sides will this
                      widget stick to the cell boundary
        """
        self.tk.call(
              ('grid', 'configure', self._w)
              + self._options(cnf, kw))
    grid = configure = config = grid_configure
    bbox = grid_bbox = Misc.grid_bbox
    columnconfigure = grid_columnconfigure = Misc.grid_columnconfigure
    def grid_forget(self):
        """Unmap this widget."""
        self.tk.call('grid', 'forget', self._w)
    forget = grid_forget
    def grid_remove(self):
        """Unmap this widget but remember the grid options."""
        self.tk.call('grid', 'remove', self._w)
    def grid_info(self):
        """Return information about the options
        for positioning this widget in a grid."""
        words = self.tk.splitlist(
            self.tk.call('grid', 'info', self._w))
        dict = {}
        for i in range(0, len(words), 2):
            key = words[i][1:]
            value = words[i+1]
            if value[:1] == '.':
                value = self._nametowidget(value)
            dict[key] = value
        return dict
    info = grid_info
    location = grid_location = Misc.grid_location
    propagate = grid_propagate = Misc.grid_propagate
    rowconfigure = grid_rowconfigure = Misc.grid_rowconfigure
    size = grid_size = Misc.grid_size
    slaves = grid_slaves = Misc.grid_slaves

class BaseWidget(Misc):
    """Internal class."""
    def _setup(self, master, cnf):
        """Internal function. Sets up information about children."""
        if _support_default_root:
            global _default_root
            if not master:
                if not _default_root:
                    _default_root = Tk()
                master = _default_root
        self.master = master
        self.tk = master.tk
        name = None
        if cnf.has_key('name'):
            name = cnf['name']
            del cnf['name']
        if not name:
            name = `id(self)`
        self._name = name
        if master._w=='.':
            self._w = '.' + name
        else:
            self._w = master._w + '.' + name
        self.children = {}
        if self.master.children.has_key(self._name):
            self.master.children[self._name].destroy()
        self.master.children[self._name] = self
    def __init__(self, master, widgetName, cnf={}, kw={}, extra=()):
        """Construct a widget with the parent widget MASTER, a name WIDGETNAME
        and appropriate options."""
        if kw:
            cnf = _cnfmerge((cnf, kw))
        self.widgetName = widgetName
        BaseWidget._setup(self, master, cnf)
        classes = []
        for k in cnf.keys():
            if type(k) is ClassType:
                classes.append((k, cnf[k]))
                del cnf[k]
        self.tk.call(
            (widgetName, self._w) + extra + self._options(cnf))
        for k, v in classes:
            k.configure(self, v)
    def destroy(self):
        """Destroy this and all descendants widgets."""
        for c in self.children.values(): c.destroy()
        if self.master.children.has_key(self._name):
            del self.master.children[self._name]
        self.tk.call('destroy', self._w)
        Misc.destroy(self)
    def _do(self, name, args=()):
        # XXX Obsolete -- better use self.tk.call directly!
        return self.tk.call((self._w, name) + args)

class Widget(BaseWidget, Pack, Place, Grid):
    """Internal class.

    Base class for a widget which can be positioned with the geometry managers
    Pack, Place or Grid."""
    pass

class Toplevel(BaseWidget, Wm):
    """Toplevel widget, e.g. for dialogs."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a toplevel widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, class,
        colormap, container, cursor, height, highlightbackground,
        highlightcolor, highlightthickness, menu, relief, screen, takefocus,
        use, visual, width."""
        if kw:
            cnf = _cnfmerge((cnf, kw))
        extra = ()
        for wmkey in ['screen', 'class_', 'class', 'visual',
                  'colormap']:
            if cnf.has_key(wmkey):
                val = cnf[wmkey]
                # TBD: a hack needed because some keys
                # are not valid as keyword arguments
                if wmkey[-1] == '_': opt = '-'+wmkey[:-1]
                else: opt = '-'+wmkey
                extra = extra + (opt, val)
                del cnf[wmkey]
        BaseWidget.__init__(self, master, 'toplevel', cnf, {}, extra)
        root = self._root()
        self.iconname(root.iconname())
        self.title(root.title())
        self.protocol("WM_DELETE_WINDOW", self.destroy)

class Button(Widget):
    """Button widget."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a button widget with the parent MASTER.

        Valid resource names: activebackground, activeforeground, anchor,
        background, bd, bg, bitmap, borderwidth, command, cursor, default,
        disabledforeground, fg, font, foreground, height,
        highlightbackground, highlightcolor, highlightthickness, image,
        justify, padx, pady, relief, state, takefocus, text, textvariable,
        underline, width, wraplength."""
        Widget.__init__(self, master, 'button', cnf, kw)
    def tkButtonEnter(self, *dummy):
        self.tk.call('tkButtonEnter', self._w)
    def tkButtonLeave(self, *dummy):
        self.tk.call('tkButtonLeave', self._w)
    def tkButtonDown(self, *dummy):
        self.tk.call('tkButtonDown', self._w)
    def tkButtonUp(self, *dummy):
        self.tk.call('tkButtonUp', self._w)
    def tkButtonInvoke(self, *dummy):
        self.tk.call('tkButtonInvoke', self._w)
    def flash(self):
        self.tk.call(self._w, 'flash')
    def invoke(self):
        return self.tk.call(self._w, 'invoke')

# Indices:
# XXX I don't like these -- take them away
def AtEnd():
    return 'end'
def AtInsert(*args):
    s = 'insert'
    for a in args:
        if a: s = s + (' ' + a)
    return s
def AtSelFirst():
    return 'sel.first'
def AtSelLast():
    return 'sel.last'
def At(x, y=None):
    if y is None:
        return '@' + `x`
    else:
        return '@' + `x` + ',' + `y`

class Canvas(Widget):
    """Canvas widget to display graphical elements like lines or text."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a canvas widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, closeenough,
        confine, cursor, height, highlightbackground, highlightcolor,
        highlightthickness, insertbackground, insertborderwidth,
        insertofftime, insertontime, insertwidth, offset, relief,
        scrollregion, selectbackground, selectborderwidth, selectforeground,
        state, takefocus, width, xscrollcommand, xscrollincrement,
        yscrollcommand, yscrollincrement."""
        Widget.__init__(self, master, 'canvas', cnf, kw)
    def addtag(self, *args):
        """Internal function."""
        self.tk.call((self._w, 'addtag') + args)
    def addtag_above(self, newtag, tagOrId):
        """Add tag NEWTAG to all items above TAGORID."""
        self.addtag(newtag, 'above', tagOrId)
    def addtag_all(self, newtag):
        """Add tag NEWTAG to all items."""
        self.addtag(newtag, 'all')
    def addtag_below(self, newtag, tagOrId):
        """Add tag NEWTAG to all items below TAGORID."""
        self.addtag(newtag, 'below', tagOrId)
    def addtag_closest(self, newtag, x, y, halo=None, start=None):
        """Add tag NEWTAG to item which is closest to pixel at X, Y.
        If several match take the top-most.
        All items closer than HALO are considered overlapping (all are
        closests). If START is specified the next below this tag is taken."""
        self.addtag(newtag, 'closest', x, y, halo, start)
    def addtag_enclosed(self, newtag, x1, y1, x2, y2):
        """Add tag NEWTAG to all items in the rectangle defined
        by X1,Y1,X2,Y2."""
        self.addtag(newtag, 'enclosed', x1, y1, x2, y2)
    def addtag_overlapping(self, newtag, x1, y1, x2, y2):
        """Add tag NEWTAG to all items which overlap the rectangle
        defined by X1,Y1,X2,Y2."""
        self.addtag(newtag, 'overlapping', x1, y1, x2, y2)
    def addtag_withtag(self, newtag, tagOrId):
        """Add tag NEWTAG to all items with TAGORID."""
        self.addtag(newtag, 'withtag', tagOrId)
    def bbox(self, *args):
        """Return a tuple of X1,Y1,X2,Y2 coordinates for a rectangle
        which encloses all items with tags specified as arguments."""
        return self._getints(
            self.tk.call((self._w, 'bbox') + args)) or None
    def tag_unbind(self, tagOrId, sequence, funcid=None):
        """Unbind for all items with TAGORID for event SEQUENCE  the
        function identified with FUNCID."""
        self.tk.call(self._w, 'bind', tagOrId, sequence, '')
        if funcid:
            self.deletecommand(funcid)
    def tag_bind(self, tagOrId, sequence=None, func=None, add=None):
        """Bind to all items with TAGORID at event SEQUENCE a call to function FUNC.

        An additional boolean parameter ADD specifies whether FUNC will be
        called additionally to the other bound function or whether it will
        replace the previous function. See bind for the return value."""
        return self._bind((self._w, 'bind', tagOrId),
                  sequence, func, add)
    def canvasx(self, screenx, gridspacing=None):
        """Return the canvas x coordinate of pixel position SCREENX rounded
        to nearest multiple of GRIDSPACING units."""
        return getdouble(self.tk.call(
            self._w, 'canvasx', screenx, gridspacing))
    def canvasy(self, screeny, gridspacing=None):
        """Return the canvas y coordinate of pixel position SCREENY rounded
        to nearest multiple of GRIDSPACING units."""
        return getdouble(self.tk.call(
            self._w, 'canvasy', screeny, gridspacing))
    def coords(self, *args):
        """Return a list of coordinates for the item given in ARGS."""
        # XXX Should use _flatten on args
        return map(getdouble,
                           self.tk.splitlist(
                   self.tk.call((self._w, 'coords') + args)))
    def _create(self, itemType, args, kw): # Args: (val, val, ..., cnf={})
        """Internal function."""
        args = _flatten(args)
        cnf = args[-1]
        if type(cnf) in (DictionaryType, TupleType):
            args = args[:-1]
        else:
            cnf = {}
        return getint(apply(
            self.tk.call,
            (self._w, 'create', itemType)
            + args + self._options(cnf, kw)))
    def create_arc(self, *args, **kw):
        """Create arc shaped region with coordinates x1,y1,x2,y2."""
        return self._create('arc', args, kw)
    def create_bitmap(self, *args, **kw):
        """Create bitmap with coordinates x1,y1."""
        return self._create('bitmap', args, kw)
    def create_image(self, *args, **kw):
        """Create image item with coordinates x1,y1."""
        return self._create('image', args, kw)
    def create_line(self, *args, **kw):
        """Create line with coordinates x1,y1,...,xn,yn."""
        return self._create('line', args, kw)
    def create_oval(self, *args, **kw):
        """Create oval with coordinates x1,y1,x2,y2."""
        return self._create('oval', args, kw)
    def create_polygon(self, *args, **kw):
        """Create polygon with coordinates x1,y1,...,xn,yn."""
        return self._create('polygon', args, kw)
    def create_rectangle(self, *args, **kw):
        """Create rectangle with coordinates x1,y1,x2,y2."""
        return self._create('rectangle', args, kw)
    def create_text(self, *args, **kw):
        """Create text with coordinates x1,y1."""
        return self._create('text', args, kw)
    def create_window(self, *args, **kw):
        """Create window with coordinates x1,y1,x2,y2."""
        return self._create('window', args, kw)
    def dchars(self, *args):
        """Delete characters of text items identified by tag or id in ARGS (possibly
        several times) from FIRST to LAST character (including)."""
        self.tk.call((self._w, 'dchars') + args)
    def delete(self, *args):
        """Delete items identified by all tag or ids contained in ARGS."""
        self.tk.call((self._w, 'delete') + args)
    def dtag(self, *args):
        """Delete tag or id given as last arguments in ARGS from items
        identified by first argument in ARGS."""
        self.tk.call((self._w, 'dtag') + args)
    def find(self, *args):
        """Internal function."""
        return self._getints(
            self.tk.call((self._w, 'find') + args)) or ()
    def find_above(self, tagOrId):
        """Return items above TAGORID."""
        return self.find('above', tagOrId)
    def find_all(self):
        """Return all items."""
        return self.find('all')
    def find_below(self, tagOrId):
        """Return all items below TAGORID."""
        return self.find('below', tagOrId)
    def find_closest(self, x, y, halo=None, start=None):
        """Return item which is closest to pixel at X, Y.
        If several match take the top-most.
        All items closer than HALO are considered overlapping (all are
        closests). If START is specified the next below this tag is taken."""
        return self.find('closest', x, y, halo, start)
    def find_enclosed(self, x1, y1, x2, y2):
        """Return all items in rectangle defined
        by X1,Y1,X2,Y2."""
        return self.find('enclosed', x1, y1, x2, y2)
    def find_overlapping(self, x1, y1, x2, y2):
        """Return all items which overlap the rectangle
        defined by X1,Y1,X2,Y2."""
        return self.find('overlapping', x1, y1, x2, y2)
    def find_withtag(self, tagOrId):
        """Return all items with TAGORID."""
        return self.find('withtag', tagOrId)
    def focus(self, *args):
        """Set focus to the first item specified in ARGS."""
        return self.tk.call((self._w, 'focus') + args)
    def gettags(self, *args):
        """Return tags associated with the first item specified in ARGS."""
        return self.tk.splitlist(
            self.tk.call((self._w, 'gettags') + args))
    def icursor(self, *args):
        """Set cursor at position POS in the item identified by TAGORID.
        In ARGS TAGORID must be first."""
        self.tk.call((self._w, 'icursor') + args)
    def index(self, *args):
        """Return position of cursor as integer in item specified in ARGS."""
        return getint(self.tk.call((self._w, 'index') + args))
    def insert(self, *args):
        """Insert TEXT in item TAGORID at position POS. ARGS must
        be TAGORID POS TEXT."""
        self.tk.call((self._w, 'insert') + args)
    def itemcget(self, tagOrId, option):
        """Return the resource value for an OPTION for item TAGORID."""
        return self.tk.call(
            (self._w, 'itemcget') + (tagOrId, '-'+option))
    def itemconfigure(self, tagOrId, cnf=None, **kw):
        """Configure resources of an item TAGORID.

        The values for resources are specified as keyword
        arguments. To get an overview about
        the allowed keyword arguments call the method without arguments.
        """
        if cnf is None and not kw:
            cnf = {}
            for x in self.tk.split(
                self.tk.call(self._w,
                         'itemconfigure', tagOrId)):
                cnf[x[0][1:]] = (x[0][1:],) + x[1:]
            return cnf
        if type(cnf) == StringType and not kw:
            x = self.tk.split(self.tk.call(
                self._w, 'itemconfigure', tagOrId, '-'+cnf))
            return (x[0][1:],) + x[1:]
        self.tk.call((self._w, 'itemconfigure', tagOrId) +
                 self._options(cnf, kw))
    itemconfig = itemconfigure
    # lower, tkraise/lift hide Misc.lower, Misc.tkraise/lift,
    # so the preferred name for them is tag_lower, tag_raise
    # (similar to tag_bind, and similar to the Text widget);
    # unfortunately can't delete the old ones yet (maybe in 1.6)
    def tag_lower(self, *args):
        """Lower an item TAGORID given in ARGS
        (optional below another item)."""
        self.tk.call((self._w, 'lower') + args)
    lower = tag_lower
    def move(self, *args):
        """Move an item TAGORID given in ARGS."""
        self.tk.call((self._w, 'move') + args)
    def postscript(self, cnf={}, **kw):
        """Print the contents of the canvas to a postscript
        file. Valid options: colormap, colormode, file, fontmap,
        height, pageanchor, pageheight, pagewidth, pagex, pagey,
        rotate, witdh, x, y."""
        return self.tk.call((self._w, 'postscript') +
                    self._options(cnf, kw))
    def tag_raise(self, *args):
        """Raise an item TAGORID given in ARGS
        (optional above another item)."""
        self.tk.call((self._w, 'raise') + args)
    lift = tkraise = tag_raise
    def scale(self, *args):
        """Scale item TAGORID with XORIGIN, YORIGIN, XSCALE, YSCALE."""
        self.tk.call((self._w, 'scale') + args)
    def scan_mark(self, x, y):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x, y)
    def scan_dragto(self, x, y):
        """Adjust the view of the canvas to 10 times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        self.tk.call(self._w, 'scan', 'dragto', x, y)
    def select_adjust(self, tagOrId, index):
        """Adjust the end of the selection near the cursor of an item TAGORID to index."""
        self.tk.call(self._w, 'select', 'adjust', tagOrId, index)
    def select_clear(self):
        """Clear the selection if it is in this widget."""
        self.tk.call(self._w, 'select', 'clear')
    def select_from(self, tagOrId, index):
        """Set the fixed end of a selection in item TAGORID to INDEX."""
        self.tk.call(self._w, 'select', 'from', tagOrId, index)
    def select_item(self):
        """Return the item which has the selection."""
        return self.tk.call(self._w, 'select', 'item') or None
    def select_to(self, tagOrId, index):
        """Set the variable end of a selection in item TAGORID to INDEX."""
        self.tk.call(self._w, 'select', 'to', tagOrId, index)
    def type(self, tagOrId):
        """Return the type of the item TAGORID."""
        return self.tk.call(self._w, 'type', tagOrId) or None
    def xview(self, *args):
        """Query and change horizontal position of the view."""
        if not args:
            return self._getdoubles(self.tk.call(self._w, 'xview'))
        self.tk.call((self._w, 'xview') + args)
    def xview_moveto(self, fraction):
        """Adjusts the view in the window so that FRACTION of the
        total width of the canvas is off-screen to the left."""
        self.tk.call(self._w, 'xview', 'moveto', fraction)
    def xview_scroll(self, number, what):
        """Shift the x-view according to NUMBER which is measured in "units" or "pages" (WHAT)."""
        self.tk.call(self._w, 'xview', 'scroll', number, what)
    def yview(self, *args):
        """Query and change vertical position of the view."""
        if not args:
            return self._getdoubles(self.tk.call(self._w, 'yview'))
        self.tk.call((self._w, 'yview') + args)
    def yview_moveto(self, fraction):
        """Adjusts the view in the window so that FRACTION of the
        total height of the canvas is off-screen to the top."""
        self.tk.call(self._w, 'yview', 'moveto', fraction)
    def yview_scroll(self, number, what):
        """Shift the y-view according to NUMBER which is measured in "units" or "pages" (WHAT)."""
        self.tk.call(self._w, 'yview', 'scroll', number, what)

class Checkbutton(Widget):
    """Checkbutton widget which is either in on- or off-state."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a checkbutton widget with the parent MASTER.

        Valid resource names: activebackground, activeforeground, anchor,
        background, bd, bg, bitmap, borderwidth, command, cursor,
        disabledforeground, fg, font, foreground, height,
        highlightbackground, highlightcolor, highlightthickness, image,
        indicatoron, justify, offvalue, onvalue, padx, pady, relief,
        selectcolor, selectimage, state, takefocus, text, textvariable,
        underline, variable, width, wraplength."""
        Widget.__init__(self, master, 'checkbutton', cnf, kw)
    def deselect(self):
        """Put the button in off-state."""
        self.tk.call(self._w, 'deselect')
    def flash(self):
        """Flash the button."""
        self.tk.call(self._w, 'flash')
    def invoke(self):
        """Toggle the button and invoke a command if given as resource."""
        return self.tk.call(self._w, 'invoke')
    def select(self):
        """Put the button in on-state."""
        self.tk.call(self._w, 'select')
    def toggle(self):
        """Toggle the button."""
        self.tk.call(self._w, 'toggle')

class Entry(Widget):
    """Entry widget which allows to display simple text."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct an entry widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, cursor,
        exportselection, fg, font, foreground, highlightbackground,
        highlightcolor, highlightthickness, insertbackground,
        insertborderwidth, insertofftime, insertontime, insertwidth,
        invalidcommand, invcmd, justify, relief, selectbackground,
        selectborderwidth, selectforeground, show, state, takefocus,
        textvariable, validate, validatecommand, vcmd, width,
        xscrollcommand."""
        Widget.__init__(self, master, 'entry', cnf, kw)
    def delete(self, first, last=None):
        """Delete text from FIRST to LAST (not included)."""
        self.tk.call(self._w, 'delete', first, last)
    def get(self):
        """Return the text."""
        return self.tk.call(self._w, 'get')
    def icursor(self, index):
        """Insert cursor at INDEX."""
        self.tk.call(self._w, 'icursor', index)
    def index(self, index):
        """Return position of cursor."""
        return getint(self.tk.call(
            self._w, 'index', index))
    def insert(self, index, string):
        """Insert STRING at INDEX."""
        self.tk.call(self._w, 'insert', index, string)
    def scan_mark(self, x):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x)
    def scan_dragto(self, x):
        """Adjust the view of the canvas to 10 times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        self.tk.call(self._w, 'scan', 'dragto', x)
    def selection_adjust(self, index):
        """Adjust the end of the selection near the cursor to INDEX."""
        self.tk.call(self._w, 'selection', 'adjust', index)
    select_adjust = selection_adjust
    def selection_clear(self):
        """Clear the selection if it is in this widget."""
        self.tk.call(self._w, 'selection', 'clear')
    select_clear = selection_clear
    def selection_from(self, index):
        """Set the fixed end of a selection to INDEX."""
        self.tk.call(self._w, 'selection', 'from', index)
    select_from = selection_from
    def selection_present(self):
        """Return whether the widget has the selection."""
        return self.tk.getboolean(
            self.tk.call(self._w, 'selection', 'present'))
    select_present = selection_present
    def selection_range(self, start, end):
        """Set the selection from START to END (not included)."""
        self.tk.call(self._w, 'selection', 'range', start, end)
    select_range = selection_range
    def selection_to(self, index):
        """Set the variable end of a selection to INDEX."""
        self.tk.call(self._w, 'selection', 'to', index)
    select_to = selection_to
    def xview(self, index):
        """Query and change horizontal position of the view."""
        self.tk.call(self._w, 'xview', index)
    def xview_moveto(self, fraction):
        """Adjust the view in the window so that FRACTION of the
        total width of the entry is off-screen to the left."""
        self.tk.call(self._w, 'xview', 'moveto', fraction)
    def xview_scroll(self, number, what):
        """Shift the x-view according to NUMBER which is measured in "units" or "pages" (WHAT)."""
        self.tk.call(self._w, 'xview', 'scroll', number, what)

class Frame(Widget):
    """Frame widget which may contain other widgets and can have a 3D border."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a frame widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, class,
        colormap, container, cursor, height, highlightbackground,
        highlightcolor, highlightthickness, relief, takefocus, visual, width."""
        cnf = _cnfmerge((cnf, kw))
        extra = ()
        if cnf.has_key('class_'):
            extra = ('-class', cnf['class_'])
            del cnf['class_']
        elif cnf.has_key('class'):
            extra = ('-class', cnf['class'])
            del cnf['class']
        Widget.__init__(self, master, 'frame', cnf, {}, extra)

class Label(Widget):
    """Label widget which can display text and bitmaps."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a label widget with the parent MASTER.

        Valid resource names: anchor, background, bd, bg, bitmap,
        borderwidth, cursor, fg, font, foreground, height,
        highlightbackground, highlightcolor, highlightthickness, image,
        justify, padx, pady, relief, takefocus, text, textvariable,
        underline, width, wraplength."""
        Widget.__init__(self, master, 'label', cnf, kw)

class Listbox(Widget):
    """Listbox widget which can display a list of strings."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a listbox widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, cursor,
        exportselection, fg, font, foreground, height, highlightbackground,
        highlightcolor, highlightthickness, relief, selectbackground,
        selectborderwidth, selectforeground, selectmode, setgrid, takefocus,
        width, xscrollcommand, yscrollcommand, listvariable."""
        Widget.__init__(self, master, 'listbox', cnf, kw)
    def activate(self, index):
        """Activate item identified by INDEX."""
        self.tk.call(self._w, 'activate', index)
    def bbox(self, *args):
        """Return a tuple of X1,Y1,X2,Y2 coordinates for a rectangle
        which encloses the item identified by index in ARGS."""
        return self._getints(
            self.tk.call((self._w, 'bbox') + args)) or None
    def curselection(self):
        """Return list of indices of currently selected item."""
        # XXX Ought to apply self._getints()...
        return self.tk.splitlist(self.tk.call(
            self._w, 'curselection'))
    def delete(self, first, last=None):
        """Delete items from FIRST to LAST (not included)."""
        self.tk.call(self._w, 'delete', first, last)
    def get(self, first, last=None):
        """Get list of items from FIRST to LAST (not included)."""
        if last:
            return self.tk.splitlist(self.tk.call(
                self._w, 'get', first, last))
        else:
            return self.tk.call(self._w, 'get', first)
    def index(self, index):
        """Return index of item identified with INDEX."""
        i = self.tk.call(self._w, 'index', index)
        if i == 'none': return None
        return getint(i)
    def insert(self, index, *elements):
        """Insert ELEMENTS at INDEX."""
        self.tk.call((self._w, 'insert', index) + elements)
    def nearest(self, y):
        """Get index of item which is nearest to y coordinate Y."""
        return getint(self.tk.call(
            self._w, 'nearest', y))
    def scan_mark(self, x, y):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x, y)
    def scan_dragto(self, x, y):
        """Adjust the view of the listbox to 10 times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        self.tk.call(self._w, 'scan', 'dragto', x, y)
    def see(self, index):
        """Scroll such that INDEX is visible."""
        self.tk.call(self._w, 'see', index)
    def selection_anchor(self, index):
        """Set the fixed end oft the selection to INDEX."""
        self.tk.call(self._w, 'selection', 'anchor', index)
    select_anchor = selection_anchor
    def selection_clear(self, first, last=None):
        """Clear the selection from FIRST to LAST (not included)."""
        self.tk.call(self._w,
                 'selection', 'clear', first, last)
    select_clear = selection_clear
    def selection_includes(self, index):
        """Return 1 if INDEX is part of the selection."""
        return self.tk.getboolean(self.tk.call(
            self._w, 'selection', 'includes', index))
    select_includes = selection_includes
    def selection_set(self, first, last=None):
        """Set the selection from FIRST to LAST (not included) without
        changing the currently selected elements."""
        self.tk.call(self._w, 'selection', 'set', first, last)
    select_set = selection_set
    def size(self):
        """Return the number of elements in the listbox."""
        return getint(self.tk.call(self._w, 'size'))
    def xview(self, *what):
        """Query and change horizontal position of the view."""
        if not what:
            return self._getdoubles(self.tk.call(self._w, 'xview'))
        self.tk.call((self._w, 'xview') + what)
    def xview_moveto(self, fraction):
        """Adjust the view in the window so that FRACTION of the
        total width of the entry is off-screen to the left."""
        self.tk.call(self._w, 'xview', 'moveto', fraction)
    def xview_scroll(self, number, what):
        """Shift the x-view according to NUMBER which is measured in "units" or "pages" (WHAT)."""
        self.tk.call(self._w, 'xview', 'scroll', number, what)
    def yview(self, *what):
        """Query and change vertical position of the view."""
        if not what:
            return self._getdoubles(self.tk.call(self._w, 'yview'))
        self.tk.call((self._w, 'yview') + what)
    def yview_moveto(self, fraction):
        """Adjust the view in the window so that FRACTION of the
        total width of the entry is off-screen to the top."""
        self.tk.call(self._w, 'yview', 'moveto', fraction)
    def yview_scroll(self, number, what):
        """Shift the y-view according to NUMBER which is measured in "units" or "pages" (WHAT)."""
        self.tk.call(self._w, 'yview', 'scroll', number, what)
    def itemcget(self, index, option):
        """Return the resource value for an ITEM and an OPTION."""
        return self.tk.call(
            (self._w, 'itemcget') + (index, '-'+option))
    def itemconfigure(self, index, cnf=None, **kw):
        """Configure resources of an ITEM.

        The values for resources are specified as keyword arguments.
        To get an overview about the allowed keyword arguments
        call the method without arguments.
        Valid resource names: background, bg, foreground, fg,
        selectbackground, selectforeground."""
        if cnf is None and not kw:
            cnf = {}
            for x in self.tk.split(
                self.tk.call(self._w, 'itemconfigure', index)):
                cnf[x[0][1:]] = (x[0][1:],) + x[1:]
            return cnf
        if type(cnf) == StringType and not kw:
            x = self.tk.split(self.tk.call(
                self._w, 'itemconfigure', index, '-'+cnf))
            return (x[0][1:],) + x[1:]
        self.tk.call((self._w, 'itemconfigure', index) +
                     self._options(cnf, kw))
    itemconfig = itemconfigure

class Menu(Widget):
    """Menu widget which allows to display menu bars, pull-down menus and pop-up menus."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct menu widget with the parent MASTER.

        Valid resource names: activebackground, activeborderwidth,
        activeforeground, background, bd, bg, borderwidth, cursor,
        disabledforeground, fg, font, foreground, postcommand, relief,
        selectcolor, takefocus, tearoff, tearoffcommand, title, type."""
        Widget.__init__(self, master, 'menu', cnf, kw)
    def tk_bindForTraversal(self):
        pass # obsolete since Tk 4.0
    def tk_mbPost(self):
        self.tk.call('tk_mbPost', self._w)
    def tk_mbUnpost(self):
        self.tk.call('tk_mbUnpost')
    def tk_traverseToMenu(self, char):
        self.tk.call('tk_traverseToMenu', self._w, char)
    def tk_traverseWithinMenu(self, char):
        self.tk.call('tk_traverseWithinMenu', self._w, char)
    def tk_getMenuButtons(self):
        return self.tk.call('tk_getMenuButtons', self._w)
    def tk_nextMenu(self, count):
        self.tk.call('tk_nextMenu', count)
    def tk_nextMenuEntry(self, count):
        self.tk.call('tk_nextMenuEntry', count)
    def tk_invokeMenu(self):
        self.tk.call('tk_invokeMenu', self._w)
    def tk_firstMenu(self):
        self.tk.call('tk_firstMenu', self._w)
    def tk_mbButtonDown(self):
        self.tk.call('tk_mbButtonDown', self._w)
    def tk_popup(self, x, y, entry=""):
        """Post the menu at position X,Y with entry ENTRY."""
        self.tk.call('tk_popup', self._w, x, y, entry)
    def activate(self, index):
        """Activate entry at INDEX."""
        self.tk.call(self._w, 'activate', index)
    def add(self, itemType, cnf={}, **kw):
        """Internal function."""
        self.tk.call((self._w, 'add', itemType) +
                 self._options(cnf, kw))
    def add_cascade(self, cnf={}, **kw):
        """Add hierarchical menu item."""
        self.add('cascade', cnf or kw)
    def add_checkbutton(self, cnf={}, **kw):
        """Add checkbutton menu item."""
        self.add('checkbutton', cnf or kw)
    def add_command(self, cnf={}, **kw):
        """Add command menu item."""
        self.add('command', cnf or kw)
    def add_radiobutton(self, cnf={}, **kw):
        """Addd radio menu item."""
        self.add('radiobutton', cnf or kw)
    def add_separator(self, cnf={}, **kw):
        """Add separator."""
        self.add('separator', cnf or kw)
    def insert(self, index, itemType, cnf={}, **kw):
        """Internal function."""
        self.tk.call((self._w, 'insert', index, itemType) +
                 self._options(cnf, kw))
    def insert_cascade(self, index, cnf={}, **kw):
        """Add hierarchical menu item at INDEX."""
        self.insert(index, 'cascade', cnf or kw)
    def insert_checkbutton(self, index, cnf={}, **kw):
        """Add checkbutton menu item at INDEX."""
        self.insert(index, 'checkbutton', cnf or kw)
    def insert_command(self, index, cnf={}, **kw):
        """Add command menu item at INDEX."""
        self.insert(index, 'command', cnf or kw)
    def insert_radiobutton(self, index, cnf={}, **kw):
        """Addd radio menu item at INDEX."""
        self.insert(index, 'radiobutton', cnf or kw)
    def insert_separator(self, index, cnf={}, **kw):
        """Add separator at INDEX."""
        self.insert(index, 'separator', cnf or kw)
    def delete(self, index1, index2=None):
        """Delete menu items between INDEX1 and INDEX2 (not included)."""
        self.tk.call(self._w, 'delete', index1, index2)
    def entrycget(self, index, option):
        """Return the resource value of an menu item for OPTION at INDEX."""
        return self.tk.call(self._w, 'entrycget', index, '-' + option)
    def entryconfigure(self, index, cnf=None, **kw):
        """Configure a menu item at INDEX."""
        if cnf is None and not kw:
            cnf = {}
            for x in self.tk.split(self.tk.call(
                (self._w, 'entryconfigure', index))):
                cnf[x[0][1:]] = (x[0][1:],) + x[1:]
            return cnf
        if type(cnf) == StringType and not kw:
            x = self.tk.split(self.tk.call(
                (self._w, 'entryconfigure', index, '-'+cnf)))
            return (x[0][1:],) + x[1:]
        self.tk.call((self._w, 'entryconfigure', index)
              + self._options(cnf, kw))
    entryconfig = entryconfigure
    def index(self, index):
        """Return the index of a menu item identified by INDEX."""
        i = self.tk.call(self._w, 'index', index)
        if i == 'none': return None
        return getint(i)
    def invoke(self, index):
        """Invoke a menu item identified by INDEX and execute
        the associated command."""
        return self.tk.call(self._w, 'invoke', index)
    def post(self, x, y):
        """Display a menu at position X,Y."""
        self.tk.call(self._w, 'post', x, y)
    def type(self, index):
        """Return the type of the menu item at INDEX."""
        return self.tk.call(self._w, 'type', index)
    def unpost(self):
        """Unmap a menu."""
        self.tk.call(self._w, 'unpost')
    def yposition(self, index):
        """Return the y-position of the topmost pixel of the menu item at INDEX."""
        return getint(self.tk.call(
            self._w, 'yposition', index))

class Menubutton(Widget):
    """Menubutton widget, obsolete since Tk8.0."""
    def __init__(self, master=None, cnf={}, **kw):
        Widget.__init__(self, master, 'menubutton', cnf, kw)

class Message(Widget):
    """Message widget to display multiline text. Obsolete since Label does it too."""
    def __init__(self, master=None, cnf={}, **kw):
        Widget.__init__(self, master, 'message', cnf, kw)

class Radiobutton(Widget):
    """Radiobutton widget which shows only one of several buttons in on-state."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a radiobutton widget with the parent MASTER.

        Valid resource names: activebackground, activeforeground, anchor,
        background, bd, bg, bitmap, borderwidth, command, cursor,
        disabledforeground, fg, font, foreground, height,
        highlightbackground, highlightcolor, highlightthickness, image,
        indicatoron, justify, padx, pady, relief, selectcolor, selectimage,
        state, takefocus, text, textvariable, underline, value, variable,
        width, wraplength."""
        Widget.__init__(self, master, 'radiobutton', cnf, kw)
    def deselect(self):
        """Put the button in off-state."""

        self.tk.call(self._w, 'deselect')
    def flash(self):
        """Flash the button."""
        self.tk.call(self._w, 'flash')
    def invoke(self):
        """Toggle the button and invoke a command if given as resource."""
        return self.tk.call(self._w, 'invoke')
    def select(self):
        """Put the button in on-state."""
        self.tk.call(self._w, 'select')

class Scale(Widget):
    """Scale widget which can display a numerical scale."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a scale widget with the parent MASTER.

        Valid resource names: activebackground, background, bigincrement, bd,
        bg, borderwidth, command, cursor, digits, fg, font, foreground, from,
        highlightbackground, highlightcolor, highlightthickness, label,
        length, orient, relief, repeatdelay, repeatinterval, resolution,
        showvalue, sliderlength, sliderrelief, state, takefocus,
        tickinterval, to, troughcolor, variable, width."""
        Widget.__init__(self, master, 'scale', cnf, kw)
    def get(self):
        """Get the current value as integer or float."""
        value = self.tk.call(self._w, 'get')
        try:
            return getint(value)
        except ValueError:
            return getdouble(value)
    def set(self, value):
        """Set the value to VALUE."""
        self.tk.call(self._w, 'set', value)
    def coords(self, value=None):
        """Return a tuple (X,Y) of the point along the centerline of the
        trough that corresponds to VALUE or the current value if None is
        given."""

        return self._getints(self.tk.call(self._w, 'coords', value))
    def identify(self, x, y):
        """Return where the point X,Y lies. Valid return values are "slider",
        "though1" and "though2"."""
        return self.tk.call(self._w, 'identify', x, y)

class Scrollbar(Widget):
    """Scrollbar widget which displays a slider at a certain position."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a scrollbar widget with the parent MASTER.

        Valid resource names: activebackground, activerelief,
        background, bd, bg, borderwidth, command, cursor,
        elementborderwidth, highlightbackground,
        highlightcolor, highlightthickness, jump, orient,
        relief, repeatdelay, repeatinterval, takefocus,
        troughcolor, width."""
        Widget.__init__(self, master, 'scrollbar', cnf, kw)
    def activate(self, index):
        """Display the element at INDEX with activebackground and activerelief.
        INDEX can be "arrow1","slider" or "arrow2"."""
        self.tk.call(self._w, 'activate', index)
    def delta(self, deltax, deltay):
        """Return the fractional change of the scrollbar setting if it
        would be moved by DELTAX or DELTAY pixels."""
        return getdouble(
            self.tk.call(self._w, 'delta', deltax, deltay))
    def fraction(self, x, y):
        """Return the fractional value which corresponds to a slider
        position of X,Y."""
        return getdouble(self.tk.call(self._w, 'fraction', x, y))
    def identify(self, x, y):
        """Return the element under position X,Y as one of
        "arrow1","slider","arrow2" or ""."""
        return self.tk.call(self._w, 'identify', x, y)
    def get(self):
        """Return the current fractional values (upper and lower end)
        of the slider position."""
        return self._getdoubles(self.tk.call(self._w, 'get'))
    def set(self, *args):
        """Set the fractional values of the slider position (upper and
        lower ends as value between 0 and 1)."""
        self.tk.call((self._w, 'set') + args)

class Text(Widget):
    """Text widget which can display text in various forms."""
    # XXX Add dump()
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a text widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, cursor,
        exportselection, fg, font, foreground, height,
        highlightbackground, highlightcolor, highlightthickness,
        insertbackground, insertborderwidth, insertofftime,
        insertontime, insertwidth, padx, pady, relief,
        selectbackground, selectborderwidth, selectforeground,
        setgrid, spacing1, spacing2, spacing3, state, tabs, takefocus,
        width, wrap, xscrollcommand, yscrollcommand."""
        Widget.__init__(self, master, 'text', cnf, kw)
    def bbox(self, *args):
        """Return a tuple of (x,y,width,height) which gives the bounding
        box of the visible part of the character at the index in ARGS."""
        return self._getints(
            self.tk.call((self._w, 'bbox') + args)) or None
    def tk_textSelectTo(self, index):
        self.tk.call('tk_textSelectTo', self._w, index)
    def tk_textBackspace(self):
        self.tk.call('tk_textBackspace', self._w)
    def tk_textIndexCloser(self, a, b, c):
        self.tk.call('tk_textIndexCloser', self._w, a, b, c)
    def tk_textResetAnchor(self, index):
        self.tk.call('tk_textResetAnchor', self._w, index)
    def compare(self, index1, op, index2):
        """Return whether between index INDEX1 and index INDEX2 the
        relation OP is satisfied. OP is one of <, <=, ==, >=, >, or !=."""
        return self.tk.getboolean(self.tk.call(
            self._w, 'compare', index1, op, index2))
    def debug(self, boolean=None):
        """Turn on the internal consistency checks of the B-Tree inside the text
        widget according to BOOLEAN."""
        return self.tk.getboolean(self.tk.call(
            self._w, 'debug', boolean))
    def delete(self, index1, index2=None):
        """Delete the characters between INDEX1 and INDEX2 (not included)."""
        self.tk.call(self._w, 'delete', index1, index2)
    def dlineinfo(self, index):
        """Return tuple (x,y,width,height,baseline) giving the bounding box
        and baseline position of the visible part of the line containing
        the character at INDEX."""
        return self._getints(self.tk.call(self._w, 'dlineinfo', index))
    def get(self, index1, index2=None):
        """Return the text from INDEX1 to INDEX2 (not included)."""
        return self.tk.call(self._w, 'get', index1, index2)
    # (Image commands are new in 8.0)
    def image_cget(self, index, option):
        """Return the value of OPTION of an embedded image at INDEX."""
        if option[:1] != "-":
            option = "-" + option
        if option[-1:] == "_":
            option = option[:-1]
        return self.tk.call(self._w, "image", "cget", index, option)
    def image_configure(self, index, cnf={}, **kw):
        """Configure an embedded image at INDEX."""
        if not cnf and not kw:
            cnf = {}
            for x in self.tk.split(
                    self.tk.call(
                    self._w, "image", "configure", index)):
                cnf[x[0][1:]] = (x[0][1:],) + x[1:]
            return cnf
        apply(self.tk.call,
              (self._w, "image", "configure", index)
              + self._options(cnf, kw))
    def image_create(self, index, cnf={}, **kw):
        """Create an embedded image at INDEX."""
        return apply(self.tk.call,
                 (self._w, "image", "create", index)
                 + self._options(cnf, kw))
    def image_names(self):
        """Return all names of embedded images in this widget."""
        return self.tk.call(self._w, "image", "names")
    def index(self, index):
        """Return the index in the form line.char for INDEX."""
        return self.tk.call(self._w, 'index', index)
    def insert(self, index, chars, *args):
        """Insert CHARS before the characters at INDEX. An additional
        tag can be given in ARGS. Additional CHARS and tags can follow in ARGS."""
        self.tk.call((self._w, 'insert', index, chars) + args)
    def mark_gravity(self, markName, direction=None):
        """Change the gravity of a mark MARKNAME to DIRECTION (LEFT or RIGHT).
        Return the current value if None is given for DIRECTION."""
        return self.tk.call(
            (self._w, 'mark', 'gravity', markName, direction))
    def mark_names(self):
        """Return all mark names."""
        return self.tk.splitlist(self.tk.call(
            self._w, 'mark', 'names'))
    def mark_set(self, markName, index):
        """Set mark MARKNAME before the character at INDEX."""
        self.tk.call(self._w, 'mark', 'set', markName, index)
    def mark_unset(self, *markNames):
        """Delete all marks in MARKNAMES."""
        self.tk.call((self._w, 'mark', 'unset') + markNames)
    def mark_next(self, index):
        """Return the name of the next mark after INDEX."""
        return self.tk.call(self._w, 'mark', 'next', index) or None
    def mark_previous(self, index):
        """Return the name of the previous mark before INDEX."""
        return self.tk.call(self._w, 'mark', 'previous', index) or None
    def scan_mark(self, x, y):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x, y)
    def scan_dragto(self, x, y):
        """Adjust the view of the text to 10 times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        self.tk.call(self._w, 'scan', 'dragto', x, y)
    def search(self, pattern, index, stopindex=None,
           forwards=None, backwards=None, exact=None,
           regexp=None, nocase=None, count=None):
        """Search PATTERN beginning from INDEX until STOPINDEX.
        Return the index of the first character of a match or an empty string."""
        args = [self._w, 'search']
        if forwards: args.append('-forwards')
        if backwards: args.append('-backwards')
        if exact: args.append('-exact')
        if regexp: args.append('-regexp')
        if nocase: args.append('-nocase')
        if count: args.append('-count'); args.append(count)
        if pattern[0] == '-': args.append('--')
        args.append(pattern)
        args.append(index)
        if stopindex: args.append(stopindex)
        return self.tk.call(tuple(args))
    def see(self, index):
        """Scroll such that the character at INDEX is visible."""
        self.tk.call(self._w, 'see', index)
    def tag_add(self, tagName, index1, *args):
        """Add tag TAGNAME to all characters between INDEX1 and index2 in ARGS.
        Additional pairs of indices may follow in ARGS."""
        self.tk.call(
            (self._w, 'tag', 'add', tagName, index1) + args)
    def tag_unbind(self, tagName, sequence, funcid=None):
        """Unbind for all characters with TAGNAME for event SEQUENCE  the
        function identified with FUNCID."""
        self.tk.call(self._w, 'tag', 'bind', tagName, sequence, '')
        if funcid:
            self.deletecommand(funcid)
    def tag_bind(self, tagName, sequence, func, add=None):
        """Bind to all characters with TAGNAME at event SEQUENCE a call to function FUNC.

        An additional boolean parameter ADD specifies whether FUNC will be
        called additionally to the other bound function or whether it will
        replace the previous function. See bind for the return value."""
        return self._bind((self._w, 'tag', 'bind', tagName),
                  sequence, func, add)
    def tag_cget(self, tagName, option):
        """Return the value of OPTION for tag TAGNAME."""
        if option[:1] != '-':
            option = '-' + option
        if option[-1:] == '_':
            option = option[:-1]
        return self.tk.call(self._w, 'tag', 'cget', tagName, option)
    def tag_configure(self, tagName, cnf={}, **kw):
        """Configure a tag TAGNAME."""
        if type(cnf) == StringType:
            x = self.tk.split(self.tk.call(
                self._w, 'tag', 'configure', tagName, '-'+cnf))
            return (x[0][1:],) + x[1:]
        self.tk.call(
              (self._w, 'tag', 'configure', tagName)
              + self._options(cnf, kw))
    tag_config = tag_configure
    def tag_delete(self, *tagNames):
        """Delete all tags in TAGNAMES."""
        self.tk.call((self._w, 'tag', 'delete') + tagNames)
    def tag_lower(self, tagName, belowThis=None):
        """Change the priority of tag TAGNAME such that it is lower
        than the priority of BELOWTHIS."""
        self.tk.call(self._w, 'tag', 'lower', tagName, belowThis)
    def tag_names(self, index=None):
        """Return a list of all tag names."""
        return self.tk.splitlist(
            self.tk.call(self._w, 'tag', 'names', index))
    def tag_nextrange(self, tagName, index1, index2=None):
        """Return a list of start and end index for the first sequence of
        characters between INDEX1 and INDEX2 which all have tag TAGNAME.
        The text is searched forward from INDEX1."""
        return self.tk.splitlist(self.tk.call(
            self._w, 'tag', 'nextrange', tagName, index1, index2))
    def tag_prevrange(self, tagName, index1, index2=None):
        """Return a list of start and end index for the first sequence of
        characters between INDEX1 and INDEX2 which all have tag TAGNAME.
        The text is searched backwards from INDEX1."""
        return self.tk.splitlist(self.tk.call(
            self._w, 'tag', 'prevrange', tagName, index1, index2))
    def tag_raise(self, tagName, aboveThis=None):
        """Change the priority of tag TAGNAME such that it is higher
        than the priority of ABOVETHIS."""
        self.tk.call(
            self._w, 'tag', 'raise', tagName, aboveThis)
    def tag_ranges(self, tagName):
        """Return a list of ranges of text which have tag TAGNAME."""
        return self.tk.splitlist(self.tk.call(
            self._w, 'tag', 'ranges', tagName))
    def tag_remove(self, tagName, index1, index2=None):
        """Remove tag TAGNAME from all characters between INDEX1 and INDEX2."""
        self.tk.call(
            self._w, 'tag', 'remove', tagName, index1, index2)
    def window_cget(self, index, option):
        """Return the value of OPTION of an embedded window at INDEX."""
        if option[:1] != '-':
            option = '-' + option
        if option[-1:] == '_':
            option = option[:-1]
        return self.tk.call(self._w, 'window', 'cget', index, option)
    def window_configure(self, index, cnf={}, **kw):
        """Configure an embedded window at INDEX."""
        if type(cnf) == StringType:
            x = self.tk.split(self.tk.call(
                self._w, 'window', 'configure',
                index, '-'+cnf))
            return (x[0][1:],) + x[1:]
        self.tk.call(
              (self._w, 'window', 'configure', index)
              + self._options(cnf, kw))
    window_config = window_configure
    def window_create(self, index, cnf={}, **kw):
        """Create a window at INDEX."""
        self.tk.call(
              (self._w, 'window', 'create', index)
              + self._options(cnf, kw))
    def window_names(self):
        """Return all names of embedded windows in this widget."""
        return self.tk.splitlist(
            self.tk.call(self._w, 'window', 'names'))
    def xview(self, *what):
        """Query and change horizontal position of the view."""
        if not what:
            return self._getdoubles(self.tk.call(self._w, 'xview'))
        self.tk.call((self._w, 'xview') + what)
    def xview_moveto(self, fraction):
        """Adjusts the view in the window so that FRACTION of the
        total width of the canvas is off-screen to the left."""
        self.tk.call(self._w, 'xview', 'moveto', fraction)
    def xview_scroll(self, number, what):
        """Shift the x-view according to NUMBER which is measured
        in "units" or "pages" (WHAT)."""
        self.tk.call(self._w, 'xview', 'scroll', number, what)
    def yview(self, *what):
        """Query and change vertical position of the view."""
        if not what:
            return self._getdoubles(self.tk.call(self._w, 'yview'))
        self.tk.call((self._w, 'yview') + what)
    def yview_moveto(self, fraction):
        """Adjusts the view in the window so that FRACTION of the
        total height of the canvas is off-screen to the top."""
        self.tk.call(self._w, 'yview', 'moveto', fraction)
    def yview_scroll(self, number, what):
        """Shift the y-view according to NUMBER which is measured
        in "units" or "pages" (WHAT)."""
        self.tk.call(self._w, 'yview', 'scroll', number, what)
    def yview_pickplace(self, *what):
        """Obsolete function, use see."""
        self.tk.call((self._w, 'yview', '-pickplace') + what)

class _setit:
    """Internal class. It wraps the command in the widget OptionMenu."""
    def __init__(self, var, value, callback=None):
        self.__value = value
        self.__var = var
        self.__callback = callback
    def __call__(self, *args):
        self.__var.set(self.__value)
        if self.__callback:
            apply(self.__callback, (self.__value,)+args)

class OptionMenu(Menubutton):
    """OptionMenu which allows the user to select a value from a menu."""
    def __init__(self, master, variable, value, *values, **kwargs):
        """Construct an optionmenu widget with the parent MASTER, with
        the resource textvariable set to VARIABLE, the initially selected
        value VALUE, the other menu values VALUES and an additional
        keyword argument command."""
        kw = {"borderwidth": 2, "textvariable": variable,
              "indicatoron": 1, "relief": RAISED, "anchor": "c",
              "highlightthickness": 2}
        Widget.__init__(self, master, "menubutton", kw)
        self.widgetName = 'tk_optionMenu'
        menu = self.__menu = Menu(self, name="menu", tearoff=0)
        self.menuname = menu._w
        # 'command' is the only supported keyword
        callback = kwargs.get('command')
        if kwargs.has_key('command'):
            del kwargs['command']
        if kwargs:
            raise TclError, 'unknown option -'+kwargs.keys()[0]
        menu.add_command(label=value,
                 command=_setit(variable, value, callback))
        for v in values:
            menu.add_command(label=v,
                     command=_setit(variable, v, callback))
        self["menu"] = menu

    def __getitem__(self, name):
        if name == 'menu':
            return self.__menu
        return Widget.__getitem__(self, name)

    def destroy(self):
        """Destroy this widget and the associated menu."""
        Menubutton.destroy(self)
        self.__menu = None

class Image:
    """Base class for images."""
    _last_id = 0
    def __init__(self, imgtype, name=None, cnf={}, master=None, **kw):
        self.name = None
        if not master:
            master = _default_root
            if not master:
                raise RuntimeError, 'Too early to create image'
        self.tk = master.tk
        if not name:
            Image._last_id += 1
            name = "pyimage" +`Image._last_id` # tk itself would use image<x>
            # The following is needed for systems where id(x)
            # can return a negative number, such as Linux/m68k:
            if name[0] == '-': name = '_' + name[1:]
        if kw and cnf: cnf = _cnfmerge((cnf, kw))
        elif kw: cnf = kw
        options = ()
        for k, v in cnf.items():
            if callable(v):
                v = self._register(v)
            options = options + ('-'+k, v)
        self.tk.call(('image', 'create', imgtype, name,) + options)
        self.name = name
    def __str__(self): return self.name
    def __del__(self):
        if self.name:
            try:
                self.tk.call('image', 'delete', self.name)
            except TclError:
                # May happen if the root was destroyed
                pass
    def __setitem__(self, key, value):
        self.tk.call(self.name, 'configure', '-'+key, value)
    def __getitem__(self, key):
        return self.tk.call(self.name, 'configure', '-'+key)
    def configure(self, **kw):
        """Configure the image."""
        res = ()
        for k, v in _cnfmerge(kw).items():
            if v is not None:
                if k[-1] == '_': k = k[:-1]
                if callable(v):
                    v = self._register(v)
                res = res + ('-'+k, v)
        self.tk.call((self.name, 'config') + res)
    config = configure
    def height(self):
        """Return the height of the image."""
        return getint(
            self.tk.call('image', 'height', self.name))
    def type(self):
        """Return the type of the imgage, e.g. "photo" or "bitmap"."""
        return self.tk.call('image', 'type', self.name)
    def width(self):
        """Return the width of the image."""
        return getint(
            self.tk.call('image', 'width', self.name))

class PhotoImage(Image):
    """Widget which can display colored images in GIF, PPM/PGM format."""
    def __init__(self, name=None, cnf={}, master=None, **kw):
        """Create an image with NAME.

        Valid resource names: data, format, file, gamma, height, palette,
        width."""
        apply(Image.__init__, (self, 'photo', name, cnf, master), kw)
    def blank(self):
        """Display a transparent image."""
        self.tk.call(self.name, 'blank')
    def cget(self, option):
        """Return the value of OPTION."""
        return self.tk.call(self.name, 'cget', '-' + option)
    # XXX config
    def __getitem__(self, key):
        return self.tk.call(self.name, 'cget', '-' + key)
    # XXX copy -from, -to, ...?
    def copy(self):
        """Return a new PhotoImage with the same image as this widget."""
        destImage = PhotoImage()
        self.tk.call(destImage, 'copy', self.name)
        return destImage
    def zoom(self,x,y=''):
        """Return a new PhotoImage with the same image as this widget
        but zoom it with X and Y."""
        destImage = PhotoImage()
        if y=='': y=x
        self.tk.call(destImage, 'copy', self.name, '-zoom',x,y)
        return destImage
    def subsample(self,x,y=''):
        """Return a new PhotoImage based on the same image as this widget
        but use only every Xth or Yth pixel."""
        destImage = PhotoImage()
        if y=='': y=x
        self.tk.call(destImage, 'copy', self.name, '-subsample',x,y)
        return destImage
    def get(self, x, y):
        """Return the color (red, green, blue) of the pixel at X,Y."""
        return self.tk.call(self.name, 'get', x, y)
    def put(self, data, to=None):
        """Put row formated colors to image starting from
        position TO, e.g. image.put("{red green} {blue yellow}", to=(4,6))"""
        args = (self.name, 'put', data)
        if to:
            if to[0] == '-to':
                to = to[1:]
            args = args + ('-to',) + tuple(to)
        self.tk.call(args)
    # XXX read
    def write(self, filename, format=None, from_coords=None):
        """Write image to file FILENAME in FORMAT starting from
        position FROM_COORDS."""
        args = (self.name, 'write', filename)
        if format:
            args = args + ('-format', format)
        if from_coords:
            args = args + ('-from',) + tuple(from_coords)
        self.tk.call(args)

class BitmapImage(Image):
    """Widget which can display a bitmap."""
    def __init__(self, name=None, cnf={}, master=None, **kw):
        """Create a bitmap with NAME.

        Valid resource names: background, data, file, foreground, maskdata, maskfile."""
        apply(Image.__init__, (self, 'bitmap', name, cnf, master), kw)

def image_names(): return _default_root.tk.call('image', 'names')
def image_types(): return _default_root.tk.call('image', 'types')

######################################################################
# Extensions:

class Studbutton(Button):
    def __init__(self, master=None, cnf={}, **kw):
        Widget.__init__(self, master, 'studbutton', cnf, kw)
        self.bind('<Any-Enter>',       self.tkButtonEnter)
        self.bind('<Any-Leave>',       self.tkButtonLeave)
        self.bind('<1>',               self.tkButtonDown)
        self.bind('<ButtonRelease-1>', self.tkButtonUp)

class Tributton(Button):
    def __init__(self, master=None, cnf={}, **kw):
        Widget.__init__(self, master, 'tributton', cnf, kw)
        self.bind('<Any-Enter>',       self.tkButtonEnter)
        self.bind('<Any-Leave>',       self.tkButtonLeave)
        self.bind('<1>',               self.tkButtonDown)
        self.bind('<ButtonRelease-1>', self.tkButtonUp)
        self['fg']               = self['bg']
        self['activebackground'] = self['bg']

######################################################################
# Test:

def _test():
    root = Tk()
    text = "This is Tcl/Tk version %s" % TclVersion
    if TclVersion >= 8.1:
        try:
            text = text + unicode("\nThis should be a cedilla: \347",
                                  "iso-8859-1")
        except NameError:
            pass # no unicode support
    label = Label(root, text=text)
    label.pack()
    test = Button(root, text="Click me!",
              command=lambda root=root: root.test.configure(
                  text="[%s]" % root.test['text']))
    test.pack()
    root.test = test
    quit = Button(root, text="QUIT", command=root.destroy)
    quit.pack()
    # The following three commands are needed so the window pops
    # up on top on Windows...
    root.iconify()
    root.update()
    root.deiconify()
    root.mainloop()

if __name__ == '__main__':
    _test()
