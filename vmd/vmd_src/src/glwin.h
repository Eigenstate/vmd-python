/*
 * glwin.h -- Simple self-contained code for opening an 
 *            OpenGL-capable display window with a double 
 *            buffered mono or stereoscopic visual.
 *            This code is primarily meant for 2-D image display
 *            usage or for trivial 3-D rendering usage without
 *            any GLX/WGL extensions that have to be enumerated
 *            prior to window creation.
 I 
 *            This file is part of the Tachyon ray tracer.
 *            John E. Stone - john.stone@gmail.com
 *
 * $Id: glwin.h,v 1.1 2015/05/19 16:05:09 johns Exp $
 */

#ifdef  __cplusplus
extern "C" {
#endif

#define GLWIN_EV_POLL_NONBLOCK    0
#define GLWIN_EV_POLL_BLOCK       1

#define GLWIN_EV_NONE             0

#define GLWIN_EV_KBD              1 /**< all non-special chars */

#define GLWIN_EV_KBD_UP           2
#define GLWIN_EV_KBD_DOWN         3
#define GLWIN_EV_KBD_LEFT         4
#define GLWIN_EV_KBD_RIGHT        5
#define GLWIN_EV_KBD_PAGE_UP      6
#define GLWIN_EV_KBD_PAGE_DOWN    7
#define GLWIN_EV_KBD_HOME         8
#define GLWIN_EV_KBD_END          9
#define GLWIN_EV_KBD_INSERT      10
#define GLWIN_EV_KBD_DELETE      11

#define GLWIN_EV_KBD_F1          12
#define GLWIN_EV_KBD_F2          13
#define GLWIN_EV_KBD_F3          14
#define GLWIN_EV_KBD_F4          15
#define GLWIN_EV_KBD_F5          16
#define GLWIN_EV_KBD_F6          17
#define GLWIN_EV_KBD_F7          18
#define GLWIN_EV_KBD_F8          19
#define GLWIN_EV_KBD_F9          20
#define GLWIN_EV_KBD_F10         21
#define GLWIN_EV_KBD_F11         22
#define GLWIN_EV_KBD_F12         23

#define GLWIN_EV_KBD_ESC         24

#define GLWIN_EV_MOUSE_MOVE      31

#define GLWIN_EV_MOUSE_LEFT      32
#define GLWIN_EV_MOUSE_MIDDLE    34
#define GLWIN_EV_MOUSE_RIGHT     35
#define GLWIN_EV_MOUSE_WHEELUP   36
#define GLWIN_EV_MOUSE_WHEELDOWN 37

#define GLWIN_EV_WINDOW_CLOSE    128 /**< window manager close event */

void * glwin_create(const char * wintitle, int width, int height);
void glwin_destroy(void * voidhandle);
void glwin_swap_buffers(void * voidhandle);
int glwin_handle_events(void * voidhandle, int evblockmode);
int glwin_get_wininfo(void * voidhandle, int *instereo, int *havestencil);
int glwin_get_winsize(void * voidhandle, int *xsize, int *ysize);
int glwin_get_winpos(void * voidhandle, int *xpos, int *ypos);
int glwin_get_mousepointer(void *voidhandle, int *x, int *y);
int glwin_get_lastevent(void * voidhandle, int *evdev, int *evval, char *evkey);
int glwin_get_spaceball(void *voidhandle, int *rx, int *ry, int *rz, int *tx, int *ty, int *tz, int *buttons);
int glwin_spaceball_available(void *voidhandle);
int glwin_resize(void *voidhandle, int width, int height);
int glwin_reposition(void *voidhandle, int xpos, int ypos);
int glwin_fullscreen(void * voidhandle, int fson, int xinescreen);
int glwin_query_extension(void *voidhandle, const char *extname);
void glwin_draw_image(void * voidhandle, int xsize, int ysize, unsigned char * img);

#ifdef  __cplusplus
}
#endif

