/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: Win32OpenGLDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.124 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Subclass of DisplayDevice, this object has routines used by all the
 *   different display devices that are OpenGL-specific.  Will render drawing
 *   commands into a single window.
 ***************************************************************************/

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0400 // hack to allow access to mouse wheel events
#endif
#include <windows.h>        // Mouse wheel events and related macros
#include <winuser.h>        // Mouse wheel events and related macros

#include "VMDApp.h"
#include "OpenGLDisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"   // VMD version strings etc

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/gl.h>

#include "../msvc/winvmd/res/resource.h" // VMD icon resource

#if 1
// 
// Compile-time constant to provide hybrid graphics drivers with a hint to 
// favor the use of the high performance GPU when one exists.
// 
extern "C" {
// trigger AMD PowerXpress drivers to use the high performance GPU
__declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 1;

// trigger NVIDIA Optimus drivers to use the high performance GPU
__declspec(dllexport) DWORD NvOptimusEnablement = 1;
}
#endif


// NOTE: you may have to get copies of the latest OpenGL extension headers
// from the OpenGL web site if your Linux or Win32 machine lacks them:
//   http://oss.sgi.com/projects/ogl-sample/registry/
#include <GL/glext.h>   // include OpenGL extension headers
#include <GL/wglext.h>  // include OpenGL extension headers

// static data for this object
static const char *glStereoNameStr[OPENGL_STEREO_MODES] = 
 { "Off", 
   "QuadBuffered", 
   "HDTV SideBySide",
   "Checkerboard",
   "ColumnInterleaved",
   "RowInterleaved",
   "Anaglyph",
   "SideBySide", 
   "AboveBelow",
   "Left", 
   "Right" };

static const char *glRenderNameStr[OPENGL_RENDER_MODES] =
{ "Normal",
  "GLSL",
  "Acrobat3D" };

static const char *glCacheNameStr[OPENGL_CACHE_MODES] = 
{ "Off",
  "On" };

static char szAppName[] = "VMD";
static char szAppTitle[]="VMD " VMDVERSION " OpenGL Display";

LRESULT WINAPI vmdWindowProc( HWND, UINT, WPARAM, LPARAM );

static int OpenWin32Connection(wgldata * glwsrv) {
  WNDCLASS  wc;
  HINSTANCE hInstance = GetModuleHandle(NULL);

  /* Clear (important!) and then fill in the window class structure. */
  memset(&wc, 0, sizeof(WNDCLASS));
  wc.style         = CS_OWNDC;
  wc.lpfnWndProc   = (WNDPROC) vmdWindowProc;
  wc.hInstance     = hInstance;
#if 1
  // use our VMD icon
  wc.hIcon         = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
#else
  wc.hIcon         = LoadIcon(NULL, IDI_WINLOGO);
#endif
  wc.hCursor       = LoadCursor(hInstance, IDC_ARROW);
  wc.hbrBackground = NULL; /* Default color */
  wc.lpszMenuName  = NULL;
  wc.lpszClassName = szAppName;

  if(!RegisterClass(&wc)) {
    printf("Cannot register window class.\n");
    return -1;
  }

  // get screen size 
  // XXX There's no Win32 API to get the full multi-monitor desktop,
  //     so this code doesn't correctly handle multi-monitor systems yet.
  //     To correctly handle multiple monitors, we'd have to 
  //     walk the device tree, take into account monitor layout/positioning, 
  //     and compute the desktop dimensions from that.  Since these values
  //     are currently only used by do_reposition_window() method, we can
  //     live with primary-monitor values for the time being.
  glwsrv->scrwidth  = GetSystemMetrics(SM_CXSCREEN);
  glwsrv->scrheight = GetSystemMetrics(SM_CYSCREEN);

  return 0;
}

static int PFDHasStereo(int ID, HDC hDC) {
  PIXELFORMATDESCRIPTOR pfd;
  DescribePixelFormat(hDC, ID, sizeof(PIXELFORMATDESCRIPTOR), &pfd); 

#if 0
  // print a message if we find out we've got an accelerated mode
  if ((pfd.dwFlags & PFD_GENERIC_ACCELERATED) ||
      !(pfd.dwFlags & PFD_GENERIC_FORMAT))  
    msgInfo << "Hardware 3D Acceleration enabled." << sendmsg;
  else
    msgInfo << "No hardware 3D Acceleration found." << sendmsg;
#endif

  if (pfd.dwFlags & PFD_STEREO) 
    return 1;
  
  return 0;
} 

#if 0
static void PrintPFD(int ID, HDC hDC) {
  PIXELFORMATDESCRIPTOR pfd;
  FILE * ofp;

  if (ID == 0) {
    int i, num;
    num = DescribePixelFormat(hDC, 1, sizeof(PIXELFORMATDESCRIPTOR), &pfd); 
    for (i=1; i<num; i++) 
      PrintPFD(i, hDC);

    return;
  }

  DescribePixelFormat(hDC, ID, sizeof(PIXELFORMATDESCRIPTOR), &pfd); 

  ofp=fopen("c:/video.txt", "a+");
  if (ofp == NULL) 
    ofp=stdout;

  if (pfd.cColorBits < 15) {
    fprintf(ofp, "Windows Pixel Format ID: %d -- not enough color bits\n", ID);
  }
  else {
    fprintf(ofp, "\nWindows Pixel Format ID: %d\n", ID);
    fprintf(ofp, "  Color Buffer Depth: %d bits\n", pfd.cColorBits);
    fprintf(ofp, "      Z Buffer Depth: %d bits\n", pfd.cDepthBits);
    if (pfd.dwFlags & PFD_DOUBLEBUFFER)
      fprintf(ofp, "    PFD_DOUBLEBUFFER\n"); 
    if (pfd.dwFlags & PFD_STEREO)
      fprintf(ofp, "    PFD_STEREO\n"); 
    if (pfd.dwFlags & PFD_DRAW_TO_WINDOW)
      fprintf(ofp, "    PFD_DRAW_TO_WINDOW\n"); 
    if (pfd.dwFlags & PFD_SUPPORT_GDI)
      fprintf(ofp, "    PFD_SUPPORT_GDI\n"); 
    if (pfd.dwFlags & PFD_SUPPORT_OPENGL)
      fprintf(ofp, "    PFD_SUPPORT_OPENGL\n"); 
    if (pfd.dwFlags & PFD_SWAP_EXCHANGE) 
      fprintf(ofp, "    PFD_SWAP_EXCHANGE\n"); 
    if (pfd.dwFlags & PFD_SWAP_COPY) 
      fprintf(ofp, "    PFD_SWAP_COPY\n"); 
    if (pfd.dwFlags & PFD_SWAP_LAYER_BUFFERS) 
      fprintf(ofp, "    PFD_SWAP_LAYER_BUFFERS\n"); 
    if (pfd.dwFlags & PFD_GENERIC_ACCELERATED)
      fprintf(ofp, "    PFD_GENERIC_ACCELERATED\n"); 
    if (pfd.dwFlags & PFD_GENERIC_FORMAT)
      fprintf(ofp, "    PFD_GENERIC_FORMAT\n"); 
  } 
  if (ofp != NULL && ofp != stdout) 
    fclose(ofp);
}
#endif

static HGLRC SetupOpenGL(wgldata *glwsrv) {
  int ID;
  HDC hDC;
  HGLRC hRC;

  PIXELFORMATDESCRIPTOR pfd = {
    sizeof (PIXELFORMATDESCRIPTOR), /* struct size      */
    1,                              /* Version number   */
    PFD_DRAW_TO_WINDOW      /* Flags, draw to a window, */
      | PFD_DOUBLEBUFFER    /* Requires Doublebuffer hw */
      | PFD_STEREO          /* we want stereo if possible */ 
      | PFD_SUPPORT_OPENGL, /* use OpenGL               */
    PFD_TYPE_RGBA,          /* RGBA pixel values        */
    16,                     /* 24-bit color             */
    0, 0, 0,                /* RGB bits & shift sizes.  */
    0, 0, 0,                /* Don't care about them    */
    0, 0,                   /* No alpha buffer info     */
    0, 0, 0, 0, 0,          /* No accumulation buffer   */
    16,                     /* depth buffer             */
    1,                      /* stencil buffer           */
    0,                      /* No auxiliary buffers     */
    PFD_MAIN_PLANE,         /* Layer type               */
    0,                      /* Reserved (must be 0)     */
    0,                      /* No layer mask            */
    0,                      /* No visible mask          */
    0                       /* No damage mask           */
  };

  hDC = GetDC(glwsrv->hWnd);
  ID = ChoosePixelFormat(hDC, &pfd);

  /*
   * catch errors here.
   * If ID is zero, then there's
   * something wrong... most likely the window's
   * style bits are incorrect (in CreateWindow() )
   * or OpenGL isn't installed on this machine
   */

  if (ID == 0) {
    printf("Error selecting OpenGL Pixel Format!!\n");
    return NULL;
  }

  glwsrv->PFDisStereo = PFDHasStereo(ID, hDC);
  //PrintPFD(ID, hDC);
  //printf("*** Setting Windows OpenGL Pixel Format to ID %d ***\n", ID); 
  SetPixelFormat( hDC, ID, &pfd );

  hRC = wglCreateContext(hDC);
  ReleaseDC(glwsrv->hWnd, hDC);

  return hRC;
}

static int myCreateWindow(OpenGLDisplayDevice *ogldispdev,
                          int xpos, int ypos, int xs, int ys) {
  /* Create a main window for this application instance. */
  ogldispdev->glwsrv.hWnd =
        CreateWindow(
              szAppName,          /* app name */
              szAppTitle,         /* Text for window title bar */
              WS_OVERLAPPEDWINDOW /* Window style */
               | WS_CLIPCHILDREN
               | WS_CLIPSIBLINGS, /* NEED THESE for OpenGL calls to work! */
              xpos, ypos,
              xs, ys,
              NULL,                  /* no parent window                */
              NULL,                  /* Use the window class menu.      */
              GetModuleHandle(NULL), /* This instance owns this window  */
              ogldispdev             /* We pass in the caller class ptr */
        );

  if (!ogldispdev->glwsrv.hWnd) {
    printf("Couldn't Open Window!!\n");
    return -1;
  }

  ogldispdev->glwsrv.hDC = GetDC(ogldispdev->glwsrv.hWnd);
  wglMakeCurrent(ogldispdev->glwsrv.hDC, ogldispdev->glwsrv.hRC);

  /* Make the window visible & update its client area */
  ShowWindow(ogldispdev->glwsrv.hWnd, SW_SHOW);   /* Show the window    */
  UpdateWindow(ogldispdev->glwsrv.hWnd );         /* Sends WM_PAINT msg */
  DragAcceptFiles(ogldispdev->glwsrv.hWnd, TRUE); /* Enable Drag & Drop */

  return 0;
}

static void vmd_transwin32mouse(OpenGLDisplayDevice * d, LPARAM l) {
  int x, y;
  x = LOWORD(l);
  y = HIWORD(l);
  if(x & 1 << 15) x -= (1 << 16); // handle mouse capture in negative range
  if(y & 1 << 15) y -= (1 << 16); // handle mouse capture in negative range
  d->glwsrv.MouseX = x;
  d->glwsrv.MouseY = (d->ySize) - y; // translate to coords VMD likes (GL-like)
}


#ifdef VMDSPACEWARE
// Windows code to talk to Spaceball device
static void vmd_setupwin32spaceball(wgldata *glwsrv) {
  SiOpenData oData;
  enum SpwRetVal res;

  // init the sball pointer to NULL by default, used to determine if we
  // had a healthy init later on.
  glwsrv->sball = NULL;

  switch (SiInitialize()) {
    case SPW_NO_ERROR:
      break;

    case SPW_DLL_LOAD_ERROR:
      msgInfo << "Spaceball driver not installed.  Spaceball interface disabled." << sendmsg;
      return;

    default:
      msgInfo << "Spaceball did not initialize properly.  Spaceball interface disabled." << sendmsg;
      return;
  }

  SiOpenWinInit(&oData, glwsrv->hWnd);            // init win platform data
  SiSetUiMode(glwsrv->sball, SI_UI_ALL_CONTROLS); // config softbutton display

  // actually start a connection to the device now that the UI mode
  // and window system data are setup.
  glwsrv->sball = SiOpen("VMD", SI_ANY_DEVICE, SI_NO_MASK, SI_EVENT, &oData);
  if ((glwsrv->sball == NULL) || (glwsrv->sball == SI_NO_HANDLE)) {
    SiTerminate(); // shutdown spaceware input library
    msgInfo << "Spaceball is unresponsive.  Spaceball interface disabled." << sendmsg;
    glwsrv->sball = NULL; // NULL out the handle for sure.
    return;
  }

  res = SiBeep(glwsrv->sball, "CcCc"); // beep the spaceball
  if ((glwsrv->sball != NULL) && (glwsrv->sball != SI_NO_HANDLE))
    msgInfo << "Spaceball found, software interface initialized." << sendmsg;
}

static void vmd_closewin32spaceball(wgldata *glwsrv) {
  enum SpwRetVal res;

  if (glwsrv->sball != NULL) {
    res = SiClose(glwsrv->sball); // close spaceball device
    if (res != SPW_NO_ERROR) 
      msgInfo << "An error occured while shutting down the Spaceball device." << sendmsg;

    SiTerminate();          // shutdown spaceware input library
  }

  glwsrv->sball = NULL;   // NULL out the handle.
}

static int vmd_processwin32spaceballevent(wgldata *glwsrv, UINT msg, WPARAM wParam, LPARAM lParam) {

  if (glwsrv == NULL)
    return 0;

  if (glwsrv->sball == NULL) 
    return 0;  // no spaceball attached/running
 
  // Check to see if this message is a spaceball message
  SiGetEventWinInit(&glwsrv->spwedata, msg, wParam, lParam);

  if (SiGetEvent(glwsrv->sball, 0, &glwsrv->spwedata, &glwsrv->spwevent) == SI_IS_EVENT) {
    return 1;
  }

  return 0;
}
#endif


LRESULT WINAPI vmdWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  PAINTSTRUCT   ps; /* Paint structure. */

  // XXX this enum has to be replicated here since its otherwise 
  //     private to the DisplayDevice class and children.
  enum EventCodes { WIN_REDRAW, WIN_LEFT, WIN_MIDDLE, WIN_RIGHT,
                    WIN_WHEELUP, WIN_WHEELDOWN, WIN_MOUSEX, WIN_MOUSEY, 
                    WIN_KBD, 
                    WIN_KBD_ESCAPE,
                    WIN_KBD_UP,
                    WIN_KBD_DOWN,
                    WIN_KBD_LEFT,
                    WIN_KBD_RIGHT,
                    WIN_KBD_PAGE_UP,
                    WIN_KBD_PAGE_DOWN,
                    WIN_KBD_HOME,
                    WIN_KBD_END,
                    WIN_KBD_INSERT,
                    WIN_KBD_DELETE,
                    WIN_KBD_F1,  WIN_KBD_F2,  WIN_KBD_F3,  WIN_KBD_F4,
                    WIN_KBD_F5,  WIN_KBD_F6,  WIN_KBD_F7,  WIN_KBD_F8,
                    WIN_KBD_F9,  WIN_KBD_F10, WIN_KBD_F11, WIN_KBD_F12,
                    WIN_NOEVENT };
  wgldata *glwsrv;
  OpenGLDisplayDevice * ogldispdev;

  // Upon first window creation, immediately set our user-data field
  // to store caller-provided handles for this window instance
  if (msg == WM_NCCREATE) {
#if defined(_M_X64) || defined(_WIN64) || defined(_Wp64)
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR) (((CREATESTRUCT *) lParam)->lpCreateParams));
#else
    SetWindowLong(hwnd, GWL_USERDATA, (LONG) (((CREATESTRUCT *) lParam)->lpCreateParams));
#endif
  }

  // check to make sure we have a valid window data structure in case
  // it is destroyed while there are still pending messages...
#if defined(_M_X64) || defined(_WIN64) || defined(_Wp64)
  ogldispdev = (OpenGLDisplayDevice *) GetWindowLongPtr(hwnd, GWLP_USERDATA);
#else
  ogldispdev = (OpenGLDisplayDevice *) GetWindowLong(hwnd, GWL_USERDATA);
#endif

  // when VMD destroys its window data structures it is possible that
  // the window could still get messages briefly thereafter, this prevents
  // us from attempting to handle any messages when the VMD state that goes
  // with the window has already been destructed. (most notably when using
  // the spaceball..)  If we have a NULL pointer, let windows handle the
  // event for us using the default window proc.
  if (ogldispdev == NULL)
    return DefWindowProc(hwnd, msg, wParam, lParam);
 
  glwsrv = &ogldispdev->glwsrv;

#ifdef VMDSPACEWARE
  // see if it is a spaceball event, if so do something about it.
  if (vmd_processwin32spaceballevent(glwsrv, msg, wParam, lParam))
    return 0; //
#endif

  switch(msg) {
    case WM_CREATE:
      glwsrv->hWnd = hwnd;
      glwsrv->hRC = SetupOpenGL(glwsrv);
      glwsrv->WEvents = WIN_REDRAW;
      return 0;

    case WM_SIZE:
      wglMakeCurrent(glwsrv->hDC, glwsrv->hRC);
      ogldispdev->xSize = LOWORD(lParam);
      ogldispdev->ySize = HIWORD(lParam);
      ogldispdev->reshape();
      glViewport(0, 0, (GLsizei) ogldispdev->xSize, (GLsizei) ogldispdev->ySize);
      glwsrv->WEvents = WIN_REDRAW;
      return 0;

    case WM_SIZING:
      wglMakeCurrent(glwsrv->hDC, glwsrv->hRC);
      glClear(GL_COLOR_BUFFER_BIT);
      SwapBuffers(glwsrv->hDC);
      glDrawBuffer(GL_BACK);
      return 0;

    case WM_CLOSE:
      PostQuitMessage(0);
      return 0;

    case WM_PAINT:
      BeginPaint(hwnd, &ps);
      EndPaint(hwnd, &ps);
      glwsrv->WEvents = WIN_REDRAW;
      return 0;

    case WM_KEYDOWN:
      glwsrv->KeyFlag = MapVirtualKey((UINT) wParam, 2); // map to ASCII
      glwsrv->WEvents = WIN_KBD;
      if (glwsrv->KeyFlag == 0) {
        unsigned int keysym = wParam;
        switch (keysym) {
          case VK_ESCAPE:    glwsrv->WEvents = WIN_KBD_ESCAPE;    break;
          case VK_UP:        glwsrv->WEvents = WIN_KBD_UP;        break;
          case VK_DOWN:      glwsrv->WEvents = WIN_KBD_DOWN;      break;
          case VK_LEFT:      glwsrv->WEvents = WIN_KBD_LEFT;      break;
          case VK_RIGHT:     glwsrv->WEvents = WIN_KBD_RIGHT;     break;
          case VK_PRIOR:     glwsrv->WEvents = WIN_KBD_PAGE_UP;   break;
          case VK_NEXT:      glwsrv->WEvents = WIN_KBD_PAGE_DOWN; break;
          case VK_HOME:      glwsrv->WEvents = WIN_KBD_HOME;      break;
          case VK_END:       glwsrv->WEvents = WIN_KBD_END;       break;
          case VK_INSERT:    glwsrv->WEvents = WIN_KBD_INSERT;    break;
          case VK_DELETE:    glwsrv->WEvents = WIN_KBD_DELETE;    break;
          case VK_F1:        glwsrv->WEvents = WIN_KBD_F1;        break;
          case VK_F2:        glwsrv->WEvents = WIN_KBD_F2;        break;
          case VK_F3:        glwsrv->WEvents = WIN_KBD_F3;        break;
          case VK_F4:        glwsrv->WEvents = WIN_KBD_F4;        break;
          case VK_F5:        glwsrv->WEvents = WIN_KBD_F5;        break;
          case VK_F6:        glwsrv->WEvents = WIN_KBD_F6;        break;
          case VK_F7:        glwsrv->WEvents = WIN_KBD_F7;        break;
          case VK_F8:        glwsrv->WEvents = WIN_KBD_F8;        break;
          case VK_F9:        glwsrv->WEvents = WIN_KBD_F9;        break;
          case VK_F10:       glwsrv->WEvents = WIN_KBD_F10;       break;
          case VK_F11:       glwsrv->WEvents = WIN_KBD_F11;       break;
          case VK_F12:       glwsrv->WEvents = WIN_KBD_F12;       break;
          default:
            glwsrv->WEvents = WIN_NOEVENT;
            break;
        }
      }
      return 0;

    case WM_MOUSEMOVE:
      vmd_transwin32mouse(ogldispdev, lParam);
      glwsrv->MouseFlags = (long) wParam;
      return 0;

    case WM_MOUSEWHEEL:
      {
        int zDelta = ((short) HIWORD(wParam));
        // XXX
        // zDelta is in positive or negative multiples of WHEEL_DELTA for
        // clicky type scroll wheels on existing mice, may need to
        // recode this for continuous wheels at some future point in time.
        // WHEEL_DELTA is 120 in current versions of Windows.
        // We only activate an event if the user moves the mouse wheel at
        // least half of WHEEL_DELTA, so that they don't do it by accident
        // all the time.
        if (zDelta > (WHEEL_DELTA / 2)) {
          glwsrv->WEvents = WIN_WHEELUP;
        } else if (zDelta < -(WHEEL_DELTA / 2)) {
          glwsrv->WEvents = WIN_WHEELDOWN;
        }
      }
      return 0;

    case WM_LBUTTONDOWN:
      SetCapture(hwnd);
      vmd_transwin32mouse(ogldispdev, lParam);
      glwsrv->MouseFlags = (long) wParam;
      glwsrv->WEvents = WIN_LEFT;
      return 0;

    case WM_LBUTTONUP:
      vmd_transwin32mouse(ogldispdev, lParam);
      glwsrv->MouseFlags = (long) wParam;
      glwsrv->WEvents = WIN_LEFT;
      if (!(glwsrv->MouseFlags & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON))) 
        ReleaseCapture();
      return 0;

    case WM_MBUTTONDOWN:
      SetCapture(hwnd);
      vmd_transwin32mouse(ogldispdev, lParam);
      glwsrv->MouseFlags = (long) wParam;
      glwsrv->WEvents = WIN_MIDDLE;
      return 0;

    case WM_MBUTTONUP:
      vmd_transwin32mouse(ogldispdev, lParam);
      glwsrv->MouseFlags = (long) wParam;
      glwsrv->WEvents = WIN_MIDDLE;
      if (!(glwsrv->MouseFlags & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON))) 
        ReleaseCapture();
      return 0;

    case WM_RBUTTONDOWN:
      SetCapture(hwnd);
      vmd_transwin32mouse(ogldispdev, lParam);
      glwsrv->MouseFlags = (long) wParam;
      glwsrv->WEvents = WIN_RIGHT;
      return 0;

    case WM_RBUTTONUP:
      vmd_transwin32mouse(ogldispdev, lParam);
      glwsrv->MouseFlags = (long) wParam;
      glwsrv->WEvents = WIN_RIGHT;
      if (!(glwsrv->MouseFlags & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON))) 
        ReleaseCapture();
      return 0;

    case WM_SETCURSOR:
      // We process the mouse cursor hit test codes here, they tell us
      // what part of the window we're over, which helps us set the cursor
      // to the correct style for sizing borders, moves, etc.
      switch (LOWORD(lParam)) {
        case HTBOTTOM:
        case HTTOP:
          SetCursor(LoadCursor(NULL, IDC_SIZENS));
          break;

        case HTLEFT:
        case HTRIGHT:
          SetCursor(LoadCursor(NULL, IDC_SIZEWE));
          break;

        case HTTOPRIGHT:
        case HTBOTTOMLEFT:
          SetCursor(LoadCursor(NULL, IDC_SIZENESW));
          break;

        case HTTOPLEFT:
        case HTBOTTOMRIGHT:
          SetCursor(LoadCursor(NULL, IDC_SIZENWSE));
          break;

        case HTCAPTION:
          SetCursor(LoadCursor(NULL, IDC_ARROW));
          break;
          
        case HTCLIENT:
        default:
          ogldispdev->set_cursor(glwsrv->cursornum);
      }
      return 0;

    // 
    // Handle Windows File Drag and Drop Operations  
    // This code needs to be linked against SHELL32.DLL
    // 
    case WM_DROPFILES: 
      {
        char lpszFile[4096];
        UINT numfiles, fileindex, numc;
        HDROP hDropInfo = (HDROP)wParam;
        
        // Get the number of simultaneous dragged/dropped files.
        numfiles = DragQueryFile(hDropInfo, (DWORD)(-1), (LPSTR)NULL, 0);
  
        msgInfo << "Ignoring Drag and Drop operation, received " 
                << ((int) numfiles) << " files:" << sendmsg;

        FileSpec spec;       
        for (fileindex=0; fileindex<numfiles; fileindex++) {
          // lpszFile: complete pathname with device, colon and backslashes
          numc = DragQueryFile(hDropInfo, fileindex, (char *) &lpszFile, 4096);
  
          // VMD loads the file(s) here, or queues them up in its own
          // list to decide how to cope with them.  Deciding how to deal
          // with these files is definitely the tricky part.
          msgInfo << "  File(" << ((int) fileindex) << "): " << lpszFile 
                  << " (numc=" << ((int) numc) << ")" << sendmsg;

          // attempt to load the file into a new molecule
          ogldispdev->vmdapp->molecule_load(-1, lpszFile, NULL, &spec);
        }  
        DragFinish(hDropInfo); // finish drop operation and release memory
      }
      return 0;

    default:
      return DefWindowProc(hwnd, msg, wParam, lParam);
  }

  return 0;
}


/////////////////////////  constructor and destructor  

OpenGLDisplayDevice::OpenGLDisplayDevice() 
: OpenGLRenderer("VMD " VMDVERSION " OpenGL Display") {
  // set up data possible before opening window
  stereoNames = glStereoNameStr;
  stereoModes = OPENGL_STEREO_MODES;

  renderNames = glRenderNameStr;
  renderModes = OPENGL_RENDER_MODES;

  cacheNames = glCacheNameStr;
  cacheModes = OPENGL_CACHE_MODES;

  memset(&glwsrv, 0, sizeof(glwsrv));
  have_window = FALSE;
  screenX = screenY = 0;
  vmdapp = NULL;
}

// init ... open a window and set initial default values
int OpenGLDisplayDevice::init(int argc, char **argv, VMDApp *app, int *size, int *loc) {
  vmdapp = app; // save VMDApp handle for use by drag-and-drop handlers

  // open the window
  if (open_window(name, size, loc, argc, argv) != 0) return FALSE;
  if (!have_window) return FALSE;

  // get screen size 
  // XXX There's no Win32 API to get the full multi-monitor desktop,
  //     so this code doesn't correctly handle multi-monitor systems yet.
  //     To correctly handle multiple monitors, we'd have to 
  //     walk the device tree, take into account monitor layout/positioning, 
  //     and compute the desktop dimensions from that.  Since these values
  //     are currently only used by do_reposition_window() method, we can
  //     live with primary-monitor values for the time being.
  screenX = GetSystemMetrics(SM_CXSCREEN);
  screenY = GetSystemMetrics(SM_CYSCREEN);

  // set flags for the capabilities of this display
  ext->hasmultisample = FALSE;      // no code for this extension yet
  ext->nummultisamples = 0;
  aaAvailable = FALSE;

  // set default settings
  if (ext->hasmultisample) {
    aa_on();  // enable fast multisample based antialiasing by default
              // other antialiasing techniques are slow, so only multisample
              // makes sense to enable by default.
  }

  cueingAvailable = TRUE;
  cueing_on(); // leave depth cueing on by default, despite the speed hit.

  cullingAvailable = TRUE;
  culling_off();

  set_sphere_mode(sphereMode);
  set_sphere_res(sphereRes);
  set_line_width(lineWidth);
  set_line_style(lineStyle);

  // reshape and clear the display, which initializes some other variables
  reshape();
  normal();
  clear();
  update();

  // successfully created window
  return TRUE;
}

// destructor ... close the window
OpenGLDisplayDevice::~OpenGLDisplayDevice(void) {
  if (have_window) {
    // close and delete windows, contexts, and display connections
    free_opengl_ctx(); // free display lists, textures, etc

#if VMDSPACEWARE
    vmd_closewin32spaceball(&glwsrv);
#endif
  }
}


/////////////////////////  protected nonvirtual routines  

// create a new window and set it's characteristics
int OpenGLDisplayDevice::open_window(char *nm, int *size, int *loc,
                                     int argc, char** argv) {
  int SX = 596, SY = 190;
  if (loc) {
    SX = loc[0];
    // X screen uses Y increasing from upper-left corner down; this is
    // opposite to what GL does, which is the way VMD was set up originally
    SY = screenY - loc[1] - size[1];
  }
  glwsrv.cursornum = 0; // initialize cursor number
  
  // window opening stuff goes here
  int rc = OpenWin32Connection(&glwsrv);
  if (rc != 0) {
    return -1;
  }

  xOrig = 0;
  yOrig = 0;
  xSize = size[0]; 
  ySize = size[1]; 
  glwsrv.width = xSize; 
  glwsrv.height = ySize; 
  rc = myCreateWindow(this, 0, 0, glwsrv.width, glwsrv.height);
  if (rc != 0) {
    return -1;
  }

  // Determine if stereo is available
  if (glwsrv.PFDisStereo == 0) {
    ext->hasstereo = FALSE;
  } else {
    ext->hasstereo = TRUE;
  }
  ext->stereodrawforced = FALSE; // don't force stereo draws initially
  
  setup_initial_opengl_state();

#ifdef VMDSPACEWARE
  vmd_setupwin32spaceball(&glwsrv); 
#endif

  // normal return: window was successfully created
  have_window = TRUE;
  // return window id
  return 0;
}


int OpenGLDisplayDevice::prepare3D(int do_clear) {
  // force reset of OpenGL context back to ours in case something
  // else modified the OpenGL state
  wglMakeCurrent(glwsrv.hDC, glwsrv.hRC);

  return OpenGLRenderer::prepare3D(do_clear);
}

void OpenGLDisplayDevice::do_resize_window(int width, int height) {
  RECT rcClient, rcWindow;
  POINT ptDiff;
  GetClientRect(glwsrv.hWnd, &rcClient);
  GetWindowRect(glwsrv.hWnd, &rcWindow);
  ptDiff.x = (rcWindow.right - rcWindow.left) - rcClient.right;
  ptDiff.y = (rcWindow.bottom - rcWindow.top) - rcClient.bottom;
  MoveWindow(glwsrv.hWnd, rcWindow.left, rcWindow.top, width + ptDiff.x, height + ptDiff.y, TRUE);
}

void OpenGLDisplayDevice::do_reposition_window(int xpos, int ypos) {
  RECT rcClient, rcWindow;
  GetClientRect(glwsrv.hWnd, &rcClient);
  GetWindowRect(glwsrv.hWnd, &rcWindow);
  MoveWindow(glwsrv.hWnd, xpos, ypos, rcWindow.right-rcWindow.left, rcWindow.bottom-rcWindow.top, TRUE);
}


/////////////////////////  public virtual routines  

//
// get the current state of the device's pointer (i.e. cursor if it has one)
//

// abs X pos of cursor from lower-left corner of display
int OpenGLDisplayDevice::x(void) {
  return glwsrv.MouseX;
}

// same, for Y direction
int OpenGLDisplayDevice::y(void) {
  return glwsrv.MouseY;
}

// return the current state of the shift, control, and alt keys
int OpenGLDisplayDevice::shift_state(void) {
  int retval = 0;

  if ((glwsrv.MouseFlags & MK_SHIFT) != 0)
    retval |= SHIFT;
  
  if ((glwsrv.MouseFlags & MK_CONTROL) != 0)
    retval |= CONTROL;

  return retval; 
}

// return the spaceball state, if any
int OpenGLDisplayDevice::spaceball(int *rx, int *ry, int *rz, int *tx, int *ty, int *tz, int *buttons) {

#ifdef VMDSPACEWARE
  if (glwsrv.sball != NULL) {
    *rx = glwsrv.spwevent.u.spwData.mData[SI_RX];
    *ry = glwsrv.spwevent.u.spwData.mData[SI_RY];
    *rz = glwsrv.spwevent.u.spwData.mData[SI_RZ];
    *tx = glwsrv.spwevent.u.spwData.mData[SI_TX];
    *ty = glwsrv.spwevent.u.spwData.mData[SI_TY];
    *tz = glwsrv.spwevent.u.spwData.mData[SI_TZ];
    *buttons = glwsrv.spwevent.u.spwData.bData.current;
    return 1;
  }
#endif

  return 0;
}


// set the Nth cursor shape as the current one.
void OpenGLDisplayDevice::set_cursor(int n) {
  glwsrv.cursornum = n; // hack to save cursor state when mouse enters/leaves

  switch (n) {
    default:
    case DisplayDevice::NORMAL_CURSOR:
      SetCursor(LoadCursor(NULL, IDC_ARROW));
      break;

    case DisplayDevice::TRANS_CURSOR:
      SetCursor(LoadCursor(NULL, IDC_SIZEALL));
      break;
 
    case DisplayDevice::SCALE_CURSOR:
      SetCursor(LoadCursor(NULL, IDC_SIZEWE));
      break;

    case DisplayDevice::PICK_CURSOR:
      SetCursor(LoadCursor(NULL, IDC_CROSS));
      break;

    case DisplayDevice::WAIT_CURSOR:
      SetCursor(LoadCursor(NULL, IDC_WAIT));
      break;
  }
}


//
// event handling routines
//

// queue the standard events (need only be called once ... but this is
// not done automatically by the window because it may not be necessary or
// even wanted)
void OpenGLDisplayDevice::queue_events(void) {
}

// read the next event ... returns an event type (one of the above ones),
// and a value.  Returns success, and sets arguments.
int OpenGLDisplayDevice::read_event(long &retdev, long &retval) {
  MSG msg;

  // This pumps the Windows message queue, forcing WEvents to be updated
  // by the time we return from DispatchMessage.
  if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
    TranslateMessage(&msg); // translate the message
    DispatchMessage(&msg);  // fire it off to the window proc
  } 

  retdev = glwsrv.WEvents;

  switch (glwsrv.WEvents) {
    case WIN_REDRAW:
      glwsrv.WEvents = WIN_NOEVENT;
      // reshape() is already called from within the window proc.
      _needRedraw = 1;
      return FALSE;

    case WIN_KBD:
      if (glwsrv.KeyFlag != '\0') {
        retval = glwsrv.KeyFlag;
        glwsrv.WEvents = WIN_NOEVENT;
        return TRUE;
      }
      break; 

    case WIN_KBD_ESCAPE:
    case WIN_KBD_UP:
    case WIN_KBD_DOWN:
    case WIN_KBD_LEFT:
    case WIN_KBD_RIGHT:
    case WIN_KBD_PAGE_UP:
    case WIN_KBD_PAGE_DOWN:
    case WIN_KBD_HOME:
    case WIN_KBD_END:
    case WIN_KBD_INSERT:
    case WIN_KBD_DELETE:
    case WIN_KBD_F1:
    case WIN_KBD_F2:
    case WIN_KBD_F3:
    case WIN_KBD_F4:
    case WIN_KBD_F5:
    case WIN_KBD_F6:
    case WIN_KBD_F7:
    case WIN_KBD_F8:
    case WIN_KBD_F9:
    case WIN_KBD_F10:
    case WIN_KBD_F11:
    case WIN_KBD_F12:
      retval = glwsrv.KeyFlag;
      glwsrv.WEvents = WIN_NOEVENT;
      return TRUE;

    case WIN_WHEELUP:
      retval = 1;
      glwsrv.WEvents = WIN_NOEVENT;
      return TRUE;

    case WIN_WHEELDOWN:
      retval = 1;
      glwsrv.WEvents = WIN_NOEVENT;
      return TRUE;
  
    case WIN_LEFT:
      // retval _must_ be either 1 or 0, nothing else...
      retval = (glwsrv.MouseFlags & MK_LBUTTON) != 0; 
      glwsrv.WEvents = WIN_NOEVENT;
      return TRUE;

    case WIN_MIDDLE:
      // retval _must_ be either 1 or 0, nothing else...
      retval = (glwsrv.MouseFlags & MK_MBUTTON) != 0; 
      glwsrv.WEvents = WIN_NOEVENT;
      return TRUE;

    case WIN_RIGHT:
      // retval _must_ be either 1 or 0, nothing else...
      retval = (glwsrv.MouseFlags & MK_RBUTTON) != 0; 
      glwsrv.WEvents = WIN_NOEVENT;
      return TRUE;
  }

  retval = 0; 
  glwsrv.WEvents = WIN_NOEVENT;
  return FALSE;
}


//
// virtual routines for preparing to draw, drawing, and finishing drawing
//

// reshape the display after a shape change
void OpenGLDisplayDevice::reshape(void) {
  // this code assumes that the xSize and ySize variables have
  // been updated (magically) already by the time this gets called.

  switch (inStereo) {
    case OPENGL_STEREO_SIDE:
      set_screen_pos(0.5f * (float)xSize / (float)ySize);
      break;

    case OPENGL_STEREO_ABOVEBELOW:
      set_screen_pos(2.0f * (float)xSize / (float)ySize);
      break;

    case OPENGL_STEREO_STENCIL_CHECKERBOARD:
    case OPENGL_STEREO_STENCIL_COLUMNS:
    case OPENGL_STEREO_STENCIL_ROWS:
      enable_stencil_stereo(inStereo);
      set_screen_pos((float)xSize / (float)ySize);
      break;
 
    default:
      set_screen_pos((float)xSize / (float)ySize);
      break;
  }
}

unsigned char * OpenGLDisplayDevice::readpixels(int &x, int &y) {
  unsigned char * img;

  x = xSize;
  y = ySize;

  if ((img = (unsigned char *) malloc(x * y * 3)) != NULL) {
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, x, y, GL_RGB, GL_UNSIGNED_BYTE, img);
  } else {
    x = 0;
    y = 0;
  } 

  return img; 
}


// update after drawing
void OpenGLDisplayDevice::update(int do_update) {
  glFlush();

  if(do_update) 
    SwapBuffers(glwsrv.hDC);

  glDrawBuffer(GL_BACK);
}


