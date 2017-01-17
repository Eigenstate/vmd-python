/*
 * eventio.c - Input device event I/O for modern Linux kernels,
 *             for joysticks and other input devices connected by
 *             hot-plug USB and Bluetooth interfaces
 * 
 *  $Id: eventio.c,v 1.3 2016/02/12 03:39:14 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <linux/input.h>

#include "eventio.h"

/*
 * Docs for low-level Linux kernel event I/O API:
 *   https://www.kernel.org/doc/Documentation/input/event-codes.txt
 #
 * Compile stand-alone test program:
 *   cc -DTEST_MAIN=1 eventio.c -o /tmp/evtest
 */

/* 
 * xorg.conf entry to prevent X from using a joystick to control the pointer
 */
/*
    /etc/X11/xorg.conf.d/50-joystick.conf
    Section "InputClass"
            Identifier "joystick catchall"
            MatchIsJoystick "on"
            MatchDevicePath "/dev/input/event*"
            Driver "joystick"
            Option "StartKeysEnabled" "False"       #Disable mouse
            Option "StartMouseEnabled" "False"      #support
    EndSection

  Another scheme to disable X interference:
    sudo apt-get install xserver-xorg-input-joystick
    xinput list
    xinput --set-prop "DEVICE NAME" "Device Enabled" 0

*/


#ifdef __cplusplus
extern "C" {
#endif

/*
 * Add macros that are missing for older linux kernels
 */
#if !defined(REL_RX)
#define REL_RX                  0x03
#endif
#if !defined(REL_RY)
#define REL_RY                  0x04
#endif
#if !defined(REL_RZ)
#define REL_RZ                  0x05
#endif

/* macro to round up bit count into next whole long, avoid use of */
/* linux kernel header macros that are not universally available  */
#define EVIO_LBIT (8*sizeof(long))
#define EVIO_BITSTOLONGS(bnr) (((bnr) + EVIO_LBIT - 1) / EVIO_LBIT)

/* mask off selected bit in array of longs and test it */
#define EVIO_TESTBIT(bnr, array) \
	(((1UL << ((bnr)&(EVIO_LBIT-1))) & ((array)[(bnr)/EVIO_LBIT]))!=0)

typedef struct {
  int fd;
  char devpath[2048];
  char devname[2048];
  struct input_id devid;
  unsigned long    evbit[EVIO_BITSTOLONGS( EV_MAX)];
  unsigned long   keybit[EVIO_BITSTOLONGS(KEY_MAX)];
  unsigned long keystate[EVIO_BITSTOLONGS(KEY_MAX)];
  unsigned long   absbit[EVIO_BITSTOLONGS(ABS_MAX)];
  unsigned long   relbit[EVIO_BITSTOLONGS(REL_MAX)];
  struct input_event inpev;
  int devjoystick;
  int devspaceball;
} evio;


evio_handle evio_open(const char *devpath) {
  evio *evp = NULL;
  int fd;
  if ((fd = open(devpath, O_RDONLY, 0)) < 0) {
    printf("Failed to open device '%s' ...\n", devpath);
    return NULL;
  }

  evp = (evio *) calloc(1, sizeof(evio));
  evp->fd = fd; 
  strncpy(evp->devpath, devpath, sizeof(evp->devpath));

  if (ioctl(evp->fd, EVIOCGNAME(sizeof(evp->devname)), evp->devname) < 0) {
    printf("Error) EVIOCGNAME ...\n");
    free(evp);
    return NULL;
  }

  if ((ioctl(evp->fd, EVIOCGBIT(0,      sizeof(evp->evbit)),  evp->evbit) < 0) ||
      (ioctl(evp->fd, EVIOCGBIT(EV_KEY, sizeof(evp->keybit)), evp->keybit) < 0) ||
      (ioctl(evp->fd, EVIOCGBIT(EV_ABS, sizeof(evp->absbit)), evp->absbit) < 0) || 
      (ioctl(evp->fd, EVIOCGBIT(EV_REL, sizeof(evp->relbit)), evp->relbit) < 0)) {
    printf("Error) ioctl() calls ...\n");
    free(evp);
    return NULL;
  }

  /* 
   * classify device as joystick, spaceball, or something else
   * see if it has buttons and absolute x/y position at a minimum 
   */
  if (EVIO_TESTBIT(EV_KEY, evp->evbit) && EVIO_TESTBIT(EV_ABS, evp->evbit) &&
      EVIO_TESTBIT(ABS_X, evp->absbit) && EVIO_TESTBIT(ABS_Y, evp->absbit) &&
      EVIO_TESTBIT(ABS_RX, evp->absbit) && EVIO_TESTBIT(ABS_RY, evp->absbit)) {
    evp->devjoystick = EVENTIO_JOYSTICK_LOGIF310;
  }

  if (!evp->devjoystick &&
      EVIO_TESTBIT(EV_KEY, evp->evbit) && EVIO_TESTBIT(EV_ABS, evp->evbit) &&
      EVIO_TESTBIT(ABS_X, evp->absbit) && EVIO_TESTBIT(ABS_Y, evp->absbit) &&
      EVIO_TESTBIT(ABS_Z, evp->absbit) && EVIO_TESTBIT(ABS_RZ, evp->absbit)) {
    evp->devjoystick = EVENTIO_JOYSTICK_NYKO;
  }

  if (!evp->devjoystick &&
      EVIO_TESTBIT(EV_KEY, evp->evbit) && EVIO_TESTBIT(EV_ABS, evp->evbit) &&
      EVIO_TESTBIT(ABS_X, evp->absbit) && EVIO_TESTBIT(ABS_Y, evp->absbit)) {
    evp->devjoystick = EVENTIO_JOYSTICK_STD;
  }

  /* see if it has buttons and relative x/y/z rx/ry/rz at a minimum */
  if (!evp->devjoystick &&
      EVIO_TESTBIT(EV_KEY, evp->evbit) && 
      EVIO_TESTBIT(EV_REL, evp->evbit) &&
      EVIO_TESTBIT(REL_X, evp->relbit) && 
      EVIO_TESTBIT(REL_Y, evp->relbit) && 
      EVIO_TESTBIT(REL_Z, evp->relbit) && 
      EVIO_TESTBIT(REL_RX, evp->relbit) && 
      EVIO_TESTBIT(REL_RY, evp->relbit) && 
      EVIO_TESTBIT(REL_RZ, evp->relbit)) {
    evp->devspaceball = EVENTIO_SPACEBALL_STD;
  }

  /* query device ID info */
  if (ioctl(evp->fd, EVIOCGID, &evp->devid) < 0) {
    printf("Error) EVIOCGID ...\n");
    free(evp);
    return NULL;
  }

  fcntl(evp->fd, F_SETFL, O_NONBLOCK); /* set device to non-blocking I/O */

  return evp;
}


int evio_close(evio_handle v) {
  evio *evp = (evio *) v;
  close(evp->fd);
  free(evp);

  return 0;
}


int evio_dev_joystick(evio_handle v) {
  evio *evp = (evio *) v;

  /* see if it has buttons and absolute x/y position at a minimum */
  if (evp->devjoystick)
    return 1;

  return 0;
}


int evio_dev_spaceball(evio_handle v) {
  evio *evp = (evio *) v;

  /* see if it has buttons and relative x/y/z rx/ry/rz at a minimum */
  if (evp->devspaceball)
    return 1;

  return 0;
}


int evio_dev_recognized(evio_handle v) {
  evio *evp = (evio *) v;

  if (evp->devjoystick || evp->devspaceball)
    return 1;

  return 0;
}


int evio_read_events(evio_handle v) {
  evio *evp = (evio *) v;
  int bcnt=0;
  int rc=0;

  do {
    bcnt = read(evp->fd, &evp->inpev, sizeof(evp->inpev));
  } while (bcnt == -1 && errno == EINTR);     

  if (bcnt > 0) {
    switch (evp->inpev.type) {
      case EV_SYN:
        printf("EV_SYN: '%s'      %ld:%ld s    \n",
               (evp->inpev.code == SYN_REPORT) ? "SYN_REPORT" : "Other",
               evp->inpev.time.tv_sec, evp->inpev.time.tv_usec);
        rc=1;
        break;

      case EV_KEY:
        printf("EV_KEY[0x%04x]: %d               \n",
               evp->inpev.code, evp->inpev.value);
        rc=1;
        break;

      case EV_ABS:
        printf("EV_ABS[%d]: %d               \n", 
               evp->inpev.code - ABS_X, evp->inpev.value);
        rc=1;
        break;

      case EV_REL:
        printf("EV_REL[%d]: %d               \n", 
               evp->inpev.code - REL_X, evp->inpev.value);
        rc=1;
        break;

      case EV_MSC:
        printf("EV_MSC[%d]                   \n", evp->inpev.code);
        rc=1;
        break;

      default:
        printf("Unknown event type: 0x%x     \n",
               evp->inpev.code - REL_X);
        rc=1;
        break;

    }
  } else {
    /* handle unexpected device disconnects and similar */
    if (errno != EAGAIN) {
      printf("Error reading events from input device\n");
      return -1;
    }
  }

  return rc; 
}


#if 0
int evio_print_absinfo(void) {
  printf("REL_X Values = { %d, %d, %d, %d, %d }\n",
         absinfo.value, absinfo.minimum, absinfo.maximum,
         absinfo.fuzz, absinfo.flat);
}
#endif


float evio_absinfo2float(struct input_absinfo *absinfo) {
  int val = absinfo->value - absinfo->minimum;
  int range = absinfo->maximum - absinfo->minimum;
  return 2.0f * ((float) val / (float) range) - 1.0f;
}


int evio_get_button_status(evio_handle v, int evbtn, int btnflag) {
  evio *evp = (evio *) v;
  int btmp=0;

  if (EVIO_TESTBIT(evbtn, evp->keystate))
    btmp |= btnflag;

  return btmp;  
}


int evio_get_joystick_status(evio_handle v, float *abs_x1, float *abs_y1,
                             float *abs_x2, float *abs_y2, int *buttons) {
  struct input_absinfo absinfo;
  int xax2, yax2;
  evio *evp = (evio *) v;
  int rc=0;

  memset(&absinfo, 0, sizeof(absinfo));

  *abs_x1 = 0;
  *abs_y1 = 0;
  *abs_x2 = 0;
  *abs_y2 = 0;
  *buttons = 0;

  if (ioctl(evp->fd, EVIOCGABS(ABS_X), &absinfo) < 0) {
    return 0;
  } else {
    rc |= 1;
    *abs_x1 = evio_absinfo2float(&absinfo);
  }
  if (ioctl(evp->fd, EVIOCGABS(ABS_Y), &absinfo) < 0) {
    return 0;
  } else {
    rc |= 1;
    *abs_y1 = evio_absinfo2float(&absinfo);
  }
 
  switch (evp->devjoystick) {
    case EVENTIO_JOYSTICK_LOGIF310:
      xax2 = ABS_RX; 
      yax2 = ABS_RY; 
      break;

    case EVENTIO_JOYSTICK_NYKO:
      xax2 = ABS_Z; 
      yax2 = ABS_RZ; 
      break;

    default:
    case EVENTIO_JOYSTICK_STD:
      xax2 = 0; 
      yax2 = 0; 
      break;
  }

  if (xax2 && yax2) { 
    if (ioctl(evp->fd, EVIOCGABS(xax2), &absinfo) < 0) {
      return 0;
    } else {
      rc |= 1;
      *abs_x2 = evio_absinfo2float(&absinfo);
    }
    if (ioctl(evp->fd, EVIOCGABS(yax2), &absinfo) < 0) {
      return 0;
    } else {
      rc |= 1;
      *abs_y2 = evio_absinfo2float(&absinfo);
    }
  }


  if (ioctl(evp->fd, EVIOCGKEY(sizeof(evp->keystate)), evp->keystate) >= 0) {
#if 1
    int btmp=0;
    btmp |= evio_get_button_status(v, BTN_BACK, EVENTIO_BACK);
    btmp |= evio_get_button_status(v, BTN_TASK, EVENTIO_TASK);
    btmp |= evio_get_button_status(v, BTN_START, EVENTIO_START);

    btmp |= evio_get_button_status(v, BTN_A, EVENTIO_GAMEPAD_A);
    btmp |= evio_get_button_status(v, BTN_B, EVENTIO_GAMEPAD_B);
    btmp |= evio_get_button_status(v, BTN_X, EVENTIO_GAMEPAD_X);
    btmp |= evio_get_button_status(v, BTN_Y, EVENTIO_GAMEPAD_Y);

    btmp |= evio_get_button_status(v, BTN_TL, EVENTIO_TL);
    btmp |= evio_get_button_status(v, BTN_TR, EVENTIO_TR);
    btmp |= evio_get_button_status(v, BTN_THUMBL, EVENTIO_THUMBL);
    btmp |= evio_get_button_status(v, BTN_THUMBR, EVENTIO_THUMBR);
#else      
    int i, btmp=0;
    for (i=0; i<31; i++) {
      if (EVIO_TESTBIT(BTN_GAMEPAD+i, evp->keybit))
        btmp |= (1 << i);
    }
#endif

    *buttons = btmp;
  }

  return rc;
}


int evio_get_spaceball_status(evio_handle v,
                              int *rel_x,
                              int *rel_y,
                              int *rel_z,
                              int *rel_rx,
                              int *rel_ry,
                              int *rel_rz,
                              int *buttons) {
  struct input_absinfo absinfo;
  evio *evp = (evio *) v;
  int rc=0;

  *rel_x = 0;
  *rel_y = 0;
  *rel_z = 0;
  *rel_rx = 0;
  *rel_ry = 0;
  *rel_rz = 0;
  *buttons = 0;

  memset(&absinfo, 0, sizeof(absinfo));

  if (ioctl(evp->fd, EVIOCGABS(REL_X), &absinfo) < 0) {
    return 0;
  } else {
    rc |= 1;
    *rel_x = absinfo.value;
  }
  if (ioctl(evp->fd, EVIOCGABS(REL_Y), &absinfo) < 0) {
    return 0;
  } else {
    rc |= 1;
    *rel_y = absinfo.value;
  }
  if (ioctl(evp->fd, EVIOCGABS(REL_Z), &absinfo) < 0) {
    return 0;
  } else {
    rc |= 1;
    *rel_z = absinfo.value;
  }

  if (ioctl(evp->fd, EVIOCGABS(REL_RX), &absinfo) < 0) {
    return 0;
  } else {
    rc |= 1;
    *rel_rx = absinfo.value;
  }
  if (ioctl(evp->fd, EVIOCGABS(REL_RY), &absinfo) < 0) {
    return 0;
  } else {
    rc |= 1;
    *rel_ry = absinfo.value;
  }
  if (ioctl(evp->fd, EVIOCGABS(REL_RZ), &absinfo) < 0) {
    return 0;
  } else {
    rc |= 1;
    *rel_rz = absinfo.value;
  }


  if (ioctl(evp->fd, EVIOCGBIT(EV_KEY, sizeof(evp->keybit)), evp->keybit) >= 0) {
    int i, btmp=0;
    for (i=0; i<31; i++) {
      if (EVIO_TESTBIT(BTN_MISC+i, evp->keybit))
        btmp |= (1 << i);
    }
    *buttons = btmp;
  }

  return rc; 
}


int evio_print_devinfo(evio_handle v) {
  evio *evp = (evio *) v;
  const char *busstr;

  if (!evio_dev_recognized(v)) {
    printf("Unrecognized device type\n");
    return EVENTIO_ERROR;
  }

  if (evp->devid.bustype == BUS_USB) {
    busstr = "USB";
  } else if (evp->devid.bustype == BUS_BLUETOOTH) {
    busstr = "Bluetooth";
  } else if (evp->devid.bustype == BUS_PCI) {
    busstr = "PCI";
  } else {
    busstr = "Other";
  }

  if (evp->devjoystick) {
    printf("Joystick at '%s':\n", evp->devpath);
  } else if (evp->devspaceball) {
    printf("Spaceball at '%s':\n", evp->devpath);
  }

  printf("  '%s'\n", evp->devname);
  printf("  bus: %s ", busstr);
  printf("  vendor: 0x%x", evp->devid.vendor);
  printf("  product: 0x%x \n", evp->devid.product);

  return EVENTIO_SUCCESS;
}



#if defined(TEST_MAIN)

int dev_valid(const char *devpath) {
  evio * evp = (evio *) evio_open(devpath);
  if (!evp)
    return EVENTIO_ERROR;

  if (evio_dev_recognized(evp)) {
    evio_print_devinfo(evp);
  } else {
    printf("Unrecognized device type: '%s' (%s)\n", evp->devpath, evp->devname);
  }

  evio_close(evp);
  return EVENTIO_SUCCESS;
}


int dev_test(const char *devpath) {
  evio * evp = (evio *) evio_open(devpath);
  if (!evp)
    return EVENTIO_ERROR;

  if (!evio_dev_recognized(evp)) {
    printf("Unrecognized device type: '%s' (%s)\n", evp->devpath, evp->devname);
    evio_close(evp);
    return EVENTIO_ERROR;
  }

  printf("Running loop test on device '%s'...\n", devpath);

#if 1
  if (evio_dev_joystick(evp)) {
    while (1) {
      float ax1, ay1, ax2, ay2;
      int buttons;
      if (evio_get_joystick_status(evp, &ax1, &ay1, &ax2, &ay2, &buttons)) {
        printf("Joystick: %5.2f %5.2f  %5.2f %5.2f  0x%08x  \r", 
               ax1, ay1, ax2, ay2, buttons);
      }
      if (buttons) 
        break; 
    }
  }
#endif

#if 0
  if (evio_dev_spaceball(evp)) {
    while (1) {
      int tx, ty, tz, rx, ry, rz, buttons;
      if (evio_get_spaceball_status(evp, &tx, &ty, &tz, 
                                    &rx, &ry, &rz, &buttons)) {
        printf("Spaceball: %6d %6d %6d  %6d %6d %6d  0x%08x  \r", 
               tx, ty, tz, rx, ry, rz, buttons);

        if (buttons) 
          break; 
      }
    }
  }
#endif


#if 1
  if (!evio_dev_joystick(evp) && !evio_dev_spaceball(evp)) {
    while (1) {
      int ev=0, hadev=0;
      do { 
        ev=evio_read_events(evp);
        hadev |= (ev > 0);
      } while (ev);
      if (hadev)
        printf("End of event report\n");
    }
  }
#endif

  evio_close(evp);
  return EVENTIO_SUCCESS;
}


int main(int argc, char **argv) {
  int rc, i;

  if (argc < 2) {
    for (i=0; i<40; i++) {
      char devpath[2048];
      sprintf(devpath, "/dev/input/event%d", i);
      if (dev_valid(devpath) == EVENTIO_SUCCESS)
        dev_test(devpath);
    }
  } else {
    for (i=1; i<argc; i++) {
      char *devpath=argv[i];
      if (dev_valid(devpath) == EVENTIO_SUCCESS)
        dev_test(devpath);
    }
  }

  return 0;
}

#endif


#ifdef __cplusplus
}
#endif


