/*
 * eventio.h - Input device event I/O for modern Linux kernels,
 *             for joysticks and other input devices connected by
 *             hot-plug USB and Bluetooth interfaces
 * 
 *  $Id: eventio.h,v 1.2 2016/02/12 03:39:14 johns Exp $
 */

#ifndef EVENTIO_INC
#define EVENTIO_INC 1

#ifdef __cplusplus
extern "C" {
#endif

#define EVENTIO_SUCCESS        0
#define EVENTIO_ERROR         -1

#define EVENTIO_BACK         0x0001
#define EVENTIO_TASK         0x0002
#define EVENTIO_START        0x0004
#define EVENTIO_UNUSED08     0x0008

#define EVENTIO_UNUSED10     0x0010
#define EVENTIO_UNUSED20     0x0020
#define EVENTIO_UNUSED40     0x0040
#define EVENTIO_UNUSED80     0x0080

#define EVENTIO_GAMEPAD_A    0x0100
#define EVENTIO_GAMEPAD_B    0x0200
#define EVENTIO_GAMEPAD_X    0x0400
#define EVENTIO_GAMEPAD_Y    0x0800

#define EVENTIO_TL           0x1000
#define EVENTIO_TR           0x2000
#define EVENTIO_THUMBL       0x4000
#define EVENTIO_THUMBR       0x8000

#define EVENTIO_JOYSTICK_STD        0x01
#define EVENTIO_JOYSTICK_LOGIF310   0x02
#define EVENTIO_JOYSTICK_NYKO       0x04

#define EVENTIO_SPACEBALL_STD       0x01

typedef void * evio_handle;
evio_handle evio_open(const char *devpath);
int evio_close(evio_handle);
int evio_is_joystick(evio_handle);
int evio_print_devinfo(evio_handle);
int evio_get_joystick_status(evio_handle, float *abs_x1, float *abs_y1,
                             float *abs_x2, float *abs_y2, int *buttons);

#ifdef __cplusplus
}
#endif

#endif


