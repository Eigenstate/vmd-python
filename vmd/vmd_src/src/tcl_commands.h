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
 *      $RCSfile: tcl_commands.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.29 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Fundamental VMD Tcl text commands.
 ***************************************************************************/


extern int text_cmd_animate(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_color(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_display(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_light(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_point_light(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_axes(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_stage(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_imd(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_label(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_material(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_menu(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_mol(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_mouse(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_mobile(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_spaceball(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_plugin(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_render(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_tkrender(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_tool(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_rotmat(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_rotate(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_translate(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_scale(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_rock(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_user(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_sleep(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_log(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_gopython(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_collab(ClientData, Tcl_Interp *, int, const char *argv[]);
extern int text_cmd_videostream(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_vmdbench(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_parallel(ClientData, Tcl_Interp *, int, const char *[]);
extern int text_cmd_profile(ClientData, Tcl_Interp *, int, const char *[]);
// XXX hack for getting around Swift/T issues
extern int swift_mpi_init(Tcl_Interp *);

extern int cmd_rawtimestep(ClientData, Tcl_Interp *, int, Tcl_Obj *const[]);
extern int cmd_gettimestep(ClientData, Tcl_Interp *, int, Tcl_Obj *const[]);

