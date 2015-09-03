/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

#include <stdio.h>
#include <FL/fl_draw.H>
#include <FL/forms.H>
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Hold_Browser.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Value_Slider.H>

#include "ColorFltkMenu.h"
#include "Command.h"
#include "Scene.h"
#include "VMDApp.h"
#include "Inform.h"

/// class to maintain a GUI-usable image of a ColorScale
class ColorscaleImage : public Fl_Box {
  VMDApp *app;
  unsigned char *data;
public:
  ColorscaleImage(int myx, int myy, int myw, int myh, VMDApp *vmdapp) 
  : Fl_Box(myx, myy, myw, myh), app(vmdapp) {
    data = new unsigned char[3*w()*h()];
  }
  ~ColorscaleImage() { delete [] data; }
protected:
  virtual void draw() {
    for (int i=0; i<w(); i++) {
      const float *rgb = app->scene->color_value(i*MAPCLRS/w()+REGCLRS);
      unsigned char r = (unsigned char)(255*rgb[0]);
      unsigned char g = (unsigned char)(255*rgb[1]);
      unsigned char b = (unsigned char)(255*rgb[2]);
      for (int j=0; j<h(); j++) {
        data[3*w()*j + 3*i + 0] = r;
        data[3*w()*j + 3*i + 1] = g;
        data[3*w()*j + 3*i + 2] = b;
      }
    }
    fl_draw_image(data, x(), y(), w(), h());
  }
};


void ColorFltkMenu::make_window() {
  size(400, 305);
  { 
    { Fl_Hold_Browser* o = categorybrowser = new Fl_Hold_Browser(10, 55, 125, 100, "Categories");
      o->align(FL_ALIGN_TOP);
      o->color(VMDMENU_BROWSER_BG, VMDMENU_BROWSER_SEL);
      o->callback(category_cb, this);
      VMDFLTKTOOLTIP(o, "Select color category then name to set active color")
    }
    { Fl_Hold_Browser* o = itembrowser = new Fl_Hold_Browser(140, 55, 120, 100, "Names");
      o->align(FL_ALIGN_TOP);
      o->color(VMDMENU_BROWSER_BG, VMDMENU_BROWSER_SEL);
      o->callback(item_cb, this);
      VMDFLTKTOOLTIP(o, "Select color category then name to set active color")
    }
    { Fl_Hold_Browser* o = colorbrowser = new Fl_Hold_Browser(265, 55, 125, 100, "Colors");
      o->align(FL_ALIGN_TOP);
      o->color(VMDMENU_BROWSER_BG, VMDMENU_BROWSER_SEL);
      o->callback(color_cb, this);
      VMDFLTKTOOLTIP(o, "Select color category then name to set active color")
    }
    new Fl_Box(10, 10, 190, 25, "Assign colors to categories:");
    { Fl_Tabs* o = new Fl_Tabs(0, 165, 400, 150);
#if defined(VMDMENU_WINDOW)
      o->color(VMDMENU_WINDOW, FL_GRAY);
      o->selection_color(VMDMENU_WINDOW);
#endif

      { Fl_Group* o = new Fl_Group(0, 185, 400, 125, "Color Definitions");
#if defined(VMDMENU_WINDOW)
        o->color(VMDMENU_WINDOW, FL_GRAY);
        o->selection_color(VMDMENU_WINDOW);
#endif
        { Fl_Hold_Browser* o = colordefbrowser = new Fl_Hold_Browser(15, 195, 135, 100);
          o->labeltype(FL_NO_LABEL);
          o->color(VMDMENU_BROWSER_BG, VMDMENU_BROWSER_SEL);
          o->callback(colordef_cb, this);
          VMDFLTKTOOLTIP(o, "Select color name to adjust RGB color definition")
        }
        { Fl_Value_Slider* o = redscale = new Fl_Value_Slider(160, 195, 225, 20);
          o->type(FL_HORIZONTAL);
          o->color(VMDMENU_COLOR_RSLIDER);
          o->callback(rgb_cb, this);
          VMDFLTKTOOLTIP(o, "Adjust slider to change RGB color definition")
        }
        { Fl_Value_Slider* o = greenscale = new Fl_Value_Slider(160, 215, 225, 20);
          o->type(FL_HORIZONTAL);
          o->color(VMDMENU_COLOR_GSLIDER);
          o->callback(rgb_cb, this);
          VMDFLTKTOOLTIP(o, "Adjust slider to change RGB color definition")
        }
        { Fl_Value_Slider* o = bluescale = new Fl_Value_Slider(160, 235, 225, 20);
          o->type(FL_HORIZONTAL);
          o->color(VMDMENU_COLOR_BSLIDER);
          o->callback(rgb_cb, this);
          VMDFLTKTOOLTIP(o, "Adjust slider to change RGB color definition")
        }
        { Fl_Button* o = grayscalebutton = new Fl_Button(165, 265, 85, 25, "Grayscale");
          o->type(FL_TOGGLE_BUTTON);
#if defined(VMDMENU_WINDOW)
          o->color(VMDMENU_WINDOW, FL_GRAY);
#endif
          VMDFLTKTOOLTIP(o, "Lock sliders for grayscale color")
        }
        defaultbutton = new Fl_Button(290, 265, 85, 25, "Default");
#if defined(VMDMENU_WINDOW)
        defaultbutton->color(VMDMENU_WINDOW, FL_GRAY);
#endif
        defaultbutton->callback(default_cb, this);
        VMDFLTKTOOLTIP(defaultbutton, "Reset to original RGB color")
        o->end();
      }
      { Fl_Group* o = new Fl_Group(0, 185, 400, 125, "Color Scale");
#if defined(VMDMENU_WINDOW)
        o->color(VMDMENU_WINDOW, FL_GRAY);
        o->selection_color(VMDMENU_WINDOW);
#endif
        o->hide();
        { Fl_Choice* o = scalemethod = new Fl_Choice(15, 220, 80, 25, "Method");
          o->color(VMDMENU_CHOOSER_BG, VMDMENU_CHOOSER_SEL);
          o->down_box(FL_BORDER_BOX);
          o->align(FL_ALIGN_TOP);
          o->callback(scalemethod_cb, this);
        }
        offsetvalue = new Fl_Value_Slider(160, 205, 180, 20, "Offset");
        offsetvalue->type(FL_HORIZONTAL);
        offsetvalue->color(VMDMENU_SLIDER_BG, VMDMENU_SLIDER_FG);
        offsetvalue->align(FL_ALIGN_LEFT);
        offsetvalue->range(-1.0, 1.0);
        offsetvalue->callback(scalesettings_cb, this);
        { Fl_Value_Slider* o = midpointvalue = new Fl_Value_Slider(160, 235, 180, 20, "Midpoint");
          o->type(FL_HORIZONTAL);
          midpointvalue->align(FL_ALIGN_LEFT);
          midpointvalue->color(VMDMENU_SLIDER_BG, VMDMENU_SLIDER_FG);
          o->range(0.0, 1.0);
          o->callback(scalesettings_cb, this);
        }
        image = new ColorscaleImage(10, 265, 380, 25, app);
        o->end();
      }
      o->end();
    }
    end();
  }
}

ColorFltkMenu::ColorFltkMenu(VMDApp *vmdapp) 
: VMDFltkMenu("color", "Color Controls", vmdapp) {
  make_window();

  command_wanted(Command::COLOR_SCALE_METHOD);
  command_wanted(Command::COLOR_SCALE_SETTINGS);
  command_wanted(Command::COLOR_SCALE_COLORS);
  command_wanted(Command::COLOR_CHANGE);
  command_wanted(Command::COLOR_NAME);
  command_wanted(Command::MOL_RENAME);
  command_wanted(Command::COLOR_ADD_ITEM);

  // set up color scale values
  for (int j=0; j<vmdapp->num_colorscale_methods(); j++)
    scalemethod->add(app->colorscale_method_name(j));

  reset_color_categories();
  reset_color_names();
  reset_color_scale();
}

void ColorFltkMenu::reset_color_categories() {
  categorybrowser->clear();
  int n = app->num_color_categories();
  for (int j=0; j<n; j++)
    categorybrowser->add(app->color_category(j));
  categorybrowser->value(0);  //nothing selected
}

void ColorFltkMenu::reset_color_names() {
  colorbrowser->clear();
  colordefbrowser->clear();
  int n = app->num_regular_colors();
  for (int j=0; j<n; j++) {
    char buf[128];
    sprintf(buf, "%d %s", j, app->color_name(j));
    colorbrowser->add(buf);
    colordefbrowser->add(buf);
  }
  colorbrowser->value(0);    // nothing selected
  colordefbrowser->value(0); // nothing selected
}

void ColorFltkMenu::reset_color_scale() {
  scalemethod->value(app->colorscale_method_current());
  float mid, min, max;
  app->colorscale_info(&mid, &min, &max);
  offsetvalue->value(min);
  midpointvalue->value(mid);
  image->redraw();
}

void ColorFltkMenu::update_chosen_color() {
  int catval = categorybrowser->value();
  int itemval = itembrowser->value();
  if (!catval || !itemval) {
    colordefbrowser->value(0);
    return;
  }
  const char *category = categorybrowser->text(catval);
  const char *item = itembrowser->text(itemval);
  const char *color = app->color_mapping(category, item);
  int index = app->color_index(color);
  if (index < 0) {
    msgErr << "ColorFltkMenu::update_chosen_color: invalid color" << sendmsg;
    return;
  }
  colorbrowser->value(index+1);
  colorbrowser->visible(index+1);
  colordefbrowser->value(index+1);
  colordefbrowser->visible(index+1);
  update_color_definition();
}

void ColorFltkMenu::update_color_definition() {
  float r, g, b;
  const char *colorname = app->color_name(colordefbrowser->value()-1);
  if (!app->color_value(colorname, &r, &g, &b)) return;
  redscale->value(r);
  greenscale->value(g);
  bluescale->value(b);
}

int ColorFltkMenu::act_on_command(int type, Command *) {
  switch (type) {
    case Command::COLOR_SCALE_METHOD:
    case Command::COLOR_SCALE_SETTINGS:
    case Command::COLOR_SCALE_COLORS:
      reset_color_scale();
      break;
    case Command::COLOR_CHANGE:
      update_color_definition();
      break;
    case Command::COLOR_NAME:
    case Command::MOL_RENAME:
      update_chosen_color();
      break;
    case Command::COLOR_ADD_ITEM:
      category_cb(NULL, this);
      break;
    default:
      ;
  }
  return FALSE;
}

void ColorFltkMenu::category_cb(Fl_Widget *, void *v) {
  ColorFltkMenu *self = (ColorFltkMenu *)v;
  int val = self->categorybrowser->value();
  if (!val) return;
  const char *category = self->categorybrowser->text(val);
  int n = self->app->num_color_category_items(category);
  self->itembrowser->clear();
  for (int i=0; i<n; i++)
    self->itembrowser->add(self->app->color_category_item(category, i));
  self->itembrowser->value(0);
  self->colorbrowser->value(0);
}

void ColorFltkMenu::item_cb(Fl_Widget *, void *v) {
  ColorFltkMenu *self = (ColorFltkMenu *)v;
  self->update_chosen_color();
}

void ColorFltkMenu::color_cb(Fl_Widget *, void *v) {
  ColorFltkMenu *self = (ColorFltkMenu *)v;
  int catval = self->categorybrowser->value();
  int itemval = self->itembrowser->value();
  int colorval = self->colorbrowser->value();
  if (!catval || !itemval || !colorval) return;
  const char *category = self->categorybrowser->text(catval);
  const char *item = self->itembrowser->text(itemval);
  const char *color = self->app->color_name(colorval-1);
  self->app->color_changename(category, item, color);
  self->update_chosen_color();
}

void ColorFltkMenu::colordef_cb(Fl_Widget *, void *v) {
  ColorFltkMenu *self = (ColorFltkMenu *)v;
  self->update_color_definition();
}

void ColorFltkMenu::rgb_cb(Fl_Widget *w, void *v) {
  ColorFltkMenu *self = (ColorFltkMenu *)v;
  int val = self->colordefbrowser->value();
  if (!val) return;
  const char *color = self->colordefbrowser->text(val);
  float r, g, b;
  if (self->grayscalebutton->value()) {
    r = g = b = (float)((Fl_Value_Slider *)w)->value();
  } else {
    r = (float)self->redscale->value();
    g = (float)self->greenscale->value();
    b = (float)self->bluescale->value();
  }
  self->app->color_changevalue(color, r, g, b);
}

void ColorFltkMenu::default_cb(Fl_Widget *, void *v) {
  ColorFltkMenu *self = (ColorFltkMenu *)v;
  int val = self->colordefbrowser->value();
  if (!val) return;
  const char *color = self->colordefbrowser->text(val);
  float r, g, b;
  if (!self->app->color_default_value(color, &r, &g, &b)) {
    msgErr << "ColorFltkMenu::default_cb(): invalid color" << sendmsg;
    return;
  }
  self->app->color_changevalue(color, r, g, b);
}

void ColorFltkMenu::scalemethod_cb(Fl_Widget *w, void *v) {
  ColorFltkMenu *self = (ColorFltkMenu *)v;
  Fl_Choice *choice = (Fl_Choice *)w;
  self->app->colorscale_setmethod(choice->value());
}

void ColorFltkMenu::scalesettings_cb(Fl_Widget *, void *v) {
  ColorFltkMenu *self = (ColorFltkMenu *)v;
  float mid, min, max;
  self->app->colorscale_info(&mid, &min, &max);
  mid = (float)self->midpointvalue->value();
  min = (float)self->offsetvalue->value();
  self->app->colorscale_setvalues(mid, min, max);
}
