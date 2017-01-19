#==============================================================================
# Contains procedures that create various bitmap and photo images.  The
# argument w specifies a canvas displaying a sort arrow, while the argument win
# stands for a tablelist widget.
#
# Copyright (c) 2006-2015  Csaba Nemethi (E-mail: csaba.nemethi@t-online.de)
#==============================================================================

#------------------------------------------------------------------------------
# tablelist::flat5x3Arrows
#------------------------------------------------------------------------------
proc tablelist::flat5x3Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp5x3_width 5
#define triangleUp5x3_height 3
static unsigned char triangleUp5x3_bits[] = {
   0x04, 0x0e, 0x1f};
"
    image create bitmap triangleDn$w -data "
#define triangleDn5x3_width 5
#define triangleDn5x3_height 3
static unsigned char triangleDn5x3_bits[] = {
   0x1f, 0x0e, 0x04};
"
}

#------------------------------------------------------------------------------
# tablelist::flat5x4Arrows
#------------------------------------------------------------------------------
proc tablelist::flat5x4Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp5x4_width 5
#define triangleUp5x4_height 4
static unsigned char triangleUp5x4_bits[] = {
   0x04, 0x0e, 0x1f, 0x1f};
"
    image create bitmap triangleDn$w -data "
#define triangleDn5x4_width 5
#define triangleDn5x4_height 4
static unsigned char triangleDn5x4_bits[] = {
   0x1f, 0x1f, 0x0e, 0x04};
"
}

#------------------------------------------------------------------------------
# tablelist::flat6x4Arrows
#------------------------------------------------------------------------------
proc tablelist::flat6x4Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp6x4_width 6
#define triangleUp6x4_height 4
static unsigned char triangleUp6x4_bits[] = {
   0x0c, 0x1e, 0x3f, 0x3f};
"
    image create bitmap triangleDn$w -data "
#define triangleDn6x4_width 6
#define triangleDn6x4_height 4
static unsigned char triangleDn7x4_bits[] = {
   0x3f, 0x3f, 0x1e, 0x0c};
"
}

#------------------------------------------------------------------------------
# tablelist::flat7x4Arrows
#------------------------------------------------------------------------------
proc tablelist::flat7x4Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp7x4_width 7
#define triangleUp7x4_height 4
static unsigned char triangleUp7x4_bits[] = {
   0x08, 0x1c, 0x3e, 0x7f};
"
    image create bitmap triangleDn$w -data "
#define triangleDn7x4_width 7
#define triangleDn7x4_height 4
static unsigned char triangleDn7x4_bits[] = {
   0x7f, 0x3e, 0x1c, 0x08};
"
}

#------------------------------------------------------------------------------
# tablelist::flat7x5Arrows
#------------------------------------------------------------------------------
proc tablelist::flat7x5Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp7x5_width 7
#define triangleUp7x5_height 5
static unsigned char triangleUp7x5_bits[] = {
   0x08, 0x1c, 0x3e, 0x7f, 0x7f};
"
    image create bitmap triangleDn$w -data "
#define triangleDn7x5_width 7
#define triangleDn7x5_height 5
static unsigned char triangleDn7x5_bits[] = {
   0x7f, 0x7f, 0x3e, 0x1c, 0x08};
"
}

#------------------------------------------------------------------------------
# tablelist::flat7x7Arrows
#------------------------------------------------------------------------------
proc tablelist::flat7x7Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp7x7_width 7
#define triangleUp7x7_height 7
static unsigned char triangleUp7x7_bits[] = {
   0x08, 0x1c, 0x1c, 0x3e, 0x3e, 0x7f, 0x7f};
"
    image create bitmap triangleDn$w -data "
#define triangleDn7x7_width 7
#define triangleDn7x7_height 7
static unsigned char triangleDn7x7_bits[] = {
   0x7f, 0x7f, 0x3e, 0x3e, 0x1c, 0x1c, 0x08};
"
}

#------------------------------------------------------------------------------
# tablelist::flat8x5Arrows
#------------------------------------------------------------------------------
proc tablelist::flat8x5Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp8x5_width 8
#define triangleUp8x5_height 5
static unsigned char triangleUp8x5_bits[] = {
   0x18, 0x3c, 0x7e, 0xff, 0xff};
"
    image create bitmap triangleDn$w -data "
#define triangleDn8x5_width 8
#define triangleDn8x5_height 5
static unsigned char triangleDn8x5_bits[] = {
   0xff, 0xff, 0x7e, 0x3c, 0x18};
"
}

#------------------------------------------------------------------------------
# tablelist::flat9x5Arrows
#------------------------------------------------------------------------------
proc tablelist::flat9x5Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp9x5_width 9
#define triangleUp9x5_height 5
static unsigned char triangleUp9x5_bits[] = {
   0x10, 0x00, 0x38, 0x00, 0x7c, 0x00, 0xfe, 0x00, 0xff, 0x01};
"
    image create bitmap triangleDn$w -data "
#define triangleDn9x5_width 9
#define triangleDn9x5_height 5
static unsigned char triangleDn9x5_bits[] = {
   0xff, 0x01, 0xfe, 0x00, 0x7c, 0x00, 0x38, 0x00, 0x10, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::flat9x6Arrows
#------------------------------------------------------------------------------
proc tablelist::flat9x6Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp9x6_width 9
#define triangleUp9x6_height 6
static unsigned char triangleUp9x6_bits[] = {
   0x10, 0x00, 0x38, 0x00, 0x7c, 0x00, 0xfe, 0x00, 0xff, 0x01, 0xff, 0x01};
"
    image create bitmap triangleDn$w -data "
#define triangleDn9x6_width 9
#define triangleDn9x6_height 6
static unsigned char triangleDn9x6_bits[] = {
   0xff, 0x01, 0xff, 0x01, 0xfe, 0x00, 0x7c, 0x00, 0x38, 0x00, 0x10, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::flatAngle7x4Arrows
#------------------------------------------------------------------------------
proc tablelist::flatAngle7x4Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp7x4_width 7
#define triangleUp7x4_height 4
static unsigned char triangleUp7x4_bits[] = {
   0x08, 0x1c, 0x36, 0x63};
"
    image create bitmap triangleDn$w -data "
#define triangleDn7x4_width 7
#define triangleDn7x4_height 4
static unsigned char triangleDn7x4_bits[] = {
   0x63, 0x36, 0x1c, 0x08};
"
}

#------------------------------------------------------------------------------
# tablelist::flatAngle7x5Arrows
#------------------------------------------------------------------------------
proc tablelist::flatAngle7x5Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp7x5_width 7
#define triangleUp7x5_height 5
static unsigned char triangleUp7x5_bits[] = {
   0x08, 0x1c, 0x3e, 0x77, 0x63};
"
    image create bitmap triangleDn$w -data "
#define triangleDn7x5_width 7
#define triangleDn7x5_height 5
static unsigned char triangleDn7x5_bits[] = {
   0x63, 0x77, 0x3e, 0x1c, 0x08};
"
}

#------------------------------------------------------------------------------
# tablelist::flatAngle9x5Arrows
#------------------------------------------------------------------------------
proc tablelist::flatAngle9x5Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp9x5_width 9
#define triangleUp9x5_height 5
static unsigned char triangleUp9x5_bits[] = {
   0x10, 0x00, 0x38, 0x00, 0x6c, 0x00, 0xc6, 0x00, 0x83, 0x01};
"
    image create bitmap triangleDn$w -data "
#define triangleDn9x5_width 9
#define triangleDn9x5_height 5
static unsigned char triangleDn9x5_bits[] = {
   0x83, 0x01, 0xc6, 0x00, 0x6c, 0x00, 0x38, 0x00, 0x10, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::flatAngle9x6Arrows
#------------------------------------------------------------------------------
proc tablelist::flatAngle9x6Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp9x6_width 9
#define triangleUp9x6_height 6
static unsigned char triangleUp9x6_bits[] = {
   0x10, 0x00, 0x38, 0x00, 0x7c, 0x00, 0xee, 0x00, 0xc7, 0x01, 0x83, 0x01};
"
    image create bitmap triangleDn$w -data "
#define triangleDn9x6_width 9
#define triangleDn9x6_height 6
static unsigned char triangleDn9x6_bits[] = {
   0x83, 0x01, 0xc7, 0x01, 0xee, 0x00, 0x7c, 0x00, 0x38, 0x00, 0x10, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::flatAngle9x7Arrows
#------------------------------------------------------------------------------
proc tablelist::flatAngle9x7Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp9x7_width 9
#define triangleUp9x7_height 7
static unsigned char triangleUp9x7_bits[] = {
   0x10, 0x00, 0x38, 0x00, 0x7c, 0x00, 0xfe, 0x00, 0xef, 0x01, 0xc7, 0x01,
   0x83, 0x01};
"
    image create bitmap triangleDn$w -data "
#define triangleDn9x7_width 9
#define triangleDn9x7_height 7
static unsigned char triangleDn9x7_bits[] = {
   0x83, 0x01, 0xc7, 0x01, 0xef, 0x01, 0xfe, 0x00, 0x7c, 0x00, 0x38, 0x00,
   0x10, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::flatAngle10x6Arrows
#------------------------------------------------------------------------------
proc tablelist::flatAngle10x6Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp10x6_width 10
#define triangleUp10x6_height 6
static unsigned char triangleUp10x6_bits[] = {
   0x30, 0x00, 0x78, 0x00, 0xfc, 0x00, 0xce, 0x01, 0x87, 0x03, 0x03, 0x03};
"
    image create bitmap triangleDn$w -data "
#define triangleDn10x6_width 10
#define triangleDn10x6_height 6
static unsigned char triangleDn10x6_bits[] = {
   0x03, 0x03, 0x87, 0x03, 0xce, 0x01, 0xfc, 0x00, 0x78, 0x00, 0x30, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::flatAngle10x7Arrows
#------------------------------------------------------------------------------
proc tablelist::flatAngle10x7Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp10x7_width 10
#define triangleUp10x7_height 7
static unsigned char triangleUp10x7_bits[] = {
   0x30, 0x00, 0x78, 0x00, 0xfc, 0x00, 0xfe, 0x01, 0xcf, 0x03, 0x87, 0x03,
   0x03, 0x03};
"
    image create bitmap triangleDn$w -data "
#define triangleDn10x7_width 10
#define triangleDn10x7_height 7
static unsigned char triangleDn10x6_bits[] = {
   0x03, 0x03, 0x87, 0x03, 0xcf, 0x03, 0xfe, 0x01, 0xfc, 0x00, 0x78, 0x00,
   0x30, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::photo7x7Arrows
#------------------------------------------------------------------------------
proc tablelist::photo7x7Arrows w {
    foreach dir {Up Dn} {
	image create photo triangle$dir$w
    }

    triangleUp$w put "
iVBORw0KGgoAAAANSUhEUgAAAAcAAAAHCAYAAADEUlfTAAAABGdBTUEAALGPC/xhBQAAACBjSFJN
AAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAACXBIWXMAAA7DAAAOwwHHb6hk
AAAAGnRFWHRTb2Z0d2FyZQBQYWludC5ORVQgdjMuNS4xMDD0cqEAAABCSURBVBhXXY4BCgAgCAP9
T//R9/Ryc+ZEHCyb40CB3D1n6OAZuQOKi9klPhUsjNJ6VwUp+tOLopOGNkXncToWw6IPjiowJNyp
gu8AAAAASUVORK5CYII=
"
    triangleDn$w put "
iVBORw0KGgoAAAANSUhEUgAAAAcAAAAHCAYAAADEUlfTAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwwAADsMBx2+oZAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAP0lE
QVQYV22LgQ0AIAjD9g//yD1ejoBoFpRkISsUPsMzPwkOIcARmJlvKMGIJq9jt+Uem51Wscfe1hkq
8VAdWKBfMCRjQcZZAAAAAElFTkSuQmCC
"
}

#------------------------------------------------------------------------------
# tablelist::sunken8x7Arrows
#------------------------------------------------------------------------------
proc tablelist::sunken8x7Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp8x7_width 8
#define triangleUp8x7_height 7
static unsigned char triangleUp8x7_bits[] = {
   0x18, 0x3c, 0x3c, 0x7e, 0x7e, 0xff, 0xff};
"
    image create bitmap darkLineUp$w -data "
#define darkLineUp8x7_width 8
#define darkLineUp8x7_height 7
static unsigned char darkLineUp8x7_bits[] = {
   0x08, 0x0c, 0x04, 0x06, 0x02, 0x03, 0x00};
"
    image create bitmap lightLineUp$w -data "
#define lightLineUp8x7_width 8
#define lightLineUp8x7_height 7
static unsigned char lightLineUp8x7_bits[] = {
   0x10, 0x30, 0x20, 0x60, 0x40, 0xc0, 0xff};
"
    image create bitmap triangleDn$w -data "
#define triangleDn8x7_width 8
#define triangleDn8x7_height 7
static unsigned char triangleDn8x7_bits[] = {
   0xff, 0xff, 0x7e, 0x7e, 0x3c, 0x3c, 0x18};
"
    image create bitmap darkLineDn$w -data "
#define darkLineDn8x7_width 8
#define darkLineDn8x7_height 7
static unsigned char darkLineDn8x7_bits[] = {
   0xff, 0x03, 0x02, 0x06, 0x04, 0x0c, 0x08};
"
    image create bitmap lightLineDn$w -data "
#define lightLineDn8x7_width 8
#define lightLineDn8x7_height 7
static unsigned char lightLineDn8x7_bits[] = {
   0x00, 0xc0, 0x40, 0x60, 0x20, 0x30, 0x10};
"
}

#------------------------------------------------------------------------------
# tablelist::sunken10x9Arrows
#------------------------------------------------------------------------------
proc tablelist::sunken10x9Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp10x9_width 10
#define triangleUp10x9_height 9
static unsigned char triangleUp10x9_bits[] = {
   0x30, 0x00, 0x78, 0x00, 0x78, 0x00, 0xfc, 0x00, 0xfc, 0x00, 0xfe, 0x01,
   0xfe, 0x01, 0xff, 0x03, 0xff, 0x03};
"
    image create bitmap darkLineUp$w -data "
#define darkLineUp10x9_width 10
#define darkLineUp10x9_height 9
static unsigned char darkLineUp10x9_bits[] = {
   0x10, 0x00, 0x18, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x04, 0x00, 0x06, 0x00,
   0x02, 0x00, 0x03, 0x00, 0x00, 0x00};
"
    image create bitmap lightLineUp$w -data "
#define lightLineUp10x9_width 10
#define lightLineUp10x9_height 9
static unsigned char lightLineUp10x9_bits[] = {
   0x20, 0x00, 0x60, 0x00, 0x40, 0x00, 0xc0, 0x00, 0x80, 0x00, 0x80, 0x01,
   0x00, 0x01, 0x00, 0x03, 0xff, 0x03};
"
    image create bitmap triangleDn$w -data "
#define triangleDn10x9_width 10
#define triangleDn10x9_height 9
static unsigned char triangleDn10x9_bits[] = {
   0xff, 0x03, 0xff, 0x03, 0xfe, 0x01, 0xfe, 0x01, 0xfc, 0x00, 0xfc, 0x00,
   0x78, 0x00, 0x78, 0x00, 0x30, 0x00};
"
    image create bitmap darkLineDn$w -data "
#define darkLineDn10x9_width 10
#define darkLineDn10x9_height 9
static unsigned char darkLineDn10x9_bits[] = {
   0xff, 0x03, 0x03, 0x00, 0x02, 0x00, 0x06, 0x00, 0x04, 0x00, 0x0c, 0x00,
   0x08, 0x00, 0x18, 0x00, 0x10, 0x00};
"
    image create bitmap lightLineDn$w -data "
#define lightLineDn10x9_width 10
#define lightLineDn10x9_height 9
static unsigned char lightLineDn10x9_bits[] = {
   0x00, 0x00, 0x00, 0x03, 0x00, 0x01, 0x80, 0x01, 0x80, 0x00, 0xc0, 0x00,
   0x40, 0x00, 0x60, 0x00, 0x20, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::sunken12x11Arrows
#------------------------------------------------------------------------------
proc tablelist::sunken12x11Arrows w {
    image create bitmap triangleUp$w -data "
#define triangleUp12x11_width 12
#define triangleUp12x11_height 11
static unsigned char triangleUp12x11_bits[] = {
   0x60, 0x00, 0xf0, 0x00, 0xf0, 0x00, 0xf8, 0x01, 0xf8, 0x01, 0xfc, 0x03,
   0xfc, 0x03, 0xfe, 0x07, 0xfe, 0x07, 0xff, 0x0f, 0xff, 0x0f};
"
    image create bitmap darkLineUp$w -data "
#define darkLineUp12x11_width 12
#define darkLineUp12x11_height 11
static unsigned char darkLineUp12x11_bits[] = {
   0x20, 0x00, 0x30, 0x00, 0x10, 0x00, 0x18, 0x00, 0x08, 0x00, 0x0c, 0x00,
   0x04, 0x00, 0x06, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00};
"
    image create bitmap lightLineUp$w -data "
#define lightLineUp12x11_width 12
#define lightLineUp12x11_height 11
static unsigned char lightLineUp12x11_bits[] = {
   0x40, 0x00, 0xc0, 0x00, 0x80, 0x00, 0x80, 0x01, 0x00, 0x01, 0x00, 0x03,
   0x00, 0x02, 0x00, 0x06, 0x00, 0x04, 0x00, 0x0c, 0xff, 0x0f};
"
    image create bitmap triangleDn$w -data "
#define triangleDn12x11_width 12
#define triangleDn12x11_height 11
static unsigned char triangleDn12x11_bits[] = {
   0xff, 0x0f, 0xff, 0x0f, 0xfe, 0x07, 0xfe, 0x07, 0xfc, 0x03, 0xfc, 0x03,
   0xf8, 0x01, 0xf8, 0x01, 0xf0, 0x00, 0xf0, 0x00, 0x60, 0x00};
"
    image create bitmap darkLineDn$w -data "
#define darkLineDn12x11_width 12
#define darkLineDn12x11_height 11
static unsigned char darkLineDn12x11_bits[] = {
   0xff, 0x0f, 0x03, 0x00, 0x02, 0x00, 0x06, 0x00, 0x04, 0x00, 0x0c, 0x00,
   0x08, 0x00, 0x18, 0x00, 0x10, 0x00, 0x30, 0x00, 0x20, 0x00};
"
    image create bitmap lightLineDn$w -data "
#define lightLineDn12x11_width 12
#define lightLineDn12x11_height 11
static unsigned char lightLineDn12x11_bits[] = {
   0x00, 0x00, 0x00, 0x0c, 0x00, 0x04, 0x00, 0x06, 0x00, 0x02, 0x00, 0x03,
   0x00, 0x01, 0x80, 0x01, 0x80, 0x00, 0xc0, 0x00, 0x40, 0x00};
"
}

#------------------------------------------------------------------------------
# tablelist::createSortRankImgs
#------------------------------------------------------------------------------
proc tablelist::createSortRankImgs win {
    image create bitmap sortRank1$win -data "
#define sortRank1_width 4
#define sortRank1_height 6
static unsigned char sortRank1_bits[] = {
   0x04, 0x06, 0x04, 0x04, 0x04, 0x04};
"
    image create bitmap sortRank2$win -data "
#define sortRank2_width 4
#define sortRank2_height 6
static unsigned char sortRank2_bits[] = {
   0x06, 0x09, 0x08, 0x04, 0x02, 0x0f};
"
    image create bitmap sortRank3$win -data "
#define sortRank3_width 4
#define sortRank3_height 6
static unsigned char sortRank3_bits[] = {
   0x0f, 0x08, 0x06, 0x08, 0x09, 0x06};
"
    image create bitmap sortRank4$win -data "
#define sortRank4_width 4
#define sortRank4_height 6
static unsigned char sortRank4_bits[] = {
   0x04, 0x06, 0x05, 0x0f, 0x04, 0x04};
"
    image create bitmap sortRank5$win -data "
#define sortRank5_width 4
#define sortRank5_height 6
static unsigned char sortRank5_bits[] = {
   0x0f, 0x01, 0x07, 0x08, 0x09, 0x06};
"
    image create bitmap sortRank6$win -data "
#define sortRank6_width 4
#define sortRank6_height 6
static unsigned char sortRank6_bits[] = {
   0x06, 0x01, 0x07, 0x09, 0x09, 0x06};
"
    image create bitmap sortRank7$win -data "
#define sortRank7_width 4
#define sortRank7_height 6
static unsigned char sortRank7_bits[] = {
   0x0f, 0x08, 0x04, 0x04, 0x02, 0x02};
"
    image create bitmap sortRank8$win -data "
#define sortRank8_width 4
#define sortRank8_height 6
static unsigned char sortRank8_bits[] = {
   0x06, 0x09, 0x06, 0x09, 0x09, 0x06};
"
    image create bitmap sortRank9$win -data "
#define sortRank9_width 4
#define sortRank9_height 6
static unsigned char sortRank9_bits[] = {
   0x06, 0x09, 0x09, 0x0e, 0x08, 0x06};
"
}

#------------------------------------------------------------------------------
# tablelist::createCheckbuttonImgs
#------------------------------------------------------------------------------
proc tablelist::createCheckbuttonImgs {} {
    variable checkedImg [image create bitmap tablelist_checkedImg -data "
#define checked_width 9
#define checked_height 9
static unsigned char checked_bits[] = {
   0x00, 0x00, 0x80, 0x00, 0xc0, 0x00, 0xe2, 0x00, 0x76, 0x00, 0x3e, 0x00,
   0x1c, 0x00, 0x08, 0x00, 0x00, 0x00};
"]

    variable uncheckedImg [image create bitmap tablelist_uncheckedImg -data "
#define unchecked_width 9
#define unchecked_height 9
static unsigned char unchecked_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
"]
}

#------------------------------------------------------------------------------
# tablelist::adwaitaTreeImgs
#------------------------------------------------------------------------------
proc tablelist::adwaitaTreeImgs {} {
    foreach mode {collapsed expanded collapsedSel expandedSel
		  collapsedAct expandedAct collapsedSelAct expandedSelAct} {
	variable adwaita_${mode}Img \
		 [image create photo tablelist_adwaita_${mode}Img]
    }

    variable pngSupported
    if {$pngSupported} {
	tablelist_adwaita_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwQAADsEBuJFr7QAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAVUlE
QVQoU2P4//8/yRirICGMymFgCELm48KoHAaGCiAORxbDhlE5EE3FhDSiciCaCGpE5SA0wTQmIcvD
1aFwKNREsvPICgiSg5z0yCUWYxXEj/8zAACoLdL8k+To/wAAAABJRU5ErkJggg==
"
	tablelist_adwaita_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwQAADsEBuJFr7QAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAWklE
QVQoU5WLQQ7AIAgEufQbfrIXf0+LWhLrxG1J5sDAmLv/BqUCpQKlAqUiptzUD5SMRhnhOY5vwmeQ
0SZcgvY7LXOIQftbRA8jwiBgaXaQf0CpQKlAucftAn9twSV64sz7AAAAAElFTkSuQmCC
"
	tablelist_adwaita_collapsedSelImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwQAADsEBuJFr7QAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAUElE
QVQoU2P4//8/yRirICGMLhCLxseK0QU6gTgFTQwDowuANLUAMV6N6AIgTQQ1ogvANME0FgExuhrq
aiLZeWQFBMlBTlbkEoWxCuLH/xkA08cuiUfbFjwAAAAASUVORK5CYII=
"
	tablelist_adwaita_expandedSelImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwQAADsEBuJFr7QAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAWElE
QVQoU5XLMQ6AMBADwTS8hK/S8OrDLjiBsoqhGERW51FVv2FMMCYYE4yJP7ucH/iuR/fwEDp274H1
j9BwGtjrIc8hDmwK4kOPcGAYZYPWMCYYE4xrNS6APy6jKYbrTgAAAABJRU5ErkJggg==
"
	tablelist_adwaita_collapsedActImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwAAADsABataJCQAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAV0lE
QVQoU2P4//8/yRirICGMymFg6EHm48KoHAaGs0A8BVkMG0blQDQdIaQRlQPRRFAjKgehCYSPAvFy
ZHm4OhQOhZpIdh5ZAUFykJMeucRirIL48X8GANuH2/YmUibsAAAAAElFTkSuQmCC
"
	tablelist_adwaita_expandedActImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwAAADsABataJCQAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAW0lE
QVQoU5WLMQ7AMAjEWPue/H/smDfRkqRIaaxci+QBg83df4NSgVKBUoFSEVNu6gdKRqOM8BzHN+Ez
yGgTLkH7nZY5xKD9LaKHEWEQsDQ7yD+gVKBUoNzjdgEnwso211sfFwAAAABJRU5ErkJggg==
"
	tablelist_adwaita_collapsedSelActImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOvwAADr8BOAVTJAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAUUlE
QVQoU2P4//8/yRirICGMLrAOjY8VowuAwE4gRhdHwegCIPAZiPFqRBeAAbwa0QWQwTcgvgjE6Gqo
q4lk55EVECQHOVmRSxTGKogf/2cAACEfOvsNAw2AAAAAAElFTkSuQmCC
"
	tablelist_adwaita_expandedSelActImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwAAADsABataJCQAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAWElE
QVQoU5XLwQ2AMBADwRSQnui/Ib6H/eBElBWGSIPCKh5V9RvGBGOCMcGY+HPIl+N3PbqHp9Bx74H1
RWi4DWz5kecQB7YF8UMfHBhGmdAaxgRjgvFdjQtoujswws5A0wAAAABJRU5ErkJggg==
"
    } else {
	tablelist_adwaita_collapsedImg put "
R0lGODlhDQAOAPcAAAAAAIeHh4yMjJ2dnaioqK2trQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgxAP8JHEiwoMGD/wogJBiAwEKBAQQ4XBgg4sSDFSMOQJhRwEaMFilKfNjwYUKTKFEGBAA7
"
	tablelist_adwaita_expandedImg put "
R0lGODlhDQAOAPcAAAAAAIeHh4yMjAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgpAP8JHEiwoMGDCBMqXBigoUOHBQU8DCDgoMSGFRFKzJgwwMKPIEMmDAgAOw==
"
	tablelist_adwaita_collapsedSelImg put "
R0lGODlhDQAOAPcAAAAAAMzMzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgrAP8JHEiwoMGD/wIgJBhA4cKEDR82jIhwIkWDFh1inLiQY0eNFR+KHCkwIAA7
"
	tablelist_adwaita_expandedSelImg put "
R0lGODlhDQAOAPcAAAAAAMzMzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgiAP8JHEiwoMGDCBMqXBigoUOHBR9CNCgx4USLCzNq3JgwIAA7
"
	tablelist_adwaita_collapsedActImg put "
R0lGODlhDQAOAPcAAAAAADIyMjo6Ojs7O1hYWGtra3NzcwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgxAP8JHEiwoMGD/wwgJBigwEKBAQY4XBgg4sSDFQMIIIAw48aOFilKfNjwYUKTKFEGBAA7
"
	tablelist_adwaita_expandedActImg put "
R0lGODlhDQAOAPcAAAAAADIyMjo6OgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgpAP8JHEiwoMGDCBMqXBigoUOHBQU8DCDgoMSGFRFKzJgwwMKPIEMmDAgAOw==
"
	tablelist_adwaita_collapsedSelActImg put "
R0lGODlhDQAOAPcAAAAAAP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgrAP8JHEiwoMGD/wIgJBhA4cKEDR82jIhwIkWDFh1inLiQY0eNFR+KHCkwIAA7
"
	tablelist_adwaita_expandedSelActImg put "
R0lGODlhDQAOAPcAAAAAAP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgiAP8JHEiwoMGDCBMqXBigoUOHBR9CNCgx4USLCzNq3JgwIAA7
"
    }
}

#------------------------------------------------------------------------------
# tablelist::ambianceTreeImgs
#------------------------------------------------------------------------------
proc tablelist::ambianceTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable ambiance_${mode}Img \
		 [image create photo tablelist_ambiance_${mode}Img]
    }

    tablelist_ambiance_collapsedImg put "
R0lGODlhEwAPAPcAAAAAADw7N9/Wxd/Wxt/WyODYyeLZyuHazeTdz+Pd0eTd0uXf0+Xf1efg1OXg
1ujh0+jg1Onj1+nj2Ork2O3m3Ozm3e7p4e/q4e7s5u/s6PHs5PHs5fHs5vHu6fPw6vTw6vTw6/bz
7vbz7/b07vb07/b08Pb08fj28/n49Pr59fr59vv5+Pv6+Pr6+vz6+fz7+v39/QAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAATAA8A
AAjAAP/9OyCgoEGDBwQqFFigQwsYECPCaJGhwMKBGVawcOHihUeOLFZgSKhQgIkTKFCkSKFiZcoT
JQQsFBBCBIkSJkwEyFmCxAgSMkt+8NDBAwgQAY568PDhQ1CBAjhI5aBhQ4ANG6ZyePpPwIULFgKI
HRvg6wWuAipUoKC2QoC2bdFOiCChroQAdiNMmID2AYS/fwMAhtDgAVoFCBIrXoxgAdcDDAxInkzZ
gAOSCgskIDCgs+cBBBJYvDjwoGnMCgMCADs=
"
    tablelist_ambiance_expandedImg put "
R0lGODlhEwAPAPcAAAAAADw7N9/Wxd/Wxt/WyODYyeLZyuHazeTdz+Pd0eTd0uXf0+Xf1efg1OXg
1ujh0+jg1Onj1+nj2Ork2O3m3Ozm3e7p4e/q4e7s5u/s6PHs5PHs5fHs5vHu6fPw6vTw6vTw6/bz
7vbz7/b07vb07/b08Pb08fj18fj28/n49Pr59fr59vv5+Pv6+Pr6+vz6+fz7+v39/QAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAATAA8A
AAi5AP/9OyCgoEGDBwQqFFigg4sYECPGcJGhwMKBGVi0ePEChkeOLVhgSKhQgAkUKVKoULFiZUoU
JQQsFBBCBIkSJkycyFmCxAgSMkt+8NDBA4ijRz14+PAhqEABHKJy0LCBw4aqUp3+E3DhgoUAYMMG
6HpBq4AKFSigXcu2gtkJESTInSs3woQJZh9A2Mu3b4MHZhUgGEy4MIIFWg8wMMC4sWMDDkgqLJCA
wIDLmAcQSGDx4sCDoCUrDAgAOw==
"
}

#------------------------------------------------------------------------------
# tablelist::aquaTreeImgs
#------------------------------------------------------------------------------
proc tablelist::aquaTreeImgs {} {
    foreach mode {collapsed expanded collapsedSel expandedSel} {
	variable aqua_${mode}Img \
		 [image create photo tablelist_aqua_${mode}Img]
    }

    variable pngSupported
    variable winSys
    scan $::tcl_platform(osVersion) "%d" majorOSVersion
    if {[string compare $winSys "aqua"] == 0 && $majorOSVersion > 10} {
	set osVerPost10 1
    } else {
	set osVerPost10 0
    }

    if {$pngSupported} {
	if {$osVerPost10} {
	    tablelist_aqua_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAABUAAAAOCAYAAADABlfOAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwgAADsIBFShKgAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAaklE
QVQ4T2MYBSjg////DMXFxQ+B2A/GR6bJBkAD/0PxViBWAhmICxMNkAwF4Z9AXIXNQBAmGqAZCsNX
gdie2obCcAC6wUQZjsUgEMZwKUkAzTCMMCULIBmIEvtkAZhmoEHUT6eDEDAwAACJ1s4t5kg57QAA
AABJRU5ErkJggg==
"
	    tablelist_aqua_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAABUAAAAOCAYAAADABlfOAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwgAADsIBFShKgAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAgElE
QVQ4T82Quw3AIAxEmYVF2ABlFzr2YhQGYIxU5E5yJIsiiQkFTzrxke8VdlvQe5ebgbfSPtKUUkP6
Q5qMfgelOEjGRBm1gWIZRHeKjNjgzlD2yKlkDN9+aqdExFkJmTwlZEkHoirCqv/N6DJkQaRB//8G
woPnMiFZJtLYpc5dCuytTkpAX+QAAAAASUVORK5CYII=
"
	} else {
	    tablelist_aqua_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAABUAAAAOCAYAAADABlfOAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwgAADsIBFShKgAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAVUlE
QVQ4T2MYBRigra3tIRD7QbnUAUAD/0PxViBWggpTBpAMBeGfQFwFlSIfoBkKw1eB2B6qhHSAZhg6
DoAqIw1gMQiEqepSqocpVWOf+ul0EAIGBgD3QGlokXBrxgAAAABJRU5ErkJggg==
"
	    tablelist_aqua_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAABUAAAAOCAYAAADABlfOAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwgAADsIBFShKgAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAbklE
QVQ4T2MYBbQHbW1tD4H4Px78EKqUeADU5IdmCDr2gyolDQA1bkUzCIa3QpWQDoCalYD4J5JhIAzi
K/3//58BhMkCQAOqkAwE4SqoFGUAaNBVqIFXoUKUA6Bh9lBD7aFC1AFAAwOgzGEHGBgA7s9vYvwQ
9+MAAAAASUVORK5CYII=
"
	}

	tablelist_aqua_collapsedSelImg put "
iVBORw0KGgoAAAANSUhEUgAAABUAAAAOCAYAAADABlfOAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwwAADsMBx2+oZAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAVElE
QVQ4T2MYBRjg////N4HYEcqlDgAaCANLgFgCKkwZgJgHB5+AOAsqRT6AmIUBTgOxKVQJ6QBiBk7g
DFVGGoBqRgdUdSnVw5SqsU/9dDoIAQMDAK4npwmnhxytAAAAAElFTkSuQmCC
"
	tablelist_aqua_expandedSelImg put "
iVBORw0KGgoAAAANSUhEUgAAABUAAAAOCAYAAADABlfOAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwwAADsMBx2+oZAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAbElE
QVQ4T2MYBbQH////vwnE+MBNqFLiAVCTI0QvTuAIVUoaAGpcAtGPAZZAlZAOgJolgPgT2BgEAPEl
oErIA0ADssBGIUAWVIoyADToNMS8/6ehQpQDoGGmEDP/m0KFqAOABjpDmcMOMDAAAJg2pwkFRUVJ
AAAAAElFTkSuQmCC
"
    } else {
	if {$osVerPost10} {
	    tablelist_aqua_collapsedImg put "
R0lGODlhFQAOAPcAAAAAAHNzc3Z2doODg4qKipubmwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAVAA4A
AAg5AP8JHEiwoMGDCBMqXChwAEOEAQIUeFgwYgABFAdajEgg40aLFD92DBkRY8Z/ESee/OdwpcuX
AgMCADs=
"
	    tablelist_aqua_expandedImg put "
R0lGODlhFQAOAPcAAAAAAHNzc3Z2doODg4qKipubmwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAVAA4A
AAg2AP8JHEiwoMGDCBMqXMhQ4YAAECMGGNBQYsSG/wpYLIDxn4CIAjoKvCjyH4EABEqOVMmypcqA
ADs=
"
	} else {
	    tablelist_aqua_collapsedImg put "
R0lGODlhFQAOAPcAAAAAAIaGhoiIiJSUlJmZmampqQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAVAA4A
AAg5AP8JHEiwoMGDCBMqXChwAEOEAQIUeFgwYgABFAdajEgg40aLFD92DBkRY8Z/ESee/OdwpcuX
AgMCADs=

"
	    tablelist_aqua_expandedImg put "
R0lGODlhFQAOAPcAAAAAAIaGhoiIiJSUlJmZmampqQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAVAA4A
AAg2AP8JHEiwoMGDCBMqXMhQ4YAAECMGGNBQYsSG/wpYLIDxn4CIAjoKvCjyH4EABEqOVMmypcqA
ADs=
"
	}

	tablelist_aqua_collapsedSelImg put "
R0lGODlhFQAOAPcAAAAAAP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAVAA4A
AAgyAP8JHEiwoMGDCBMqXCgwAEOEASI+LBhR4sSGFS0+zFhxIkeHHjte/CdyJMiRKFMSDAgAOw==
"
	tablelist_aqua_expandedSelImg put "
R0lGODlhFQAOAPcAAAAAAP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAVAA4A
AAgvAP8JHEiwoMGDCBMqXMhQYYCHECE2jChxIsUADQVSzDiwIsd/Hj9i/EiypEmCAQEAOw==
"
    }
}

#------------------------------------------------------------------------------
# tablelist::baghiraTreeImgs
#------------------------------------------------------------------------------
proc tablelist::baghiraTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable baghira_${mode}Img \
		 [image create photo tablelist_baghira_${mode}Img]
    }

    tablelist_baghira_collapsedImg put "
R0lGODlhEAAOAPcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgoAP8JHEiwoMGDCBMqXCgQAMOBABw+jChxIcWKCSlOxKiQ48OPIBkGBAA7
"
    tablelist_baghira_expandedImg put "
R0lGODlhEAAOAPcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgkAP8JHEiwoMGDCBMqXMiw4UEAECMCUChx4sKIDi063Mixo8KAADs=
"
}

#------------------------------------------------------------------------------
# tablelist::dustTreeImgs
#------------------------------------------------------------------------------
proc tablelist::dustTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable dust_${mode}Img \
		 [image create photo tablelist_dust_${mode}Img]
    }

    tablelist_dust_collapsedImg put "
R0lGODlhEwAPAPcAAAAAADIyMrConLGpncC6scC7ssG8s8K8tMK9tdDMxtDMx9LOyNrVztvVz9vW
ztvWz9vX0NzW0N3Y0d/a0uDd1uHe1+Lf2OPg2uTh2+Xj3efk3+jl4Ojm4enm4unn4+nn5Orn5evo
5Ovo5evq5ezp5e3q5u3q5+7s6O/t6e7t6vDu6/Hv7PHv7fLw7PLw7vT08vf39fj39fn49vn49wAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAATAA8A
AAjCAP8RGCCgoEGDAwj8W7jQwIIXMGLImEFRRgwYLxQcYPhvAIsSJ1CoWNFihQoUJ0a4GMBRAIsO
IUiUMEGzBIkQHVgIaJkCQwYNGzgE4LBBQwYMKXYyFGCCQgULFy4EiGqhAgUTShcKECGhq4QJASZ4
lSAi6z8BHRhECMC2bYAHDzqYRQv3gYMHARrUjTvXw168fz30/ft2r+CWISIoXsw4QgizAzxAaMwY
ggeWDA8oACGis2fPHxIg4CiQ4MGDAwqQDggAOw==
"
    tablelist_dust_expandedImg put "
R0lGODlhEwAPAPcAAAAAADIyMrConLGpncS+tcS/tsbBuMfBucfCucfCutfTzdjUztnUz9vWztvW
z9vX0NzW0N3Y0d/a0t/a0+Dd1uHe1+Lf2OPg2uTh2+Xj3efk3+jl4Ojm4enm4unn4+vo5Ovo5evp
5uvq5ezp5e3q5u3q5+7s6O/t6e7t6vDu6/Hv7PLv7fLw7PHw7fLx7vX08vf39fj39fn49vn49wAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAATAA8A
AAi9AP8RGCCgoEGDAwj8W7jQwIIXMGLImEFRRgwYLxQgYPhvQAsSJk6kUMFCRYoTJkS4GMBRwIoO
H0aQKEGTxIgPHVYIaIkCQwYNGzgI3aAhAwYUOxkKKEGhgoULFzBAtVCBQomkCwWAiMA1goQJErpG
AIH1n4AOCyAEWMs2gAMHHcqefeugwVsGdOHK9ZC371sPe/32BdzyA4TDiBND+FB2wIcHihM/+MCS
4QEGIUBo3sx5QQKOAgkePDigAOiAADs=
"
}

#------------------------------------------------------------------------------
# tablelist::dustSandTreeImgs
#------------------------------------------------------------------------------
proc tablelist::dustSandTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable dustSand_${mode}Img \
		 [image create photo tablelist_dustSand_${mode}Img]
    }

    tablelist_dustSand_collapsedImg put "
R0lGODlhEwAPAPcAAAAAADIyMpuWjJyXjrSxqbWxqre0rbi1rrm2r8K+tsO+t8TAuMXBusbCu8nF
vcnGvsrHv8vHwMnGwcrIwczIwMzIwc3Jw87LxMzKxc/MxdDMxdDMx9HOx9HOyNLPydPPytPQydPQ
y9TRy9TRzNXTzNXSzdfTztbUzNbUzdfVz9nV0NjW0NnW0drY097c2eDe2eXi3uXj3+jm4+nn5Oro
5QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAATAA8A
AAjCAP8RGCCgoEGDAwj8W7jQwAQYMmTMmEFjYsQYEg4w/DfgRYkSJlSsYLFChYkSI14M2CjAxYYO
H0SMmCniQ4cNLgSwbFHBwoUMGgJoyHDBQoUWOhkKSOHgAYQIFAJQiADhgYMUSRcKCMGgK4MGARp4
ZRAi6z8BHBIoCMC2bQAFCjiYRQsXboIAauvKZcmhLty3fvcq7eB3beEOcz0sWMy48QIPZgeUcEyZ
xEqGBzCgAMG5c+cTGBBs/FeA4MGDAwqMDggAOw==
"
    tablelist_dustSand_expandedImg put "
R0lGODlhEwAPAPcAAAAAADIyMpuWjJyXjrSxqbWxqre0rbi1rrm2r8K+tsO+t8TAuMXBusbCu8bC
vMnFvcnGvsrHv8vHwMnGwcrIwczIwMzIwc3Jw87LxMzKxc/Mxc/MxtDMxdDMx9HOx9HOyNLPydPP
ytPQydPQy9TRy9TRzNXTzNXSzdfTztbUzNbUzdfVz9nV0NjW0NnW0drY097c2eDe2eXi3uXj3+jm
4+nn5Oro5QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAATAA8A
AAi/AP8RGCCgoEGDAwj8W7jQAAUZNGjUqGFjYsQZEw4w/DcgxokTKFi0cNGCBYoTJWIM2CgARocP
IUiUmEkixIcOMASwfGHhAgYNHDZw0IDhgoUXOhkKWPEAQgQJFSxUkBABwoMVSRcKGMGgK4MGDhp4
ZTAi6z8BHhIoCMC2bQAFCjyYRQsXboK7dePO9ZC3L1y5LD/47fthLogFiBMrXgDC7IATiyObWMnw
QAYVIjJr1pwiA4KN/woQPHhwQAHQAQEAOw==
"
}

#------------------------------------------------------------------------------
# tablelist::gtkTreeImgs
#------------------------------------------------------------------------------
proc tablelist::gtkTreeImgs {} {
    foreach mode {collapsed expanded collapsedAct expandedAct} {
	variable gtk_${mode}Img \
		 [image create photo tablelist_gtk_${mode}Img]
    }

    variable pngSupported
    if {$pngSupported} {
	tablelist_gtk_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAABIAAAAOCAYAAAAi2ky3AAAABGdBTUEAALGPC/xhBQAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMtfuaUMAAABTSURBVDhPtdLBDQAgCANA376YgTmcj8UY
TG2iiV9t7QAX0lLKxzSJbWZ9QjwGyN15DFBm8hgghMY2RGMnBCwi+hqgXq0pv0jSEY2gA+kfST77
FRmz+lZUJ0vkXgAAAABJRU5ErkJggg==
"
	tablelist_gtk_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAABIAAAAOCAYAAAAi2ky3AAAABGdBTUEAALGPC/xhBQAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMtfuaUMAAABRSURBVDhPY2AYBXQNARsBAYH/xGKgy2zw
uc5GSUnp/+HDh/9jAyBxkDwhQ2AWYDWMVEOwGkauISiGtbW1keQdXOEGjgBiw4RQ0uAipGBwywMA
wOZWVA37acsAAAAASUVORK5CYII=
"
	tablelist_gtk_collapsedActImg put "
iVBORw0KGgoAAAANSUhEUgAAABIAAAAOCAYAAAAi2ky3AAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuND6NzHYAAABLSURBVDhPY2CgIbChitkCAgL/gQZRbhjI
IAkJCcoNAxlEFcNgBlFsGLJBFBmGbhCMD4wALpJikyYuoij2qBprFLkEFpBUS9nkZg8AkLUuN8tq
YDwAAAAASUVORK5CYII=
"
	tablelist_gtk_expandedActImg put "
iVBORw0KGgoAAAANSUhEUgAAABIAAAAOCAYAAAAi2ky3AAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuND6NzHYAAABCSURBVDhPY2AYBXQNgUwBAYH/xGKgyzLx
uY4owwgZArMAr2HEGoLXMFINwTBMSUnpP7mGoBhGqSEww0TomnaobhkAI24v01R6oWMAAAAASUVO
RK5CYII=
"
    } else {
	tablelist_gtk_collapsedImg put "
R0lGODlhEgAOAPcAAAAAABAQECIiIoaGhsPDw////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAASAA4A
AAg5AP8JHEiwoMGDCAkGSIgwgACGBgMQeAhxYIACEysKvIiRIkSOBQYs/Figo8aLGU+mPOnxpMaX
BAMCADs=
"
	tablelist_gtk_expandedImg put "
R0lGODlhEgAOAPcAAAAAABAQECIiIoaGhsPDw////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAASAA4A
AAg1AP8JHEiwoMGDCBMqXMiwYYCHECMGQCiAQIGLFwkIUFgRo0aGHT82FDBgY0OBE0+qXMlSYEAA
Ow==
"
	tablelist_gtk_collapsedActImg put "
R0lGODlhEgAOAPcAAAAAABAQEBgYGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAASAA4A
AAg0AP8JHEiwoMGDCAkGSIgwgACGBgM4hKhQ4kOK/yRaxKhRI8WOFyFqDClyIsaMJD+eXCkwIAA7
"
	tablelist_gtk_expandedActImg put "
R0lGODlhEgAOAPcAAAAAABAQECIiIgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAASAA4A
AAgsAP8JHEiwoMGDCBMqXMiwYYCHECMGQCgxosKKExdKbPjvoQCOAjOCHEkSZEAAOw==
"
    }
}

#------------------------------------------------------------------------------
# tablelist::klearlooksTreeImgs
#------------------------------------------------------------------------------
proc tablelist::klearlooksTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable klearlooks_${mode}Img \
		 [image create photo tablelist_klearlooks_${mode}Img]
    }

    tablelist_klearlooks_collapsedImg put "
R0lGODlhEAAOAPcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgoAP8JHEiwoMGDCBMqXPgPAACGAh0+hCgRYkOHDCsu1LhxosWPIBkGBAA7
"
    tablelist_klearlooks_expandedImg put "
R0lGODlhEAAOAPcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgoAP8JHEiwoMGDCBMqXMiQIAAAAx8mfAiRokKKEhdabAixocePID8GBAA7
"
}

#------------------------------------------------------------------------------
# tablelist::mateTreeImgs
#------------------------------------------------------------------------------
proc tablelist::mateTreeImgs {} {
    foreach mode {collapsed expanded collapsedAct expandedAct} {
	variable mate_${mode}Img \
		 [image create photo tablelist_mate_${mode}Img]
    }

    variable pngSupported
    if {$pngSupported} {
	tablelist_mate_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAadEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41
LjEwMPRyoQAAAI5JREFUKFOVzsENQEAQBdApQAEaUIAyFKARVxGJCsRBAdwUgLg4KGx9VnYmMmvj
8JKf2f+TJWPMb+oxRD2GcCCK5MMXDkQLZPLRhwPRATP0kMjSGwc7qmF4cgHqlznYYgkNtLDDCrEc
3F0XeNTBBiOksuy6LtjRBNcgl6U3DnZUyUcfDp6vaNRjiHr8ZugEGr3O4KKrCEAAAAAASUVORK5C
YII=
"
	tablelist_mate_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAadEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41
LjEwMPRyoQAAAI1JREFUKFOV0L0NwjAQhuEbgAG8AAMwgJdhDbr0UFKkygIoKZEiKgYzb5zEnMkp
J4rHsr/7KSwppb+ZoccMPWbomQ+RiPeO42ZoGbyjw0V54KwHcm+5iARMW29o0KLHQQ/k3urBVgy4
4oWo66WverAV0/YnRl3TtsH3U8JvbWWHIicrX5mhxww9ZrgvyQcolMcXU9O+JgAAAABJRU5ErkJg
gg==
"
	tablelist_mate_collapsedActImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAadEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41
LjEwMPRyoQAAAI1JREFUKFOVkMENgzAMRf8WTNEpmIFZOgGDcKsqem3FBSE6ERO4P8SKLWSIenjK
i/MtW4GI/E1YrBEWa5gAjX+4wgRYyd0/nmECbNo4kdaHjpgAXzKSWX0g4comOfggL/LWe5p88w17
tog1fUgKp4mdD5dskdy06Nn70BGTHH6S6tebnKwSERZrhMVrBD8rI88WMxe4BQAAAABJRU5ErkJg
gg==
"
	tablelist_mate_expandedActImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAadEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41
LjEwMPRyoQAAAH9JREFUKFOVkNENgDAIBZnFSbqHU7hDp3EDExNTHQyh0kotKfHjVI73+BAQ8Tem
9DClhyk9ngdAJK4Bc1eSYiJOYlNwYdWFnK0fAEFCB7ETfITnSRdythnoqgS5yO+o9zXXDHRVwkzS
O00v3p8SvruCLQEWyxdM6WFKD1OOQbgBANvLoUTXnyUAAAAASUVORK5CYII=
"
    } else {
	tablelist_mate_collapsedImg put "
R0lGODlhDQAOAPcAAAAAADw8PEBAQEVFRUdHR0hISElJSVVVVVxcXF1dXWlpaW1tbW9vb319fYCA
gIKCgoSEhI2Njba2tgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAg0AP8JHEiwoMGDCBMmLCBB4b8ABhQoDOAgQYAICANAaMBAAIGDGhcMQJDxwICJDxyqXKkyIAA7
"
	tablelist_mate_expandedImg put "
R0lGODlhDQAOAPcAAAAAADw8PEJCQkREREdHR09PT1BQUFJSUl9fX2VlZWlpaXFxcXJycn19fYSE
hJ2dnQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgzAP8JHEiwoMGDCBMmDMCwYQCDChA4mHjgwcEACxokMJCwAAMBCg0QGKDw38OSKFOqRBgQADs=
"
	tablelist_mate_collapsedActImg put "
R0lGODlhDQAOAPcAAAAAADw8PD09PT8/P0FBQUZGRkhISElJSVJSUlNTU1VVVVdXV15eXoiIiAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgzAP8JHEiwoMGDCBMmFNBA4b8AAgooDJCAQAAGCAMsQHAA4kGNBgQQyDggwEQFDlOqTBkQADs=
"
	tablelist_mate_expandedActImg put "
R0lGODlhDQAOAPcAAAAAADw8PD09PT4+PkBAQENDQ0ZGRlVVVQAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgxAP8JHEiwoMGDCBMmDMCwYQCDAwQYmBjgwMEABAoMeIgQI8eOAQYo/PdxpMmTKA0GBAA7
"
    }
}

#------------------------------------------------------------------------------
# tablelist::mintTreeImgs
#------------------------------------------------------------------------------
proc tablelist::mintTreeImgs {} {
    foreach mode {collapsed expanded collapsedSel expandedSel} {
	variable mint_${mode}Img \
		 [image create photo tablelist_mint_${mode}Img]
    }

    variable pngSupported
    if {$pngSupported} {
	tablelist_mint_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOvwAADr8BOAVTJAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAV0lE
QVQoU2P4//8/yRirICGMymFgmI3Mx4VROQwM94B4MbIYNozKgWi6SkgjKgeiiaBGVA5CEwhfB+Lt
yPJwdSgcCjWR7DyyAoLkICc9conFWAXx4/8MALGp3eTNJauCAAAAAElFTkSuQmCC
"
	tablelist_mint_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOvwAADr8BOAVTJAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAW0lE
QVQoU5WL0Q2AMAgFmcGBuv8EfnYVlLaS1F76lOQ+ODhz99+gVKBUoFSgVMSUm/qBktEoIzzH8U34
DDLahEvQfqdlDjFof4voYUQYBCzNDvIPKBUoFSj3uF1KHMxW02hYDgAAAABJRU5ErkJggg==
"
	tablelist_mint_collapsedSelImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOvgAADr4B6kKxwAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAUUlE
QVQoU2P4//8/yRirICGMLjAPjY8Vowu8AOLlaGIYGF0ApOk+EOPViC4A0kRQI7oATBNM4z4gRldD
XU0kO4+sgCA5yMmKXKIwVkH8+D8DAIioOIlmRAZbAAAAAElFTkSuQmCC
"
	tablelist_mint_expandedSelImg put "
iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOvwAADr8BOAVTJAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAWklE
QVQoU5XLsQ2AMBBD0TTsxP6L0LLFYRecQPmKoXiIfJ1HVf2GMcGYYEwwJv7scn7gux7dw0Po2L0H
1j9Cw2lgr4c8hziwKYgPPcKBYZQNWsOYYEwwrtW4AJnXOKReFQYYAAAAAElFTkSuQmCC
"
    } else {
	tablelist_mint_collapsedImg put "
R0lGODlhDQAOAPcAAAAAACEhISgoKCoqKkhISFxcXGRkZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgxAP8JHEiwoMGD/wwgJBigwEKBAQY4XBgg4sSDFQMIIIAw48aOFilKfNjwYUKTKFEGBAA7
"
	tablelist_mint_expandedImg put "
R0lGODlhDQAOAPcAAAAAACEhISkpKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgpAP8JHEiwoMGDCBMqXBigoUOHBQU8DCDgoMSGFRFKzJgwwMKPIEMmDAgAOw==
"
	tablelist_mint_collapsedSelImg put "
R0lGODlhDQAOAPcAAAAAAP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgrAP8JHEiwoMGD/wIgJBhA4cKEDR82jIhwIkWDFh1inLiQY0eNFR+KHCkwIAA7
"
	tablelist_mint_expandedSelImg put "
R0lGODlhDQAOAPcAAAAAAP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAANAA4A
AAgiAP8JHEiwoMGDCBMqXBigoUOHBR9CNCgx4USLCzNq3JgwIAA7
"
    }
}

#------------------------------------------------------------------------------
# tablelist::newWaveTreeImgs
#------------------------------------------------------------------------------
proc tablelist::newWaveTreeImgs {} {
    foreach mode {collapsed expanded collapsedAct expandedAct} {
	variable newWave_${mode}Img \
		 [image create photo tablelist_newWave_${mode}Img]
    }

    variable pngSupported
    if {$pngSupported} {
	tablelist_newWave_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAABIAAAAOCAYAAAAi2ky3AAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuNUmK/OAAAADDSURBVDhPY2AYtMDHx4cPiGf4+vqKUuRI
oCECHh4eH9zd3e8D2YVAzEqWgSCDnJ2db7GxseWbmZlt8vb2vggU8yLZMJBBLi4uV4EaI0BYVFQ0
y9HR8RBQfCsQaxBtIMggNze3i0ANbshYQ0MjH+jdy0D5fpAaggaCFAE1nAEqNEHGOjo6KcCwA3mT
eIM8PT1PAA1RAmFVVVVroAs3k+U1Ly+vw7KyspJAAzqABpwjO7CBGh8DMcgbFEU/dRIkwdiglwIA
fVBDhQiKWqEAAAAASUVORK5CYII=
"
	tablelist_newWave_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAABIAAAAOCAYAAAAi2ky3AAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuNUmK/OAAAACySURBVDhPY2AY3sDHx4cPiAUIYD6CoQA0
YIaHh8cHJyenu9gwSA6khqBBvr6+okDF99jY2PKBirOQMUgMJAdSQ9AgkAKgjYUWFhargcwIZAwS
A8kRZQjUIFZvb+8LEhIS0UC+GwiD2CAxoEGsRBsENczLxcVlB5BtCsIgNtAQL5IMgSkGatxqYmIS
BsIgNlmGQF2lAfTOKRAGGqRBtkFQw/qBhvRTZAjUIHDipNggcgwAAMDwRMOkXl9fAAAAAElFTkSu
QmCC
"
	tablelist_newWave_collapsedActImg put "
iVBORw0KGgoAAAANSUhEUgAAABIAAAAOCAYAAAAi2ky3AAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuNUmK/OAAAADOSURBVDhPY2AYtMDExITP2Nh4BhCLUuRI
oEECdqbGf+xNjd8ADSsE8lnJMhBkkI+F8Y9ZHoZP0u2MPlqYGF8HGuhFsmFggyyNvy/yNXoAwlO8
jB6H2xh/ARq2DYg1iDYQZJCvlfG3pYHGd5Fxp5fxEw9L489Aw/qBWICggWCDrI2/rgg1uYmMu3xM
HgEN+kSSQX7WJl/XRJpdA+E5Iaa3Iu1NPpLlNT8bky+rYswv57iZvrAwNb5GdmDbmxv/crQweUVp
9FMnQRKMDXopAACu/llcB/jCVQAAAABJRU5ErkJggg==
"
	tablelist_newWave_expandedActImg put "
iVBORw0KGgoAAAANSUhEUgAAABIAAAAOCAYAAAAi2ky3AAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuNUmK/OAAAAC7SURBVDhPY2AY3sDExIQPiAUIYD6CoWBs
bDzD1tT4j7e50U9sGCQHUkOMQaL2psZvZrgaPJvjhopBYiA5oEGiBA0CKQAqLEy3N/qw0MfoITIG
iYHkiDIEpAgYPqwWJsbXpvsaP1gaaHwXhEFskBhIjmiDoK7yirAz+bgq3PQGCIPYQNd4kWQITDFQ
47a+QNO7IGwCZJNlCNRVGp7WJh9AGGioBtkGQQ3rBxrST5EhUIMEgAYJUGwQOQYAAONtWMaau7wh
AAAAAElFTkSuQmCC
"
    } else {
	tablelist_newWave_collapsedImg put "
R0lGODlhEgAOAPcAAAAAAFJSUl5eXl9fX2JiYmZmZmdnZ2lpaWtra2xsbG5ubnBwcHJycnR0dH19
fX9/f5KSkpSUlJqampycnJ2dnaGhoaenp6ysrLm5ucvLy8zMzN3d3ejo6AAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAASAA4A
AAhNAP8JHEiwoMGDCAdGSIgwgACGBgVAYIAA4kACFixQUPDAogEMIDFUKCCBIYIMKDNcOFAy4YIN
GzQ46AixAYcJCiwKHJBAp0KfQIMWDAgAOw==
"
	tablelist_newWave_expandedImg put "
R0lGODlhEgAOAPcAAAAAAFJSUltbW19fX2xsbHBwcHh4eHl5eX9/f5KSkpSUlJWVlZqamqenp6ys
rLm5ubq6usrKygAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAASAA4A
AAhKAP8JHEiwoMGDCBMqXHhQQQABECMGUJBwQIIFGBckGLDwQIOPDQ4wJODgwQMHBBj+MxAhggGV
/xBAgIAA5r8CBWz+Y8BAp0+YAQEAOw==
"
	tablelist_newWave_collapsedActImg put "
R0lGODlhEgAOAPcAAAAAAEA3NUI6N1A9OFE/OVNAOlNCO1BCPFFDPVRCO1VEPF5EO2JJP29IPU1E
QlBIRFdTU1dVU3lWR21tbZVdSJ9dSJdgS6RbRq1iSqpsUrFoTrRuUrlzVrh4WYWFhYyMjAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAASAA4A
AAhNAP8JHEiwoMGDCAd6SIgwgAOGBgdcaAAB4kACGDBUWDDBYgENIDVQOPCBoYENKDdYQFAyYQIO
HDIw6AhRQQcJESwKFPBAp0KfQIMWDAgAOw==
"
	tablelist_newWave_expandedActImg put "
R0lGODlhEgAOAPcAAAAAAD83NU87NlNGP2FIP29KPk1EQldUU21tbZ1oT6JXRKVYRKNiS6xhSrFo
TrdwU4WFhYyMjAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAASAA4A
AAhKAP8JHEiwoMGDCBMqXHgQQgABECMGgJDQgIIFGBcoMLCwQIOPDQowPMDAgQMGBxj+I/DgAQGV
/xAkSIAA5r8BA2z+ixBBp0+YAQEAOw==
"
    }
}

#------------------------------------------------------------------------------
# tablelist::oxygen1TreeImgs
#------------------------------------------------------------------------------
proc tablelist::oxygen1TreeImgs {} {
    foreach mode {collapsed expanded} {
	variable oxygen1_${mode}Img \
		 [image create photo tablelist_oxygen1_${mode}Img]
    }

    tablelist_oxygen1_collapsedImg put "
R0lGODlhEAAOAPcAAAAAABQTEgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgjAP8JHEiwoMGDCBMqXMjQYICGAh8yDECRYkOJFyFq3MixYUAAOw==
"
    tablelist_oxygen1_expandedImg put "
R0lGODlhEAAOAPcAAAAAABQTEgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgeAP8JHEiwoMGDCBMqXMiwocOEASJGfEixosWLGAMCADs=
"
}

#------------------------------------------------------------------------------
# tablelist::oxygen2TreeImgs
#------------------------------------------------------------------------------
proc tablelist::oxygen2TreeImgs {} {
    foreach mode {collapsed expanded collapsedAct expandedAct} {
	variable oxygen2_${mode}Img \
		 [image create photo tablelist_oxygen2_${mode}Img]
    }

    variable pngSupported
    if {$pngSupported} {
	tablelist_oxygen2_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAABAAAAAOCAYAAAAmL5yKAAAABGdBTUEAALGPC/xhBQAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuODc7gF0AAABOSURBVDhPY2AYBegh4AIU8KMkWDyBmh8C
cTUlhpgANV8B4gVAzE2uQdZAjc+BuJQcA6yAmq4B8Swg5iDVAG+ghkfk2gyyzBGIQQE5kgAA2mMJ
YLECIngAAAAASUVORK5CYII=
"
	tablelist_oxygen2_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAABAAAAAOCAYAAAAmL5yKAAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuODc7gF0AAABRSURBVDhPY2AYBVQNAUegad5oJnoC+S7E
2gJS/AiIraAaTID0QyD2I9YAkLpSIL4GxNZAfAWIq0nRDFM7C8h4DsQLyNEM0sMBdQk3uQaMNH0A
PvwJYFasmD4AAAAASUVORK5CYII=
"
	tablelist_oxygen2_collapsedActImg put "
iVBORw0KGgoAAAANSUhEUgAAABAAAAAOCAYAAAAmL5yKAAAABGdBTUEAALGPC/xhBQAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuODc7gF0AAABbSURBVDhPY2AYBSghkHDhvwsQ+5EdLEDN
nkD8EIirKTHEBGjAFSBeAMTcZBkE1GgNxM+BuJRkA4CarID4GhDPAmIOkgwAavAG4kdk2QyyCajR
ERSQJNk69BUDAI0BNzsiLc7tAAAAAElFTkSuQmCC
"
	tablelist_oxygen2_expandedActImg put "
iVBORw0KGgoAAAANSUhEUgAAABAAAAAOCAYAAAAmL5yKAAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuODc7gF0AAABcSURBVDhPY2AYBdQLgYQL/x2B2BvZRCDf
E4hdiLIFqvgRkLYCaQDSJkD8EIj9iDIAqqkUqOEaEFsD8RUgriZaM0whUNMsIH4OxAtI1gx1BQdQ
M8gl3GQZMAI1AQCLtzc7P7D1iQAAAABJRU5ErkJggg==
"
    } else {
	tablelist_oxygen2_collapsedImg put "
R0lGODlhEAAOAPcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgoAP8JHEiwoMGDCBMqXPgPAACGAh0+hCgRYkOHDCsu1LhxosWPIBkGBAA7
"
	tablelist_oxygen2_expandedImg put "
R0lGODlhEAAOAPcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgoAP8JHEiwoMGDCBMqXMiQIAAAAx8mfAiRokKKEhdabAixocePID8GBAA7
"
	tablelist_oxygen2_collapsedActImg put "
R0lGODlhEAAOAPcAAAAAAGDQ/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgoAP8JHEiwoMGDCBMqXPgvQACGAh0+hCgRYkOHDCsu1LhxosWPIBkGBAA7
"
	tablelist_oxygen2_expandedActImg put "
R0lGODlhEAAOAPcAAAAAAGDQ/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgoAP8JHEiwoMGDCBMqXMiQYIAAAx8mfAiRokKKEhdabAixocePID8GBAA7
"
    }
}

#------------------------------------------------------------------------------
# tablelist::phaseTreeImgs
#------------------------------------------------------------------------------
proc tablelist::phaseTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable phase_${mode}Img \
		 [image create photo tablelist_phase_${mode}Img]
    }

    tablelist_phase_collapsedImg put "
R0lGODlhEAAOAPcAAAAAAMfHxwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgsAP8JHEiwoMGDCBMaDKCwYACGDQU+hNhwIsWEFi8etBhxYsR/Dz+CFEmSZEAAOw==
"
    tablelist_phase_expandedImg put "
R0lGODlhEAAOAPcAAAAAAMfHxwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAgkAP8JHEiwoMGDCBMqXMhwYICHECEijCgxIUWGFTE23MixY8eAADs=
"
}

#------------------------------------------------------------------------------
# tablelist::plastikTreeImgs
#------------------------------------------------------------------------------
proc tablelist::plastikTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable plastik_${mode}Img \
		 [image create photo tablelist_plastik_${mode}Img]
    }

    tablelist_plastik_collapsedImg put "
R0lGODlhDwAOAPcAAAAAAHZ2drW1tf///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAhEAP8JHEiwoMGDCBP+ExCgYUMBBxkOmDgxAMSCASgCoBjAYMaJGyt6HACgZMkBHTFq5DgSJMuC
EimivAjT4UOFOHMSDAgAOw==
"
    tablelist_plastik_expandedImg put "
R0lGODlhDwAOAPcAAAAAAHZ2drW1tf///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAhCAP8JHEiwoMGDCBP+ExCgYUMBBxkOmDgxAMSCAShqDGAwo8aKHQcAGDlyAEeMH0GiTHmSoMSN
FwsydGhRoc2bBAMCADs=
"
}

#------------------------------------------------------------------------------
# tablelist::plastiqueTreeImgs
#------------------------------------------------------------------------------
proc tablelist::plastiqueTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable plastique_${mode}Img \
		 [image create photo tablelist_plastique_${mode}Img]
    }

    tablelist_plastique_collapsedImg put "
R0lGODlhEAAOAPcAAAAAAHp4eH59fa+trfHx8fPz8/X19ff39/n5+fv7+/39/QAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAhaAP8JHEiwoMGDCBMOHCCgoUMBAw4KUECxogIBEhNoDKAxAUaDAhCIDCASwceCAg4EWLnywEmC
AgzIDCDTwMuBAgroDKCzwE2BAggIHUrg5z+GDxtGVMi0acGAADs=
"
    tablelist_plastique_expandedImg put "
R0lGODlhEAAOAPcAAAAAAHp4eH59fa+trfHx8fPz8/X19ff39/n5+fv7+/39/QAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAQAA4A
AAhWAP8JHEiwoMGDCBMOHCCgoUMBAw4KUECxogIBEhNo3JgAo0EBCEKKROCxoIADAVKmPFCSoAAD
MGMaaDlQQIGbOAvQFCiAgM+fBHb+Y/iwYUSFSJMWDAgAOw==
"
}

#------------------------------------------------------------------------------
# tablelist::radianceTreeImgs
#------------------------------------------------------------------------------
proc tablelist::radianceTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable radiance_${mode}Img \
		 [image create photo tablelist_radiance_${mode}Img]
    }

    tablelist_radiance_collapsedImg put "
R0lGODlhEwAPAPcAAAAAAEBAQOTe1eTe1uTf1+bh2ejj2ujk3erl3uvn4Ozo4u3p4+7q5O/r5u/s
5+/t6PDt6PPw7PLw7fXz8Pb08ff18vb18/f28vj28fj28vj28/j39Pj39fj49fn49/n5+Pr6+Pv7
+fz8+fz8+vz8+/39/P39/f7+/QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAATAA8A
AAi3AP/9OyCgoEGDBwQqFFgAwwgTJyJGNDGCQoGFAy2QIFGio8cSGy0kVCgABIgQIkaoVCkihEkB
CwV4mPmhZoCaH2Z6gEmSA4cNGzh0CNDhp08OPAUK0KAhgwYKFQJUqHCBqYak/wRQoDAhgNevAbZS
wCpAQgQJaCUESJuWLIQHcOEGiAsXAtkFePMGyIuXAdkECAILHoxAAdYDDQwoXszYgIORCgsoIDCg
suUBBBRcxDjwoGfICgMCADs=
"
    tablelist_radiance_expandedImg put "
R0lGODlhEwAPAPcAAAAAAEBAQOTe1eTe1uTf1+bh2ejj2ujk3erl3uvn4Ozo4u3p4+7q5O/r5u/s
5+/t6PDt6PPw7PLw7fXz8Pb08ff18vb18/f28vj28fj28vj28/j39Pj39fj49fn49/n5+Pr6+Pv7
+fz8+fz8+vz8+/39/P39/f7+/QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAATAA8A
AAiwAP/9OyCgoEGDBwQqFFgAwwgTJyJGNDGCQoGFAy2QIFGio8cSGy0kVCgABIgQIkaoVCkihEkB
CwV4mPmhps0PMz3AJMmBw4YNHDoI9dmTw06BAjRoyKCBQoULFZ4q1XD0nwAKFCYE2Mo1AFYKVQVI
iCChrNmzEsJCeMC2rdsHEMIumEu37lwGYRMg2Mu3LwIFVQ80MEC4sGEDDkYqLKCAwIDHkAcQUHAR
48CDmBUrDAgAOw==
"
}

#------------------------------------------------------------------------------
# tablelist::ubuntuTreeImgs
#------------------------------------------------------------------------------
proc tablelist::ubuntuTreeImgs {} {
    foreach mode {collapsed expanded collapsedSel expandedSel} {
	variable ubuntu_${mode}Img \
		 [image create photo tablelist_ubuntu_${mode}Img]
    }

    variable pngSupported
    if {$pngSupported} {
	tablelist_ubuntu_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwwAADsMBx2+oZAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAeUlE
QVQoU5XRTQ5AMBAF4HcJCXEp4lIssLXFQmLlhuONn+gIVYtv8TovbdNCRII9Lr6xAYjdfGcDMFPm
rrlsABYaqaXUnW1zE/ZySR3pKYWZm3CVVUMD6SnJV1lVNB0iX1l37qkm787nnXMzN2EvB7/Gr3cO
/0E/wQq/q1nj7KhNMQAAAABJRU5ErkJggg==
"
	tablelist_ubuntu_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwwAADsMBx2+oZAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAi0lE
QVQoU43QQQ5AMBAF0LmDFXEN1xDO0YuwwdYWiaSJlQPWn6IpnQrJi/T3N1PIGPObGMaIYYwYxvCT
gob9wwb5faKEBRrBDLXteSMGGMEv8npwHa+cwQo9cJHfWilVBOXrAF9nghZ4fPXY9xc2IOqAJ7jx
bi8Izr/DH5sHe+/AhkSJmEthjBjKDB0QEltZUh1wpAAAAABJRU5ErkJggg==
"
	tablelist_ubuntu_collapsedSelImg put "
iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwwAADsMBx2+oZAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAh0lE
QVQoU43QMQrDMAyF4VykkCFbT9H7ZAm5X+gQKKV7IBcIgUzp2EX9Ba6xFDfY8A16FpJxJSLFsuE/
Pri62vDBG73LIh98sOCFwxZTQM8dM3aYLWmj0slD8IBueaLBafPPhhWXs+YRRZMn6Js7xPu0UaW/
UYcsMgV0WuuyyAc3VxvZME+qL4Ax7PyHSW8ZAAAAAElFTkSuQmCC
"
	tablelist_ubuntu_expandedSelImg put "
iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAABGdBTUEAALGPC/xhBQAAAAlwSFlz
AAAOwwAADsMBx2+oZAAAABp0RVh0U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMTAw9HKhAAAAhUlE
QVQoU43OMQqAMBAEwDzB1kKw8GMi+DfBQgQ/YC9YiGDvP85dMeGCF7EYw21Osk5EfjPDFDNMMcMU
fkrY4fxwQOX/amCB3sC8hajGBDN0Cmfm945eLmCDEbjIk/VY87VMNfg6PFkv3OtFbwC+EJ73ouGR
wwqsFd1Fg5IZWXLZZIY2cRe5CulmXcWbqAAAAABJRU5ErkJggg==
"
    } else {
	tablelist_ubuntu_collapsedImg put "
R0lGODlhCwAOAPcAAAAAAExMTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAALAA4A
AAgpAP8JHEiwoMGDCAMgFBigYcKGCg1ChFhwIsWBFiNivFjR4UGNC0MeDAgAOw==
"
	tablelist_ubuntu_expandedImg put "
R0lGODlhCwAOAPcAAAAAAExMTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAALAA4A
AAgkAP8JHEiwoMGDCBMGWMiQ4cCGDh9CDFAQ4sGIBjFmTMixI8KAADs=
"
	tablelist_ubuntu_collapsedSelImg put "
R0lGODlhCwAOAPcAAAAAAP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAALAA4A
AAgpAP8JHEiwoMGDCAMgFBigYcKGCg1ChFhwIsWBFiNivFjR4UGNC0MeDAgAOw==
"
	tablelist_ubuntu_expandedSelImg put "
R0lGODlhCwAOAPcAAAAAAP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAALAA4A
AAgkAP8JHEiwoMGDCBMGWMiQ4cCGDh9CDFAQ4sGIBjFmTMixI8KAADs=
"
    }
}

#------------------------------------------------------------------------------
# tablelist::vistaAeroTreeImgs
#------------------------------------------------------------------------------
proc tablelist::vistaAeroTreeImgs {{treeStyle "vistaAero"}} {
    foreach mode {collapsed expanded collapsedAct expandedAct} {
	variable ${treeStyle}_${mode}Img \
		 [image create photo tablelist_${treeStyle}_${mode}Img]
    }

    variable pngSupported
    if {$pngSupported} {
	tablelist_${treeStyle}_collapsedImg put "
iVBORw0KGgoAAAANSUhEUgAAAAoAAAAOCAYAAAAWo42rAAAABGdBTUEAALGPC/xhBQAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMtfuaUMAAABXSURBVChTY2AYcGAOdIEAMa6IAiqKAGIF
QoqjXr169R+oKAGIlfEpjvrz58//ly9fghRnArEILsVghSD87NkzkOJqIBbFpph4E4l2I7G+Jjoc
CQUf8fIA2yAo8LH45OMAAAAASUVORK5CYII=
"
	tablelist_${treeStyle}_expandedImg put "
iVBORw0KGgoAAAANSUhEUgAAAAoAAAAOCAYAAAAWo42rAAAABGdBTUEAALGPC/xhBQAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuMtfuaUMAAAA4SURBVChTY2AYmsAV6OxyQk4HKboJxTjV
ghVFRkb+x6cQrgifQhRF+BSCHA5zF4wm6BlCnh0IeQCHFRknat3YpwAAAABJRU5ErkJggg==
"
	tablelist_${treeStyle}_collapsedActImg put "
iVBORw0KGgoAAAANSUhEUgAAAAoAAAAOCAYAAAAWo42rAAAABGdBTUEAALGPC/xhBQAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuND6NzHYAAABZSURBVChTY2AYWCBz5Ls5ELsQdAVQUR0Q
1wBxAF7FIIXrnv38D6TXAnEQTsUghWfe/vq/FqK4E4jdsSqGKQQp3vTiF0jxdKyKSTKRaDcS62vi
wpFgQJOiAABE8FkFIoNbxwAAAABJRU5ErkJggg==
"
	tablelist_${treeStyle}_expandedActImg put "
iVBORw0KGgoAAAANSUhEUgAAAAoAAAAOCAYAAAAWo42rAAAABGdBTUEAALGPC/xhBQAAABl0RVh0
U29mdHdhcmUAUGFpbnQuTkVUIHYzLjUuND6NzHYAAAA5SURBVChTY2AYgkDmyHdXIC7H63SooptA
+iZOhTBFTfd//8epEFkRToXoivApLAdZhYbxe2aQRhIAqTxF2Z07dnMAAAAASUVORK5CYII=
"
    } else {
	tablelist_${treeStyle}_collapsedImg put "
R0lGODlhCgAOAPcAAAAAAISEhJaWlp+fn6Wlpaenp+bm5unp6erq6vz8/AAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAKAA4A
AAgtAP8JHEiwoMGDAwkg/EegAEICCAYcJJDggACDFBMYCFCQokWMESc6nLiw5MKAADs=
"
	tablelist_${treeStyle}_expandedImg put "
R0lGODlhCgAOAPcAAAAAACYmJllZWQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAKAA4A
AAgmAP8JHEiwoMGDCBMGQBhgocEAAhwShBixIMWKAxtqlJiwo0ePAQEAOw==
"
	tablelist_${treeStyle}_collapsedActImg put "
R0lGODlhCgAOAPcAAAAAABzE9y3I963m+a/n+rLn+rLo+szt+s/u+gAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAKAA4A
AAgwAP8JHEiwoMGDAwMg/BdAAMIAAxwaDHAg4sQDCAwoJBgAAQGJHAuA5DiS48KTCwMCADs=
"
	tablelist_${treeStyle}_expandedActImg put "
R0lGODlhCgAOAPcAAAAAABzE94Lf+wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAKAA4A
AAgmAP8JHEiwoMGDCBMGQBhgocEAAhwShBixIMWKAxtqlJiwo0ePAQEAOw==
"
    }
}

#------------------------------------------------------------------------------
# tablelist::vistaClassicTreeImgs
#------------------------------------------------------------------------------
proc tablelist::vistaClassicTreeImgs {{treeStyle "vistaClassic"}} {
    foreach mode {collapsed expanded} {
	variable ${treeStyle}_${mode}Img \
		 [image create photo tablelist_${treeStyle}_${mode}Img]
    }

    tablelist_${treeStyle}_collapsedImg put "
R0lGODlhDAAOAPcAAAAAAICAgP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAMAA4A
AAg9AP8JHEiwoMGDCAMoXLhwYAABECMKCOAwIoCIFAU+hHgRYsZ/DwGIFDmxIkeMJgV0LKlRIkqN
DBkinEkzIAA7
"
    tablelist_${treeStyle}_expandedImg put "
R0lGODlhDAAOAPcAAAAAAICAgP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAMAA4A
AAg4AP8JHEiwoMGDCAMoXLhwYAABECMKCOBQYkSKAh9anFgRgEePHDNuDPlPo0WMJUeiZMgSocuX
AQEAOw==
"
}

#------------------------------------------------------------------------------
# tablelist::winnativeTreeImgs
#------------------------------------------------------------------------------
proc tablelist::winnativeTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable winnative_${mode}Img \
		 [image create photo tablelist_winnative_${mode}Img]
    }

    tablelist_winnative_collapsedImg put "
R0lGODlhDwAOAPcAAAAAAICAgP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAg/AP8JHEiwoMGDCBMGWMiQocEAAiJKFBDgoUQAEisWhBgRY0SNBCECGDmSosWOGU8K8Ghy48SU
Gxs2TEizpsGAADs=
"
    tablelist_winnative_expandedImg put "
R0lGODlhDwAOAPcAAAAAAICAgP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAg7AP8JHEiwoMGDCBMGWMiQocEAAiJKFBDg4USJFQtCvEjRIoCPHztq5CiS4MaLGU2STDmwocuE
MGMaDAgAOw==
"
}

#------------------------------------------------------------------------------
# tablelist::win7AeroTreeImgs
#------------------------------------------------------------------------------
proc tablelist::win7AeroTreeImgs {} {
    vistaAeroTreeImgs "win7Aero"
}

#------------------------------------------------------------------------------
# tablelist::win7ClassicTreeImgs
#------------------------------------------------------------------------------
proc tablelist::win7ClassicTreeImgs {} {
    vistaClassicTreeImgs "win7Classic"
}

#------------------------------------------------------------------------------
# tablelist::winxpBlueTreeImgs
#------------------------------------------------------------------------------
proc tablelist::winxpBlueTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable winxpBlue_${mode}Img \
		 [image create photo tablelist_winxpBlue_${mode}Img]
    }

    tablelist_winxpBlue_collapsedImg put "
R0lGODlhDwAOAPcAAAAAAHiYtbDC08C3psG4p8K4qMO6qsa+rs/Iu9LMv9LMwNbRxtjTydvWzNzY
z9/b0uPg2eTh2eXh2urp4+3t5/Hw6/Dw7PLy7vX18ff28/b29Pf39fz8+vz8+////wAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAhjAP8JHEiwoMGDCBMKCMCwYQABBgN4mEjRQ4CIHTIC6MAhw8WCATZs0ABAQwUJHwkGwACgZcsH
KQcGuGCBAoAJEBrEFBggggMGABYoQLDzX4AEBwoQIDDAQNGFDhlCTEi1asGAADs=
"
    tablelist_winxpBlue_expandedImg put "
R0lGODlhDwAOAPcAAAAAAHiYtbDC08C3psG4p8K4qMO6qsa+rs/Iu9LMv9LMwNbRxtfSx9jTydvW
zNzYz9/b0uPg2eTh2eXh2urp4+zr5u3t5/Hw6/Dw7PLy7vX18ff28/b29Pf39fz8+vz8+////wAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAhhAP8JHEiwoMGDCBMKCMCwYQABBgOAmEgRRICIHzJm9LDhYsEAHTpwGHlhgkeCATQAWLkSwsmB
ATJgsFCBQgQHLwUGkPCgAYMFChDk/BcgwYECBAgMMDB0oUOGEBNKnVowIAA7
"
}

#------------------------------------------------------------------------------
# tablelist::winxpOliveTreeImgs
#------------------------------------------------------------------------------
proc tablelist::winxpOliveTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable winxpOlive_${mode}Img \
		 [image create photo tablelist_winxpOlive_${mode}Img]
    }

    tablelist_winxpOlive_collapsedImg put "
R0lGODlhDwAOAPcAAAAAAI6ZfcC3psG4p8K4qMO6qsa+rs/Iu9LMv9LMwNbRxtjTydvWzNzYz9/b
0uPg2eTh2eXh2urp4+3t5/Hw6/Dw7PLy7vX18ff28/b29Pf39fz8+vz8+////wAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAhgAP8JHEiwoMGDCBP+C8CwYYCDATpInNjhYcEAHDIC4LABg0WCATRoyAAgA4UIHwcGuACgZUsH
KQUGsFBhAgAJDxjEXAihwQIAChIc2BkAgQECAwYIKLBzoUOGCqNKJRgQADs=
"
    tablelist_winxpOlive_expandedImg put "
R0lGODlhDwAOAPcAAAAAAI6ZfcC3psG4p8K4qMO6qsa+rs/Iu9LMv9LMwNbRxtfSx9jTydvWzNzY
z9/b0uPg2eTh2eXh2urp4+zr5u3t5/Hw6/Dw7PLy7vX18ff28/b29Pf39fz8+vz8+////wAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAheAP8JHEiwoMGDCBP+C8CwYYCDAT5InPjhYcEAHjJm7KDBIsEAHDhsGGlBgseBATIAWLnywUmB
ATBcqEBhAoQGLxdGcMBggYIEB3IGQGCAwIABAgrkXOiQocKnUAkGBAA7
"
}

#------------------------------------------------------------------------------
# tablelist::winxpSilverTreeImgs
#------------------------------------------------------------------------------
proc tablelist::winxpSilverTreeImgs {} {
    foreach mode {collapsed expanded} {
	variable winxpSilver_${mode}Img \
		 [image create photo tablelist_winxpSilver_${mode}Img]
    }

    tablelist_winxpSilver_collapsedImg put "
R0lGODlhDwAOAPcAAAAAAJSVosTO2MXP2cbO2svT3NPZ4tXb5Nnf5trg593i6d/l6uDm6+bq7ufr
7+zv8+/y9PLz9vT39/b3+ff4+vn6+v39/f///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAheAP8JHEiwoMGDCBP+C8CwYYCDAS5InHjhYcEAFjICyEjBIsEAFSpQAEAhggOPAwNMAMCSJQOU
AgNIiAABwIMGCmAubLAgAQAEBwzoDHCgwICjAgjoXOiQocKnUAkGBAA7
"
    tablelist_winxpSilver_expandedImg put "
R0lGODlhDwAOAPcAAAAAAJSVosTO2MXP2cbO2svT3NPZ4tXb5Nnf5trg593i6d/l6uDm6+bq7ufr
7+zv8+/x8+/y9PLz9vT39/b3+ff4+vn6+v39/f///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAP8ALAAAAAAPAA4A
AAhcAP8JHEiwoMGDCBP+C8CwYYCDATBInIjhYcEAFzJqrGCRYAALFiqIlOCg48AAFACoVMnApMAA
EyREgPCggQKXCxssSJAAwQEDOAMcKDCgqAACOBc6ZKiwqVOCAQEAOw==
"
}

#------------------------------------------------------------------------------
# tablelist::createTreeImgs
#------------------------------------------------------------------------------
proc tablelist::createTreeImgs {treeStyle depth} {
    set baseWidth  [image width  tablelist_${treeStyle}_collapsedImg]
    set baseHeight [image height tablelist_${treeStyle}_collapsedImg]

    #
    # Get the width of the images to create for the specified depth and
    # the destination x coordinate for copying the base images into them
    #
    set width [expr {$depth * $baseWidth}]
    set x [expr {($depth - 1) * $baseWidth}]
    if {[regexp \
	 {^(baghira|klearlooks|oxygen.?|phase|plasti.+|winnative|winxp.+)$} \
	 $treeStyle]} {
	set delta [expr {($depth - 1) * 4}]
	incr width $delta
	incr x $delta
    } elseif {[regexp {^(mate|ubuntu|vistaClassic|win7Classic)$} $treeStyle]} {
	set delta [expr {($depth - 1) * 2}]
	incr width -$delta
	incr x -$delta
    }

    foreach mode {indented collapsed expanded} {
	image create photo tablelist_${treeStyle}_${mode}Img$depth \
	    -width $width -height $baseHeight
    }

    foreach mode {collapsed expanded} {
	tablelist_${treeStyle}_${mode}Img$depth copy \
	    tablelist_${treeStyle}_${mode}Img -to $x 0

	foreach modif {Sel Act SelAct} {
	    variable ${treeStyle}_${mode}${modif}Img
	    if {[info exists ${treeStyle}_${mode}${modif}Img]} {
		image create photo \
		    tablelist_${treeStyle}_${mode}${modif}Img$depth \
		    -width $width -height $baseHeight
		tablelist_${treeStyle}_${mode}${modif}Img$depth copy \
		    tablelist_${treeStyle}_${mode}${modif}Img -to $x 0
	    }
	}
    }
}
