
from vmdcallbacks import *
import mouse

def hotkey(s):
  if s == 'r':
    mouse.mode(mouse.ROTATE)
  elif s == 's':
    mouse.mode(mouse.SCALE)
  elif s == 't':
    mouse.mode(mouse.TRANSLATE)
  elif s == 'c':
    mouse.mode(mouse.PICK, 1)
  elif s == '1':
    mouse.mode(mouse.PICK, 2)
  elif s == '2':
    mouse.mode(mouse.PICK, 3)
  elif s == '3':
    mouse.mode(mouse.PICK, 4)
  elif s == '4':
    mouse.mode(mouse.PICK, 5)
  elif s == '5':
    mouse.mode(mouse.PICK, 6)
  elif s == '6':
    mouse.mode(mouse.PICK, 7)
  elif s == '7':
    mouse.mode(mouse.PICK, 8)
  elif s == '8':
    mouse.mode(mouse.PICK, 9)
  elif s == '9':
    mouse.mode(mouse.PICK, 13)
  elif s == '%':
    mouse.mode(mouse.PICK, 10)  # force atom
  elif s == '^':
    mouse.mode(mouse.PICK, 11)  # force residue
  elif s == '&':
    mouse.mode(mouse.PICK, 12)  # force fragment

add_callback('userkey', hotkey)

