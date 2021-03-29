## shorthand for obj_declare_property 
##
macro(property)
  obj_declare_property(${this} ${ARGN})
endmacro()
