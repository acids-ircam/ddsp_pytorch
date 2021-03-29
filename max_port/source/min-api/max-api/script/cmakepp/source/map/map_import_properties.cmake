## imports the specified properties into the current scope
## e.g map = {a:1,b:2,c:3}
## map_import_properties(${map} a c)
## -> ${a} == 1 ${b} == 2
macro(map_import_properties __map)
  foreach(key ${ARGN})
    map_tryget("${__map}" "${key}")
    ans("${key}")
  endforeach()
endmacro()
