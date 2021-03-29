
## optimized version
macro(map_new)
  address_new()
  set_property(GLOBAL PROPERTY "${__ans}.__keys__" "") ## set keys (duck typing for map is that it has property keys)  
endmacro()