
# returns all known target names
macro(target_list)
  map_tryget(global target_names)
endmacro()
