# returns all keys for the specified map
macro(map_keys this)
  get_property(__ans GLOBAL PROPERTY "${this}.__keys__")
  #return_ref(keys)
endmacro()
# returns all keys for the specified map
#function(map_keys this)
#  get_property(keys GLOBAL PROPERTY "${this}")
#  return_ref(keys)
#endfunction()
