
#creates a unique id
function(make_guid)
  string(RANDOM LENGTH 10 id)
   return_ref(id)
endfunction()

## faster
macro(make_guid)
  string(RANDOM LENGTH 10 __ans)
  #set(__ans ${id} PARENT_SCOPE)
endmacro()
