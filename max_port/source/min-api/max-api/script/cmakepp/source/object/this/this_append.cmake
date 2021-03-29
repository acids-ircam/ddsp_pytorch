# appends the value(s) to the specified member variable
function(this_append member_name)
  obj_get("${this}" "${member_name}")
  ans(value)
  obj_set("${this}" "${member_name}" ${value} "${ARGN}")
endfunction()