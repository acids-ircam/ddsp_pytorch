macro(this_get member_name)
	obj_get("${this}" "${member_name}")
  ans("${member_name}")
endmacro()