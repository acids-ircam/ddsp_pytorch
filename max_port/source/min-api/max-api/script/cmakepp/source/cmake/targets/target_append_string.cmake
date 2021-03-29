
function(target_append_string tgt_name key)
	set_property(
		TARGET "${tgt_name}"
		APPEND_STRING
		PROPERTY "${key}"
		${ARGN})
	return()
endfunction()
