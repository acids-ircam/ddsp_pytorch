
function(target_append tgt_name key)
	set_property(
		TARGET "${tgt_name}"
		APPEND
		PROPERTY "${key}"
		${ARGN})
	return()
endfunction()
