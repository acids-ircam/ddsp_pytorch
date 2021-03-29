
function(target_set tgt_name key)
	set_property(
		TARGET "${tgt_name}"
		PROPERTY "${key}"
		${ARGN}
		)
	return()
endfunction()