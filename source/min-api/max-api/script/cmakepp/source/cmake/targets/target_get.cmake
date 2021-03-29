

function(target_get tgt_name key)
	get_property(
		val
		TARGET "${tgt_name}"
		PROPERTY "${key}"
		)
	return_ref(val)
endfunction()
