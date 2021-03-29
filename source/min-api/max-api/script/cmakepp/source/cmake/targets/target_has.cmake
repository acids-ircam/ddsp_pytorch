
function(target_has tgt_name key)
	get_property(
		val
		TARGET "${tgt_name}"
		PROPERTY "${key}"
		SET)
	return_ref(val)
endfunction()
