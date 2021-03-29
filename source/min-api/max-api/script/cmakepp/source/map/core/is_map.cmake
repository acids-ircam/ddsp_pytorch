## `(<any>...)-><bool>`
##
## returns true iff the specified value is a map
## note to self: cannot make this a macro because string will be evaluated
function(is_map)
	get_property(is_map GLOBAL PROPERTY "${ARGN}.__keys__" SET)
	set(__ans "${is_map}" PARENT_SCOPE)
endfunction()