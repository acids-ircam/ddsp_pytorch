# returns the var called ${ref}
# this inderection is needed when returning escaped string, else macro will evaluate the string
macro(return_ref __return_ref_ref)
  set(__ans "${${__return_ref_ref}}" PARENT_SCOPE)
  _return()
endmacro()

