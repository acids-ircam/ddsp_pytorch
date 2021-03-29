## `(<string...>) -> <void>`
## 
## writes the specified string(s) to the templates output stream
## fails if not called inside a template
##
function(template_out)
    template_guard()
    ans(ref)
    address_append_string(${ref} "${ARGN}")
    return()
endfunction()

