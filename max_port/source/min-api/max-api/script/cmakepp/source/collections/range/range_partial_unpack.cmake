##
function(range_partial_unpack ref)
    if(NOT ${ref})
      set(${ref} ${ARGN})
    endif()
    set(partial ${${ref}})

    string(REPLACE ":" ";" parts ${partial})
    list(GET parts 0 begin)
    list(GET parts 1 end)
    list(GET parts 2 increment)
    list(GET parts 3 inclusive_begin)
    list(GET parts 4 inclusive_end)
    list(GET parts 5 length)

    set(${ref}.inclusive_begin ${inclusive_begin} PARENT_SCOPE)
    set(${ref}.inclusive_end ${inclusive_end} PARENT_SCOPE)    
    set(${ref}.begin ${begin} PARENT_SCOPE)
    set(${ref}.end ${end} PARENT_SCOPE)
    set(${ref}.increment ${increment} PARENT_SCOPE)
    set(${ref}.length  ${length} PARENT_SCOPE)
endfunction()

