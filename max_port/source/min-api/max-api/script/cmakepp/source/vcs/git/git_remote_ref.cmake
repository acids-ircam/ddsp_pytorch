
# checks the remote uri if a ref exists ref_type can be * to match any
# else it can be tags heads or HEAD
# returns the corresponding ref object
function(git_remote_ref uri ref_name ref_type)
  git_remote_refs( "${uri}")
  ans(refs)
  foreach(current_ref ${refs})
    map_navigate(name "current_ref.name")
    if("${name}" STREQUAL "${ref_name}")
      if(ref_type STREQUAL "*")
        return(${current_ref})
      else()
        map_navigate(type "current_ref.type")
        if(${type} STREQUAL "${ref_type}")
          return("${current_ref}")
        endif()
        return()
      endif()
    endif()
  endforeach()
  return()
endfunction()


