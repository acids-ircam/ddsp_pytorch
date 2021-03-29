# compares the semver on the left and right
# returns -1 if left is more up to date
# returns 1 if right is more up to date
# returns 0 if they are the same
function(semver_compare  left right)
 semver_parse(${left} )
 ans(left)
 semver_parse(${right})
 ans(right)


  scope_import_map(${left} left_)
  scope_import_map(${right} right_)

 semver_component_compare( ${left_major} ${right_major})
 ans(cmp)
 if(NOT ${cmp} STREQUAL 0)
  return(${cmp})
endif()
 semver_component_compare( ${left_minor} ${right_minor})
 ans(cmp)
 if(NOT ${cmp} STREQUAL 0)
  return(${cmp})
endif()
 
 semver_component_compare( ${left_patch} ${right_patch})
 ans(cmp)
 if(NOT ${cmp} STREQUAL 0)
  return(${cmp})
endif()


 if(right_prerelease AND NOT left_prerelease)
  return(-1)
 endif()

 if(left_prerelease AND NOT right_prerelease)
  return(1)
 endif()
 # iterate through all identifiers of prerelease
 while(true)
    list_pop_front(left_tags)
    ans(left_current)

    list_pop_front(right_tags)
    ans(right_current)

    # check for larger set
    if(right_current AND NOT left_current)
      return(1)
    elseif(left_current AND NOT right_current)
      return(-1)
    elseif(NOT left_current AND NOT right_current)
      # equal
      return(0)
    endif()

      # compare component
   semver_component_compare( ${left_current} ${right_current})
ans(cmp)

   #   message("asd '${left_current}'  '${right_current}' -> ${cmp}")
   if(NOT ${cmp} STREQUAL 0)
    return(${cmp})
   endif()



    
 endwhile()
 
 return(0)

endfunction()