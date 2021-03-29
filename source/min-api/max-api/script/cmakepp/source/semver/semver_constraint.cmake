

function(semver_constraint constraint_ish)
  map_get_special(${constraint_ish} "semver_constraint")
  ans(is_semver_constraint)
  if(is_semver_constraint)
    return_ref(constraint_ish)
  endif()

  is_map(${constraint_ish})
  ans(ismap)
  if(ismap)
    return()
  endif()

  # return cached value if it exists
 # cache_return_hit("${constraint_ish}")

  # compute and cache value
  semver_constraint_compile("${constraint_ish}")
  ans(constraint)
  # cache_update("${constraint_ish}" "${constraint}" const)

  return_ref(constraint)

endfunction()