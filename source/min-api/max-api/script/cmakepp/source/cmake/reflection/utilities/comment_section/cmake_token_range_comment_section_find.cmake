## `(<cmake token range> <section_name:<string>>)-><cmake token range>`
##
## finds the correct comment section or returns nothing 
function(cmake_token_range_comment_section_find range regex_section_name)

  set(regex_section_begin_specific "^[# ]*<section[ ]+name[ ]*=[ ]*\"${regex_section_name}\"[ ]*>[ #]*$")
  set(regex_section_begin_any "^[# ]*<section.*>[ #]*$")
  set(regex_section_end "^[# ]*<\\/[ ]*section[ ]*>[# ]*$")



  list_extract(range current end)


  cmake_token_range_find_next_by_type("${current};${end}" "^line_comment$" "${regex_section_begin_specific}")
  ans(current)

  if(NOT current)
    error("section ${section_name} not found")
    return()
  endif()

  set(section_begin_token ${current})

  cmake_token_advance(current)


  set(section_depth 1)
  set(section_end_token)

  while(current)

    cmake_token_range_find_next_by_type("${current};${end}" "^line_comment$" "(${regex_section_begin_any})|(${regex_section_end})")
    ans(current)

    map_tryget(${current} literal_value)
    ans(literal_value)

    if("${literal_value}" MATCHES "${regex_section_begin_any}")
      math(EXPR section_depth "${section_depth} + 1")
    else()
      math(EXPR section_depth "${section_depth} - 1")
    endif()

    if(NOT section_depth)
      set(section_end_token ${current})
      break()
    endif()

    map_tryget(${current} next)
    ans(current)

  endwhile()

  if(NOT section_end_token)
    error("unbalanced section close")
    return()
  endif()
  
  ## advance twice: comment->newline->begin of section
  cmake_token_advance(section_begin_token)
  cmake_token_advance(section_begin_token)

  return(${section_begin_token} ${section_end_token})
endfunction()