
  function(cmake_token_range_comment_section_find_all range regex_section_name)
    cmake_token_range("${range}")
    ans_extract(current end)

    set(sections)
    while(current)
      cmake_token_range_comment_section_find("${current};${end}" "${regex_section_name}")
      ans(section)
      ans_extract(section_begin section_end)

      if(NOT section)
        break()
      endif()
      list(APPEND sections ${section}) 

      map_tryget(${section_end} next)
      ans(current)
    endwhile()  

    return_ref(sections)

  endfunction()
