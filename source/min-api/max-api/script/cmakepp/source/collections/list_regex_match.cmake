## matches all elements of lst to regex
## all elements in list which match the regex are returned
function(list_regex_match __list_regex_match_lst )
  set(__list_regex_match_result)
  foreach(__list_regex_match_item ${${__list_regex_match_lst}})
    foreach(__list_regex_match_regex ${ARGN})
      if("${__list_regex_match_item}" MATCHES "${__list_regex_match_regex}")
        list(APPEND __list_regex_match_result "${__list_regex_match_item}")
        break() ## break inner loop on first match
      endif()
    endforeach()
  endforeach()
  return_ref(__list_regex_match_result)
endfunction()
