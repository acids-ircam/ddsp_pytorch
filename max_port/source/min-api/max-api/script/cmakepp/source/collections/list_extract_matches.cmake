## `(<&> <regex>...)-><any...>`
##
## removes all matches from the list and returns them
## sideffect: matches are removed from list
function(list_extract_matches __list_extract_matches_lst)
  list_regex_match(${__list_extract_matches_lst} ${ARGN})
  ans(matches)
  list_remove(${__list_extract_matches_lst} ${matches})
  #print_vars(matches __list_extract_matches_lst ${__list_extract_matches_lst})
  set(${__list_extract_matches_lst} ${${__list_extract_matches_lst}} PARENT_SCOPE)
  return_ref(matches)
endfunction()