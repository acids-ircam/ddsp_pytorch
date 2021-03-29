## returns every element of lst that matches any of the given regexes
## and does not match any regex that starts with !
  function(list_regex_match_ignore lst)
    set(regexes ${ARGN})
    list_regex_match(regexes "^[!]")
    ans(negs)
    set(negatives)
    foreach(negative ${negs})
      string(SUBSTRING "${negative}" 1 -1 negative )
      list(APPEND negatives "${negative}")
    endforeach()

    list_regex_match(regexes "^[^!]")
    ans(positives)


    list_regex_match(${lst} ${positives})
    ans(matches)

    list_regex_match(matches ${negatives})
    ans(ignores)

    list(REMOVE_ITEM matches ${ignores})

    return_ref(matches)

  endfunction()
