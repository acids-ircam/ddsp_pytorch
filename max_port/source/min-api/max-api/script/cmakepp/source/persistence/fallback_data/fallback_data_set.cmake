

  function(fallback_data_set dirs id nav)
    list_pop_front(dirs)
    ans(dir)

    file_data_set("${dir}" "${id}" "${nav}" ${ARGN})
    return_ans()
  endfunction()