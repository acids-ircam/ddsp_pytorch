
  function(await handle)
    process_wait(${handle})

    map_tryget("${handle}" result_file)
    ans(result_file)
    qm_read("${result_file}")
    return_ans()
  endfunction()