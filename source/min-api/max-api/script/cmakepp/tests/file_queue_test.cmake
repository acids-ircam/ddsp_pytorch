function(test)


  return()
  function(file_queue_new)
    set(args ${ARGN})

  endfunction()

  function(file_queue_push)


  endfunction()

  function(file_queue_pop)
    set(queue ${ARGN})
    file_lock("${queue}")
    ans(lock)


    file_unlock("${lock}")
  endfunction()


  function(pid)
    string_random()
    ans(pid)
    eval("
      macro(pid)
        set(__ans ${pid})
      endmacro()
      ")
    pid()
    return_ans()
  endfunction()

  function(file_lock)
    pid()
    ans(pid)
    set(path ${ARGN})
    path_qualify(path)
    while(true)
      if(NOT EXISTS "${path}.lock")
        file(WRITE "${path}" "${pid}")
        break()
      endif()
    endwhile()
    return_ref(path)
  endfunction()

  function(file_unlock)
    set(path ${ARGN})
    pid()
    ans(pid)

  endfunction()




endfunction()