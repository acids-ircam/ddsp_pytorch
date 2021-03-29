function(test)
    set(session_file ".cmakepp-session")

  function(get_session)

    map_get(global "sess")
    return_ans()
  endfunction()

  function(set_session)
    map_set(global "sess" ${ARGN})
  endfunction()

  function(session)
    file_find_anchor("${session_file}")
    ans(anchor)
    if(NOT anchor)
      fwrite("${session_file}" "")
    endif()
    cmake_read("${session_file}")
    ans(sessionData)
    set_session("${sessionData}")
    json_print("${sessionData}")
    return_ref(sessionData)


  endfunction()

  function(session_write)
    get_session()
    ans(sess)
    cmake_write("${session_file}" "${sess}")
  endfunction()



  set_session(asdasdasd)
  session_write()

  




  session()
  ans(session)




  assert(EXISTS "${test_dir}/.cmakepp-session")


endfunction()