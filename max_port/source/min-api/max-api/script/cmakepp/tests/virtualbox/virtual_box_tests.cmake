function(test)


  function(search_paths)
    set(args ${ARGN})

    set(paths ${args})

    paths($ENV{PATH})
    ans_append(paths)

    paths($ENV{INCLUDE})
    ans_append(paths)

    return_ref(paths)
  endfunction()

  function(directories)
    set(result)
    foreach(path ${ARGN})
      path_qualify(path)
      if(IS_DIRECTORY "${path}")
        list(APPEND result ${path})
      endif()
    endforeach()
    return_ref(result)
  endfunction()

  function(find globs)
    set(args ${ARGN})

    list_extract_flag(args --directory)
    ans(only_directories)

    search_paths(${args})
    ans(paths)

    set(found)

    pushd()
    foreach(path ${paths})
      if(EXISTS "${path}")
        cd("${path}")
        glob(${globs})
        ans(found)
        if(found)
          break()
        endif()
      endif()
    endforeach()
    popd()



    list_remove_duplicates(found)
      
    if(only_directories)
      directories(${found})
      ans(found)
    endif()

    return_ref(found)
  endfunction()
  timer_start(t1)
  find_file(riraresult VBoxManage.exe)
  timer_print_elapsed(t1)
  set(result asd)

  function(beh)

  endfunction()

  message("${riraresult}")
  timer_start(t1)
  find(VBoxManage.exe)
  ans(res)
  timer_print_elapsed(t1)
  message("${res}")


  return()
  wrap_executable(vbox "VBoxManage.exe")


  vbox()
  ans(res)

  message("${res}")


endfunction()