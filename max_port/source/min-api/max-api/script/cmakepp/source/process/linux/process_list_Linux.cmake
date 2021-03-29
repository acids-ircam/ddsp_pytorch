# linux specific implementation of process_list 
# returns a list of <process handle> which only contains pid


  function(process_list_Linux)

    linux_ps_lean()
    ans_extract(error)
    ans(res)

   # print_vars(error res)

    string_lines("${res}")
    ans(lines)

    list_pop_front(lines)
    ans(headers)

    set(handles)
    set(ps_regex " *([1-9][0-9]*)[ ]*")
    #set(ps_regex " *([1-9][0-9]*)[ ]*([^ ]+)[ ]*([0-9][0-9]):([0-9][0-9]):([0-9][0-9]) *([^ ].*)")
    foreach(line ${lines})
      string(REGEX REPLACE "${ps_regex}" "\\1" pid "${line}")
      #string(REGEX REPLACE "${ps_regex}" "\\2" tty "${line}")
      #string(REGEX REPLACE "${ps_regex}" "\\3" hh "${line}")
      #string(REGEX REPLACE "${ps_regex}" "\\4" mm "${line}")
      #string(REGEX REPLACE "${ps_regex}" "\\5" ss "${line}")
      #string(REGEX REPLACE "${ps_regex}" "\\6" cmd "${line}")
      #string(STRIP "${cmd}" cmd)

      process_handle("${pid}")
      ans(handle)
      #map_capture(${handle} tty hh mm ss cmd) 
      
      list(APPEND handles ${handle})
    endforeach()
    return_ref(handles)
  endfunction()