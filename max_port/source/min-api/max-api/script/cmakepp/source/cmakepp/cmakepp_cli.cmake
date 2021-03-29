
function(cmakepp_cli)
  set(args ${ARGN})

  if(NOT args)
    ## get command line args and remove executable -P and script file
    commandline_args_get(--no-script)
    ans(args)
  endif()


  list_extract_flag(args --timer)
  ans(timer)
  list_extract_flag(args --silent)
  ans(silent)
  list_extract_labelled_value(args --select)
  ans(select)

  ## get format
  list_extract_flag(args --json)
  ans(json)
  list_extract_flag(args --qm)
  ans(qm)
  list_extract_flag(args --table)
  ans(table)
  list_extract_flag(args --csv)
  ans(csv)
  list_extract_flag(args --xml)
  ans(xml)
  list_extract_flag(args --plain)
  ans(plain)
  list_extract_flag(args --ini)
  ans(ini)

  set(lazy_cmake_code)
  foreach(arg ${args})
    cmake_string_escape("${arg}")
    set(lazy_cmake_code "${lazy_cmake_code} ${__ans}")
  endforeach()

  #string_combine(" " ${args})
  #ans(lazy_cmake_code)

  lazy_cmake("${lazy_cmake_code}")
  ans(cmake_code)

  ## execute code
  set_ans("")
  if(timer)
    timer_start(timer)
  endif()
  eval("${cmake_code}")
  ans(result)

  if(timer)
    timer_print_elapsed(timer)
  endif()

  if(select)
    string(REGEX REPLACE "@([^ ]*)" "{result.\\1}" select "${select}")
    format("${select}")
    ans(result)
   # assign(result = "result${select}")
  endif()


  ## serialize code
   if(json)
    json_indented("${result}")
    ans(result)
  elseif(ini)
    ini_serialize("${result}")
    ans(result)
   elseif(qm)
    qm_serialize("${result}")
    ans(result)
   elseif(table)
      table_serialize("${result}")
      ans(result)
    elseif(csv)
      csv_serialize("${result}")
      ans(result)
    elseif(xml)
      xml_serialize("${result}")
      ans(result)
    elseif(plain)

    else()
      json_indented("${result}")
      ans(result)
   endif()



  ## print code
  if(NOT silent)
    echo("${result}")
  endif()
  return_ref(result)
endfunction()


