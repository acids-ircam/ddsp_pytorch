function(environment_processor_count)
  # from http://www.cmake.org/pipermail/cmake/2010-October/040122.html
  if(NOT DEFINED processor_count)
    # Unknown:
    set(processor_count 0)

    # Linux:
    set(cpuinfo_file "/proc/cpuinfo")
    if(EXISTS "${cpuinfo_file}")
      file(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
      list(LENGTH procs processor_count)
    endif()

    # Mac:
    if(APPLE)
      find_program(cmd_sys_pro "system_profiler")
      if(cmd_sys_pro)
        execute_process(COMMAND ${cmd_sys_pro} OUTPUT_VARIABLE info)
        string(REGEX REPLACE "^.*Total Number Of Cores: ([0-9]+).*$" "\\1" processor_count "${info}")
      endif()
    endif()

    # Windows:
    if(WIN32)
      set(processor_count "$ENV{NUMBER_OF_PROCESSORS}")
    endif()
  endif()

  eval("
  function(environment_processor_count)
    set(__ans ${processor_count} PARENT_SCOPE)
  endfunction()
  ")
  environment_processor_count()
  return_ans()
endfunction()