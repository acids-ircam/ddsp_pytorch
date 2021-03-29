## windows implementation for start process
## newer faster version
 function(process_start_Windows process_handle)
    ## create a process handle from pid
    process_handle_register(${process_handle})

    map_tryget(${process_handle} start_info)
    ans(start_info)


    map_tryget(${start_info} command)
    ans(command)

    map_tryget(${start_info} command_arguments)
    ans(command_arguments)

    command_line_args_combine(${command_arguments})
    ans(command_arguments_string)

    set(command_string "\"${command}\" ${command_arguments_string}")

    map_tryget(${start_info} working_directory)
    ans(working_directory)

    ## create temp dir where process specific files are stored
    mktemp()
    ans(dir)
    ## files where to store stdout and stderr
    set(outputfile "${dir}/stdout.txt")
    set(errorfile "${dir}/stderr.txt")
    set(returncodefile "${dir}/retcode.txt")
    set(pidfile "${dir}/pid.txt")

    fwrite("${outputfile}" "")
    fwrite("${errorfile}" "")
    fwrite("${returncodefile}" "")


    ## creates a temporary batch file
    ## which gets the process id (get the parent process id wmic....)
    ## output pid to file output command_string to 
    fwrite_temp("
      @echo off
      cd \"${working_directory}\"
      wmic process get parentprocessid,name|find \"WMIC\" > ${pidfile}
      ${command_string} > ${outputfile} 2> ${errorfile}
      echo %errorlevel% > ${returncodefile}
      exit
    " ".bat")
    ans(path)


    process_handle_change_state(${process_handle} starting)
    win32_powershell_lean("start-process -File ${path} -WindowStyle Hidden")


    ## wait until the pidfile exists and contains a valid pid
    ## this seems very hackisch but is necessary as i have not found
    ## a simpler way to do it
    while(true)
      if(EXISTS "${pidfile}")
        fread("${pidfile}")
        ans(pid)
        if("${pid}" MATCHES "[0-9]+" )
          set(pid "${CMAKE_MATCH_0}")
          break()
        endif()
      endif()
    endwhile()
    map_set(${process_handle} pid "${pid}")
    
    process_handle_change_state(${process_handle} running)

    
    ## set the output files for process_handle
    map_set(${process_handle} stdout_file ${outputfile})
    map_set(${process_handle} stderr_file ${errorfile})
    map_set(${process_handle} return_code_file  ${returncodefile})

    assign(!process_handle.windows.process_data_dir = dir) 

    return_ref(process_handle)
  endfunction()
