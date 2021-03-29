

# process_fork implementation specific to linux
# uses bash and nohup to start a process 
function(process_start_Linux process_handle)
  process_handle_register(${process_handle})
  map_tryget(${process_handle} start_info)

  ans(process_start_info)

  map_tryget(${process_start_info} command)
  ans(command)

  map_tryget(${process_start_info} command_arguments)
  ans(command_arguments)

  map_tryget(${process_start_info} working_directory)
  ans(working_directory)

  command_line_args_combine(${command_arguments})
  ans(command_arguments_string)

  set(command_string "${command} ${command_arguments_string}")

  # define output files        
  fwrite_temp("")
  ans(stdout)
  fwrite_temp("")
  ans(stderr)
  fwrite_temp("")
  ans(return_code)
  fwrite_temp("")
  ans(pid_out)

  process_handle_change_state(${process_handle} starting)
  # create a temporary shell script 
  # which starts bash with the specified command 
  # output of the command is stored in stdout file 
  # error of the command is stored in stderr file 
  # return_code is stored in return_code file 
  # and the created process id is stored in pid_out
  shell_tmp_script("( bash -c \"(cd ${working_directory}; ${command_string} > ${stdout} 2> ${stderr})\" ; echo $? > ${return_code}) & echo $! > ${pid_out}")
  ans(script)
  ## execute the script in bash with nohup 
  ## which causes the script to run detached from process
  bash_lean(-c "nohup ${script} > /dev/null 2> /dev/null")
  ans(error)

  if(error)
    message(FATAL_ERROR "could not start process '${command_string}'")
  endif()



  fread("${pid_out}")
  ans(pid)

  string(STRIP "${pid}" pid)

  map_set(${process_handle} pid "${pid}")

  process_handle_change_state(${process_handle} running)


  ## set output of process
  map_set(${process_handle} stdout_file ${stdout})
  map_set(${process_handle} stderr_file ${stderr})
  map_set(${process_handle} return_code_file ${return_code})
  

  process_refresh_handle("${process_handle}")

  return_ref(process_handle)
endfunction()
