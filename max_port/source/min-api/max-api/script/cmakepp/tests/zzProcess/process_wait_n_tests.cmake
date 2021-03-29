function(test)



  cmakepp_config(base_dir)
  ans(base_dir)

  process_start_script("
    include(${base_dir}/cmakepp.cmake)
    sleep(0)
  ")
  ans(h4)
  process_start_script("
    include(${base_dir}/cmakepp.cmake)
    sleep(0)
  ")
  ans(h5)

  process_start_script("
    include(${base_dir}/cmakepp.cmake)
    sleep(6)
  ")
  ans(h1)
  process_start_script("
    include(${base_dir}/cmakepp.cmake)
    sleep(10)
  ")
  ans(h2)
  process_start_script("
    include(${base_dir}/cmakepp.cmake)
    sleep(15)
  ")
  ans(h3)


  process_wait_n(4 ${h1} ${h2} ${h5} ${h3} ${h4} --idle-callback "[]() status_line('running: {{running_processes}} -- terminated: {{terminated_processes}}  ')")
  ans(processes)
  assert(${processes} CONTAINS ${h1})
  assert(${processes} CONTAINS ${h4})
  assert(${processes} CONTAINS ${h5})
  assert(${processes} CONTAINS ${h2})




endfunction()