function(test)
  cmakepp_config(base_dir)
  ans(base_dir)

  process_start_script("
    include(${base_dir}/cmakepp.cmake)
    sleep(2)
  ")
  ans(h1)
  process_start_script("
    include(${base_dir}/cmakepp.cmake)
    sleep(3)
  ")
  ans(h2)
  process_start_script("
    include(${base_dir}/cmakepp.cmake)
    sleep(4)
  ")
  ans(h3)

  map_new()
  ans(context)

  process_wait_all(${h1} ${h2} ${h3} 
    --idle-callback "[]()map_set(${context} idlecalled true)"
    --task-complete-callback "[](handle)map_append(${context} complete {{handle}})"
  )
  ans(res)


  assert(${res} CONTAINS ${h1})
  assert(${res} CONTAINS ${h2})
  assert(${res} CONTAINS ${h3})


  assertf({context.complete} CONTAINS ${h1})
  assertf({context.complete} CONTAINS ${h2})
  assertf({context.complete} CONTAINS ${h3})

  assertf({context.idlecalled})



  process_wait_all(${h1} ${h2} ${h3})
  ans(res)
  assert(${res} EQUALS ${h1} ${h2} ${h3})


endfunction()