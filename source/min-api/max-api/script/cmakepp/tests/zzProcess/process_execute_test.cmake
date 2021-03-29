function(test)
  function(uut)
    arguments_encoded_list(0 ${ARGC})
    ans(args)

    process_start_info_new(${args})
    ans(startinfo)

    process_handle_new(${startinfo})
    ans(handle)
    
    process_execute(${handle})
    return_ans()
  endfunction()

  uut("unkowncommand")
  ans(res)
  assert(res)
  assertf({res.state} STREQUAL "terminated")
  assertf({res.pid} ISNULL)


  uut(cmake -E echo_append "asd;bsd" csd dsd)
  ans(res)
  assert(res)
  assertf({res.exit_code} EQUAL 0)
  assertf({res.stdout} EQUALS "asd;bsd csd dsd")
  assertf({res.state} STREQUAL "terminated")
  assertf({res.pid} STREQUAL "-1")


  uut(cmake -E echo_append abc bcd cde)
  ans(res)
  assertf({res.stdout} STREQUAL "abc bcd cde")
  assertf({res.state} STREQUAL "terminated")
  assertf({res.pid} STREQUAL "-1")


  uut(COMMAND cmake -E echo_append abc bcd cde WORKING_DIRECTORY ../ )
  ans(res)
  assertf({res.stdout} STREQUAL "abc bcd cde")
  assertf({res.state} STREQUAL "terminated")
  assertf({res.pid} STREQUAL "-1")


  

endfunction()