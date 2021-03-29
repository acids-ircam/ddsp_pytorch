function(test)

  timer_start(t1)
  cmake_environment(--update-cache)
  ans(env)
  timer_print_elapsed(t1)

  assertf({env.host_name} ISNOTNULL)
  assertf({env.architecture} ISNOTNULL)
  assertf({env.os.name} ISNOTNULL)
  assertf({env.os.version} ISNOTNULL)
  assertf({env.os.family} ISNOTNULL)



endfunction()