function(test)
  set(lstA  9 1 8 2 7 3 6 4 5)
  set(lstB 1)
  set(lstC)


  list_sort(lstA "[](lhs rhs)return_math('{{lhs}} - {{rhs}}')")
  ans(res)
  assert(${res} EQUALS 9 8 7 6 5 4 3 2 1)

  list_sort(lstB "[](lhs rhs)return_math('{{lhs}} - {{rhs}}')")
  ans(res)
  assert(${res} EQUALS 1)


  list_sort(lstC "[](lhs rhs)return_math('{{lhs}} - {{rhs}}')")
  ans(res)
  assert(${res} ISNULL)


    foreach(i RANGE 1 100)
      string(RANDOM LENGTH 3 ALPHABET "1234567890" r)
      list(APPEND lstA ${r})
    endforeach()

    timer_start(timer)
    list_sort(lstA "[](lhs rhs)return_math('{{rhs}} - {{lhs}}')")
    ans(res)
    timer_print_elapsed(timer)



endfunction()



