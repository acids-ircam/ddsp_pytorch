# removes the specifed range from the list
# and returns remaining elements
function(list_without_range __list_without_range_lst start_index end_index)

  list_normalize_index(${__list_without_range_lst} -1)
  ans(list_end)

  list_slice(${__list_without_range_lst} 0 ${start_index})
  ans(part1)
  list_slice(${__list_without_range_lst} ${end_index} ${list_end})
  ans(part2)

  set(res ${part1} ${part2})
  return_ref(res)
endfunction()