
function(alias_exists name)
  alias_list()
  ans(aliases)
  list_contains(aliases "${name}")
  ans(res)
  return_ref(res)
endfunction()