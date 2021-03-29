  ## returns the key at the specified position
  function(map_key_at map idx)
    map_keys(${map})
    ans(keys)
    list_normalize_index(keys ${idx})
    ans(idx)
    list_get(keys ${idx})
    ans(key)
    return_ref(key)
  endfunction()

