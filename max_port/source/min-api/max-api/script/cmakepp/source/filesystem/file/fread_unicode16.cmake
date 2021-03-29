

  ## this is a hard hack to read unicode 16 files
  ##  it reads the file by lines and concatenates the result which removes all linebreaks  
  ## please don't use this :)
  function(fread_unicode16 path)
    path("${path}")
    ans(path)
    file(STRINGS "${path}" lines)  
    string(REPLACE ";" "" res "${lines}")
   # string(CONCAT res ${lines})
    return_ref(res)
  endfunction()
