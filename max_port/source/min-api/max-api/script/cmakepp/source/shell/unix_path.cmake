
# fully qualifies the path into a unix path (even windows paths)
# transforms C:/... to /C/...
function(unix_path path)
  path("${path}")
  ans(path)
  string(REGEX REPLACE "^_([a-zA-Z]):\\/" "/\\1/" path "_${path}")
  return_ref(path)
endfunction()