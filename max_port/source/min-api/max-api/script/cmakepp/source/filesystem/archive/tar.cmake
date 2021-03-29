# tar command 
# use cvzf to compress files relative to pwd() to a tgz file 
# use xzf to uncompress a tgz file to the pwd()
function(tar)
  cmake(-E tar ${ARGN})
  return_ans()
endfunction()

