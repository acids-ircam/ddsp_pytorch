## ''
##
## wraps nuget inside an easy to use function, downloading it if it does not exist
function(nuget)
  if(NOT WIN32)
    message(FATAL_ERROR "you currently cannot use nuget on non windows systems")
  endif()

  download_cached("https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" --readonly)
  ans(nuget_exe)

  if(NOT EXISTS "${nuget_exe}")
    message(FATAL_ERROR "nuget.exe could not be located")
  endif()

  wrap_executable(nuget "${nuget_exe}")
  nuget(${ARGN})
  return_ans()
endfunction()  
