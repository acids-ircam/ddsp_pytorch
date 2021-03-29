



#C:\ProgramData\Oracle\Java\javapath;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files (x86)\ATI Technologies\ATI.ACE\Core-Static;C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\;C:\Program Files\Microsoft SQL Server\110\Tools\Binn\;C:\Program Files (x86)\Git\cmd;C:\Program Files\Mercurial\;C:\Program Files\nodejs\;C:\Program Files (x86)\Microsoft SDKs\TypeScript\1.0\;C:\Program Files\Microsoft SQL Server\120\Tools\Binn\
#C:\ProgramData\chocolatey\bin;C:\Program Files\Mercurial;C:\Users\Tobi\AppData\Roaming\npm


# creates the bash string using the map env which contains key value pairs
function(bash_profile_compile env)
  set(res)
  map_keys(${env})
  ans(keys)
  foreach(key ${keys})
    map_tryget(${env} ${key})
    ans(val)
    set(res "${res}export ${key}=\"${val}\"\n")
  endforeach()
  return_ref(res)
endfunction()

# creates and writes the bash profile env to path (see bash_profile_compile)
function(bash_profile_write path env)
  bash_profile_compile(${env})
  ans(str)
  bash_script_create("${path}" "${str}")
  return_ans()
endfunction()

function(bash_autostart_read)
  set(session_profile_path "$ENV{HOME}/.profile")
  if(NOT EXISTS "${session_profile_path}")
    return()
  endif()
  fread("${session_profile_path}")
  ans(res)
  return_ref(res)
endfunction()

# registers
function(bash_autostart_register)
  set(session_profile_path "$ENV{HOME}/.profile")
  if(NOT EXISTS "${session_profile_path}")
    touch("${session_profile_path}")
  endif()
  fread("${session_profile_path}")
  ans(profile)

  set(profile_path "$ENV{HOME}/cmakepp.profile.sh")

  if(NOT EXISTS "${profile_path}")
    shell_script_create("${profile_path}" "")
  endif()

  if("${profile}" MATCHES "${profile_path}\n")
    return()
  endif()

  unix_path("${profile_path}")
  ans(profile_path)
  set(profile "${profile}\n${profile_path}\n")
  fwrite("${session_profile_path}" "${profile}")

  return()
endfunction()

# removes the cmake profile from $ENV{HOME}/.profile
function(bash_autostart_unregister)
  set(session_profile_path "$ENV{HOME}/.profile")
  if(NOT EXISTS "${session_profile_path}")
    return()
  endif()
  fread("${session_profile_path}")
  ans(content)
  string_regex_escape("${session_profile_path}")
  ans(escaped)
  string(REGEX REPLACE "${escaped}" "" content "${content}")
  fwrite("${session_profile_path}" "${content}")
  return()
endfunction()


# returs true if the cmakepp session profile (environment variables)are registered
function(bash_autostart_isregistered)
  set(session_profile_path "$ENV{HOME}/.profile")
  if(NOT EXISTS "${session_profile_path}")
    return(false)
  endif()
  fread("${session_profile_path}")
  ans(content)
  string_regex_escape("${session_profile_path}")
  ans(escaped)
  if("${content}" MATCHES "${escaped}")
    return(true)
  endif()
  return(false)
endfunction()
