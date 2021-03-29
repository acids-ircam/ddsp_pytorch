function(test)
  



  data("{a:1,b:{c:2}}")
  ans(data)
  template_run("@data.a@data.b.c@data.a")
  ans(res)
  assert("${res}" STREQUAL "121")

  template_run("@data.a/@data.b.c/@data.a@")
  ans(res)
  assert("${res}" STREQUAL "1/2/1")


  return()

  obj("{id:1,b:{c:3}}")
  ans(data)
  template_run("[<%={data.id}%>](<%={data.id}%>)")
  ans(res)
  assert("${res}" STREQUAL "[1](1)")



  ## this test shows a problem with cmake and template syntax 
  ## @<identifier>@ is replaced by the variable during string evalutation
  set(val1 abc)
  template_run("@set(val1 def)@val1@")
  ans(res)
  assert("${res}" STREQUAL "abc")
  ## however escaping the @ will alleviate the problem
  set(val1 abc)
  template_run("@set(val1 def)\@val1@")
  ans(res)
  assert("${res}" STREQUAL "def")

  # @ will evaluate to ""
  template_run("@")
  ans(res)
  assert("${res}_" STREQUAL "_")

  ## @ only works when escaped
  template_run("@@")
  ans(res)
  assert("${res}" STREQUAL "@")


  set(i 123)
  template_run("@foreach(i RANGE 1 3)\@i\@endforeach()")
  ans(res)
  assert("${res}" STREQUAL 123)
  ## allow storage of code fragment in variable with '<%><varname> ' (space is importand)
  template_run("<%>hello_you template_out(\${hello_you})%>@hello_you")
  ans(res)
  assert("${res}" STREQUAL "template_out(\${hello_you})")


  template_run("@@")
  ans(res)
  assert("${res}" STREQUAL "@")

  set(asd 123)
  template_run("@asd")
  ans(res)
  assert("${res}" STREQUAL "123")

  function(test_fu a b)
    return("hello ${a} ${b}")
  endfunction()

  template_run("@test_fu(\"Tobias\" \"Becker\")")
  ans(res)
  assert("${res}" STREQUAL "hello Tobias Becker")


  template_run("<%% %%>")
  ans(res)
  assert("${res}" STREQUAL "<% %>")



  template_run("
    Hello My Friend
    <% foreach(i RANGE 1 3) %><%=\${i}%><% endforeach() %>
    ByBy!
  ")
  ans(res)
  assert("${res}" STREQUAL "
    Hello My Friend
    123
    ByBy!
  ")


  ## tests wether the <%= expression works as expected

  template_run("<%={data.b.c}%>")
  ans(res)
  assert("${res}" STREQUAL "3")

  template_run("<%=abcdefg%>")
  ans(res)
  assert("${res}" STREQUAL abcdefg)

  set(input 123)
  template_run("<%=${input}%>")
  ans(res)
  assert("${res}" STREQUAL "123")

  ## spaces in string should be kep
  template_run("<%=\"  123  ${input}  \"%>")
  ans(res)
  assert("${res}" STREQUAL "  123  123  ")

  ## shoudl generate a list
  template_run("<%= 1 2 3%>")
  ans(res)
  assert(${res} EQUALS 1 2 3)



endfunction()