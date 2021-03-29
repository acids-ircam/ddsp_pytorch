function(test) 
  # single word changed to upper case
  set(str "else")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "Else")

  # v changed to lower case
  set(str "this V that")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "This v That")

  # v changed to lower case
  set(str "this V that")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "This v That")

  # Example sentence 
  set(str "the function string_totitle works")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "The Function string_totitle Works")

  # small after a subsentence (1/2)
  set(str "Subsentence: a Sub")
  string_to_title("${str}")
  ans(res)

  # Every word upper case
  set(str "abcdefg hello")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "Abcdefg Hello")

  # Small letter 'a' stays small
  set(str "abcdefg a hello")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "Abcdefg a Hello")

  # Keep whitespaces
  set(str "abcdefg a   hello")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "Abcdefg a   Hello")

  # words with dot inside stay small
  set(str "abcdefg a hel.lo")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "Abcdefg a hel.lo")

  # words with big letter elsewhere stay small
  set(str "abcdefg a heLLo")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "Abcdefg a heLLo")

  # beginning of sentence has 'small' -> uppercase
  set(str "a abcdefg hello a superTrump?")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "A Abcdefg Hello a superTrump?")

  # small after a subsentence (1/2)
  set(str "Subsentence: a Sub")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "Subsentence: A Sub")
  
  # small after a subsentence (2/2)
  set(str "Subsentence! a Sub")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "Subsentence! A Sub")

  # v stays lower case
  set(str "this v that")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "This v That")

  # v changed to lower case
  set(str "this Vs that")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "This vs That")

  # colo(u)r not changed to upper case
  set(str "this is not a colo(u)r?")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "This Is Not a colo(u)r?")

  # a at beginning (after quotes) changed to uppercase
  set(str "'a test'")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "'A Test'")

  # quotes and a subsequent subsentence
  set(str "'a test': the subsentence")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "'A Test': The Subsentence")

  # quotes and a subsequent subsentence
  set(str "'a test': 'the subsentence'")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "'A Test': 'The Subsentence'")

  # words like "that's", "it's" should be upper case also
  set(str "'a word': 'look, that's a subsentence isn't it?")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "'A Word': 'Look, That's a Subsentence Isn't It?")

  # "'" (and "," etc.) does not start a subsentence (1/3)
  set(str "This is a word, a word")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "This Is a Word, a Word")

  # "'" (and "," etc.) does not start a subsentence (2/3)
  set(str "This is a word' a word")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "This Is a Word' a Word")

  # "'" (and "," etc.) does not start a subsentence (3/3)
  set(str "This is a (word) a word")
  string_to_title("${str}")
  ans(res)
  assert("${res}" STREQUAL "This Is a (Word) a Word")
endfunction()
