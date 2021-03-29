# checks to see if an assertion holds true. per default this halts the program if the assertion fails
# usage:
# assert(<assertion> [MESSAGE <string>] [MESSAGE_TYPE <FATAL_ERROR|STATUS|...>] [RESULT <ref>])
# <assertion> := <truth-expression>|[<STRING|NUMBER>EQUALS <list> <list>]
# <truth-expression> := anything that can be checked in if(<truth-expression>)
# <list> := <ref>|<value>|<value>;<value>;...
# if RESULT is set then the assertion will not cause the program to fail but return the true or false
# if ACCU is set result is treated as a list and if an assertion fails the failure message is added to the end of result
# examples
#
# assert("3" STREQUAL "3") => nothing happens
# assert("3" STREQUAL "b") => FATAL_ERROR assertion failed: '"3" STREQUAL "b"'
# assert(EXISTS "none/existent/path") => FATAL_ERROR assertion failed 'EXISTS "none/existent/path"'
# assert(EQUALS a b) => FATAL_ERROR assertion failed ''
# assert(<assertion> MESSAGE "hello") => if assertion fails prints "hello"
# assert(<assertion> RESULT res) => sets result to false if assertion fails or to true if it holds
# assert(EQUALS "1;3;4;6;7" "1;3;4;6;7") => nothing happens lists are equal
# assert(EQUALS 1 2 3 4 1 2 3 4) =>nothing happes lists are equal (see list_equal)
# assert(EQUALS C<list> <list> COMPARATOR <comparator> )... todo
# todo: using the variable result as a boolean check fails because the name is used inside assert

function(assert)
    # parse arguments
    set(options == EQUALS AREEQUAL ARE_EQUAL ACCU SILENT DEREF INCONCLUSIVE ISNULL ISNOTNULL)
    set(oneValueArgs EXISTS COUNT MESSAGE RESULT MESSAGE_TYPE CONTAINS MISSING MATCH MAP_MATCHES FILE_CONTAINS)
    set(multiValueArgs CALL PREDICATE)
    set(prefix)
    cmake_parse_arguments("${prefix}" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    #_UNPARSED_ARGUMENTS
    set(result)


    #if no message type is set set FATAL_ERROR
    # so execution halts on failing assertion
    if (NOT _MESSAGE_TYPE)
        set(_MESSAGE_TYPE FATAL_ERROR)
    endif ()

    # if continue is set: set the mesype to statussage t
    if (_RESULT AND _MESSAGE_TYPE STREQUAL FATAL_ERROR)
        set(_MESSAGE_TYPE STATUS)
    endif ()

    if (_DEREF)
        #format( "${_UNPARSED_ARGUMENTS}")
        format("${_UNPARSED_ARGUMENTS}")
        ans(_UNPARSED_ARGUMENTS)
    endif ()

    ## transform call into further arguments
    if (_CALL)
        call(${_CALL})

        ans(vars)
        list(APPEND _UNPARSED_ARGUMENTS ${vars})
    endif ()

    #
    if (_EQUALS OR _ARE_EQUAL OR _AREEQUAL)
        if (NOT _MESSAGE)
            set(_MESSAGE "assertion failed: lists not equal [${_UNPARSED_ARGUMENTS}]")
        endif ()
        list_equal(${_UNPARSED_ARGUMENTS})
        ans(result)
    elseif (_EXISTS)
        list_pop_front(_UNPARSED_ARGUMENTS)
        ans(not)
        if (NOT "${not}_" STREQUAL "NOT_")
            set(not)
        endif ()
        path("${_EXISTS}")
        ans(_EXISTS)
        if (NOT _MESSAGE)
            set(_MESSAGE "assertion failed: file does not exists ${_EXISTS}")
        endif ()

        if (${not} EXISTS "${_EXISTS}")
            set(result true)
        else ()
            set(result false)
        endif ()
    elseif (_PREDICATE)
        if (NOT _MESSAGE)
            set(_MESSAGE "assertion failed: predicate does not hold: '${_PREDICATE}'")
        endif ()
        #	message("predicate '${_PREDICATE}'")
        call(${_PREDICATE} (${_UNPARSED_ARGUMENTS}))
        ans(result)
    elseif (_FILE_CONTAINS)
        if (NOT _MESSAGE)
            set(_MESSAGE "assertion failed: file '${_FILE_CONTAINS}' does not contain: ${_UNPARSED_ARGUMENTS}")
        endif ()
        file(READ "${_FILE_CONTAINS}" contents)
        if ("${contents}" MATCHES "${_UNPARSED_ARGUMENTS}")
            set(result true)
        else ()
            set(result false)
        endif ()
    elseif (_INCONCLUSIVE)
        if (NOT _MESSAGE)
            set(_MESSAGE "assertion inconclusive")
        endif ()
        set(result true)

    elseif (_MATCH)
        if (NOT _MESSAGE)
            set(_MESSAGE "assertion failed: input does not match '${_MATCH}'")
        endif ()
        list_extract_flag_name(_UNPARSED_ARGUMENTS NOT)
        ans(not)

        if (${not} "${_UNPARSED_ARGUMENTS}" MATCHES "${_MATCH}")
            set(result true)
        else ()
            set(result false)
        endif ()
    elseif (_COUNT OR "_${_COUNT}" STREQUAL _0)
        list(LENGTH _UNPARSED_ARGUMENTS len)
        if (NOT _MESSAGE)
            set(_MESSAGE "assertion failed: expected '${_COUNT}' elements got '${len}'")
        endif ()
        eval_truth("${len}" EQUAL "${_COUNT}")
        ans(result)
    elseif (_ISNULL)
        if ("${_UNPARSED_ARGUMENTS}_" STREQUAL "_")
            set(result true)
        else ()
            set(_MESSAGE "assertion failed: '${_UNPARSED_ARGUMENTS}' is not null")
            set(result false)
        endif ()
    elseif (_ISNOTNULL)
        if ("${_UNPARSED_ARGUMENTS}_" STREQUAL "_")
            set(result false)
        else ()
            set(_MESSAGE "assertion failed: argument is null")
            set(result true)
        endif ()
    elseif (_MAP_MATCHES)
        data("${_MAP_MATCHES}")
        ans(_MAP_MATCHES)
        map_match("${_UNPARSED_ARGUMENTS}" "${_MAP_MATCHES}")
        ans(result)
        if (NOT _MESSAGE)
            json("${_MAP_MATCHES}")
            ans(expected)
            json("${_UNPARSED_ARGUMENTS}")
            ans(actual)
            set(_MESSAGE "assertion failed: match failed: expected: '${expected}' actual:'${actual}'")
        endif ()

    elseif (_CONTAINS OR _MISSING)
        if (NOT _MESSAGE)
            set(_MESSAGE "assertion failed: list does not contain '${_CONTAINS}' list:(${_UNPARSED_ARGUMENTS})")
        endif ()
        list(FIND _UNPARSED_ARGUMENTS "${_CONTAINS}" idx)

        if (${idx} LESS 0)
            if (_MISSING)
                set(result true)
            else ()
                set(result false)
            endif ()
        else ()
            if (_MISSING)
                set(result false)
            else ()
                set(result true)
            endif ()
        endif ()

    else ()
        # if nothing else is specified use _UNPARSED_ARGUMENTS as a truth expresion
        eval_truth((${_UNPARSED_ARGUMENTS}))
        ans(result)
    endif ()

    # if message is not set add default message
    if ("${_MESSAGE}_" STREQUAL "_")
        list_to_string(_UNPARSED_ARGUMENTS " ")
        ans(msg)
        set(_MESSAGE "assertion failed1: '${_UNPARSED_ARGUMENTS}'")
    endif ()

    # print message if assertion failed, SILENT is not specified or message type is FATAL_ERROR
    if (NOT result)
        if (_MESSAGE_TYPE STREQUAL "FATAL_ERROR")
            message("'${_MESSAGE}'")
            message(PUSH "log:")
            log_print(10)
            message(POP)
            message(FATAL_ERROR " ")
        endif ()

        if (NOT _SILENT OR _MESSAGE_TYPE STREQUAL "FATAL_ERROR")
            message(${_MESSAGE_TYPE} "'${_MESSAGE}'")
        endif ()
    endif ()

    # depending on wether to accumulate the results or not
    # set result to a boolean or append to result list
    if (_ACCU)
        set(lst ${_RESULT})
        list(APPEND lst ${_MESSAGE})
        set(${_RESULT} ${lst} PARENT_SCOPE)
    else ()
        set(${_RESULT} ${result} PARENT_SCOPE)
    endif ()

endfunction()