# executes the language file, input can be given by a key value list
  function(lang2 target language)
    map_from_keyvaluelist("" ${ARGN})
    ans(ctx)
    language("${language}")
    ans(language)
    
    obj_setprototype("${ctx}" "${language}")
    lang("${target}" "${ctx}")
    ans(res)

    
    return_ref(res)
  endfunction()