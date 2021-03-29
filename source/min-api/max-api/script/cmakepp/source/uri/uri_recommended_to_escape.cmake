
## characters specified in rfc2396
## 37 %  (percent)
## 126 ~ (tilde) 
## 1-32 (control chars) (nul is not allowed) 
## 127 (del)
## 32 (space)
## 35 (#) sharp fragment identifer
## 60 (<) 62 (>) 34 (") delimiters 
## unwise 
## 123 { 125 } 124 | 92 \ 94 ^ 91 [ 93 ] 96 `

function(uri_recommended_to_escape)
  ## control chars
  index_range(1 31)
  ans(dec_codes)

  
  list(APPEND dec_codes 
    32   # space
    34   # "
    35   # #
    60   # <
    62   # >
    91   # [
    93   # ]
    94   # ^ 
    96   # ` 
    123  # {
    124  # |
    125  # }
    127  # del
    )

  set(dec_codes
      37   # %  (this is prepended - important in uri_encode )
      ${dec_codes}
      )
  return_ref(dec_codes)



endfunction()