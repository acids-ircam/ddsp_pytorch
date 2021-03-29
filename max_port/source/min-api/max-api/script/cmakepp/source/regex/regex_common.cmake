## defines common regular expressions used in many places
macro(regex_common)

  set(regex_hex "[a-fA-F0-9]")
  set(regex_hex_2 "${regex_hex}${regex_hex}")
  set(regex_hex_4 "${regex_hex_2}${regex_hex_2}")
  set(regex_hex_8 "${regex_hex_4}${regex_hex_4}")
  set(regex_hex_12 "${regex_hex_8}${regex_hex_4}")

  set(regex_guid_ms "{(${regex_hex_8})\\-(${regex_hex_4})\\-(${regex_hex_4})\\-(${regex_hex_4})\\-(${regex_hex_12})}")



endmacro()