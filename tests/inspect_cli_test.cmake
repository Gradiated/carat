foreach(variable IN ITEMS FIXTURES INSPECT WORK_DIRECTORY)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "${variable} is required")
  endif()
endforeach()

execute_process(COMMAND "${FIXTURES}" "${WORK_DIRECTORY}" RESULT_VARIABLE status)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "fixture writer exited ${status}")
endif()

execute_process(COMMAND "${INSPECT}" "${WORK_DIRECTORY}/tiny"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "valid model exited ${status}: ${error}")
endif()
foreach(expected IN ITEMS "model=gemma4_text" "layers=2 sliding_layers=1 global_layers=1"
    "text_parameters=35402" "execution_segments=24" "validation=ok")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "valid model output lacks '${expected}':\n${output}")
  endif()
endforeach()

execute_process(COMMAND "${INSPECT}" RESULT_VARIABLE status ERROR_VARIABLE error)
if(NOT status EQUAL 64)
  message(FATAL_ERROR "missing argument exited ${status}, expected 64")
endif()
string(FIND "${error}" "usage: carat-inspect MODEL_DIRECTORY" position)
if(NOT position EQUAL 0)
  message(FATAL_ERROR "missing argument printed no usage: ${error}")
endif()

execute_process(COMMAND "${INSPECT}" "${WORK_DIRECTORY}/corrupt"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 1)
  message(FATAL_ERROR "corrupt shard exited ${status}, expected 1")
endif()
string(FIND "${error}" "carat-inspect: " position)
if(NOT position EQUAL 0 OR NOT output STREQUAL "")
  message(FATAL_ERROR "corrupt shard diagnostic is malformed: ${error}")
endif()

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
