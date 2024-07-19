#!/bin/bash

declare -A args  # Declare an associative array to store arguments and values

args["task"]=""
args["model_config"]=""

# Loop through the arguments
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    # Check if the argument starts with "--"
    if [[ "$arg" == --* ]]; then
        arg_name="${arg:2}"  # Remove leading "--"
        valueid=$((i+1))
        # Get the value of the argument if it exists
        if ((i+1 <= $#)); then
            args["$arg_name"]="${!valueid}"
            i=$((i+1))  # Skip the next argument (its value)
        else
            args["$arg_name"]=""  # Set empty value if no value provided
        fi
    fi
done

# Print the values of the arguments
echo "----------- CMD args ----------"
for key in "${!args[@]}"; do
    echo "$key: ${args[$key]}"
done
echo "--------- END CMD args --------"

# --------------- 运行参数 ---------------
OPTS+=" --model_config ./model_configs/"${args['model_config']}".json"

CMD="python ${args['task']}.py${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

$CMD