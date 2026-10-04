#!/usr/bin/env bash

# 

object_name=${1}
object_id=${2}

# 

if [ -z "$object_name" ]; then
    echo "Error: object_name is required."
    echo "Usage: $0 <object_name> [object_id]"
    exit 1
fi

# object_id 

if [ -z "$object_id" ]; then
    # object_id 

    python utils/generate_object_description.py "$object_name" 
else
    # object_id 

    python utils/generate_object_description.py "$object_name" --index "$object_id"
fi