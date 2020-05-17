#!/bin/bash

LEARNING_RESULT_DIR="workspace/02-learning"

cd ${LEARNING_RESULT_DIR}
zip -r buckup history weight option -x "*.DS_Store"

