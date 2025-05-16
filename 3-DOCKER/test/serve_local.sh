#!/bin/sh

image=$1

docker run -v $(realpath ../../_Model/vTestJardianceRxEgm):/opt/ml/model/vTestJardianceRxEgm \
           -p 8080:8080 \
           -e INF_CohortName="20240410_Inference" \
           -e MODEL_VERSION="vTestJardianceRxEgm" \
           -e POST_PROCESS_NAME="EngagementPredToLabel" \
           -e LoggerLevel="INFO" \
           --rm ${image} \
           serve
