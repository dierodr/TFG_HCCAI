#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

#IMAGE_NAME="liver-cnn-app:mycnn"

echo "Stopping and removing containers, volumes, and orphans..."
docker compose down --volumes --remove-orphans

#echo "Removing old image: $IMAGE_NAME (if exists)..."
#docker rmi "$IMAGE_NAME" || echo "Image $IMAGE_NAME not found or already removed."

echo "Pruning unused Docker data..."
docker system prune -f

echo "Rebuilding and starting services..."
docker compose up --build --force-recreate