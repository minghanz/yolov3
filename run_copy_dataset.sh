#!/bin/bash
### This script is to copy files from local computer to a remote machine
usr_name="minghanz"
server="mcity_wired"
src_root="/media/sda1/datasets/extracted"
tgt_root="/mnt/storage8t/minghanz/Datasets/synthetic_shapenet1"

src_folders=(
  shapenet_KoPER1_3D_black
  shapenet_KoPER1_3D_black_aligned_1
  shapenet_KoPER1_3D_black_aligned_2
  shapenet_KoPER4_3D_black
  shapenet_KoPER4_3D_black_aligned_1
  shapenet_KoPER4_3D_black_aligned_2
)
tgt_folders=(
  KoPER1
  KoPER1_aligned1
  KoPER1_aligned2
  KoPER4
  KoPER4_aligned1
  KoPER4_aligned2
)

echo "target path:${tgt_root}"
ssh ${usr_name}@${server} "mkdir -p '${tgt_root}'"

for index in ${!src_folders[*]}; do 
  echo "${src_folders[$index]} is in ${tgt_folders[$index]}"
  ssh ${usr_name}@${server} "mkdir -p '${tgt_root}/${tgt_folders[$index]}'"
  scp "${src_root}/${src_folders[$index]}/bev_like_coco.zip" ${usr_name}@${server}:"${tgt_root}/${tgt_folders[$index]}/"
done