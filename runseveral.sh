#!/bin/bash

# Define the source and destination directories
source_dir="./Models_Crop"
destination_img_dir="./FaceImages_Crop"

d1=/home/data/Proto/Face/Models_Crop/Anna_Bayle/97943c965090e874352d22cf66d15ab6.jpg
d2=/home/data/Proto/Face/Models_Crop/Andrés_Velencoso/5f24a624a0781bd35c974b6bbad2230d.jpg
d3=/home/data/Proto/Face/Models_Crop/Anna_Bayle/97943c965090e874352d22cf66d15ab6.jpg
d4=/home/data/Proto/Face/Models_Crop/Anna_Bayle/97943c965090e874352d22cf66d15ab6.jpg
d5=/home/data/Proto/Face/Models_Crop/Natty/44d26dd63372675fab3c3a67b4bdb6ac.jpg
s1=/home/data/Proto/Face/Models_Crop/Kim_Jung-ah/aa7de74f247eeeaf659b3fb262bd1d16.jpg
s2=/home/data/Proto/Face/Models_Crop/Mary_Nnenna/18a6a36ade5b5cf6af5ad2cb783d79b6.jpg
s3=/home/data/Proto/Face/Models_Crop/Nick_Bateman/7c34af85837b5bf1bd5998928adf443f.jpg
drivers=($d1 $d2 $d3 $d4 $d5)
sources=($s1 $s2 $s3)



mkdir -p "$destination_img_dir"
i=0

for s in ${sources[@]}; do
    filename=$(basename "$s")
    ext="${filename##*.}"
    name="${filename%.*}"
    for d in ${drivers[@]}; do
        filename="${name}_$i.$ext"
        python src/scripts.py --src_img $s --drv_img $d --out_dir $destination_img_dir --out_name $filename
        let i++
    done
done

# # Loop through the subfolders in the source directory
# for subfolder in "$source_dir"/*; do
#     # Extract the subfolder name
#     subfolder_name=$(basename "$subfolder")
    
#     # Loop through the files in the subfolder
#     for file in "$subfolder"/*; do
#         # Extract the file name and extension
#         file_name=$(basename "$file")
        
#         # Check if the file has a '_mask' suffix
#         if [[ "$file_name" == *".jpg" ]]; then
#             # Copy the file to the destination masks directory
#             python src/scripts.py --src_img $file --drv_img $d --out_dir $odir --out_name $file_name

#         fi
#     done
# done

# echo "Copying files completed."
