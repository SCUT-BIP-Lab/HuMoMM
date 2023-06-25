#!/bin/bash
read -p "Please enter path of Colmap_txt(default ./Colmap_txt):" FILEPATH
if [ -z "$FILEPATH" ]
then
    FILEPATH="../../dataset/scut_key_frame/Colmap_txt"
fi
if !([ -e $FILEPATH ])
then
    echo "dont have this parh ,please retry!"
else
    OUT_FILE="colmap_out"
    for SAMPLE_ID in `ls $FILEPATH`
    do
        if !([ -e $FILEPATH"/"$SAMPLE_ID"/"$OUT_FILE ])
        then
            mkdir $FILEPATH"/"$SAMPLE_ID"/"$OUT_FILE
            echo "$SAMPLE_ID does not have $OUT_FILE,create $OUT_FILE done"
        else
            echo "$SAMPLE_ID already has $OUT_FILE"
        fi


        if [ -e $FILEPATH"/"$SAMPLE_ID"/images.txt" ]
        then
            colmap bundle_adjuster --input_path $FILEPATH"/"$SAMPLE_ID --output_path $FILEPATH"/"$SAMPLE_ID"/"$OUT_FILE --BundleAdjustment.parameter_tolerance 1e-15 --BundleAdjustment.max_num_iterations 5000 --BundleAdjustment.refine_focal_length 0 --BundleAdjustment.refine_principal_point 0 --BundleAdjustment.refine_extra_params 0 --BundleAdjustment.refine_extrinsics 0
            colmap model_converter --input_path $FILEPATH"/"$SAMPLE_ID"/"$OUT_FILE --output_path $FILEPATH"/"$SAMPLE_ID"/"$OUT_FILE --output_type txt
            rm $FILEPATH"/"$SAMPLE_ID"/"$OUT_FILE"/images.bin"
            rm $FILEPATH"/"$SAMPLE_ID"/"$OUT_FILE"/cameras.bin"
            rm $FILEPATH"/"$SAMPLE_ID"/"$OUT_FILE"/points3D.bin"
        else
            echo "$SAMPLE_ID colmaptxt is none"
        fi

    done
    echo "Bundle Adjustment finished!"
fi
