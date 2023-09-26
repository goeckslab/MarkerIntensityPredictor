#!/bin/bash

# create folders for exacloud

# create a list of folders to be created
folders=(
 /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient/error_reports
 /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_23/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_23/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_46/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_46/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_92/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_92/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_138/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_138/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_184/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_exp_patient_sp_184/output_reports


 /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient/error_reports
 /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_23/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_23/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_46/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_46/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_92/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_92/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_138/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_138/output_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_184/error_reports
  /home/groups/OMSAtlas/Code/kirchgae/MarkerIntensityPredictor/mesmer/tumor_in_patient_sp_184/output_reports

  )


# loop through the list of folders and create them
for folder in "${folders[@]}"; do

  if [ ! -d "${folder}" ]; then
    echo "${folder} does not exists"
    mkdir -p "${folder}"
  fi
done