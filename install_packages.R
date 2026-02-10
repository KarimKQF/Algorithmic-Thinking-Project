# Install necessary packages for Credit Risk Analysis
packages <- c("tidyverse", "xgboost", "caret", "nnet", "corrplot", "pROC", "ROCR", "reshape2", "Ckmeans.1d.dp")

# Check if packages are installed, if not, install them
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]

if(length(new_packages)) {
  # Create user library if it doesn't exist
  lib_dir <- Sys.getenv("R_LIBS_USER")
  if (!dir.exists(lib_dir)) dir.create(lib_dir, recursive = TRUE)
  
  install.packages(new_packages, lib = lib_dir, repos = "https://cloud.r-project.org")
}

cat("All required packages are installed.\n")
