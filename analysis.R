# Credit Risk Analysis - Main R Script (Robust Version)

# --- Libraries ---
suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(xgboost)
  library(nnet)
  library(corrplot)
  library(pROC)
  library(reshape2)
})

cat("Libraries loaded successfully.\n")

# --- Configuration ---
Config <- list(
  DATA_PATH = file.path("data", "credit_risk_dataset(in).csv"),
  OUTPUT_DIR = "output",
  PLOTS_DIR = file.path("output", "plots"),
  SEED = 42
)

if (!dir.exists(Config$PLOTS_DIR)) {
  dir.create(Config$PLOTS_DIR, recursive = TRUE)
}

# --- 1. Data Loading ---
cat("\n[1] Loading Data...\n")
tryCatch({
  df <- read.csv(Config$DATA_PATH, sep = ";", stringsAsFactors = TRUE)
  cat("Loaded", nrow(df), "rows and", ncol(df), "columns.\n")
}, error = function(e) {
  stop("Failed to load data: ", e$message)
})

# --- 2. EDA ---
cat("\n[2] Running Exploratory Data Analysis (EDA)...\n")

save_plot <- function(filename, plot_obj = NULL) {
  filepath <- file.path(Config$PLOTS_DIR, filename)
  tryCatch({
    if (!is.null(plot_obj)) {
      ggsave(filepath, plot = plot_obj, width = 12, height = 7, dpi = 300)
    } else {
      # For base plots
      dev.copy(png, filepath, width = 1200, height = 1000, res = 150)
      dev.off()
    }
    cat("Saved:", filename, "\n")
  }, error = function(e) {
    cat("Error saving", filename, ":", e$message, "\n")
  })
}

# 2.1 Distributions (Numerical)
num_cols <- df %>% select(where(is.numeric)) %>% names()

for (col in num_cols) {
  tryCatch({
    p <- ggplot(df, aes(x = .data[[col]])) +
      geom_histogram(fill = "teal", color = "white", bins = 30) +
      theme_minimal() +
      labs(title = paste("Distribution of", col), x = col, y = "Count")
    
    save_plot(paste0("dist_", col, ".png"), p)
  }, error = function(e) {
    cat("Skipped plot for", col, ":", e$message, "\n")
  })
}

# 2.2 Categorical Counts
cat_cols <- df %>% select(where(is.factor)) %>% names()

for (col in cat_cols) {
  tryCatch({
    p <- ggplot(df, aes(x = .data[[col]])) +
      geom_bar(fill = "viridis", color = "white") +
      scale_fill_viridis_d(option = "D") +
      theme_minimal() +
      labs(title = paste("Count of", col), x = col, y = "Count") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    save_plot(paste0("count_", col, ".png"), p)
  }, error = function(e) {
      # Fallback without palette if viridis fails
      p <- ggplot(df, aes(x = .data[[col]])) +
      geom_bar(fill = "steelblue", color = "white") +
      theme_minimal() +
      labs(title = paste("Count of", col), x = col, y = "Count")
      save_plot(paste0("count_", col, ".png"), p)
  })
}

# 2.3 Correlation Matrix
tryCatch({
  cat("Generating Matrix...\n")
  num_df <- df %>% select(where(is.numeric))
  cor_matrix <- cor(num_df, use = "complete.obs")
  
  png(file.path(Config$PLOTS_DIR, "correlation_heatmap.png"), width = 1200, height = 1000, res = 150)
  corrplot(cor_matrix, method = "color", type = "upper", 
           tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.7)
  dev.off()
  cat("Saved: correlation_heatmap.png\n")
}, error = function(e) {
  cat("Correlation matrix failed:", e$message, "\n")
})


# --- 3. Preprocessing ---
cat("\n[3] Preprocessing...\n")

df <- unique(df)

impute_median <- function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
}
impute_mode <- function(x) {
  ux <- unique(x)
  mode_val <- ux[which.max(tabulate(match(x, ux)))]
  x[is.na(x)] <- mode_val
  return(x)
}

df_clean <- df %>%
  mutate(across(where(is.numeric), impute_median)) %>%
  mutate(across(where(is.factor), impute_mode))

# Encode Target
df_clean$loan_status_fac <- factor(df_clean$loan_status, labels = c("Repaid", "Default"))

set.seed(Config$SEED)
trainIndex <- createDataPartition(df_clean$loan_status_fac, p = .8, list = FALSE, times = 1)
dataTrain <- df_clean[ trainIndex,]
dataTest  <- df_clean[-trainIndex,]

# --- 4. Modeling ---
cat("\n[4] Training Models...\n")

fitControl <- trainControl(method = "none", classProbs = TRUE, summaryFunction = twoClassSummary)

# 4.1 GLM
cat("Training GLM...\n")
tryCatch({
  model_glm <- train(loan_status_fac ~ . -loan_status, data = dataTrain, 
                     method = "glm", family = "binomial", trControl = fitControl)
}, error = function(e) cat("GLM Failed:", e$message, "\n"))

# 4.2 XGBoost
cat("Training XGBoost...\n")
tryCatch({
  X_train_xgb <- model.matrix(loan_status ~ . -loan_status -loan_status_fac, data = dataTrain)[,-1]
  y_train_xgb <- dataTrain$loan_status
  X_test_xgb <- model.matrix(loan_status ~ . -loan_status -loan_status_fac, data = dataTest)[,-1]
  y_test_xgb <- dataTest$loan_status
  
  dtrain <- xgb.DMatrix(data = X_train_xgb, label = y_train_xgb)
  dtest <- xgb.DMatrix(data = X_test_xgb, label = y_test_xgb)
  
  params <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.05, max_depth = 6, eval_metric = "auc")
  model_xgb <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
}, error = function(e) cat("XGB Failed:", e$message, "\n"))


# 4.3 Neural Net
cat("Training NNet...\n")
tryCatch({
  preProc <- preProcess(dataTrain, method = c("center", "scale"))
  train_norm <- predict(preProc, dataTrain)
  test_norm <- predict(preProc, dataTest)
  
  model_nn <- nnet(loan_status_fac ~ . -loan_status, data = train_norm, 
                   size = 32, decay = 0.0001, maxit = 200, trace = FALSE)
}, error = function(e) cat("NNet Failed:", e$message, "\n"))

# --- 5. Evaluation ---
cat("\n[5] Evaluation...\n")

actuals <- dataTest$loan_status_fac

# Predictions & ROC
png(file.path(Config$PLOTS_DIR, "roc_curve_comparison.png"), width = 2400, height = 1800, res = 300)
plot(0,0, type="n", xlim=c(1,0), ylim=c(0,1), xlab="Specificity", ylab="Sensitivity", main="ROC Comparison")
abline(a=0, b=1, lty=2)

if (exists("model_glm")) {
  pred <- predict(model_glm, dataTest, type = "prob")[, "Default"]
  roc_glm <- roc(actuals, pred, levels = c("Repaid", "Default"), direction = "<")
  plot(roc_glm, col = "red", add = TRUE)
  cat("GLM AUC:", auc(roc_glm), "\n")
}

if (exists("model_xgb")) {
  pred <- predict(model_xgb, dtest)
  roc_xgb <- roc(actuals, pred, levels = c("Repaid", "Default"), direction = "<")
  plot(roc_xgb, col = "blue", add = TRUE)
  
  # Importance
  tryCatch({
    imp <- xgb.importance(model = model_xgb)
    p <- xgb.ggplot.importance(imp, top_n = 20)
    save_plot("feature_importance_xgboost.png", p)
  }, error = function(e) cat("XGB Importance Failed:", e$message, "\n"))
}

if (exists("model_nn")) {
  pred <- predict(model_nn, test_norm, type = "raw")[,1]
  roc_nn <- roc(actuals, pred, levels = c("Repaid", "Default"), direction = "<")
  plot(roc_nn, col = "green", add = TRUE)
}

dev.off()
cat("ROC saved.\n")

cat("\nDone!\n")
