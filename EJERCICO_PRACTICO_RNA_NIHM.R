# =============================================================================
# Workshop 4 - Redes Neuronales Artificiales
# MCI002 - Universidad de La Frontera, 2026-1
#
# Aplicacion: Prediccion de resistencia a compresion de hormigon (f'c, MPa)
# Dataset:    Concrete Compressive Strength - Yeh (1998)
#             UCI Machine Learning Repository
#             https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
#
# Variables de entrada (8):
#   cement           - Cemento Portland         (kg/m3)
#   slag             - Escoria GGBFS            (kg/m3)
#   flyash           - Ceniza volante           (kg/m3)
#   water            - Agua                     (kg/m3)
#   superplasticizer - Superplastificante       (kg/m3)
#   coarseaggregate  - Arido grueso             (kg/m3)
#   fineaggregate    - Arido fino               (kg/m3)
#   age              - Edad de ensayo           (dias)
#
# Variable de salida (1):
#   strength         - Resistencia a compresion (MPa)
#
# Autor: Nicole Hernandez Morales
# Fecha: junio 2026
# =============================================================================


# -----------------------------------------------------------------------------
# 0. PAQUETES 
# -----------------------------------------------------------------------------

packages <- c("neuralnet", "ggplot2", "dplyr")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(neuralnet)   # Red neuronal
library(ggplot2)     # Visualizaciones
library(dplyr)       # Manipulacion de datos

set.seed(123)        # Reproducibilidad global


# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------------------

url_datos <- "https://raw.githubusercontent.com/nighdezm/Workshop-4---MCI002/main/concrete.csv"

cargar_datos <- function(url) {
  tmp <- tempfile(fileext = ".csv")
  ok <- tryCatch({
    download.file(url, tmp, mode = "wb", quiet = TRUE)
    TRUE
  }, error = function(e) FALSE)
  
  if (ok && file.exists(tmp) && file.size(tmp) > 1000) {
    df <- read.csv(tmp)
    message("Datos cargados desde GitHub correctamente.")
    return(df)
  }
  
  # Respaldo: copia local en la misma carpeta del script
  if (file.exists("concrete.csv")) {
    message("URL no disponible. Cargando copia local 'concrete.csv'...")
    return(read.csv("concrete.csv"))
  }
  
  stop("No se pudo cargar el dataset ni desde GitHub ni localmente.\n",
       "Verifica tu conexion o coloca 'concrete.csv' junto al script.")
}

concrete <- cargar_datos(url_datos)

# Asegurar nombres de columnas legibles y consistentes
colnames(concrete) <- c(
  "cement", "slag", "flyash", "water",
  "superplasticizer", "coarseaggregate", "fineaggregate",
  "age", "strength"
)

cat("=== DATASET CARGADO ===\n")
cat("Dimensiones:", nrow(concrete), "filas x", ncol(concrete), "columnas\n\n")


# -----------------------------------------------------------------------------
# 2. EXPLORACION INICIAL
# -----------------------------------------------------------------------------

cat("=== RESUMEN ESTADISTICO ===\n")
print(summary(concrete))

cat("\n=== VALORES FALTANTES ===\n")
print(colSums(is.na(concrete)))

# Distribucion de la variable objetivo
p_hist <- ggplot(concrete, aes(x = strength)) +
  geom_histogram(bins = 30, fill = "#7F77DD", color = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(concrete$strength), color = "#D85A30",
             linetype = "dashed", linewidth = 0.8) +
  labs(
    title = "Distribucion de resistencia a compresion",
    subtitle = paste0("Media = ", round(mean(concrete$strength), 1), " MPa  |  ",
                      "DE = ", round(sd(concrete$strength), 1), " MPa"),
    x = "Resistencia (MPa)", y = "Frecuencia"
  ) +
  theme_minimal(base_size = 12)
print(p_hist)

# Correlacion con la variable objetivo (R base, sin corrplot)
cat("\n=== CORRELACION CON VARIABLE OBJETIVO ===\n")
cors <- cor(concrete)
print(round(sort(cors[, "strength"], decreasing = TRUE), 3))


# -----------------------------------------------------------------------------
# 3. PREPROCESAMIENTO
# -----------------------------------------------------------------------------

# 3.1 Normalizacion min-max al rango [0, 1]
# Imprescindible en redes: evita que variables de rango grande
# (arido ~1000) dominen sobre las de rango pequeno (superplast. ~5).
normalize <- function(x) (x - min(x)) / (max(x) - min(x))

# Guardar min/max de 'strength' para desnormalizar predicciones luego
strength_min <- min(concrete$strength)
strength_max <- max(concrete$strength)

concrete_norm <- as.data.frame(lapply(concrete, normalize))

cat("\n=== DATOS NORMALIZADOS (primeras 3 filas) ===\n")
print(round(head(concrete_norm, 3), 4))

# 3.2 Division train / test (80% / 20%) con R base
# Muestreo aleatorio simple; set.seed(123) garantiza reproducibilidad.
n <- nrow(concrete_norm)
idx_train <- sample(seq_len(n), size = floor(0.80 * n))

train <- concrete_norm[ idx_train, ]
test  <- concrete_norm[-idx_train, ]

cat("\n=== DIVISION DEL DATASET ===\n")
cat("Entrenamiento:", nrow(train), "muestras (", round(nrow(train)/n*100), "%)\n")
cat("Test:         ", nrow(test),  "muestras (", round(nrow(test)/n*100),  "%)\n")

# Vectores de la variable objetivo en escala real (MPa), reutilizables
real_train_mpa <- train$strength * (strength_max - strength_min) + strength_min
real_test_mpa  <- test$strength  * (strength_max - strength_min) + strength_min


# -----------------------------------------------------------------------------
# 4. METRICAS 
# -----------------------------------------------------------------------------
rmse <- function(real, pred) sqrt(mean((real - pred)^2))
mae  <- function(real, pred) mean(abs(real - pred))
r2   <- function(real, pred) cor(real, pred)^2


# -----------------------------------------------------------------------------
# 5. MODELO REFERENCIA - REGRESION LINEAL MULTIPLE
# -----------------------------------------------------------------------------
# Entrenamos una regresion en los mismos datos para comparar.
# Justifica cuanto valor agrega realmente la RNA.

lm_model     <- lm(strength ~ ., data = train)
pred_lm_norm <- predict(lm_model, newdata = test)
pred_lm_mpa  <- pred_lm_norm * (strength_max - strength_min) + strength_min

r2_lm   <- r2(real_test_mpa,  pred_lm_mpa)
rmse_lm <- rmse(real_test_mpa, pred_lm_mpa)
mae_lm  <- mae(real_test_mpa,  pred_lm_mpa)

cat("\n=== REGRESION LINEAL (referencia) ===\n")
cat("R2  =", round(r2_lm, 4),   "\n")
cat("RMSE=", round(rmse_lm, 3), "MPa\n")
cat("MAE =", round(mae_lm, 3),  "MPa\n")


# -----------------------------------------------------------------------------
# 6. RED NEURONAL ARTIFICIAL - neuralnet
# -----------------------------------------------------------------------------

vars_entrada <- setdiff(names(train), "strength")
formula_rna  <- as.formula(
  paste("strength ~", paste(vars_entrada, collapse = " + "))
)

cat("\n=== ENTRENANDO RNA ===\n")
cat("Arquitectura: 8 entradas -> [8, 4] capas ocultas -> 1 salida\n")
cat("Activacion: logistica (sigmoide) | Salida: lineal (regresion)\n\n")

rna_model <- neuralnet(
  formula       = formula_rna,
  data          = train,
  hidden        = c(8, 4),
  linear.output = TRUE,
  act.fct       = "logistic",
  threshold     = 0.01,
  stepmax       = 1e6,
  rep           = 3
)

errores_rep <- rna_model$result.matrix["error", ]
mejor_rep   <- as.integer(which.min(errores_rep))

cat("Entrenamiento completado.\n")
cat("Mejor repeticion:", mejor_rep, "\n")
cat("Iteraciones:", rna_model$result.matrix["steps", mejor_rep], "\n")

# Visualizar arquitectura (la mejor repeticion)
plot(rna_model, rep = mejor_rep,
     col.entry = "#0F6E56", col.hidden = "#534AB7",
     col.out = "#993C1D", col.intercept = "#888780",
     information = FALSE, show.weights = FALSE)


# -----------------------------------------------------------------------------
# 7. PREDICCION Y EVALUACION EN TEST
# -----------------------------------------------------------------------------

# Forzar que newdata sea un data.frame con las columnas en el orden correcto.
# (Si queda como vector o cambia el orden, predict.nn falla.)
newdata_test <- as.data.frame(test[, vars_entrada, drop = FALSE])

# Usar el NUMERO de la mejor repeticion, no "best".
pred_rna_norm <- predict(rna_model, newdata = newdata_test, rep = mejor_rep)
pred_rna_norm <- as.numeric(pred_rna_norm)   # asegurar vector numerico limpio

pred_rna_mpa  <- pred_rna_norm * (strength_max - strength_min) + strength_min

r2_rna   <- r2(real_test_mpa,  pred_rna_mpa)
rmse_rna <- rmse(real_test_mpa, pred_rna_mpa)
mae_rna  <- mae(real_test_mpa,  pred_rna_mpa)
mape_rna <- mean(abs((real_test_mpa - pred_rna_mpa) / real_test_mpa)) * 100

cat("\n=== METRICAS RNA - SET DE TEST ===\n")
cat("R2  =", round(r2_rna, 4),   "\n")
cat("RMSE=", round(rmse_rna, 3), "MPa\n")
cat("MAE =", round(mae_rna, 3),  "MPa\n")
cat("MAPE=", round(mape_rna, 2), "%\n")

# -----------------------------------------------------------------------------
# 8. VISUALIZACIONES
# -----------------------------------------------------------------------------

# 8.1 Predicho vs Real - RNA
df_pred <- data.frame(Real = real_test_mpa, Predicho = as.vector(pred_rna_mpa))
limite  <- c(0, max(df_pred$Real, df_pred$Predicho) * 1.05)

p_pred <- ggplot(df_pred, aes(x = Real, y = Predicho)) +
  geom_point(color = "#7F77DD", alpha = 0.65, size = 2) +
  geom_abline(slope = 1, intercept = 0,
              color = "#D85A30", linewidth = 0.8, linetype = "dashed") +
  annotate("text", x = limite[1] + diff(limite) * 0.05, y = limite[2] * 0.95,
           label = paste0("R2 = ", round(r2_rna, 4),
                          "\nRMSE = ", round(rmse_rna, 2), " MPa",
                          "\nMAE  = ", round(mae_rna, 2), " MPa"),
           hjust = 0, vjust = 1, size = 3.8, color = "#3C3489", fontface = "bold") +
  scale_x_continuous(limits = limite) +
  scale_y_continuous(limits = limite) +
  labs(title = "RNA - Predicho vs Real (set de test)",
       subtitle = paste0("Arquitectura: 8-[8,4]-1  |  n = ", nrow(test), " muestras"),
       x = "Resistencia real (MPa)", y = "Resistencia predicha (MPa)") +
  theme_minimal(base_size = 12)
print(p_pred)

# 8.2 Comparacion RNA vs Regresion Lineal
df_comp <- data.frame(
  Real     = rep(real_test_mpa, 2),
  Predicho = c(as.vector(pred_rna_mpa), pred_lm_mpa),
  Modelo   = rep(c("RNA (8-4)", "Regresion Lineal"), each = length(real_test_mpa))
)

p_comp <- ggplot(df_comp, aes(x = Real, y = Predicho, color = Modelo)) +
  geom_point(alpha = 0.50, size = 1.8) +
  geom_abline(slope = 1, intercept = 0,
              color = "black", linewidth = 0.6, linetype = "dashed") +
  scale_color_manual(values = c("RNA (8-4)" = "#7F77DD",
                                "Regresion Lineal" = "#1D9E75")) +
  scale_x_continuous(limits = limite) +
  scale_y_continuous(limits = limite) +
  facet_wrap(~Modelo) +
  labs(title = "Comparacion: RNA vs Regresion Lineal",
       subtitle = "Linea punteada = prediccion perfecta (predicho = real)",
       x = "Resistencia real (MPa)", y = "Resistencia predicha (MPa)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
print(p_comp)

# 8.3 Residuos de la RNA
df_pred$Residuo <- df_pred$Real - df_pred$Predicho

p_res <- ggplot(df_pred, aes(x = Predicho, y = Residuo)) +
  geom_point(color = "#7F77DD", alpha = 0.60, size = 2) +
  geom_hline(yintercept = 0, color = "#D85A30",
             linewidth = 0.8, linetype = "dashed") +
  geom_smooth(method = "loess", se = TRUE, formula = y ~ x,
              color = "#0F6E56", fill = "#9FE1CB", alpha = 0.3, linewidth = 0.7) +
  labs(title = "Analisis de residuos - RNA",
       subtitle = "Sin patron sistematico = buen ajuste",
       x = "Resistencia predicha (MPa)", y = "Residuo (real - predicho)") +
  theme_minimal(base_size = 12)
print(p_res)


# -----------------------------------------------------------------------------
# 9. DIAGNOSTICO DE OVERFITTING
# -----------------------------------------------------------------------------
pred_train_norm <- predict(rna_model, newdata = train[, vars_entrada], rep = mejor_rep)
pred_train_norm <- as.numeric(pred_train_norm)
pred_train_mpa  <- pred_train_norm * (strength_max - strength_min) + strength_min
r2_train        <- r2(real_train_mpa, pred_train_mpa)

cat("\n=== DIAGNOSTICO DE OVERFITTING ===\n")
cat("R2 entrenamiento:", round(r2_train, 4), "\n")
cat("R2 test:         ", round(r2_rna,   4), "\n")
cat("Diferencia:      ", round(r2_train - r2_rna, 4), "\n")

if (r2_train - r2_rna > 0.05) {
  cat("ADVERTENCIA: posible sobreajuste (diferencia > 0.05).\n")
  cat("Sugerencia: reducir capas/neuronas o aumentar threshold.\n")
} else {
  cat("Sin senales de sobreajuste significativo.\n")
}


# -----------------------------------------------------------------------------
# 10. RESUMEN FINAL
# -----------------------------------------------------------------------------
cat("\n")
cat("=================================================================\n")
cat("               RESUMEN FINAL - WORKSHOP 4 RNA                    \n")
cat("=================================================================\n")
cat(sprintf("Dataset:      Yeh (1998) - %d mezclas, 8 variables\n", nrow(concrete)))
cat(sprintf("Arquitectura: 8 entradas -> [8, 4] -> 1 salida\n"))
cat(sprintf("Activacion:   Logistica (ocultas) + Lineal (salida)\n\n"))
cat(sprintf("%-20s %8s %10s %10s\n", "Modelo", "R2", "RMSE(MPa)", "MAE(MPa)"))
cat(sprintf("%-20s %8.4f %10.3f %10.3f\n", "Regresion Lineal", r2_lm,  rmse_lm,  mae_lm))
cat(sprintf("%-20s %8.4f %10.3f %10.3f\n", "RNA (8-4)",        r2_rna, rmse_rna, mae_rna))
cat("=================================================================\n")
cat(sprintf("\nMejora en R2  : %+.4f  (%+.1f%%)\n",
            r2_rna - r2_lm, (r2_rna - r2_lm) / r2_lm * 100))
cat(sprintf("Reduccion RMSE: %.3f MPa  (%+.1f%%)\n",
            rmse_lm - rmse_rna, (rmse_lm - rmse_rna) / rmse_lm * 100))
cat("\nInterpretacion:\n")
cat("  La RNA captura las interacciones no lineales entre cemento,\n")
cat("  ceniza volante, escoria y edad, logrando una mejora sobre la\n")
cat("  regresion lineal, que asume efectos aditivos independientes.\n")

# =============================================================================
# FIN DEL SCRIPT
# =============================================================================