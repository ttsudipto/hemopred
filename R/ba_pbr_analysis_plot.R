library(mcr)
library(boot)
library(ggplot2)
library(patchwork)
library(ggpubr)

plot_theme <- theme(
    plot.title=element_text(size=16, face="bold"),
    plot.title.position="plot",
    panel.background=element_rect(fill="white"), 
    panel.grid.major.x=element_line(color="grey90"),
    panel.grid.major.y=element_line(color="grey90"),
    panel.border=element_rect(color="black", fill=NA),
    axis.title=element_text(size=13), 
    axis.text.y=element_text(size=12, color="gray10",), 
    axis.text.x=element_text(size=12, color="gray10",), 
    axis.ticks.length=unit(5, "pt")
)
legend_theme <- theme(
    legend.title=element_blank(),
    legend.text=element_text(size=12),
    legend.spacing.y=unit(7, "pt"),
    # legend.background=element_rect(fill="gray95"),
    legend.direction="horizontal",
    legend.position="bottom",
    legend.margin=margin(t=0,r=0,b=0,l=0),
    legend.key.width = unit(15, "pt")
)

class_order <- c(
    "Beta Thalassemia Trait",
    "Normal",
    "HbD Punjab Trait",
    "Sickle Cell Trait",
    "Borderline high HbA2",
    "Compound or heterogeneous"
)
# mapped_class_order <- c(
#     "Beta Thalassemia Trait (n=453)",
#     "Normal (n=4729)",
#     "HbD Punjab Trait (n=48)",
#     "Sickle Cell Trait (n=21)",
#     "Borderline HbA2 (n=58)",
#     "HbD Punjab/Beta Thalassemia DH (n=5)"
# )
mapped_class_order <- c(
    "Beta Thalassemia Trait",
    "Normal",
    "HbD Punjab Trait",
    "Sickle Cell Trait",
    "Borderline HbA2",
    "HbD Punjab/Beta Thalassemia DH"
)
label_map <- data.frame(Label=class_order, MappedLabel=mapped_class_order)

colors <- c(
    "#9467bd",
    "#1f77b4",
    "#2ca02c",
    "#bcbd22",
    "#ff7f0e",
    "#d62728"
    # '#8c564b', '#e377c2', '#7f7f7f', '#17becf'
)


data <- read.csv("output/affinity/ac_pred_euclidean_rb.tsv", sep="\t")


##### Bland-Altman Plot #####

df <- data.frame(
    Class = dplyr::left_join(data.frame(Label=data$Class), label_map)$MappedLabel,
    HbA2 = data$HbA2,
    HbF = data$HbF,
    Predicted_HbA2 = data$Predicted_HbA2,
    Predicted_HbF = data$Predicted_HbF
)
df$mean_HbA2 <- (df$HbA2 + df$Predicted_HbA2) / 2
df$diff_HbA2 <- df$HbA2 - df$Predicted_HbA2
df$mean_HbF <- (df$HbF + df$Predicted_HbF) / 2
df$diff_HbF <- df$HbF - df$Predicted_HbF

get_ba_stats <- function(data, indices, target_diff_col) {
    d_resampled <- data[indices, target_diff_col]
    
    bias <- mean(d_resampled)
    sd_diff <- sd(d_resampled)
    upper_loa <- bias + (1.96 * sd_diff)
    lower_loa <- bias - (1.96 * sd_diff)
    
    return(c(bias, upper_loa, lower_loa))
}

set.seed(42)
B_replicates <- 2000

boot_HbA2 <- boot(data = df, statistic = get_ba_stats, R = B_replicates, target_diff_col = "diff_HbA2")
bias_HbA2       <- boot_HbA2$t0[1]
upper_loa_HbA2  <- boot_HbA2$t0[2]
lower_loa_HbA2  <- boot_HbA2$t0[3]
ci_bias_HbA2    <- boot.ci(boot_HbA2, type = "perc", index = 1)$percent[4:5]
ci_upper_HbA2   <- boot.ci(boot_HbA2, type = "perc", index = 2)$percent[4:5]
ci_lower_HbA2   <- boot.ci(boot_HbA2, type = "perc", index = 3)$percent[4:5]

boot_HbF <- boot(data = df, statistic = get_ba_stats, R = B_replicates, target_diff_col = "diff_HbF")
bias_HbF       <- boot_HbF$t0[1]
upper_loa_HbF  <- boot_HbF$t0[2]
lower_loa_HbF  <- boot_HbF$t0[3]
ci_bias_HbF    <- boot.ci(boot_HbF, type = "perc", index = 1)$percent[4:5]
ci_upper_HbF   <- boot.ci(boot_HbF, type = "perc", index = 2)$percent[4:5]
ci_lower_HbF   <- boot.ci(boot_HbF, type = "perc", index = 3)$percent[4:5]

agreement_count_hba2 <- 0
agreement_count_hbf <- 0
for(i in 1:nrow(df)) {
    if (df$diff_HbA2[i] < upper_loa_HbA2 & df$diff_HbA2[i] > lower_loa_HbA2)
        agreement_count_hba2 = agreement_count_hba2 + 1
    if (df$diff_HbF[i] < upper_loa_HbF & df$diff_HbF[i] > lower_loa_HbF)
        agreement_count_hbf = agreement_count_hbf + 1
}
agreement_fraction_hba2 <- agreement_count_hba2 / nrow(df)
agreement_fraction_hbf <- agreement_count_hbf / nrow(df)


ba_plot_hba2 <- ggplot(df, aes(x = mean_HbA2, y = diff_HbA2, color = factor(Class, levels=mapped_class_order))) +
    geom_point(alpha = 0.3, size = 2, position=position_jitter(width = 0.2, height = 0.2, seed = 0)) +

    # horizontal reference lines
    geom_hline(yintercept = bias_HbA2, color = "darkred", linetype = "solid", linewidth = 0.8) +
    geom_hline(yintercept = upper_loa_HbA2, color = "black", linetype = "dashed") +
    geom_hline(yintercept = lower_loa_HbA2, color = "black", linetype = "dashed") +
    # geom_hline(yintercept = 0, color = "gray60", linetype = "dotted") +

    # text annotations for lines
    annotate("text", x = max(df$mean_HbA2), y = bias_HbA2, label = paste("Mean:", round(bias_HbA2, 4), "[95% CI: ", round(ci_bias_HbA2[1], 4), " - ", round(ci_bias_HbA2[2], 4), "]"), vjust = 1.5, hjust = 0.82) +
    annotate("text", x = min(df$mean_HbA2), y = upper_loa_HbA2, label = paste("+1.96 SD:", round(upper_loa_HbA2, 4), "\n[95% CI: ", round(ci_upper_HbA2[1], 4), " - ", round(ci_upper_HbA2[2], 4), "]"), vjust = -0.3, hjust = 0.11) +
    annotate("text", x = max(df$mean_HbA2), y = lower_loa_HbA2, label = paste("-1.96 SD:", round(lower_loa_HbA2, 4), "\n[95% CI: ", round(ci_lower_HbA2[1], 4), " - ", round(ci_lower_HbA2[2], 4), "]"), vjust = 1.5, hjust = 0.75) +
    annotate("text", x = min(df$mean_HbA2), y = 4, label = paste("Proportion of samples within Limits of Agreement (LoA):", round(agreement_fraction_hba2 * 100, 2), "%"), size = 4.5, vjust = 1.5, hjust = 0.03) +

    # Styling
    # scale_color_brewer(palette = "Set2", direction = 1) +
    scale_color_manual(values = colors) +
    scale_y_continuous(limits = c(-2.35, 4), n.breaks = 6) +
    labs(title = "a.", x = "Mean of HbA2 % measurements (HPLC and optimal MoE model)", y = "Difference in HbA2 % measurements (HPLC - Optimal MoE model)") +
    plot_theme + legend_theme #+ guides(fill=guide_legend(nrow=3,byrow=TRUE))

ba_plot_hba2

ba_plot_hbf <- ggplot(df, aes(x = mean_HbF, y = diff_HbF, color = factor(Class, levels=mapped_class_order))) +
    geom_point(alpha = 0.3, size = 2, position=position_jitter(width = 0.2, height = 0.2, seed = 0)) +

    # horizontal reference lines
    geom_hline(yintercept = bias_HbF, color = "darkred", linetype = "solid", linewidth = 0.8) +
    geom_hline(yintercept = upper_loa_HbF, color = "black", linetype = "dashed") +
    geom_hline(yintercept = lower_loa_HbF, color = "black", linetype = "dashed") +
    # geom_hline(yintercept = 0, color = "gray60", linetype = "dotted") +

    # text annotations for lines
    annotate("text", x = max(df$mean_HbF), y = bias_HbF, label = paste("Mean:", round(bias_HbF, 4), "[95% CI: ", round(ci_bias_HbF[1], 4), " - ", round(ci_bias_HbF[2], 4), "]"), vjust = 1.5, hjust = 0.82) +
    annotate("text", x = min(df$mean_HbF), y = upper_loa_HbF, label = paste("+1.96 SD:", round(upper_loa_HbF, 4), "\n[95% CI: ", round(ci_upper_HbF[1], 4), " - ", round(ci_upper_HbF[2], 4), "]"), vjust = -0.3, hjust = 0.3) +
    annotate("text", x = max(df$mean_HbF), y = lower_loa_HbF, label = paste("-1.96 SD:", round(lower_loa_HbF, 4), "[95% CI: ", round(ci_lower_HbF[1], 4), " - ", round(ci_lower_HbF[2], 4), "]"), vjust = 1.5, hjust = 0.83) +
    annotate("text", x = min(df$mean_HbF), y = 4, label = paste("Proportion of samples within Limits of Agreement (LoA):", round(agreement_fraction_hbf * 100, 2), "%"), size = 4.5, vjust = 1.5, hjust = 0.1) +

    # Styling
    # scale_color_brewer(palette = "Set2", direction = 1) +
    scale_color_manual(values = colors) +
    scale_y_continuous(limits = c(-2.35, 4), n.breaks = 6) +
    labs(title = "b.", x = "Mean of HbF % measurements (HPLC and optimal MoE model)", y = "Difference in HbF measurements (HPLC - Optimal MoE model)") +
    plot_theme + legend_theme #+ guides(fill=guide_legend(nrow=3,byrow=TRUE))

ba_plot_hbf

ba_plot_hba2 + ba_plot_hbf


##### Passing-Bablock Regression Plot #####

df_hba2 <- data.frame(
    Ref = data$HbA2,
    New = data$Predicted_HbA2
)
df_hba2 <- data.frame(lapply(df_hba2, jitter))
df_hba2$Class <- dplyr::left_join(data.frame(Label=data$Class), label_map)$MappedLabel

# mcr_model_hba2 <- mcreg(
#     x = df_hba2$Ref,
#     y = df_hba2$New,
#     method.reg = "PaBaLarge",
#     method.ci  = "bootstrap",
#     method.bootstrap.ci = "quantile",
#     NBins = 1000,
#     rng.seed = 42,
#     rng.kind = "Mersenne-Twister"
# )
# saveRDS(mcr_model_hba2, "mcr_model_PaBaLarge_hba2.rds")
mcr_model_hba2 <- readRDS("output/models/mcr_pb_reg/mcr_model_PaBaLarge_hba2.rds")
print(summary(mcr_model_hba2))

summary_matrix_hba2 <- mcr_model_hba2@para
intercept_est_hba2 <- summary_matrix_hba2["Intercept", "EST"]
slope_est_hba2 <- summary_matrix_hba2["Slope", "EST"]
intercept_lci_hba2 <- summary_matrix_hba2["Intercept", "LCI"]
intercept_uci_hba2 <- summary_matrix_hba2["Intercept", "UCI"]
slope_lci_hba2 <- summary_matrix_hba2["Slope", "LCI"]
slope_uci_hba2 <- summary_matrix_hba2["Slope", "UCI"]
r_value_hba2 <- cor(df_hba2$Ref, df_hba2$New, method = "pearson")

annotation_label_hba2 <- paste0(
    "Y = ", round(intercept_est_hba2, 3), " + ", round(slope_est_hba2, 3), " * X\n",
    "Intercept (A) = ", round(intercept_est_hba2, 3), " [95% CI: ", round(intercept_lci_hba2, 4), " - ", round(intercept_uci_hba2, 4), "]\n",
    "Slope (B) = ", round(slope_est_hba2, 3), " [95% CI: ", round(slope_lci_hba2, 4), " - ", round(slope_uci_hba2, 4), "]\n"
    # "Pearson correlation coefficient = ", round(r_value_hba2, 4)
)
pb_plot_hba2 <- ggplot() +
    geom_point(data = df_hba2, aes(x = Ref, y = New, color = factor(Class, levels=mapped_class_order)), alpha = 0.4, size = 2, position=position_jitter(width = 0.2, height = 0.2, seed = 0)) +
    geom_abline(intercept = intercept_est_hba2, slope = slope_est_hba2, color = "darkred", size = 0.8) +
    # geom_abline(intercept = intercept_uci_hba2, slope = slope_uci_hba2, color = "#3498db", linetype = "dotdash", size = 0.8) +
    # geom_abline(intercept = intercept_lci_hba2, slope = slope_lci_hba2, color = "#3498db", linetype = "dotdash", size = 0.8) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", size = 0.5) +
    # annotate("rect", xmin = 0.1, xmax = 3.5, ymin = 6.7, ymax = 7.9, fill = "white", color = "gray30", alpha = 0.85, size = 0.4) +
    annotate("text", x = 0.1, y = 7.8, hjust = 0, vjust = 1, size = 4.5, parse = FALSE, label = annotation_label_hba2) +
    labs(title = "c.", x = "HPLC-derived HbA2 % (X)", y = "Optimal MoE model-derived HbA2 % (Y)") +
    scale_color_manual(values = colors) +
    scale_x_continuous(limits = c(0,8)) +
    scale_y_continuous(limits = c(0,8)) +
    plot_theme + legend_theme #+ guides(fill=guide_legend(nrow=3,byrow=TRUE))
pb_plot_hba2


df_hbf <- data.frame(
    Ref = data$HbF,
    New = data$Predicted_HbF
)
df_hbf <- data.frame(lapply(df_hbf, jitter))
df_hbf$Class <- dplyr::left_join(data.frame(Label=data$Class), label_map)$MappedLabel

# mcr_model_hbf <- mcreg(
#     x = df_hbf$Ref,
#     y = df_hbf$New,
#     method.reg = "PaBaLarge",
#     method.ci  = "bootstrap",
#     method.bootstrap.ci = "quantile",
#     NBins = 1000,
#     rng.seed = 42,
#     rng.kind = "Mersenne-Twister"
# )
# saveRDS(mcr_model_hbf, "mcr_model_PaBaLarge_hbf.rds")
mcr_model_hbf <- readRDS("output/models/mcr_pb_reg/mcr_model_PaBaLarge_hbf.rds")
print(summary(mcr_model_hbf))

summary_matrix_hbf <- mcr_model_hbf@para
intercept_est_hbf <- summary_matrix_hbf["Intercept", "EST"]
slope_est_hbf <- summary_matrix_hbf["Slope", "EST"]
intercept_lci_hbf <- summary_matrix_hbf["Intercept", "LCI"]
intercept_uci_hbf <- summary_matrix_hbf["Intercept", "UCI"]
slope_lci_hbf <- summary_matrix_hbf["Slope", "LCI"]
slope_uci_hbf <- summary_matrix_hbf["Slope", "UCI"]
r_value_hbf <- cor(df_hbf$Ref, df_hbf$New, method = "pearson")

annotation_label_hbf <- paste0(
    "Y = ", round(intercept_est_hbf, 3), " + ", round(slope_est_hbf, 3), " * X\n",
    "Intercept (A) = ", round(intercept_est_hbf, 3), " [95% CI: ", round(intercept_lci_hbf, 4), " - ", round(intercept_uci_hbf, 4), "]\n",
    "Slope (B) = ", round(slope_est_hbf, 3), " [95% CI: ", round(slope_lci_hbf, 4), " - ", round(slope_uci_hbf, 4), "]\n"
    # "Pearson correlation coefficient = ", round(r_value_hbf, 4)
)
pb_plot_hbf <- ggplot() +
    geom_point(data = df_hbf, aes(x = Ref, y = New, color = factor(Class, levels=mapped_class_order)), alpha = 0.4, size = 2, position=position_jitter(width = 0.2, height = 0.2, seed = 0)) +
    geom_abline(intercept = intercept_est_hbf, slope = slope_est_hbf, color = "darkred", size = 0.8) +
    # geom_abline(intercept = intercept_uci_hbf, slope = slope_uci_hbf, color = "#3498db", linetype = "dotdash", size = 0.8) +
    # geom_abline(intercept = intercept_lci_hbf, slope = slope_lci_hbf, color = "#3498db", linetype = "dotdash", size = 0.8) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", size = 0.5) +
    # annotate("rect", xmin = 0.1, xmax = 3.5, ymin = 3.1, ymax = 3.9, fill = "white", color = "gray30", alpha = 0.85, size = 0.4) +
    annotate("text", x = 0.05, y = 3.9, hjust = 0, vjust = 1, size = 4.5, parse = FALSE, label = annotation_label_hbf) +
    labs(title = "d.", x = "HPLC-derived HbF % (X)", y = "Optimal MoE model-derived HbF % (Y)") +
    scale_color_manual(values = colors) +
    scale_x_continuous(limits = c(0,4)) +
    scale_y_continuous(limits = c(0,4)) +
    plot_theme + legend_theme #+ guides(fill=guide_legend(nrow=3,byrow=TRUE))
pb_plot_hbf

pb_plot_hba2 + pb_plot_hbf

# # svg(filename=paste0("method_comparison.svg"), height=14, width=15)
ggarrange(ba_plot_hba2, ba_plot_hbf, pb_plot_hba2, pb_plot_hbf, ncol = 2, nrow = 2, common.legend = TRUE, legend="bottom")
# # dev.off()

