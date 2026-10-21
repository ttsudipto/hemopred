library(ggplot2)
library(dplyr)
library(forcats)
library(patchwork)

plot_theme_bar <- theme(
    plot.title=element_text(size=16, face="bold"),
    plot.title.position="plot",
    panel.background=element_rect(fill="white"), 
    panel.grid.major.x=element_line(color="grey90"),
    panel.grid.major.y=element_line(color="grey90"),
    panel.border=element_rect(color="black", fill=NA),
    axis.title.y=element_text(size=13),
    axis.title.x=element_blank(),
    axis.text.y=element_text(size=12, color="gray10",), 
    axis.text.x=element_text(size=12, color="gray10",), 
    axis.ticks.length=unit(5, "pt")
)
plot_theme_box <- theme(
    plot.title=element_text(size=16, face="bold"),
    plot.title.position="plot",
    panel.background=element_rect(fill="white"), 
    panel.grid.major.x=element_line(color="grey90"),
    panel.grid.major.y=element_line(color="grey90"),
    panel.border=element_rect(color="black", fill=NA),
    axis.title=element_text(size=13), 
    axis.text.y=element_text(size=12, color="gray10",), 
    axis.text.x=element_text(size=12, color="gray10", angle=30, hjust=1), 
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
    legend.key.width = unit(30, "pt")
)

x_axis_order <- c(
    "Beta Thalassemia Trait",
    "Normal",
    "HbD Punjab Trait",
    "Sickle Cell Trait",
    "Borderline high HbA2",
    "Compound or heterogeneous"
)
mapped_x_axis_order <- c(
    "BTT (n=453)",
    "Normal (n=4729)",
    "HbD Punjab Trait (n=48)",
    "Sickle Cell Trait (n=21)",
    "Borderline HbA2 (n=58)",
    "HbD Punjab/Beta\nThalassemia (double\nheterozygous) (n=5)"
)
label_map <- data.frame(Label=x_axis_order, MappedLabel=mapped_x_axis_order)

# study_labels <- c(
#     'Normal',
#     'Beta Thalassemia Trait',
#     'Borderline high HbA2',
#     'HbD Punjab Trait',
#     'Sickle Cell Trait',
#     # 'HbE Trait',
#     # 'HbS Homozygous (Sickle cell disease)',
#     'HbS/Beta Thalassemia (double heterozygous)',
#     'HbD-Punjab/Beta Thalassemia (double heterozygous)',
#     'HbE Homozygous (HbE disease)',
#     'HbE/Beta Thalassemia (double heterozygous)',
#     'Delta-Beta Thalassemia Trait',
#     'HbQ India/Beta Thalassemia (double heterozygous)',
#     'HbE Trait with coexisting alpha thalassemia',
#     'HbS/HbD Punjab (double heterozygous)'
# )

performance <- read.csv("output/prediction/performance_comparison.tsv", sep="\t")
performance$Model[performance$Model == "Multiple regression"] <- "Optimal MR model"
performance$Model[performance$Model == "Mixture of Experts"] <- "Optimal MoE model"
performance$Validation <- gsub('\\s', '\n', performance$Validation)
performance_hba2 <- filter(performance, (Target == "HbA2"))
performance_hbf <- filter(performance, (Target == "HbF"))
model_order <- c("Optimal MR model", "Optimal MoE model")

p_hba2_mae <- ggplot(performance_hba2, aes(x=Validation, y=MAE, fill=factor(Model, levels=model_order))) +
    geom_bar(position="dodge", stat="identity", alpha=1) +
    geom_text(aes(label=MAE), size = 4, vjust=1.6, position = position_dodge(0.9)) +
    ggtitle("a. MAE comparison for \n     HbA2 regression") +
    scale_fill_brewer(palette = "Accent", direction = -1) +
    plot_theme_bar + legend_theme + guides(fill=guide_legend(nrow=2,byrow=TRUE))
p_hba2_mae

p_hba2_r2 <- ggplot(performance_hba2, aes(x=Validation, y=R2_score, fill=factor(Model, levels=model_order))) +
    geom_bar(position="dodge", stat="identity", alpha=1) +
    geom_text(aes(label=R2_score), size = 4, vjust=1.6, position = position_dodge(0.9)) +
    ggtitle("b. R\u00B2 score comparison for \n     HbA2 regression") + ylab(bquote(R^2~"score")) +
    scale_fill_brewer(palette = "Accent", direction = -1) +
    plot_theme_bar + legend_theme + guides(fill=guide_legend(nrow=2,byrow=TRUE))
p_hba2_r2

p_hbf_mae <- ggplot(performance_hbf, aes(x=Validation, y=MAE, fill=factor(Model, levels=model_order))) +
    geom_bar(position="dodge", stat="identity", alpha=1) +
    geom_text(aes(label=MAE), size = 4, vjust=1.6, position = position_dodge(0.9)) +
    ggtitle("c. MAE comparison for \n     HbF regression") +
    scale_fill_brewer(palette = "Accent", direction = -1) +
    plot_theme_bar + legend_theme + guides(fill=guide_legend(nrow=2,byrow=TRUE))
p_hbf_mae

p_hbf_r2 <- ggplot(performance_hbf, aes(x=Validation, y=R2_score, fill=factor(Model, levels=model_order))) +
    geom_bar(position="dodge", stat="identity", alpha=1) +
    geom_text(aes(label=R2_score), size = 4, vjust=1.6, position = position_dodge(0.9)) +
    ggtitle("d. R\u00B2 score comparison for \n     HbF regression") + ylab(bquote(R^2~"score")) +
    scale_fill_brewer(palette = "Accent", direction = -1) +
    plot_theme_bar + legend_theme + guides(fill=guide_legend(nrow=2,byrow=TRUE))
p_hbf_r2

pred_hba2_moe <- read.csv("output/prediction/prediction_hba2_moe.tsv", sep="\t")
mae_hba2_moe <- pred_hba2_moe %>%
    group_by(Label) %>%
    summarize(MAE = mean(abs(HbA2 - Predicted.HbA2)),
              PCC = round(cor(HbA2, Predicted.HbA2), 2),
              y_pos_mae = max(c(max(HbA2), max(Predicted.HbA2))) + 0.7,
              y_pos_pcc = max(c(max(HbA2), max(Predicted.HbA2))) + 0.2)
mae_hba2_moe$Label <- dplyr::left_join(data.frame(Label=mae_hba2_moe$Label), label_map)$MappedLabel
pred_hbf_moe <- read.csv("output/prediction/prediction_hbf_moe.tsv", sep="\t")
mae_hbf_moe <- pred_hbf_moe %>%
    group_by(Label) %>%
    summarize(MAE = mean(abs(HbF - Predicted.HbF)),
              PCC = round(cor(HbF, Predicted.HbF), 2),
              y_pos_mae = max(c(max(HbF), max(Predicted.HbF))) + 0.4,
              y_pos_pcc = max(c(max(HbF), max(Predicted.HbF))) + 0.1)
mae_hbf_moe$Label <- dplyr::left_join(data.frame(Label=mae_hbf_moe$Label), label_map)$MappedLabel
pred_hba2_mr <- read.csv("output/prediction/prediction_hba2_mr.tsv", sep="\t")
mae_hba2_mr <- pred_hba2_mr %>%
    group_by(Label) %>%
    summarize(MAE = mean(abs(HbA2 - Predicted.HbA2)),
              PCC = round(cor(HbA2, Predicted.HbA2), 2),
              y_pos_mae = max(c(max(HbA2), max(Predicted.HbA2))) + 0.7,
              y_pos_pcc = max(c(max(HbA2), max(Predicted.HbA2))) + 0.2)
mae_hba2_mr$Label <- dplyr::left_join(data.frame(Label=mae_hba2_mr$Label), label_map)$MappedLabel
pred_hbf_mr <- read.csv("output/prediction/prediction_hbf_mr.tsv", sep="\t")
mae_hbf_mr <- pred_hbf_mr %>%
    group_by(Label) %>%
    summarize(MAE = mean(abs(HbF - Predicted.HbF)),
              PCC = round(cor(HbF, Predicted.HbF), 2),
              y_pos_mae = max(c(max(HbF), max(Predicted.HbF))) + 0.4,
              y_pos_pcc = max(c(max(HbF), max(Predicted.HbF))) + 0.1)
mae_hbf_mr$Label <- dplyr::left_join(data.frame(Label=mae_hbf_mr$Label), label_map)$MappedLabel

mean_hba2 <- mean(pred_hba2_moe$HbA2)
mean_hbf <- mean(pred_hbf_moe$HbF)


value <- c(pred_hba2_mr$HbA2, pred_hba2_mr$Predicted.HbA2)
x <- c(pred_hba2_mr$Label, pred_hba2_mr$Label)
x <- dplyr::left_join(data.frame(Label=x), label_map)$MappedLabel
group <- c(rep('Original', nrow(pred_hba2_mr)), rep('Predicted', nrow(pred_hba2_mr)))
plot_data_hba2_mr <- data.frame(Group=group, Label=x, HbA2=value)
p_hba2_mr <- ggplot(plot_data_hba2_mr) +
    geom_boxplot(aes(x=factor(Label, levels=mapped_x_axis_order), y=HbA2, fill=as.factor(Group), color=as.factor(Group)), alpha=0.5, width = 0.8) +
    # geom_violin(aes(x=Label, y=HbA2, fill=as.factor(Group)), alpha=0.6, width = 0.8) +
    # geom_text(data = mae_hba2_mr, aes(x=Label, y=y_pos_mae, label=paste("MAE=", round(MAE, 2), sep="")), position = position_dodge(width = 0.75), vjust=0) +
    # geom_text(data = mae_hba2_mr, aes(x=Label, y=y_pos_pcc, label=paste("PCC=", PCC, sep="")), position = position_dodge(width = 0.75), vjust=0) +
    geom_hline(yintercept=mean_hba2, linetype="dashed", color='purple') +
    xlab(paste0("Label")) + ylab("HbA2 (%)") +
    ggtitle("e. HbA2 prediction by optimal MR model") +
    scale_fill_brewer(palette = "Set2", direction = 1) +
    scale_color_brewer(palette = "Set2", direction = 1) +
    # coord_flip() +
    plot_theme_box + legend_theme #+ guides(fill=guide_legend(nrow=4,byrow=TRUE))
p_hba2_mr

value <- c(pred_hbf_mr$HbF, pred_hbf_mr$Predicted.HbF)
x <- c(pred_hbf_mr$Label, pred_hbf_mr$Label)
x <- dplyr::left_join(data.frame(Label=x), label_map)$MappedLabel
group <- c(rep('Original', nrow(pred_hbf_mr)), rep('Predicted', nrow(pred_hbf_mr)))
plot_data_hbf_mr <- data.frame(Group=group, Label=x, HbF=value)
p_hbf_mr <- ggplot(plot_data_hbf_mr) +
    geom_boxplot(aes(x=factor(Label, levels=mapped_x_axis_order), y=HbF, fill=as.factor(Group), color=as.factor(Group)), alpha=0.5, width = 0.8) +
    # geom_violin(aes(x=Label, y=HbA2, fill=as.factor(Group)), alpha=0.6, width = 0.8) +
    # geom_text(data = mae_hbf_mr, aes(x=Label, y=y_pos_mae, label=paste("MAE=", round(MAE, 2), sep="")), position = position_dodge(width = 0.75), vjust=0) +
    # geom_text(data = mae_hbf_mr, aes(x=Label, y=y_pos_pcc, label=paste("PCC=", PCC, sep="")), position = position_dodge(width = 0.75), vjust=0) +
    geom_hline(yintercept=mean_hbf, linetype="dashed", color='purple') +
    xlab(paste0("Label")) + ylab("HbF (%)") +
    ggtitle("f. HbF prediction by optimal MR model") +
    scale_fill_brewer(palette = "Set2", direction = 1) +
    scale_color_brewer(palette = "Set2", direction = 1) +
    # coord_flip() +
    plot_theme_box + legend_theme #+ guides(fill=guide_legend(nrow=4,byrow=TRUE))
p_hbf_mr

value <- c(pred_hba2_moe$HbA2, pred_hba2_moe$Predicted.HbA2)
x <- c(pred_hba2_moe$Label, pred_hba2_moe$Label)
x <- dplyr::left_join(data.frame(Label=x), label_map)$MappedLabel
group <- c(rep('Original', nrow(pred_hba2_moe)), rep('Predicted', nrow(pred_hba2_moe)))
plot_data_hba2_moe <- data.frame(Group=group, Label=x, HbA2=value)
p_hba2_moe <- ggplot(plot_data_hba2_moe) +
    geom_boxplot(aes(x=factor(Label, levels=mapped_x_axis_order), y=HbA2, fill=as.factor(Group), color=as.factor(Group)), alpha=0.4, width = 0.8) +
    # geom_violin(aes(x=Label, y=HbA2, fill=as.factor(Group)), alpha=0.6, width = 0.8) +
    # geom_text(data = mae_hba2_moe, aes(x=Label, y=y_pos_mae, label=paste("MAE=", round(MAE, 2), sep="")), position = position_dodge(width = 0.75), vjust=0) +
    # geom_text(data = mae_hba2_moe, aes(x=Label, y=y_pos_pcc, label=paste("PCC=", PCC, sep="")), position = position_dodge(width = 0.75), vjust=0) +
    geom_hline(yintercept=mean_hba2, linetype="dashed", color='purple') +
    xlab(paste0("Label")) + ylab("HbA2 (%)") +
    ggtitle("g. HbA2 prediction by optimal MoE model") +
    scale_fill_brewer(palette = "Set1", direction = -1) +
    scale_color_brewer(palette = "Set1", direction = -1) +
    # coord_flip() +
    plot_theme_box + legend_theme #+ guides(fill=guide_legend(nrow=4,byrow=TRUE))
p_hba2_moe

value <- c(pred_hbf_moe$HbF, pred_hbf_moe$Predicted.HbF)
x <- c(pred_hbf_moe$Label, pred_hbf_moe$Label)
x <- dplyr::left_join(data.frame(Label=x), label_map)$MappedLabel
group <- c(rep('Original', nrow(pred_hbf_moe)), rep('Predicted', nrow(pred_hbf_moe)))
plot_data_hbf_moe <- data.frame(Group=group, Label=x, HbF=value)
p_hbf_moe <- ggplot(plot_data_hbf_moe) +
    geom_boxplot(aes(x=factor(Label, levels=mapped_x_axis_order), y=HbF, fill=as.factor(Group), color=as.factor(Group)), alpha=0.4, width = 0.8) +
    # geom_violin(aes(x=Label, y=HbA2, fill=as.factor(Group)), alpha=0.6, width = 0.8) +
    # geom_text(data = mae_hbf_moe, aes(x=Label, y=y_pos_mae, label=paste("MAE=", round(MAE, 2), sep="")), position = position_dodge(width = 0.75), vjust=0) +
    # geom_text(data = mae_hbf_moe, aes(x=Label, y=y_pos_pcc, label=paste("PCC=", PCC, sep="")), position = position_dodge(width = 0.75), vjust=0) +
    geom_hline(yintercept=mean_hbf, linetype="dashed", color='purple') +
    xlab(paste0("Label")) + ylab("HbF (%)") +
    ggtitle("h. HbF prediction by optimal MoE model") +
    # scale_fill_manual(values=c('darkturquoise', 'darkred')) +
    scale_fill_brewer(palette = "Set1", direction = -1) +
    scale_color_brewer(palette = "Set1", direction = -1) +
    # coord_flip() +
    plot_theme_box + legend_theme #+ guides(fill=guide_legend(nrow=4,byrow=TRUE))
p_hbf_moe

design <- "
  ABCD
  EEFF
  GGHH
"
# # svg(filename=paste0("prediction.svg"), height=15, width=12)
(p_hba2_mae + p_hba2_r2 + p_hbf_mae + p_hbf_r2 + p_hba2_mr + p_hbf_mr + p_hba2_moe + p_hbf_moe) +
    plot_layout(design=design, heights = c(3,6,6), widths = c(3,3,3,3))
# # dev.off()

