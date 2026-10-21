library(ggplot2)
library(patchwork)
library(magick)

plot_theme <- theme(
    plot.title=element_text(size=14, face="bold"),
    plot.title.position="plot",
)

shap_global_hba2 <- image_read_pdf('output/explanation/shap_global_hba2_moe.pdf', density = 1000)
shap_global_hba2_plot <- image_ggplot(shap_global_hba2)
shap_global_hba2_plot <- shap_global_hba2_plot +
    ggtitle("a.") + plot_theme

shap_cohort_hba2 <- image_read_pdf('output/explanation/shap_cohort_hba2_moe.pdf', density = 1000)
shap_cohort_hba2_plot <- image_ggplot(shap_cohort_hba2)
shap_cohort_hba2_plot <- shap_cohort_hba2_plot +
    ggtitle("c.") + plot_theme

shap_global_hbf <- image_read_pdf('output/explanation/shap_global_hbf_moe.pdf', density = 1000)
shap_global_hbf_plot <- image_ggplot(shap_global_hbf)
shap_global_hbf_plot <- shap_global_hbf_plot +
    ggtitle("b.") + plot_theme

shap_cohort_hbf <- image_read_pdf('output/explanation/shap_cohort_hbf_moe.pdf', density = 1000)
shap_cohort_hbf_plot <- image_ggplot(shap_cohort_hbf)
shap_cohort_hbf_plot <- shap_cohort_hbf_plot +
    ggtitle("d.") + plot_theme

design <- "
  AB
  CD
"
# svg(filename=paste0("shap_moe.svg"), height=9, width=8)
(shap_global_hba2_plot + shap_global_hbf_plot + shap_cohort_hba2_plot + shap_cohort_hbf_plot) + 
    plot_layout(design=design, heights = c(3,6), widths = c(4,4))
# dev.off()
