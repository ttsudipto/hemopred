library(ggplot2)
library(patchwork)
library(magick)

plot_theme <- theme(
    plot.title=element_text(size=16, face="bold"),
    plot.title.position="plot",
)

cm_mc_cv <- image_read_pdf('output/affinity/ac_cm_euclidean_rb_mc_cv.pdf', density = 1000)
cm_mc_cv_plot <- image_ggplot(cm_mc_cv)
cm_mc_cv_plot <- cm_mc_cv_plot +
    ggtitle("a.") + plot_theme
cm_bin_cv <- image_read_pdf('output/affinity/ac_cm_euclidean_rb_bin_cv.pdf', density = 1000)
cm_bin_cv_plot <- image_ggplot(cm_bin_cv)
cm_bin_cv_plot <- cm_bin_cv_plot +
    ggtitle("b.") + plot_theme
# svg(filename=paste0("ac_cm_euclidean_rb_cv.svg"), height=7, width=11)
cm_mc_cv_plot + cm_bin_cv_plot + plot_layout(widths = c(7,4))
# dev.off()

cm_mc_cv <- image_read_pdf('output/affinity/ac_cm_euclidean_rb_mc_blind.pdf', density = 1000)
cm_mc_cv_plot <- image_ggplot(cm_mc_cv)
cm_mc_cv_plot <- cm_mc_cv_plot +
    ggtitle("a.") + plot_theme
cm_bin_cv <- image_read_pdf('output/affinity/ac_cm_euclidean_rb_bin_blind.pdf', density = 1000)
cm_bin_cv_plot <- image_ggplot(cm_bin_cv)
cm_bin_cv_plot <- cm_bin_cv_plot +
    ggtitle("b.") + plot_theme
# svg(filename=paste0("ac_cm_euclidean_rb_blind.svg"), height=7, width=11)
cm_mc_cv_plot + cm_bin_cv_plot + plot_layout(widths = c(7,4))
# dev.off()

cm_mc_cv <- image_read_pdf('output/affinity/ac_cm_euclidean_rb_mc_moepred_set_a.pdf', density = 1000)
cm_mc_cv_plot <- image_ggplot(cm_mc_cv)
cm_mc_cv_plot <- cm_mc_cv_plot +
    ggtitle("a.") + plot_theme
cm_bin_cv <- image_read_pdf('output/affinity/ac_cm_euclidean_rb_bin_moepred_set_a.pdf', density = 1000)
cm_bin_cv_plot <- image_ggplot(cm_bin_cv)
cm_bin_cv_plot <- cm_bin_cv_plot +
    ggtitle("b.") + plot_theme
# svg(filename=paste0("ac_cm_euclidean_rb_moepred_set_a.svg"), height=7, width=11)
cm_mc_cv_plot + cm_bin_cv_plot + plot_layout(widths = c(7,4))
# dev.off()

cm_mc_cv <- image_read_pdf('output/affinity/ac_cm_euclidean_rb_mc_moepred_set_b.pdf', density = 1000)
cm_mc_cv_plot <- image_ggplot(cm_mc_cv)
cm_mc_cv_plot <- cm_mc_cv_plot +
    ggtitle("a.") + plot_theme
cm_bin_cv <- image_read_pdf('output/affinity/ac_cm_euclidean_rb_bin_moepred_set_b.pdf', density = 1000)
cm_bin_cv_plot <- image_ggplot(cm_bin_cv)
cm_bin_cv_plot <- cm_bin_cv_plot +
    ggtitle("b.") + plot_theme
# svg(filename=paste0("ac_cm_euclidean_rb_moepred_set_b.svg"), height=7, width=11)
cm_mc_cv_plot + cm_bin_cv_plot + plot_layout(widths = c(7,4))
# dev.off()
