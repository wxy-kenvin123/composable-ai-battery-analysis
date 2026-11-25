suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
    library(svglite)
})

# Do not enable showtext: svglite would convert labels to glyph paths
# and Illustrator would lose editable <text>. Keep native Arial text.
windowsFonts(Arial = windowsFont("Arial"))
cli_args <- commandArgs(trailingOnly = TRUE)
svg_only <- "--svg-only" %in% cli_args

# Comparison variant of nightingale_rose_2026.R: every petal is split into an
# inner Article segment (full petal colour) and an outer Review segment (same
# hue, lighter). Petal height keeps the B2 transform (total^0.4); inside a
# petal the two segment lengths are proportional to the two counts. Labels add
# the review count as "R n". Input is annual_records_by_stream.csv
# (stream, year, doctype, records).

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_idx <- grep("^--file=", args)
  if (length(file_idx) > 0) {
    return(dirname(normalizePath(
      sub("^--file=", "", args[file_idx[1]]),
      winslash = "/", mustWork = TRUE
    )))
  }
  for (frame in sys.frames()) {
    if (!is.null(frame$ofile)) {
      return(dirname(normalizePath(
        frame$ofile, winslash = "/", mustWork = TRUE
      )))
    }
  }
  getwd()
}

this_dir <- get_script_dir()
sec_dir <- dirname(this_dir)
pos_args <- cli_args[!startsWith(cli_args, "--")]
out_dir <- if (length(pos_args)) pos_args[1] else file.path(sec_dir, "rerun")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
input_csv <- file.path(this_dir, "annual_records_by_stream.csv")
if (!file.exists(input_csv)) {
  stop("Missing input file: ", input_csv, call. = FALSE)
}
stream_map <- c(anode = "Anode", cathode = "Cathode", electrolyte = "Electrolyte")
long <- read.csv(input_csv, stringsAsFactors = FALSE) %>%
  mutate(Category = unname(stream_map[stream]),
         year = as.integer(year),
         doctype = ifelse(doctype == "Review", "Review", "Article"),
         records = as.integer(records))

prep_data <- function(df, target_n = 18) {
  base <- df %>%
    mutate(year_group = as.character(year)) %>%
    group_by(year_group) %>%
    summarise(count = sum(records),
              n_review = sum(records[doctype == "Review"]),
              .groups = "drop") %>%
    mutate(year_order = as.numeric(year_group)) %>%
    arrange(year_order)

  n_years <- nrow(base)
  if (n_years > target_n) {
    n_merge <- n_years - target_n + 1
    merged_rows <- base[1:n_merge, ]
    merged_label <- paste0("\u2264", merged_rows$year_group[n_merge])
    rest <- base[(n_merge + 1):n_years, ]
    base <- bind_rows(
      tibble(year_group = merged_label,
             count = sum(merged_rows$count),
             n_review = sum(merged_rows$n_review),
             year_order = merged_rows$year_order[n_merge]),
      rest
    )
  }
  base %>% mutate(n_article = count - n_review)
}

# Same-hue palettes. Articles keep a sequential ramp (light = few papers,
# dark = many). Reviews are not a second hue: each petal's outer band is a
# lighter tint of that petal's own colour, so the split reads as depth, not
# as a complementary clash. `default` is written to the repository root;
# every palette is also rendered as PNG into rose_2026_palettes/.
tint_hex <- function(hex, mix = 0.30, toward = "#FFF8F2") {
  rgb <- col2rgb(hex) / 255
  dest <- as.numeric(col2rgb(toward) / 255)
  lum <- 0.2126 * rgb[1, ] + 0.7152 * rgb[2, ] + 0.0722 * rgb[3, ]
  mix_i <- ifelse(lum > 0.70, 0.12, mix)
  mixed <- sweep(rgb, 2, 1 - mix_i, `*`) + dest %o% mix_i
  grDevices::rgb(mixed[1, ], mixed[2, ], mixed[3, ])
}

# Fig.1 / manuscript cluster colours (C1–C8), used as sequential ramps so
# the split rose sits in the same family as the dendrogram, donuts and map.
#   C1 #8DB2D7  C2 #F9DA8E  C3 #79BED5  C4 #F8B62D
#   C5 #274C9F  C6 #E9504F  C7 #B7DFD9  C8 #FBDED7
# Warm = C8–C2–C4–C6 (blush / butter / gold / coral).
# Cool = C8–C1–C3–C5 (blush / powder / cyan / navy).
# High end is a one-step darkening of C6 / C5 so the 2026 petal is not flat.
PALETTES <- list(
  fig1_warm = list(
    ramp = c("#FBDED7", "#F9DA8E", "#F8B62D", "#E9504F", "#B83A3C"),
    ink = "#274C9F", tint = 0.28, toward = "#FBDED7", key = "Fig.1 C8–C2–C4–C6"),
  fig1_cool = list(
    ramp = c("#FBDED7", "#8DB2D7", "#79BED5", "#274C9F", "#1B3370"),
    ink = "#1B3370", tint = 0.28, toward = "#B7DFD9", key = "Fig.1 C8–C1–C3–C5")
)
DEFAULT_PALETTE <- "fig1_warm"

plot_rose_split <- function(pdata, title_text, total_n, pal) {
  n_bars <- nrow(pdata)
  colors <- colorRampPalette(pal$ramp)(n_bars)
  review_cols <- tint_hex(colors, pal$tint, pal$toward)
  INK <- pal$ink

  pdata <- pdata %>%
    mutate(
      x_factor = factor(year_group, levels = year_group),
      bar_height = count^0.4,
      h_article = bar_height * n_article / count,
      h_review = bar_height - h_article,
      color_idx = rank(count, ties.method = "first"),
      pct = sprintf("%.1f", count / total_n * 100),
      bar_idx = row_number()
    )

  max_h <- max(pdata$bar_height)
  inner_r <- max_h * 0.14
  lab_off <- max_h * 0.05
  lab_floor <- max_h * 0.31

  wt <- pdata$bar_height^3
  wm <- sum(pdata$bar_idx * wt) / sum(wt)
  rot_frac <- (wm - 0.5) / n_bars
  start_rad <- -2 * pi * rot_frac

  pdata <- pdata %>%
    mutate(
      spoke_angle = 90 - 360 * ((bar_idx - 0.5) / n_bars) + 360 * rot_frac,
      norm_angle = ((spoke_angle + 180) %% 360) - 180,
      right_side = norm_angle > -90 & norm_angle < 90,
      lab_hjust = ifelse(right_side, 0, 1),
      label_inside = bar_height > max_h * 0.62,
      # Same label text as the main figure; the review count is carried by the
      # outer band only, not written out.
      label_text = ifelse(label_inside,
                          paste0(year_group, "\n", count, "\n(", pct, "%)"),
                          paste0(year_group, " ", count, "(", pct, "%)")),
      # Centre the inside label on the (dark) Article segment so the white
      # text never sits on the lighter Review band.
      inlab_y = pmax(h_article * 0.55, bar_height * 0.30),
      outlab_y = pmax(bar_height + lab_off, lab_floor),
      need_leader = outlab_y - bar_height > max_h * 0.06
    )

  # Long table for the stacked petals: Article inner, Review outer.
  # Review reuses the petal's own hue at a lighter tint (no second colour).
  seg <- bind_rows(
    pdata %>% transmute(x_factor, bar_idx, segment = "Article", h = h_article,
                        fill_key = paste0("A", color_idx)),
    pdata %>% transmute(x_factor, bar_idx, segment = "Review", h = h_review,
                        fill_key = paste0("R", color_idx))
  ) %>%
    mutate(segment = factor(segment, levels = c("Review", "Article")))
  fill_values <- c(
    setNames(colors, paste0("A", seq_len(n_bars))),
    setNames(review_cols, paste0("R", seq_len(n_bars)))
  )

  # Inside labels sit on the Article segment; light early petals get ink text.
  pdata <- pdata %>%
    mutate(inlab_col = ifelse(color_idx > n_bars * 0.45, "white", INK))

  name_size <- 2.3

  p <- ggplot() +
    annotate("rect", xmin = 0.5, xmax = n_bars + 0.5,
             ymin = -inner_r, ymax = 0, fill = "white", colour = NA) +
    geom_col(data = seg,
             aes(x = x_factor, y = h, fill = fill_key, group = segment),
             width = 1, position = position_stack(),
             colour = "white", linewidth = 0.3) +
    geom_segment(
      data = filter(pdata, !label_inside & need_leader),
      aes(x = bar_idx, xend = bar_idx, y = bar_height, yend = outlab_y),
      colour = "grey70", linewidth = 0.22
    ) +
    geom_text(
      data = filter(pdata, label_inside),
      aes(x = x_factor, y = inlab_y, label = label_text, colour = inlab_col),
      size = 1.8, lineheight = 0.82,
      fontface = "bold", hjust = 0.5, family = "Arial"
    ) +
    geom_text(
      data = filter(pdata, !label_inside),
      aes(x = x_factor, y = outlab_y, label = label_text, hjust = lab_hjust),
      angle = 0, color = INK, size = 2.0, lineheight = 0.82,
      fontface = "bold", family = "Arial"
    ) +
    coord_polar(theta = "x", start = start_rad, clip = "off") +
    scale_fill_manual(values = fill_values, guide = "none") +
    scale_colour_identity(guide = "none") +
    scale_y_continuous(limits = c(-inner_r, max_h * 1.18), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(
      legend.position = "none",
      plot.margin = margin(6, 6, 6, 6),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    ) +
    annotate("text", x = (n_bars + 1) / 2, y = -inner_r,
             label = paste0(title_text, "\n", total_n),
             size = name_size, color = INK, fontface = "bold",
             family = "Arial", lineheight = 0.95)

  p
}

# Same cropping as the main script, plus a one-line key drawn into the
# bottom-left corner of the cropped page (outside the polar panel).
save_cropped <- function(p, file, device_fun, side_in, crop_bottom, crop_top = 0,
                         key_text = NULL) {
  grob <- ggplot2::ggplotGrob(p)
  device_fun(file, side_in, side_in * (1 - crop_bottom - crop_top))
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(
    y = grid::unit(1, "npc") + grid::unit(crop_top * side_in, "in"),
    just = "top",
    width = grid::unit(side_in, "in"), height = grid::unit(side_in, "in")
  ))
  grid::grid.draw(grob)
  grid::popViewport()
  if (!is.null(key_text)) {
    grid::grid.text(key_text,
                    x = grid::unit(0.05, "in"), y = grid::unit(0.06, "in"),
                    just = c("left", "bottom"),
                    gp = grid::gpar(fontfamily = "Arial", fontsize = 6, col = "grey35"))
  }
  grDevices::dev.off()
}

data_list <- list(
  list(df = filter(long, Category == "Electrolyte"), name = "Electrolyte"),
  list(df = filter(long, Category == "Cathode"),     name = "Cathode"),
  list(df = filter(long, Category == "Anode"),       name = "Anode")
)

SIDE_IN <- 6
CROP_BOTTOM <- 0.22
CROP_TOP <- 0.09
key_for <- function(pal) {
  paste0("Inner: Article   Outer (same hue, lighter): Review   ",
         pal$key, "   2026: indexed to 2026-09")
}

dev_png  <- function(f, w, h) grDevices::png(f, width = w, height = h,
                                             units = "in", res = 600,
                                             type = "cairo", bg = "white")
dev_tiff <- function(f, w, h) grDevices::tiff(f, width = w, height = h,
                                              units = "in", res = 600,
                                              compression = "lzw", bg = "white")
dev_pdf  <- function(f, w, h) grDevices::cairo_pdf(f, width = w, height = h,
                                                   bg = "white")
# svglite keeps real <text font-family="Arial"> nodes (editable in Illustrator).
dev_svg  <- function(f, w, h) svglite::svglite(f, width = w, height = h, bg = "white")

# Three roses in one row; each panel reuses the same vertical crop as the singles.
save_combined <- function(plots, file, device_fun, side_in, crop_bottom, crop_top,
                          key_text = NULL) {
  n <- length(plots)
  grobs <- lapply(plots, ggplot2::ggplotGrob)
  panel_h <- side_in * (1 - crop_bottom - crop_top)
  key_extra <- if (is.null(key_text)) 0 else 0.20
  device_fun(file, side_in * n, panel_h + key_extra)
  grid::grid.newpage()
  for (i in seq_along(grobs)) {
    grid::pushViewport(grid::viewport(
      x = grid::unit((i - 0.5) / n, "npc"),
      y = grid::unit(1, "npc") + grid::unit(crop_top * side_in, "in"),
      just = c("center", "top"),
      width = grid::unit(side_in, "in"),
      height = grid::unit(side_in, "in")
    ))
    grid::grid.draw(grobs[[i]])
    grid::popViewport()
  }
  if (!is.null(key_text)) {
    grid::grid.text(key_text,
                    x = grid::unit(0.08, "in"), y = grid::unit(0.08, "in"),
                    just = c("left", "bottom"),
                    gp = grid::gpar(fontfamily = "Arial", fontsize = 7, col = "grey35"))
  }
  grDevices::dev.off()
}

pal_dir <- file.path(out_dir, "rose_2026_palettes")
dir.create(pal_dir, showWarnings = FALSE)

# Combined row order matches the reference triptych: Anode | Electrolyte | Cathode.
combined_order <- c("Anode", "Electrolyte", "Cathode")
plots_default <- vector("list", length(combined_order))
names(plots_default) <- combined_order

for (item in data_list) {
  pdata <- prep_data(item$df)
  total_n <- sum(pdata$count)

  pal <- PALETTES[[DEFAULT_PALETTE]]
  p <- plot_rose_split(pdata, item$name, total_n, pal)
  plots_default[[item$name]] <- p
  fname <- paste0("nightingale_rose_", tolower(item$name), "_2026_split")
  save_cropped(p, file.path(out_dir, paste0(fname, ".svg")),
               dev_svg, SIDE_IN, CROP_BOTTOM, CROP_TOP, key_for(pal))
  if (!svg_only) {
    save_cropped(p, file.path(out_dir, paste0(fname, ".pdf")),
                 dev_pdf, SIDE_IN, CROP_BOTTOM, CROP_TOP, key_for(pal))
    save_cropped(p, file.path(out_dir, paste0(fname, ".png")),
                 dev_png, SIDE_IN, CROP_BOTTOM, CROP_TOP, key_for(pal))
    save_cropped(p, file.path(out_dir, paste0(fname, ".tiff")),
                 dev_tiff, SIDE_IN, CROP_BOTTOM, CROP_TOP, key_for(pal))
  }
  cat("Saved:", fname, if (svg_only) "(svg only)" else "", "\n")

  if (!svg_only) {
    for (pn in names(PALETTES)) {
      pal_i <- PALETTES[[pn]]
      p_i <- plot_rose_split(pdata, item$name, total_n, pal_i)
      save_cropped(p_i, file.path(pal_dir, paste0(tolower(item$name), "_", pn, ".png")),
                   function(f, w, h) grDevices::png(f, width = w, height = h,
                                                    units = "in", res = 300,
                                                    type = "cairo", bg = "white"),
                   SIDE_IN, CROP_BOTTOM, CROP_TOP, key_for(pal_i))
    }
    cat("Saved palette variants for", item$name, "\n")
  }
}

if (any(vapply(plots_default, is.null, logical(1)))) {
  stop("Combined panel missing a category: ",
       paste(names(plots_default)[vapply(plots_default, is.null, logical(1))],
             collapse = ", "),
       call. = FALSE)
}

pal <- PALETTES[[DEFAULT_PALETTE]]
combined_stem <- "nightingale_rose_2026_split_combined"
save_combined(plots_default[combined_order],
              file.path(out_dir, paste0(combined_stem, ".svg")),
              dev_svg, SIDE_IN, CROP_BOTTOM, CROP_TOP, key_for(pal))
save_combined(plots_default[combined_order],
              file.path(out_dir, paste0(combined_stem, ".png")),
              function(f, w, h) grDevices::png(f, width = w, height = h,
                                               units = "in", res = 300,
                                               type = "cairo", bg = "white"),
              SIDE_IN, CROP_BOTTOM, CROP_TOP, key_for(pal))
cat("Saved:", combined_stem, "\n")
