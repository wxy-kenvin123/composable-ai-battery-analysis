write_dca_ordination_svg <- function(model,
                                      file,
                                      xlim = c(-3, 3),
                                      ylim = c(-3, 3),
                                      width = 720,
                                      height = 576) {
  esc <- function(x) {
    x <- as.character(x)
    x <- gsub("&", "&amp;", x, fixed = TRUE)
    x <- gsub("<", "&lt;", x, fixed = TRUE)
    x <- gsub(">", "&gt;", x, fixed = TRUE)
    x <- gsub("\"", "&quot;", x, fixed = TRUE)
    x
  }
  fmt <- function(x) {
    sub("\\.?0+$", "", sprintf("%.3f", x))
  }
  sx <- function(x) plot_left + (x - xlim[1]) / diff(xlim) * plot_width
  sy <- function(y) plot_top + (ylim[2] - y) / diff(ylim) * plot_height
  clip_xy <- function(mat) {
    mat[mat[, 1] >= xlim[1] & mat[, 1] <= xlim[2] &
          mat[, 2] >= ylim[1] & mat[, 2] <= ylim[2], , drop = FALSE]
  }

  # Pixel-aligned frame for consistent raster export.
  # Half-pixel coordinates with 1-px strokes avoid uneven antialiasing
  # on the top/right versus bottom/left sides of the plot box.
  plot_left <- 72.5
  plot_top <- 43.5
  plot_width <- 619
  plot_height <- 460
  plot_right <- plot_left + plot_width
  plot_bottom <- plot_top + plot_height

  site_scores <- as.matrix(vegan::scores(model, display="sites", choices=1:2))
  term_scores <- as.matrix(vegan::scores(model, display="species", choices=1:2))
  site_scores <- clip_xy(site_scores)
  term_scores <- clip_xy(term_scores)

  x_ticks <- seq(xlim[1], xlim[2], by = 1)
  y_ticks <- seq(ylim[1], ylim[2], by = 1)
  site_r <- 1.55
  term_half <- 1.8

  lines <- c(
    '<?xml version="1.0" encoding="UTF-8"?>',
    sprintf('<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" viewBox="0 0 %s %s">',
            fmt(width), fmt(height), fmt(width), fmt(height)),
    '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    sprintf('<defs><clipPath id="plot-area"><rect x="%s" y="%s" width="%s" height="%s"/></clipPath></defs>',
            fmt(plot_left), fmt(plot_top), fmt(plot_width), fmt(plot_height)),
    '<g id="ticks" fill="none" stroke="#000000" stroke-width="1" stroke-linecap="butt" stroke-linejoin="miter" shape-rendering="crispEdges">'
  )

  for (x in x_ticks) {
    xp <- sx(x)
    lines <- c(lines,
               sprintf('<path d="M %s %s L %s %s"/>', fmt(xp), fmt(plot_bottom), fmt(xp), fmt(plot_bottom + 7.2)))
  }
  for (y in y_ticks) {
    yp <- sy(y)
    lines <- c(lines,
               sprintf('<path d="M %s %s L %s %s"/>', fmt(plot_left), fmt(yp), fmt(plot_left - 7.2), fmt(yp)))
  }
  lines <- c(lines, '</g>')

  if (0 >= xlim[1] && 0 <= xlim[2]) {
    xp0 <- sx(0)
    lines <- c(lines,
               '<g id="zero-lines" clip-path="url(#plot-area)" fill="none" stroke="#000000" stroke-width="0.75" stroke-dasharray="0.75 2.25">',
               sprintf('<path d="M %s %s L %s %s"/>', fmt(xp0), fmt(plot_top), fmt(xp0), fmt(plot_bottom)),
               '</g>')
  }
  if (0 >= ylim[1] && 0 <= ylim[2]) {
    yp0 <- sy(0)
    lines <- c(lines,
               '<g id="zero-lines-horizontal" clip-path="url(#plot-area)" fill="none" stroke="#000000" stroke-width="0.75" stroke-dasharray="0.75 2.25">',
               sprintf('<path d="M %s %s L %s %s"/>', fmt(plot_left), fmt(yp0), fmt(plot_right), fmt(yp0)),
               '</g>')
  }

  lines <- c(lines,
             '<g id="term-scores" clip-path="url(#plot-area)" fill="none" stroke="#DF536B" stroke-width="0.75" stroke-linecap="round">')
  if (nrow(term_scores) > 0) {
    for (i in seq_len(nrow(term_scores))) {
      xp <- sx(term_scores[i, 1])
      yp <- sy(term_scores[i, 2])
      lines <- c(lines,
                 sprintf('<path d="M %s %s L %s %s M %s %s L %s %s"/>',
                         fmt(xp - term_half), fmt(yp),
                         fmt(xp + term_half), fmt(yp),
                         fmt(xp), fmt(yp - term_half),
                         fmt(xp), fmt(yp + term_half)))
    }
  }
  lines <- c(lines, '</g>',
             '<g id="article-scores" clip-path="url(#plot-area)" fill="none" stroke="#000000" stroke-width="0.75">')
  if (nrow(site_scores) > 0) {
    for (i in seq_len(nrow(site_scores))) {
      lines <- c(lines,
                 sprintf('<circle cx="%s" cy="%s" r="%s"/>',
                         fmt(sx(site_scores[i, 1])),
                         fmt(sy(site_scores[i, 2])),
                         fmt(site_r)))
    }
  }
  lines <- c(lines, '</g>',
             sprintf('<rect x="%s" y="%s" width="%s" height="%s" fill="none" stroke="#000000" stroke-width="1" shape-rendering="crispEdges"/>',
                     fmt(plot_left), fmt(plot_top), fmt(plot_width), fmt(plot_height)),
             '<g id="editable-figure-text" fill="#000000">',
             '<text x="381.6" y="28.5" text-anchor="middle" font-size="16" font-family="Arial" font-weight="bold">Detrended Correspondence Analysis</text>',
             '<text x="381.6" y="558.7" text-anchor="middle" font-size="16" font-family="Arial" font-weight="bold">DCA1</text>',
             '<text transform="translate(25.9 273.6) rotate(-90)" text-anchor="middle" font-size="16" font-family="Arial" font-weight="bold">DCA2</text>')
  for (x in x_ticks) {
    lines <- c(lines,
               sprintf('<text x="%s" y="529.9" text-anchor="middle" font-size="12" font-family="Arial" font-weight="bold">%s</text>',
                       fmt(sx(x)), esc(x)))
  }
  for (y in y_ticks) {
    lines <- c(lines,
               sprintf('<text x="54.7" y="%s" text-anchor="end" dominant-baseline="central" font-size="12" font-family="Arial" font-weight="bold">%s</text>',
                       fmt(sy(y)), esc(y)))
  }
  lines <- c(lines, '</g>', '</svg>')
  writeLines(lines, file, useBytes = TRUE)
}
