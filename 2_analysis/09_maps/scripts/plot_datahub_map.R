###############################################################################
# Fig. 2e: global distribution of public training databases and author cities.
#   - each database is a ring split by the research clusters of the articles that
#     use it (the cluster colours of panels b, c and e), numbered in the middle;
#   - arcs in neutral grey, light at the database and dark at the country;
#   - key grouped by data type; arc-width samples under the city colour bar.
#
# Inputs (read only)
#   data_sources755/  dataset_catalog.csv, dataset_by_country.csv,
#                                             dataset_by_cluster.csv, paper_summary.csv
#   data_sources755_recall/names.csv   narrow exclusions
#   map755/data/city_counts.csv
#   data/hubs.csv                              host institutions, checked on the web
# Outputs
#   data/flows.csv, data/rings.csv, data/hub_positions.csv, data/country_anchors.csv,
#   data/summary.txt, svg/world_map_databases.svg
#
# Run: "Rscript" scripts/plot_datahub_map.R
###############################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
  library(ggplot2)
  library(sf)
  library(rnaturalearth)
  library(patchwork)
})

# Package version: inputs from 05_data_sources/data and 09_maps/data, outputs to rerun/ (or the folder given).
script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
SEC   <- dirname(script_dir())
ROOT  <- dirname(SEC)
args  <- commandArgs(trailingOnly = TRUE)
OUT   <- if (length(args)) args[1] else file.path(SEC, "rerun")
dir.create(file.path(OUT, "data"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT, "svg"), recursive = TRUE, showWarnings = FALSE)
FINAL <- file.path(ROOT, "05_data_sources", "data")
OLD   <- SEC
NAMES <- file.path(FINAL, "database_names.csv")

countries7     <- c("China", "United States", "South Korea", "Germany", "Japan", "United Kingdom", "India")
cluster_colors <- c("1" = "#8DB2D7", "2" = "#F9DA8E", "3" = "#79BED5", "4" = "#F8B62D",
                    "5" = "#274C9F", "6" = "#E9504F", "7" = "#B7DFD9", "8" = "#FBDED7")  # as in plot_world_map.R
group_order    <- c("Crystals & solids", "Molecules & liquids", "Battery cycling")

names_tab   <- read_csv(NAMES, col_types = cols(.default = "c"))
narrow_drop <- names_tab$dataset[names_tab$narrow_excluded == "True"]
hubs    <- read_csv(file.path(SEC, "data/hubs.csv"), show_col_types = FALSE)
catalog <- read_csv(file.path(FINAL, "dataset_catalog.csv"), show_col_types = FALSE)
bycty   <- read_csv(file.path(FINAL, "dataset_by_country.csv"), show_col_types = FALSE)
byclu   <- read_csv(file.path(FINAL, "dataset_by_cluster.csv"), col_types = cols(cluster = "c", n_papers = "i", .default = "c"))
papers  <- read_csv(file.path(FINAL, "paper_summary.csv"), col_types = cols(.default = "c"))
cities  <- read_csv(file.path(OLD, "data/city_counts.csv"), show_col_types = FALSE)

# ---------------------------------------------------------------- data checks
ranked <- catalog %>% filter(!dataset %in% narrow_drop) %>% arrange(desc(n_papers))
stopifnot(nrow(hubs) == 11, setequal(ranked$dataset[ranked$n_papers >= 5], hubs$dataset),
          max(ranked$n_papers[ranked$n_papers < 5]) == 4)                 # a clean cut below the eleventh
hubs <- hubs %>% left_join(catalog %>% select(dataset, n_papers), by = "dataset")

pub      <- papers %>% filter(public_db == "True")
sets     <- strsplit(pub$public_datasets, ";", fixed = TRUE)
narrow   <- !vapply(sets, function(d) all(d %in% narrow_drop), logical(1))
n_narrow <- sum(narrow)
n_hubs   <- sum(vapply(sets[narrow], function(d) any(d %in% hubs$dataset), logical(1)))
n_mp     <- sum(vapply(sets[narrow], function(d) "Materials Project database" %in% d, logical(1)))
stopifnot(nrow(papers) == 755, nrow(pub) == 270, n_narrow == 257, n_mp == 142)

rings <- byclu %>% filter(dataset %in% hubs$dataset) %>% rename(k = n_papers)
ring_check <- rings %>% group_by(dataset) %>% summarise(k = sum(k), .groups = "drop") %>%
  left_join(hubs %>% select(dataset, n_papers), by = "dataset")
stopifnot(nrow(ring_check) == 11, all(ring_check$k == ring_check$n_papers))   # one cluster per article

anchors <- cities %>%
  filter(country %in% countries7) %>%
  group_by(country) %>%
  summarise(ax = weighted.mean(longitude, Total), ay = weighted.mean(latitude, Total),
            city_papers = sum(Total), .groups = "drop")        # paper-weighted centre of the author cities
flows <- bycty %>%
  filter(dataset %in% hubs$dataset, country %in% countries7) %>%
  left_join(hubs %>% select(dataset, label, lon, lat), by = "dataset") %>%
  left_join(anchors %>% select(country, ax, ay), by = "country") %>%
  arrange(desc(n_papers))
stopifnot(nrow(flows) > 0, !anyNA(flows))

# ---------------------------------------------------------------- hub markers
# Numbered by data type, then by articles (ties alphabetical). Rings that would overlap,
# or would sit on the end point of a country's arcs, are moved a few degrees; the true
# positions stay in hubs.csv and the distance moved is written to hub_positions.csv.
hubs <- hubs %>%
  mutate(group = factor(group, levels = group_order)) %>%
  arrange(group, desc(n_papers), label) %>%
  mutate(rank = row_number())
nudge <- tribble(
  ~dataset,                            ~mlon,  ~mlat,
  "ZINC",                             -133.5,   44.0,
  "NASA",                             -132.5,   30.0,
  "PubChem",                           -74.0,   42.0,
  "AFLOW",                             -81.0,   32.5,
  "ICSD",                                7.0,   57.2,
  "QM9",                                 2.5,   44.2,
  "Liverpool Ionics Dataset (LiIon)",   -9.0,   55.0
)
hubs <- hubs %>%
  left_join(nudge, by = "dataset") %>%
  mutate(mlon = coalesce(mlon, lon), mlat = coalesce(mlat, lat),
         moved_deg = round(sqrt((mlon - lon)^2 + (mlat - lat)^2), 1))

R_OUT  <- 5.0                # ring radius, degrees of longitude
R_IN   <- 0.56 * R_OUT       # hole large enough for a two-digit number
R_HALO <- R_OUT + 0.5        # white edge that separates the ring from arcs and land
KY     <- cos(mean(c(-46.5, 76)) * pi / 180)   # coord_sf draws a degree of latitude 1/KY times a degree of longitude

d_hub <- as.matrix(dist(hubs[, c("mlon", "mlat")]))
diag(d_hub) <- Inf
d_anchor <- outer(seq_len(nrow(hubs)), seq_len(nrow(anchors)), Vectorize(function(i, j)
  sqrt((hubs$mlon[i] - anchors$ax[j])^2 + (hubs$mlat[i] - anchors$ay[j])^2)))
stopifnot(min(d_hub) > 2 * R_HALO, min(d_anchor) > R_HALO + 1)

disc <- function(x0, y0, r, n = 120) {
  a <- seq(0, 2 * pi, length.out = n)
  tibble(x = x0 + r * cos(a), y = y0 + KY * r * sin(a))
}
sector <- function(x0, y0, a0, a1, r0, r1) {
  a <- seq(a0, a1, length.out = max(4, ceiling(abs(a1 - a0) / (pi / 90))))
  tibble(x = c(x0 + r1 * cos(a), x0 + r0 * cos(rev(a))),
         y = c(y0 + KY * r1 * sin(a), y0 + KY * r0 * sin(rev(a))))
}
sectors <- rings %>%
  left_join(hubs %>% select(dataset, mlon, mlat), by = "dataset") %>%
  arrange(dataset, as.integer(cluster)) %>%
  group_by(dataset) %>%
  mutate(a0 = pi / 2 - 2 * pi * (cumsum(k) - k) / sum(k),    # clockwise from 12 o'clock, C1 first
         a1 = pi / 2 - 2 * pi * cumsum(k) / sum(k)) %>%
  ungroup()
ring_df <- bind_rows(lapply(seq_len(nrow(sectors)), function(i) {
  s <- sectors[i, ]
  sector(s$mlon, s$mlat, s$a0, s$a1, R_IN, R_OUT) %>% mutate(id = i, fill = cluster_colors[[s$cluster]])
}))
halo_df <- bind_rows(lapply(seq_len(nrow(hubs)), function(i) disc(hubs$mlon[i], hubs$mlat[i], R_HALO) %>% mutate(id = i)))

# ---------------------------------------------------------------- arcs
# Quadratic Bezier from the database to the country centre, always bowing north;
# the colour runs from light at the database to dark at the country.
arc <- function(x0, y0, x1, y1, k = 0.28, n = 121) {
  dx <- x1 - x0; dy <- y1 - y0; L <- sqrt(dx^2 + dy^2)
  nx <- -dy / L; ny <- dx / L
  if (ny < 0) { nx <- -nx; ny <- -ny }
  cx <- (x0 + x1) / 2 + k * L * nx
  cy <- (y0 + y1) / 2 + k * L * ny
  t <- seq(0, 1, length.out = n)
  tibble(x = (1 - t)^2 * x0 + 2 * (1 - t) * t * cx + t^2 * x1,
         y = (1 - t)^2 * y0 + 2 * (1 - t) * t * cy + t^2 * y1)
}
# Opaque: a colour that changes along a path is drawn as short segments, and with
# transparency their joins show as light seams. The city dots are drawn above the arcs.
ARC_LIGHT <- "#CDD2D9"
ARC_DARK  <- "#4E5866"
arc_cols  <- colorRampPalette(c(ARC_LIGHT, ARC_DARK))(121)
lw_of     <- function(n) 0.35 + 0.075 * n

flows <- flows %>% left_join(hubs %>% select(dataset, mlon, mlat), by = "dataset")
paths <- bind_rows(lapply(seq_len(nrow(flows)), function(i) {
  f <- flows[i, ]
  arc(f$mlon, f$mlat, f$ax, f$ay) %>% mutate(id = i, col = arc_cols, lw = lw_of(f$n_papers))
}))
stopifnot(max(paths$y) < 74)

write_csv(flows %>% select(dataset, label, country, n_papers, lon, lat, mlon, mlat, ax, ay), file.path(OUT, "data/flows.csv"))
write_csv(anchors, file.path(OUT, "data/country_anchors.csv"))
write_csv(rings %>% arrange(dataset, as.integer(cluster)) %>% rename(n_papers = k), file.path(OUT, "data/rings.csv"))
write_csv(hubs %>% select(rank, dataset, label, group, n_papers, host_country, lon, lat, mlon, mlat, moved_deg),
          file.path(OUT, "data/hub_positions.csv"))
writeLines(c(
  sprintf("papers with a public database, broad definition: %d of 755", nrow(pub)),
  sprintf("papers with a public database, narrow definition: %d of 755 (%.1f%%)", n_narrow, 100 * n_narrow / 755),
  sprintf("of these, using at least one of the eleven mapped databases: %d (%.0f%%)", n_hubs, 100 * n_hubs / n_narrow),
  sprintf("of these, using Materials Project: %d (%.0f%%)", n_mp, 100 * n_mp / n_narrow),
  sprintf("hosts in the United States: %d of 11", sum(hubs$host_country == "United States")),
  sprintf("arcs drawn (database-country pairs with >= 1 paper, seven countries): %d", nrow(flows)),
  sprintf("highest arc point: %.1f N", max(paths$y))
), file.path(OUT, "data/summary.txt"))

# ---------------------------------------------------------------- map
land_fill  <- "#D8DCE6"
border_col <- "#7F8C9D"
dot_pal    <- c("#A6CEE3", "#6BAED6", "#3E8EC4", "#2171B5", "#08519C", "#08306B")
world_sf <- rnaturalearth::ne_countries(scale = 50, returnclass = "sf")
world_sf <- world_sf[is.na(world_sf$continent) | world_sf$continent != "Antarctica", ]

# Arc-width samples under the colour bar. A round cap reaches half the line width
# (lw * 0.754 mm; about 1 mm per degree here) beyond each end, so samples are spaced by it.
ARC_KEY <- list(x = -136, y_title = -36.5, y_line = -42.6)
cap_deg <- function(n) lw_of(n) * 0.754 / 2
arc_key <- tibble(n = c(5, 20, 50), cap = cap_deg(n), chars = nchar(n))
arc_key$x0 <- ARC_KEY$x + arc_key$cap[1]
for (i in 2:3) arc_key$x0[i] <- arc_key$x0[i - 1] + 6 + arc_key$cap[i - 1] + 1.2 + 3.1 * arc_key$chars[i - 1] + 3 + arc_key$cap[i]
arc_key <- arc_key %>% mutate(x1 = x0 + 6, lx = x1 + cap + 1.2, lw = lw_of(n))

p_map <- ggplot() +
  geom_sf(data = world_sf, fill = land_fill, color = border_col, linewidth = 0.6) +
  geom_path(data = paths, aes(x = x, y = y, group = id, colour = col, linewidth = lw), lineend = "round") +
  scale_colour_identity(guide = "none") +
  scale_linewidth_identity(guide = "none") +
  geom_point(data = arrange(cities, Total), aes(x = longitude, y = latitude, fill = Total),
             shape = 21, size = 7.2, colour = "white", stroke = 0.8, alpha = 0.95) +
  scale_fill_gradientn(
    colours = dot_pal, trans = "sqrt",
    limits = c(1, max(cities$Total)), breaks = pretty(c(1, max(cities$Total))),
    name = "No. of papers",
    guide = guide_colourbar(display = "rectangles", title.position = "top", title.hjust = 0,
                            barwidth = unit(0.5, "cm"), barheight = unit(3.6, "cm"),
                            frame.colour = "grey55", ticks.colour = "grey55")) +
  ggnewscale::new_scale_fill() +
  geom_polygon(data = halo_df, aes(x = x, y = y, group = id), fill = "white", colour = NA) +
  geom_polygon(data = ring_df, aes(x = x, y = y, group = id, fill = fill), colour = "white", linewidth = 0.35) +
  scale_fill_identity(guide = "none") +
  geom_text(data = hubs, aes(x = mlon, y = mlat, label = rank), family = "Arial", size = 4.3, colour = "#222222") +
  annotate("text", x = ARC_KEY$x, y = ARC_KEY$y_title, label = "Arc width (articles)", hjust = 0, vjust = 0.5,
           family = "Arial", size = 5.6, colour = "grey20") +
  geom_segment(data = arc_key, aes(x = x0, xend = x1, y = ARC_KEY$y_line, yend = ARC_KEY$y_line, linewidth = lw),
               colour = ARC_DARK, lineend = "round") +
  geom_text(data = arc_key, aes(x = lx, y = ARC_KEY$y_line, label = n), hjust = 0, vjust = 0.5,
            family = "Arial", size = 5.6, colour = "grey20") +
  coord_sf(xlim = c(-158, 184), ylim = c(-46.5, 76), expand = FALSE, clip = "on") +
  theme_void(base_family = "Arial") +
  theme(
    legend.position      = c(0.065, 0.30),
    legend.justification = c(0, 0.5),
    legend.background    = element_rect(fill = alpha("white", 0.88), color = NA),
    legend.text          = element_text(size = 16, face = "bold", colour = "grey20", family = "Arial"),
    legend.title         = element_text(size = 17, face = "bold", colour = "grey20", family = "Arial"),
    legend.key.size      = unit(0.7, "cm"),
    plot.margin          = margin(t = 10, r = 10, b = 4, l = 16),
    panel.background     = element_rect(fill = "white", color = NA)
  )

# ---------------------------------------------------------------- key rows
# px frame 0..4167 over the 14-inch width. Row 1: crystals and solids; row 2: molecules
# and liquids, then battery cycling. x positions from Arial text widths (systemfonts).
KX     <- 4167
KEY_FS <- 6                                             # text size in mm
text_px <- function(s) systemfonts::string_width(s, family = "Arial", size = KEY_FS * .pt, res = 72) / 72 * KX / 14
layout_row <- function(groups, row) {
  x <- 20; entries <- list(); labels <- list()
  for (g in groups) {
    labels[[g]] <- tibble(x = x, row = row, text = g)
    x <- x + text_px(g) + 45
    for (i in which(as.character(hubs$group) == g)) {
      txt <- sprintf("%s %d", hubs$label[i], as.integer(hubs$n_papers[i]))
      entries[[length(entries) + 1]] <- tibble(rank = hubs$rank[i], x = x, row = row, text = txt)
      x <- x + 95 + text_px(txt) + 60
    }
    x <- x + 70
  }
  list(entries = bind_rows(entries), labels = bind_rows(labels), end = x - 130)
}
row1 <- layout_row(group_order[1], 1)
row2 <- layout_row(group_order[2:3], 0)
cat(sprintf("key rows end at %.0f and %.0f of %d px\n", row1$end, row2$end, KX))
stopifnot(row1$end < KX, row2$end < KX)
key_db  <- bind_rows(row1$entries, row2$entries)
key_grp <- bind_rows(row1$labels, row2$labels)

p_key <- ggplot() +
  geom_text(data = key_grp, aes(x = x, y = row, label = text), hjust = 0, vjust = 0.5,
            family = "Arial", size = KEY_FS, colour = "grey45") +
  geom_point(data = key_db, aes(x = x + 40, y = row), shape = 21, size = 7.4, fill = "white",
             colour = border_col, stroke = 1.1) +
  geom_text(data = key_db, aes(x = x + 40, y = row, label = rank), family = "Arial", size = 4.3, colour = "#222222") +
  geom_text(data = key_db, aes(x = x + 95, y = row, label = text), hjust = 0, vjust = 0.5,
            family = "Arial", size = KEY_FS, colour = "grey15") +
  scale_x_continuous(limits = c(0, KX), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.6, 1.6), expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  theme_void(base_family = "Arial") +
  theme(plot.margin = margin(t = 2, r = 4, b = 2, l = 4))

panel <- p_map / p_key + plot_layout(heights = c(5, 0.62))
ggsave(file.path(OUT, "svg/world_map_databases.svg"), plot = panel, width = 14, height = 5.9,
       device = svglite::svglite, bg = "white")
cat(readLines(file.path(OUT, "data/summary.txt")), sep = "\n")
print(hubs %>% select(rank, label, n_papers, moved_deg), n = 11)

# Regenerate country_counts.csv and country_cluster_counts.csv from country_article_records.csv when absent
car_path <- file.path(SEC, "data/country_article_records.csv")
if (file.exists(car_path)) {
  car <- read_csv(car_path, show_col_types = FALSE)
  cc_path <- file.path(SEC, "data/country_counts.csv")
  if (!file.exists(cc_path)) {
    cc <- car %>% count(country, name = "Total") %>% arrange(desc(Total), country)
    write_csv(cc, cc_path)
  }
  ccc_path <- file.path(SEC, "data/country_cluster_counts.csv")
  if (!file.exists(ccc_path)) {
    ccc <- car %>% count(country, Cluster, name = "n") %>% arrange(country, Cluster)
    write_csv(ccc, ccc_path)
  }
}

# Self-check rerun CSVs against output/data/ when OUT differs from output/ and reference exists
ref_dir <- file.path(SEC, "output", "data")
out_norm <- normalizePath(file.path(OUT, "data"), winslash = "/", mustWork = FALSE)
ref_norm <- normalizePath(ref_dir, winslash = "/", mustWork = FALSE)
if (out_norm != ref_norm && all(file.exists(file.path(ref_dir, c("flows.csv", "rings.csv", "hub_positions.csv", "country_anchors.csv"))))) {
  for (fn in c("flows.csv", "rings.csv", "hub_positions.csv", "country_anchors.csv")) {
    a <- readLines(file.path(OUT, "data", fn), encoding = "UTF-8")
    b <- readLines(file.path(ref_dir, fn), encoding = "UTF-8")
    stopifnot(identical(a, b))
  }
  cat("PASS: all 4 map CSVs match output/data/ byte-for-byte\n")
} else {
  cat("Wrote map data tables and SVG to", OUT, "\n")
}
