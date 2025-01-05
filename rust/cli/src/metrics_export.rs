//! Export training metrics to CSV for plotting.

use std::path::{Path, PathBuf};

use clap::ValueEnum;

/// Metric types that can be exported.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum MetricType {
    Loss,
    Perplexity,
    Accuracy,
    LearningRate,
}

impl MetricType {
    fn filename(&self) -> &'static str {
        match self {
            MetricType::Loss => "Loss.log",
            MetricType::Perplexity => "Perplexity.log",
            MetricType::Accuracy => "Accuracy.log",
            MetricType::LearningRate => "LearningRate.log",
        }
    }

    fn column_name(&self) -> &'static str {
        match self {
            MetricType::Loss => "loss",
            MetricType::Perplexity => "perplexity",
            MetricType::Accuracy => "accuracy",
            MetricType::LearningRate => "learning_rate",
        }
    }
}

/// Configuration for the metrics export.
pub struct ExportConfig {
    pub metrics: Vec<MetricType>,
    pub include_train: bool,
    pub include_valid: bool,
    pub target_points: Option<usize>,
    pub window: Option<usize>,
}

/// A single row in the output CSV.
#[derive(Debug)]
struct MetricRow {
    experiment: String,
    epoch: usize,
    step: usize,
    global_step: usize,
    values: Vec<Option<f64>>,
}

/// Parse a metric log file, returning (step, value) pairs.
/// Steps are 1-indexed based on line number (since the step column in log files isn't reliable).
fn parse_metric_log_with_steps(path: &Path) -> Option<Vec<(usize, f64)>> {
    let content = std::fs::read_to_string(path).ok()?;
    let values: Vec<(usize, f64)> = content
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let parts: Vec<&str> = line.split(',').collect();
            let value = parts.first()?.parse().ok()?;
            Some((i + 1, value)) // 1-indexed step based on line number
        })
        .collect();

    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

/// Downsample data to approximately `target` points using bucket averaging.
fn downsample_buckets(values: &[(usize, f64)], target: usize) -> Vec<(usize, f64)> {
    if values.len() <= target || target == 0 {
        return values.to_vec();
    }

    let bucket_size = values.len() / target;
    let mut result = Vec::with_capacity(target);

    for chunk in values.chunks(bucket_size) {
        if chunk.is_empty() {
            continue;
        }
        let avg_step = chunk.iter().map(|(s, _)| *s).sum::<usize>() / chunk.len();
        let avg_value = chunk.iter().map(|(_, v)| *v).sum::<f64>() / chunk.len() as f64;
        result.push((avg_step, avg_value));
    }

    result
}

/// Apply rolling average smoothing.
fn apply_rolling_average(values: &[(usize, f64)], window: usize) -> Vec<(usize, f64)> {
    if window <= 1 || values.is_empty() {
        return values.to_vec();
    }

    let mut result = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(values.len());
        let slice = &values[start..end];
        let avg = slice.iter().map(|(_, v)| *v).sum::<f64>() / slice.len() as f64;
        result.push((values[i].0, avg));
    }

    result
}

/// Expand glob patterns to actual paths.
fn expand_globs(patterns: &[String]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();

    for pattern in patterns {
        let matches: Vec<_> = glob::glob(pattern)?
            .filter_map(std::result::Result::ok)
            .collect();

        if matches.is_empty() {
            // Treat as a literal path if no glob matches
            let path = PathBuf::from(pattern);
            if path.exists() {
                paths.push(path);
            } else {
                eprintln!("Warning: no matches for pattern '{pattern}'");
            }
        } else {
            paths.extend(matches);
        }
    }

    // Filter to only directories
    paths.retain(|p| p.is_dir());
    paths.sort();

    Ok(paths)
}

/// Get experiment name from a path (the directory name).
fn experiment_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || "unknown".to_string(),
        |n| n.to_string_lossy().to_string(),
    )
}

/// Collect all epoch directories from a metrics dir (train/ or valid/).
fn collect_epochs(metrics_dir: &Path) -> Vec<(usize, PathBuf)> {
    let mut epochs = Vec::new();

    if !metrics_dir.exists() {
        return epochs;
    }

    if let Ok(entries) = std::fs::read_dir(metrics_dir) {
        for entry in entries.filter_map(std::result::Result::ok) {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(epoch_str) = name.strip_prefix("epoch-")
                && let Ok(epoch) = epoch_str.parse::<usize>()
            {
                epochs.push((epoch, entry.path()));
            }
        }
    }

    epochs.sort_by_key(|(e, _)| *e);
    epochs
}

/// Apply smoothing and downsampling to metric data.
fn smooth_and_downsample(mut data: Vec<(usize, f64)>, config: &ExportConfig) -> Vec<(usize, f64)> {
    if let Some(window) = config.window {
        data = apply_rolling_average(&data, window);
    }
    if let Some(target) = config.target_points {
        data = downsample_buckets(&data, target);
    }
    data
}

/// Build MetricRows from processed per-metric data and epoch boundaries.
fn build_rows(
    experiment: &str,
    per_metric_processed: &[Vec<(usize, f64)>],
    epoch_boundaries: &[(usize, usize)],
) -> Vec<MetricRow> {
    let index_maps: Vec<std::collections::HashMap<usize, f64>> = per_metric_processed
        .iter()
        .map(|data| data.iter().copied().collect())
        .collect();

    let mut all_global_steps: Vec<usize> =
        index_maps.iter().flat_map(|m| m.keys().copied()).collect();
    all_global_steps.sort_unstable();
    all_global_steps.dedup();

    let mut rows = Vec::new();
    for &global_step in &all_global_steps {
        let values: Vec<Option<f64>> = index_maps
            .iter()
            .map(|m| m.get(&global_step).copied())
            .collect();

        if values.iter().all(std::option::Option::is_none) {
            continue;
        }

        // Determine epoch from boundaries (last boundary whose start <= global_step)
        let (epoch, epoch_start) = epoch_boundaries
            .iter()
            .rev()
            .find(|(_, start)| global_step >= *start)
            .copied()
            .unwrap_or((1, 0));

        rows.push(MetricRow {
            experiment: experiment.to_string(),
            epoch,
            step: global_step - epoch_start,
            global_step,
            values,
        });
    }

    rows
}

/// Collect metrics from one artifact directory for one split.
///
/// For training metrics, smoothing/downsampling spans across epoch boundaries to avoid
/// staircase artifacts. For validation metrics, it's applied per-epoch since each epoch
/// evaluates a single checkpoint and cross-epoch blending would be misleading.
fn collect_split_metrics(dir: &Path, split: &str, config: &ExportConfig) -> Vec<MetricRow> {
    let experiment = experiment_name(dir);
    let metrics_dir = dir.join(split);
    let epochs = collect_epochs(&metrics_dir);

    if epochs.is_empty() {
        return Vec::new();
    }

    let num_metrics = config.metrics.len();
    let cross_epoch = split == "train";

    let mut per_metric_raw: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_metrics];
    let mut epoch_boundaries: Vec<(usize, usize)> = Vec::new(); // (epoch, start_global_step)
    let mut global_step_offset = 0;

    let mut per_epoch_metric_raw: Vec<Vec<Vec<(usize, f64)>>> = Vec::new();

    for (epoch, epoch_path) in &epochs {
        epoch_boundaries.push((*epoch, global_step_offset));
        let mut max_step_this_epoch: usize = 0;
        let mut this_epoch_data: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_metrics];

        for (metric_idx, metric_type) in config.metrics.iter().enumerate() {
            let log_path = epoch_path.join(metric_type.filename());
            if let Some(data) = parse_metric_log_with_steps(&log_path) {
                for &(step, value) in &data {
                    let global_step = global_step_offset + step;
                    per_metric_raw[metric_idx].push((global_step, value));
                    this_epoch_data[metric_idx].push((global_step, value));
                    max_step_this_epoch = max_step_this_epoch.max(step);
                }
            }
        }

        per_epoch_metric_raw.push(this_epoch_data);
        global_step_offset += max_step_this_epoch;
    }

    if cross_epoch {
        let per_metric_processed: Vec<Vec<(usize, f64)>> = per_metric_raw
            .into_iter()
            .map(|data| smooth_and_downsample(data, config))
            .collect();

        build_rows(&experiment, &per_metric_processed, &epoch_boundaries)
    } else {
        let mut per_metric_processed: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_metrics];

        for epoch_data in per_epoch_metric_raw {
            for (metric_idx, data) in epoch_data.into_iter().enumerate() {
                let processed = smooth_and_downsample(data, config);
                per_metric_processed[metric_idx].extend(processed);
            }
        }

        build_rows(&experiment, &per_metric_processed, &epoch_boundaries)
    }
}

/// Write CSV output.
fn write_csv(
    rows: &[MetricRow],
    output: &Path,
    config: &ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = std::fs::File::create(output)?;

    // header
    let metric_columns: Vec<&str> = config.metrics.iter().map(MetricType::column_name).collect();
    writeln!(
        file,
        "experiment,epoch,step,global_step,{}",
        metric_columns.join(",")
    )?;

    // data rows
    for row in rows {
        let values_str: Vec<String> = row
            .values
            .iter()
            .map(|v| v.map(|x| format!("{x}")).unwrap_or_default())
            .collect();

        writeln!(
            file,
            "{},{},{},{},{}",
            row.experiment,
            row.epoch,
            row.step,
            row.global_step,
            values_str.join(",")
        )?;
    }

    Ok(())
}

/// Derive output path for a split from the base output path.
fn output_path_for_split(base: &str, split: &str) -> PathBuf {
    let path = Path::new(base);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy())
        .unwrap_or_default();

    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        if ext.is_empty() {
            parent.join(format!("{stem}_{split}"))
        } else {
            parent.join(format!("{stem}_{split}.{ext}"))
        }
    } else if ext.is_empty() {
        PathBuf::from(format!("{stem}_{split}"))
    } else {
        PathBuf::from(format!("{stem}_{split}.{ext}"))
    }
}

/// Main entry point for exporting metrics.
pub fn export_metrics(
    dirs: Vec<String>,
    output: &str,
    config: ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let paths = expand_globs(&dirs)?;

    if paths.is_empty() {
        return Err("No valid artifact directories found".into());
    }

    println!("Exporting metrics from {} directories", paths.len());

    let mut train_rows = Vec::new();
    let mut valid_rows = Vec::new();

    for path in &paths {
        let name = experiment_name(path);

        if config.include_train {
            let rows = collect_split_metrics(path, "train", &config);
            if !rows.is_empty() {
                println!("  {name}: {} train data points", rows.len());
                train_rows.extend(rows);
            }
        }

        if config.include_valid {
            let rows = collect_split_metrics(path, "valid", &config);
            if !rows.is_empty() {
                println!("  {name}: {} valid data points", rows.len());
                valid_rows.extend(rows);
            }
        }
    }

    if train_rows.is_empty() && valid_rows.is_empty() {
        return Err("No metrics data found in any directory".into());
    }

    // Write separate files for train and valid
    if !train_rows.is_empty() {
        let train_path = output_path_for_split(output, "train");
        write_csv(&train_rows, &train_path, &config)?;
        println!(
            "Wrote {} rows to {}",
            train_rows.len(),
            train_path.display()
        );
    }

    if !valid_rows.is_empty() {
        let valid_path = output_path_for_split(output, "valid");
        write_csv(&valid_rows, &valid_path, &config)?;
        println!(
            "Wrote {} rows to {}",
            valid_rows.len(),
            valid_path.display()
        );
    }

    Ok(())
}
