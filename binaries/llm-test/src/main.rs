use anyhow::Context;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use llm::InferenceStats;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    cmp::min,
    collections::HashMap,
    convert::Infallible,
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

#[derive(Parser)]
struct Cli {
    /// The path to the directory containing the model configurations.
    /// If not specified, the default directory will be used.
    #[clap(short, long)]
    configs: Option<PathBuf>,

    /// Whether to use memory mapping when loading the model.
    #[clap(short, long)]
    no_mmap: bool,

    /// The thread count to use when running inference.
    #[clap(short, long)]
    threads: Option<usize>,

    /// The model architecture to test. If not specified, all architectures will be tested.
    architecture: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set up the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args = Cli::parse();
    let specific_model = args.architecture.clone();

    // Initialize directories
    let cwd = env::current_dir()?;
    let configs_dir = args
        .configs
        .unwrap_or_else(|| cwd.join("binaries/llm-test/configs"));
    let download_dir = cwd.join(".tests/models");
    fs::create_dir_all(&download_dir)?;
    let results_dir = cwd.join(".tests/results");
    fs::create_dir_all(&results_dir)?;

    // Load configurations
    let test_configs: HashMap<String, TestConfig> = fs::read_dir(configs_dir)?
        .filter_map(Result::ok)
        .map(|de| de.path())
        .filter(|p| p.is_file())
        .filter(|p| p.extension().unwrap_or_default() == "json")
        .map(|path| {
            let file_name = path.file_stem().unwrap().to_string_lossy().to_string();
            let test_config: TestConfig = serde_json::from_str(&fs::read_to_string(&path)?)?;
            anyhow::Ok((file_name, test_config))
        })
        .collect::<Result<_, _>>()?;
    let model_config = ModelConfig {
        mmap: !args.no_mmap,
        threads: args.threads.unwrap_or(2),
    };

    // Test models
    let mut test_configs = if let Some(specific_architecture) = specific_model {
        vec![test_configs
            .get(&specific_architecture)
            .with_context(|| {
                format!(
                    "No config found for `{specific_architecture}`. Available configs: {:?}",
                    test_configs.keys()
                )
            })?
            .clone()]
    } else {
        test_configs.values().cloned().collect()
    };
    test_configs.sort_by_key(|tc| tc.architecture.clone());

    let test_configs_len = test_configs.len();
    for test_config in test_configs {
        test_model(&model_config, &test_config, &download_dir, &results_dir).await?;
        if test_configs_len > 1 {
            log::info!("----");
        }
    }

    log::info!("All tests passed!");
    Ok(())
}

struct ModelConfig {
    mmap: bool,
    threads: usize,
}

#[derive(Deserialize, Debug, Clone)]
struct TestConfig {
    url: String,
    filename: PathBuf,
    architecture: String,
    test_cases: Vec<TestCase>,
}

#[derive(Deserialize, Debug, Clone)]
enum TestCase {
    Inference {
        input: String,
        output: Option<String>,
        maximum_token_count: usize,
    },
}

#[derive(Serialize)]
enum Report {
    LoadFail { error: String },
    LoadSuccess { test_cases: Vec<TestCaseReport> },
}

#[derive(Serialize)]
struct TestCaseReport {
    meta: TestCaseReportMeta,
    report: TestCaseReportInner,
}

#[derive(Serialize)]
#[serde(untagged)]
enum TestCaseReportMeta {
    Error { error: String },
    Success,
}

#[derive(Serialize)]
enum TestCaseReportInner {
    Inference {
        input: String,
        expect_output: Option<String>,
        actual_output: String,
        inference_stats: Option<InferenceStats>,
    },
}

async fn test_model(
    model_config: &ModelConfig,
    test_config: &TestConfig,
    download_dir: &Path,
    results_dir: &Path,
) -> anyhow::Result<()> {
    // Load the model
    let architecture = llm::ModelArchitecture::from_str(&test_config.architecture)?;

    let local_path = if test_config.filename.is_file() {
        // If this filename points towards a valid file, use it
        test_config.filename.clone()
    } else {
        // Otherwise, use the download dir
        download_dir.join(&test_config.filename)
    };

    log::info!(
        "Testing architecture: `{}` ({})",
        test_config.architecture,
        local_path.display()
    );

    // Download the model if necessary
    download_file(&test_config.url, &local_path).await?;

    struct TestVisitor<'a> {
        model_config: &'a ModelConfig,
        test_config: &'a TestConfig,
        results_dir: &'a Path,
        local_path: &'a Path,
    }
    impl<'a> llm::ModelArchitectureVisitor<anyhow::Result<()>> for TestVisitor<'a> {
        fn visit<M: llm::KnownModel + 'static>(&mut self) -> anyhow::Result<()> {
            let Self {
                model_config,
                test_config,
                results_dir,
                local_path,
            } = *self;

            let start_time = Instant::now();

            let model = {
                let model = llm::load::<M>(
                    local_path,
                    llm::TokenizerSource::Embedded,
                    llm::ModelParameters {
                        prefer_mmap: model_config.mmap,
                        ..Default::default()
                    },
                    |progress| {
                        let print = !matches!(&progress,
                            llm::LoadProgress::TensorLoaded { current_tensor, tensor_count }
                            if current_tensor % (tensor_count / 10) != 0
                        );

                        if print {
                            log::info!("loading: {:?}", progress);
                        }
                    },
                );

                match model {
                    Ok(m) => m,
                    Err(err) => {
                        write_report(
                            test_config,
                            results_dir,
                            &Report::LoadFail {
                                error: format!("Failed to load model: {}", err),
                            },
                        )?;

                        return Err(err.into());
                    }
                }
            };

            log::info!(
                "Model fully loaded! Elapsed: {}ms",
                start_time.elapsed().as_millis()
            );

            //
            // Non-model-specific tests
            //

            // Confirm that the model can be sent to a thread, then sent back
            let model = tests::can_send(model)?;

            // Confirm that the hyperparameters can be roundtripped
            tests::can_roundtrip_hyperparameters(&model)?;

            //

            //
            // Model-specific tests
            //

            // Run the test cases
            let mut test_case_reports = vec![];
            for test_case in &test_config.test_cases {
                match test_case {
                    TestCase::Inference {
                        input,
                        output,
                        maximum_token_count,
                    } => test_case_reports.push(tests::can_infer(
                        &model,
                        model_config,
                        input,
                        output.as_deref(),
                        *maximum_token_count,
                    )?),
                }
            }
            let first_error: Option<String> =
                test_case_reports
                    .iter()
                    .find_map(|report: &TestCaseReport| match &report.meta {
                        TestCaseReportMeta::Error { error } => Some(error.clone()),
                        _ => None,
                    });

            // Save the results
            // Serialize the report to a JSON string
            write_report(
                test_config,
                results_dir,
                &Report::LoadSuccess {
                    test_cases: test_case_reports,
                },
            )?;

            // Optionally, panic if there was an error
            if let Some(err) = first_error {
                panic!("Error: {}", err);
            }

            log::info!(
                "Successfully tested architecture `{}`!",
                test_config.architecture
            );

            Ok(())
        }
    }
    architecture.visit(&mut TestVisitor {
        model_config,
        test_config,
        results_dir,
        local_path: &local_path,
    })?;

    Ok(())
}

fn write_report(
    test_config: &TestConfig,
    results_dir: &Path,
    report: &Report,
) -> anyhow::Result<()> {
    let json_report = serde_json::to_string_pretty(&report)?;
    let report_path = results_dir.join(format!("{}.json", test_config.architecture));
    fs::write(report_path, json_report)?;
    Ok(())
}

mod tests {
    use super::*;

    pub(super) fn can_send<M: llm::KnownModel + 'static>(model: M) -> anyhow::Result<M> {
        let model = std::thread::spawn(move || model)
            .join()
            .map_err(|e| anyhow::anyhow!("Failed to join thread: {e:?}"));

        log::info!("`can_send` test passed!");

        model
    }

    pub(super) fn can_roundtrip_hyperparameters<M: llm::KnownModel + 'static>(
        model: &M,
    ) -> anyhow::Result<()> {
        fn test_hyperparameters<M: llm::Hyperparameters>(
            hyperparameters: &M,
        ) -> anyhow::Result<()> {
            let mut data = vec![];
            hyperparameters.write_ggml(&mut data)?;
            let new_hyperparameters =
                <M as llm::Hyperparameters>::read_ggml(&mut std::io::Cursor::new(data))?;

            assert_eq!(hyperparameters, &new_hyperparameters);

            log::info!("`can_roundtrip_hyperparameters` test passed!");

            Ok(())
        }

        test_hyperparameters(model.hyperparameters())
    }

    pub(super) fn can_infer(
        model: &dyn llm::Model,
        model_config: &ModelConfig,
        input: &str,
        expected_output: Option<&str>,
        maximum_token_count: usize,
    ) -> anyhow::Result<TestCaseReport> {
        let mut session = model.start_session(Default::default());
        let (actual_output, res) = run_inference(
            model,
            model_config,
            &mut session,
            input,
            maximum_token_count,
        );

        // Process the results
        Ok(TestCaseReport {
            meta: match &res {
                Ok(_) => match expected_output {
                    Some(expected_output) => {
                        if expected_output == actual_output {
                            log::info!("`can_infer` test passed!");
                            TestCaseReportMeta::Success
                        } else {
                            TestCaseReportMeta::Error {
                                error: "The output did not match the expected output.".to_string(),
                            }
                        }
                    }
                    None => {
                        log::info!("`can_infer` test passed (no expected output)!");
                        TestCaseReportMeta::Success
                    }
                },
                Err(err) => TestCaseReportMeta::Error {
                    error: err.to_string(),
                },
            },
            report: TestCaseReportInner::Inference {
                input: input.into(),
                expect_output: expected_output.map(|s| s.to_string()),
                actual_output,
                inference_stats: res.ok(),
            },
        })
    }
}

fn run_inference(
    model: &dyn llm::Model,
    model_config: &ModelConfig,
    session: &mut llm::InferenceSession,
    input: &str,
    maximum_token_count: usize,
) -> (String, Result<InferenceStats, llm::InferenceError>) {
    let mut actual_output: String = String::new();
    let res = session.infer::<Infallible>(
        model,
        &mut rand::rngs::mock::StepRng::new(0, 1),
        &llm::InferenceRequest {
            prompt: input.into(),
            parameters: &llm::InferenceParameters {
                n_threads: model_config.threads,
                n_batch: 1,
                sampler: Arc::new(DeterministicSampler),
            },
            play_back_previous_tokens: false,
            maximum_token_count: Some(maximum_token_count),
        },
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                actual_output += &t;
                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );

    (actual_output, res)
}

#[derive(Debug)]
struct DeterministicSampler;
impl llm::Sampler for DeterministicSampler {
    fn sample(
        &self,
        previous_tokens: &[llm::TokenId],
        logits: &[f32],
        _rng: &mut dyn rand::RngCore,
    ) -> llm::TokenId {
        // Takes the most likely element from the logits, except if they've appeared in `previous_tokens`
        // at all
        let mut logits = logits.to_vec();
        for &token in previous_tokens {
            logits[token as usize] = f32::NEG_INFINITY;
        }

        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as llm::TokenId
    }
}

async fn download_file(url: &str, local_path: &Path) -> anyhow::Result<()> {
    if local_path.exists() {
        return Ok(());
    }

    let client = Client::new();

    let mut res = client.get(url).send().await?;
    let total_size = res
        .content_length()
        .context("Failed to get content length")?;

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
        .progress_chars("#>-"));

    let mut file = File::create(local_path)?;
    let mut downloaded: u64 = 0;

    while let Some(chunk) = res.chunk().await? {
        file.write_all(&chunk)?;
        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message("Download complete");

    Ok(())
}
