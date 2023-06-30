use std::{
    convert::Infallible,
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use clap::Parser;
use cli_args::Args;
use color_eyre::eyre::{Context, ContextCompat, Result};
use llm::{InferenceError, InferenceFeedback, InferenceResponse};
use rustyline::{
    error::ReadlineError,
    history::DefaultHistory,
    validate::{ValidationContext, ValidationResult, Validator},
    Cmd, Completer, Helper, Highlighter, Hinter, KeyCode, KeyEvent, Modifiers,
};

mod cli_args;
mod snapshot;

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();
    color_eyre::install()?;

    let args = Args::parse();
    match args {
        Args::Infer(args) => infer(&args),
        Args::Perplexity(args) => perplexity(&args),
        Args::Info(args) => info(&args),
        Args::PromptTokens(args) => prompt_tokens(&args),
        Args::Repl(args) => interactive(&args, false),
        Args::Chat(args) => interactive(&args, true),
        Args::Quantize(args) => quantize(&args),
    }
}

fn infer(args: &cli_args::Infer) -> Result<()> {
    let prompt = load_prompt_file_with_prompt(&args.prompt_file, args.prompt.as_deref());
    let inference_session_config = args.generate.inference_session_config();
    let model = args.model_load.load(args.generate.use_gpu)?;

    let (mut session, session_loaded) = snapshot::read_or_create_session(
        model.as_ref(),
        args.persist_session.as_deref(),
        args.generate.load_session.as_deref(),
        inference_session_config,
    );
    let parameters = args.generate.inference_parameters(model.eot_token_id());

    let mut rng = args.generate.rng();
    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rng,
        &llm::InferenceRequest {
            prompt: prompt.as_str().into(),
            parameters: &parameters,
            play_back_previous_tokens: session_loaded,
            maximum_token_count: args.generate.num_predict,
        },
        // OutputRequest
        &mut Default::default(),
        |r| match &r {
            InferenceResponse::PromptToken(t) | InferenceResponse::InferredToken(t) => {
                if matches!(&r, InferenceResponse::PromptToken(_)) && args.hide_prompt {
                    return Ok(InferenceFeedback::Continue);
                }

                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(InferenceFeedback::Continue)
            }
            _ => Ok(InferenceFeedback::Continue),
        },
    );
    println!();

    match res {
        Ok(stats) => {
            if args.stats {
                println!();
                println!("{}", stats);
                println!();
            }
        }
        Err(InferenceError::ContextFull) => {
            log::warn!("Context window full, stopping inference.")
        }
        Err(InferenceError::TokenizationFailed(err)) => {
            log::error!("A tokenization-related failure occurred: {}", err);
        }
        Err(InferenceError::UserCallback(_)) | Err(InferenceError::EndOfText) => {
            unreachable!("cannot fail")
        }
    }

    if let Some(session_path) = args.save_session.as_ref().or(args.persist_session.as_ref()) {
        // Write the memory to the cache file
        snapshot::write_session(session, session_path);
    }

    Ok(())
}

fn perplexity(args: &cli_args::Perplexity) -> Result<()> {
    let prompt = load_prompt_file_with_prompt(&args.prompt_file, args.prompt.as_deref());
    let inference_session_config = args.generate.inference_session_config();
    let model = args.model_load.load(args.generate.use_gpu)?;
    let (mut session, _) = snapshot::read_or_create_session(
        model.as_ref(),
        None,
        args.generate.load_session.as_deref(),
        inference_session_config,
    );
    let parameters = args.generate.inference_parameters(model.eot_token_id());

    session.perplexity(
        model.as_ref(),
        &parameters,
        prompt.as_str(),
        |chunk, perplexity| {
            println!("Perplexity[{chunk}]: {perplexity}");
        },
    )?;

    Ok(())
}

fn info(args: &cli_args::Info) -> Result<()> {
    struct InfoVisitor<'a>(&'a cli_args::Info);
    impl llm::ModelArchitectureVisitor<Result<()>> for InfoVisitor<'_> {
        fn visit<M: llm::KnownModel + 'static>(&mut self) -> Result<()> {
            let args = self.0;

            let model_path = &args.model_and_tokenizer.model_path;
            let tokenizer = args.model_and_tokenizer.to_source()?.retrieve(model_path)?;

            let file = File::open(model_path)?;
            let mut reader = BufReader::new(&file);
            let mut loader: llm::Loader<M::Hyperparameters, _> =
                llm::Loader::new(tokenizer, |_| {
                    // We purposely do not print progress here, as we are only interested in the metadata
                });

            llm::ggml_format::load(&mut reader, &mut loader)?;

            log::info!("Container type: {:?}", loader.container_type);
            log::info!("Hyperparameters: {:?}", loader.hyperparameters);
            log::info!("Tokenizer vocabulary size: {}", loader.tokenizer.len());

            if args.tokenizer {
                log::info!("Tokens:");
                for i in 0..loader.tokenizer.len() {
                    log::info!("- {}: {}", i, utf8_or_array(&loader.tokenizer.token(i)));
                }
            }

            if args.tensors {
                log::info!("Tensors:");
                for (name, tensor) in &loader.tensors {
                    log::info!("- {} ({:?} {:?})", name, tensor.element_type, tensor.dims());
                }
            }

            fn utf8_or_array(token: &[u8]) -> String {
                std::str::from_utf8(token)
                    .map(|s| s.to_owned())
                    .unwrap_or(format!("{:?}", token))
            }

            Ok(())
        }
    }

    args.model_and_tokenizer
        .architecture
        .model_architecture
        .wrap_err("a model architecture is required at present")?
        .visit(&mut InfoVisitor(args))
}

fn prompt_tokens(args: &cli_args::PromptTokens) -> Result<()> {
    let prompt = load_prompt_file_with_prompt(&args.prompt_file, args.prompt.as_deref());
    let model = args.model_load.load(false)?;
    let toks = match model.tokenizer().tokenize(&prompt, false) {
        Ok(toks) => toks,
        Err(e) => {
            log::error!("Could not tokenize prompt: {e}");
            std::process::exit(1);
        }
    };
    log::info!("=== Dumping prompt tokens:");
    log::info!(
        "{}",
        toks.iter()
            .map(|(_, tid)| tid.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    log::info!(
        "{}",
        toks.iter()
            .map(|(s, tid)| format!("{s:?}:{tid}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    Ok(())
}

#[cfg(not(windows))]
fn force_newline_event_seq() -> KeyEvent {
    KeyEvent(KeyCode::Enter, Modifiers::ALT)
}

// On Windows, `SHIFT+ENTER` is the key sequence for forcing a newline. This is
// because `ALT+ENTER` typically maximizes the window.
#[cfg(windows)]
fn force_newline_event_seq() -> KeyEvent {
    KeyEvent(KeyCode::Enter, Modifiers::SHIFT)
}

fn interactive(
    args: &cli_args::Repl,
    // If set to false, the session will be cloned after each inference
    // to ensure that previous state is not carried over.
    chat_mode: bool,
) -> Result<()> {
    let prompt_file = args.prompt_file.contents();
    let inference_session_config = args.generate.inference_session_config();
    let model = args.model_load.load(args.generate.use_gpu)?;
    let (mut session, mut session_loaded) = snapshot::read_or_create_session(
        model.as_ref(),
        None,
        args.generate.load_session.as_deref(),
        inference_session_config,
    );
    let parameters = args.generate.inference_parameters(model.eot_token_id());

    let mut rng = args.generate.rng();
    let mut rl = rustyline::Editor::<LineContinuationValidator, DefaultHistory>::new()?;
    rl.set_helper(Some(LineContinuationValidator));

    rl.bind_sequence(force_newline_event_seq(), Cmd::Newline);

    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(raw_line) => {
                let line = raw_line.replace("\\\n", "\n");

                let prompt = prompt_file
                    .as_deref()
                    .map(|pf| process_prompt(pf, &line))
                    .unwrap_or(line);

                let sp = spinoff::Spinner::new(spinoff::spinners::Dots2, "".to_string(), None);
                if let Err(InferenceError::ContextFull) = session.feed_prompt(
                    model.as_ref(),
                    &parameters,
                    &prompt,
                    // OutputRequest
                    &mut Default::default(),
                    |_| Ok::<_, Infallible>(InferenceFeedback::Continue),
                ) {
                    log::error!("Prompt exceeds context window length.")
                };
                sp.clear();

                let res = session.infer::<Infallible>(
                    model.as_ref(),
                    &mut rng,
                    &llm::InferenceRequest {
                        prompt: "".into(),
                        parameters: &parameters,
                        play_back_previous_tokens: session_loaded,
                        maximum_token_count: args.generate.num_predict,
                    },
                    // EvaluateOuputRequest
                    &mut Default::default(),
                    |r| match r {
                        InferenceResponse::PromptToken(t) | InferenceResponse::InferredToken(t) => {
                            print!("{t}");
                            std::io::stdout().flush().unwrap();

                            Ok(InferenceFeedback::Continue)
                        }
                        _ => Ok(InferenceFeedback::Continue),
                    },
                );
                println!();

                if let Err(InferenceError::ContextFull) = res {
                    log::error!("Reply exceeds context window length");
                }

                // Reload session in REPL mode
                if !chat_mode {
                    (session, session_loaded) = snapshot::read_or_create_session(
                        model.as_ref(),
                        None,
                        args.generate.load_session.as_deref(),
                        inference_session_config,
                    );
                }
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(err) => {
                log::error!("{err}");
            }
        }
    }

    Ok(())
}

fn quantize(args: &cli_args::Quantize) -> Result<()> {
    use llm::QuantizeProgress;

    struct QuantizeVisitor<'a>(&'a cli_args::Quantize);
    impl llm::ModelArchitectureVisitor<Result<()>> for QuantizeVisitor<'_> {
        fn visit<M: llm::KnownModel>(&mut self) -> Result<()> {
            let args = self.0;

            let mut source: BufReader<File> = BufReader::new(std::fs::File::open(&args.source)?);
            let mut destination: BufWriter<File> =
                BufWriter::new(std::fs::File::create(&args.destination)?);
            let tokenizer: llm::Tokenizer = args.tokenizer.to_source()?.retrieve(&args.source)?;

            llm::quantize::<M, _, _>(
                &mut source,
                &mut destination,
                tokenizer,
                args.container_type.into(),
                args.target.into(),
                |progress| match progress {
                    QuantizeProgress::HyperparametersLoaded => log::info!("Loaded hyperparameters"),
                    QuantizeProgress::TensorLoading {
                        name,
                        dims,
                        element_type,
                        n_elements,
                    } => log::info!(
                        "Loading tensor `{name}` ({n_elements} ({dims:?}) {element_type} elements)"
                    ),
                    QuantizeProgress::TensorQuantizing { name } => log::info!("Quantizing tensor `{name}`"),
                    QuantizeProgress::TensorQuantized {
                        name,
                        original_size,
                        reduced_size,
                        history,
                    } => log::info!(
                    "Quantized tensor `{name}` from {original_size} to {reduced_size} bytes ({history:?})"
                ),
                    QuantizeProgress::TensorSkipped { name, size } => {
                        log::info!("Skipped tensor `{name}` ({size} bytes)")
                    }
                    QuantizeProgress::Finished {
                        original_size,
                        reduced_size,
                        history,
                    } => log::info!(
                        "Finished quantization from {original_size} to {reduced_size} bytes ({history:?})"
                    ),
                },
            )
            .wrap_err("failed to quantize model")
        }
    }

    args.architecture
        .model_architecture
        .wrap_err("the architecture must be known for quantization")?
        .visit(&mut QuantizeVisitor(args))
}

fn load_prompt_file_with_prompt(
    prompt_file: &cli_args::PromptFile,
    prompt: Option<&str>,
) -> String {
    if let Some(prompt_file) = prompt_file.contents() {
        if let Some(prompt) = prompt {
            process_prompt(&prompt_file, prompt)
        } else {
            prompt_file
        }
    } else if let Some(prompt) = prompt {
        prompt.to_owned()
    } else {
        log::error!("No prompt or prompt file was provided. See --help");
        std::process::exit(1);
    }
}

#[derive(Completer, Helper, Highlighter, Hinter, Debug, Clone, Copy)]
struct LineContinuationValidator;

impl Validator for LineContinuationValidator {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        if ctx.input().ends_with('\\') {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }
}

fn process_prompt(raw_prompt: &str, prompt: &str) -> String {
    raw_prompt.replace("{{PROMPT}}", prompt)
}
