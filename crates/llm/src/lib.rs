//! This crate provides a unified interface for loading and using
//! Large Language Models (LLMs). The following models are supported:
//!
//! - [BLOOM](llm_bloom)
//! - [GPT-2](llm_gpt2)
//! - [GPT-J](llm_gptj)
//! - [GPT-NeoX](llm_gptneox)
//! - [LLaMA](llm_llama)
//! - [MPT](llm_mpt)
//! - Falcon (currently disabled due to incompleteness)
//!
//! At present, the only supported backend is [GGML](https://github.com/ggerganov/ggml), but this is expected to
//! change in the future.
//!
//! # Example
//!
//! ```no_run
//! use std::io::Write;
//! use llm::Model;
//!
//! // load a GGML model from disk
//! let llama = llm::load::<llm::models::Llama>(
//!     // path to GGML file
//!     std::path::Path::new("/path/to/model"),
//!     // llm::TokenizerSource
//!     llm::TokenizerSource::Embedded,
//!     // llm::ModelParameters
//!     Default::default(),
//!     // load progress callback
//!     llm::load_progress_callback_stdout
//! )
//! .unwrap_or_else(|err| panic!("Failed to load model: {err}"));
//!
//! // use the model to generate text from a prompt
//! let mut session = llama.start_session(Default::default());
//! let res = session.infer::<std::convert::Infallible>(
//!     // model to use for text generation
//!     &llama,
//!     // randomness provider
//!     &mut rand::thread_rng(),
//!     // the prompt to use for text generation, as well as other
//!     // inference parameters
//!     &llm::InferenceRequest {
//!         prompt: "Rust is a cool programming language because".into(),
//!         parameters: &llm::InferenceParameters::default(),
//!         play_back_previous_tokens: false,
//!         maximum_token_count: None,
//!     },
//!     // llm::OutputRequest
//!     &mut Default::default(),
//!     // output callback
//!     |r| match r {
//!         llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
//!             print!("{t}");
//!             std::io::stdout().flush().unwrap();
//!
//!             Ok(llm::InferenceFeedback::Continue)
//!         }
//!         _ => Ok(llm::InferenceFeedback::Continue),
//!     }
//! );
//!
//! match res {
//!     Ok(result) => println!("\n\nInference stats:\n{result}"),
//!     Err(err) => println!("\n{err}"),
//! }
//! ```
#![deny(missing_docs)]

use std::{
    error::Error,
    fmt::{Debug, Display},
    path::Path,
    str::FromStr,
};

// Try not to expose too many GGML details here.
// This is the "user-facing" API, and GGML may not always be our backend.
pub use llm_base::{
    feed_prompt_callback, ggml::format as ggml_format, load, load_progress_callback_stdout,
    quantize, samplers, ElementType, FileType, FileTypeFormat, InferenceError, InferenceFeedback,
    InferenceParameters, InferenceRequest, InferenceResponse, InferenceSession,
    InferenceSessionConfig, InferenceSnapshot, InferenceSnapshotRef, InferenceStats,
    InvalidTokenBias, KnownModel, LoadError, LoadProgress, Loader, Model, ModelKVMemoryType,
    ModelParameters, OutputRequest, Prompt, QuantizeError, QuantizeProgress, Sampler,
    SnapshotError, TokenBias, TokenId, TokenUtf8Buffer, TokenizationError, Tokenizer,
    TokenizerSource,
};

use serde::Serialize;

/// All available models.
pub mod models {
    #[cfg(feature = "bloom")]
    pub use llm_bloom::{self as bloom, Bloom};
    #[cfg(feature = "falcon")]
    pub use llm_falcon::{self as falcon, Falcon};
    #[cfg(feature = "gpt2")]
    pub use llm_gpt2::{self as gpt2, Gpt2};
    #[cfg(feature = "gptj")]
    pub use llm_gptj::{self as gptj, GptJ};
    #[cfg(feature = "gptneox")]
    pub use llm_gptneox::{self as gptneox, GptNeoX};
    #[cfg(feature = "llama")]
    pub use llm_llama::{self as llama, Llama};
    #[cfg(feature = "mpt")]
    pub use llm_mpt::{self as mpt, Mpt};
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
/// All available model architectures.
pub enum ModelArchitecture {
    #[cfg(feature = "bloom")]
    /// [BLOOM](llm_bloom)
    Bloom,
    #[cfg(feature = "gpt2")]
    /// [GPT-2](llm_gpt2)
    Gpt2,
    #[cfg(feature = "gptj")]
    /// [GPT-J](llm_gptj)
    GptJ,
    #[cfg(feature = "gptneox")]
    /// [GPT-NeoX](llm_gptneox)
    GptNeoX,
    #[cfg(feature = "llama")]
    /// [LLaMA](llm_llama)
    Llama,
    #[cfg(feature = "mpt")]
    /// [MPT](llm_mpt)
    Mpt,
    #[cfg(feature = "falcon")]
    /// [Falcon](llm_falcon)
    Falcon,
}

impl ModelArchitecture {
    /// All available model architectures
    pub const ALL: &[Self] = &[
        #[cfg(feature = "bloom")]
        Self::Bloom,
        #[cfg(feature = "gpt2")]
        Self::Gpt2,
        #[cfg(feature = "gptj")]
        Self::GptJ,
        #[cfg(feature = "gptneox")]
        Self::GptNeoX,
        #[cfg(feature = "llama")]
        Self::Llama,
        #[cfg(feature = "mpt")]
        Self::Mpt,
        #[cfg(feature = "falcon")]
        Self::Falcon,
    ];
}

/// Used to dispatch some code based on the model architecture.
pub trait ModelArchitectureVisitor<R> {
    /// Visit a model architecture.
    fn visit<M: KnownModel + 'static>(&mut self) -> R;
}
impl ModelArchitecture {
    /// Use a visitor to dispatch some code based on the model architecture.
    pub fn visit<R>(&self, visitor: &mut impl ModelArchitectureVisitor<R>) -> R {
        match self {
            #[cfg(feature = "bloom")]
            Self::Bloom => visitor.visit::<models::Bloom>(),
            #[cfg(feature = "gpt2")]
            Self::Gpt2 => visitor.visit::<models::Gpt2>(),
            #[cfg(feature = "gptj")]
            Self::GptJ => visitor.visit::<models::GptJ>(),
            #[cfg(feature = "gptneox")]
            Self::GptNeoX => visitor.visit::<models::GptNeoX>(),
            #[cfg(feature = "llama")]
            Self::Llama => visitor.visit::<models::Llama>(),
            #[cfg(feature = "mpt")]
            Self::Mpt => visitor.visit::<models::Mpt>(),
            #[cfg(feature = "falcon")]
            Self::Falcon => visitor.visit::<models::Falcon>(),
        }
    }
}

/// An unsupported model architecture was specified.
pub struct UnsupportedModelArchitecture(String);
impl Display for UnsupportedModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Error for UnsupportedModelArchitecture {}
impl Debug for UnsupportedModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("UnsupportedModelArchitecture")
            .field(&self.0)
            .finish()
    }
}

impl FromStr for ModelArchitecture {
    type Err = UnsupportedModelArchitecture;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use ModelArchitecture::*;
        match s
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .as_str()
        {
            #[cfg(feature = "bloom")]
            "bloom" => Ok(Bloom),
            #[cfg(feature = "gpt2")]
            "gpt2" => Ok(Gpt2),
            #[cfg(feature = "gptj")]
            "gptj" => Ok(GptJ),
            #[cfg(feature = "gptneox")]
            "gptneox" => Ok(GptNeoX),
            #[cfg(feature = "llama")]
            "llama" => Ok(Llama),
            #[cfg(feature = "mpt")]
            "mpt" => Ok(Mpt),
            #[cfg(feature = "falcon")]
            "falcon" => Ok(Falcon),

            _ => Err(UnsupportedModelArchitecture(format!(
                "{s} is not a supported model architecture"
            ))),
        }
    }
}

impl Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ModelArchitecture::*;

        match self {
            #[cfg(feature = "bloom")]
            Bloom => write!(f, "BLOOM"),
            #[cfg(feature = "gpt2")]
            Gpt2 => write!(f, "GPT-2"),
            #[cfg(feature = "gptj")]
            GptJ => write!(f, "GPT-J"),
            #[cfg(feature = "gptneox")]
            GptNeoX => write!(f, "GPT-NeoX"),
            #[cfg(feature = "llama")]
            Llama => write!(f, "LLaMA"),
            #[cfg(feature = "mpt")]
            Mpt => write!(f, "MPT"),
            #[cfg(feature = "falcon")]
            Falcon => write!(f, "Falcon"),
        }
    }
}

/// A helper function that loads the specified model from disk using an architecture
/// specified at runtime. If no architecture is specified, it will try to infer it
/// from the model's metadata.
///
/// A wrapper around [load] that dispatches to the correct model.
pub fn load_dynamic(
    architecture: Option<ModelArchitecture>,
    path: &Path,
    tokenizer_source: TokenizerSource,
    params: ModelParameters,
    load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Box<dyn Model>, LoadError> {
    fn load_model<M: KnownModel + 'static>(
        path: &Path,
        tokenizer_source: TokenizerSource,
        params: ModelParameters,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<Box<dyn Model>, LoadError> {
        Ok(Box::new(load::<M>(
            path,
            tokenizer_source,
            params,
            load_progress_callback,
        )?))
    }

    let architecture = architecture.ok_or_else(|| LoadError::MissingModelArchitecture {
        path: path.to_owned(),
    })?;

    struct LoadVisitor<'a, F: FnMut(LoadProgress)> {
        path: &'a Path,
        tokenizer_source: TokenizerSource,
        params: ModelParameters,
        load_progress_callback: F,
    }
    impl<'a, F: FnMut(LoadProgress)> ModelArchitectureVisitor<Result<Box<dyn Model>, LoadError>>
        for LoadVisitor<'a, F>
    {
        fn visit<M: KnownModel + 'static>(&mut self) -> Result<Box<dyn Model>, LoadError> {
            load_model::<M>(
                self.path,
                self.tokenizer_source.clone(),
                self.params.clone(),
                &mut self.load_progress_callback,
            )
        }
    }

    architecture.visit(&mut LoadVisitor {
        path,
        tokenizer_source,
        params,
        load_progress_callback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_architecture_from_str() {
        for arch in ModelArchitecture::ALL {
            assert_eq!(
                arch,
                &arch.to_string().parse::<ModelArchitecture>().unwrap()
            );
        }
    }
}
