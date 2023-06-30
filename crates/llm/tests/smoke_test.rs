use std::{convert::Infallible, path::Path};

use llm_base::{InferenceFeedback, InferenceRequest, Model, Prompt};
use llm_gptneox::GptNeoX;

mod common;

#[test]
fn smoke_test() {
    let temp = tempfile::tempdir().unwrap();
    let model: GptNeoX = common::load_gz_model(Path::new("gptneox.bin.gz"), temp.path());
    let mut session = model.start_session(Default::default());

    let mut output: Vec<String> = vec![];
    session
        .infer::<Infallible>(
            &model,
            &mut rand::thread_rng(),
            &InferenceRequest {
                maximum_token_count: Some(1),
                parameters: &Default::default(),
                prompt: Prompt::Text(""),
                play_back_previous_tokens: false,
            },
            &mut Default::default(),
            |t| {
                match t {
                    llm::InferenceResponse::InferredToken(t) => output.push(t),
                    _ => (),
                }
                Ok(InferenceFeedback::Continue)
            },
        )
        .unwrap();

    assert_eq!(output.len(), 1);

    // Test files are large, panic if failed to clean up
    temp.close().unwrap();
}
