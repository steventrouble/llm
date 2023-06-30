use std::{
    fs::File,
    io,
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use llm_base::{ModelParameters, KnownModel};

/// Un-gzips the model with `filename` into the `output` file.
fn unzip_test_file(filename: &Path, output: &Path) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/models")
        .join(filename);
    assert!(path.exists());

    let gzipped = File::open(path).expect("Could not open file");
    let mut unzipped = GzDecoder::new(gzipped);
    let mut output = File::create(output).expect("Could not create file");

    io::copy(&mut unzipped, &mut output).unwrap();
}

/// Loads a gzipped model file as a model.
pub fn load_gz_model<M: KnownModel>(name: &Path, tempdir: &Path) -> M {
    let model_path = tempdir.join(name.with_extension(""));
    unzip_test_file(name, &model_path);
    llm::load::<M>(
        &model_path,
        llm::TokenizerSource::Embedded,
        ModelParameters::default(),
        |_| (),
    )
    .unwrap()
}
