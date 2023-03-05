use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{fs, io};

use curl::easy::Easy;
use std::fs::File;

use anyhow::{Result, Error, Context};

pub fn download_extract<'a, P, T>(
    base_dir: P,
    urls: impl Iterator<Item = T>
) -> Result<()> where T: AsRef<Path>, P: AsRef<Path> {
    let base_dir = base_dir.as_ref();
    if !base_dir.exists() {
        fs::create_dir_all(base_dir)?;
    }

    for url in urls {
        let url: &Path = &url.as_ref();
        let file_name = url.file_name().ok_or(Error::msg("failed to parse url base path"))?;
        let file_path = base_dir.join(file_name);
        download_zip(url, &file_path)?;
        extract(&file_path.with_extension("gz"), &file_path.with_extension(""))?;
    }
    Ok(())
}

fn download_zip(
    url: &Path,
    file_path: &Path,
) -> Result<()> {
    let mut easy = Easy::new();
    
    if file_path.exists() {
        println!(
            "  File {:?} already exists, skipping downloading.",
            file_path.to_str().unwrap()
        );
    } else {
        println!(
            "- Downloading from file from {} and saving to file as: {}",
            url.to_str().unwrap(),
            file_path.to_str().unwrap()
        );
        if !file_path.parent().expect("download zip expects a parent directory in file_path").exists() {
            std::fs::create_dir_all(file_path.parent().unwrap())?;
        }
        let mut file = File::create(file_path)?;

        easy.url(url.to_str().unwrap())?;
        easy.write_function(move |data| {
            file.write_all(data).unwrap();
            Ok(data.len())
        })?;
        easy.perform()?;
    }

    Ok(())
}

fn extract(archive_path: &Path, extract_to: &Path) -> Result<()> {
    if extract_to.exists() {
        println!(
            "  Extracted file {:?} already exists, skipping extraction.",
            extract_to
        );
    } else {
        println!("Extracting archive {:?} to {:?}...", archive_path, extract_to);
        let file_in = fs::File::open(archive_path)
            .context(format!("Failed to open archive {:?}", archive_path))?;
        let file_in = io::BufReader::new(file_in);
        let file_out = fs::File::create(&extract_to).context(
            format!("Failed to create extracted file {:?}", archive_path
        ))?;
        let mut file_out = io::BufWriter::new(file_out);
        let mut gz = flate2::bufread::GzDecoder::new(file_in);
        let mut v: Vec<u8> = Vec::with_capacity(10 * 1024 * 1024);
        gz.read_to_end(&mut v).context(format!("Failed to extract archive {:?}", archive_path))?;
        file_out.write_all(&v).context(format!("Failed to write extracted data to {:?}", archive_path))?;
    }
    Ok(())
}

#[test]
fn test_extract() {
    let url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
    let url = Path::new(url);
    download_extract("assets/cifar", [url].iter()).unwrap();
}