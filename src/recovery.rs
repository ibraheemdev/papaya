use crate::utils::*;
use crate::{FileSystem, Result};

use std::io;

pub fn backup_non_segment_files(mut fs: impl FileSystem) -> Result<()> {
    log::info!("Backing up non-segment files");

    for file in fs.read_dir(".")? {
        let file = file?;

        let path = file.path();

        let name = path
            .file_name()
            .map(|s| s.to_str())
            .flatten()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Invalid file name"))?;

        let ext = path
            .extension()
            .map(|s| s.to_str())
            .flatten()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Invalid file name"))?;

        if ext == SEGMENT_EXT || name == LOCK_FILE {
            continue;
        }

        let dst = [name, BACKUP_EXT].join("");
        fs.rename(name, &dst)?;
        log::info!("Moved {} to {}", name, &dst);
    }

    Ok(())
}
