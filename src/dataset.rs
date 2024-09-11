use csv::ReaderBuilder;
use ndarray::Array2;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::error::Error;

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
struct Record {
    noteId: String,
    raterParticipantId: String,
    helpfulnessLevel: String,
}

pub fn extract_matrix_from_dataset(
    file_path: &str,
    max_raters: usize,
) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().delimiter(b'\t').from_path(file_path)?;

    let mut note_ids = FxHashMap::default();
    let mut rater_ids = FxHashMap::default();
    let mut records: Vec<Record> = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        if !record.helpfulnessLevel.is_empty() {
            if !note_ids.contains_key(&record.noteId) {
                note_ids.insert(record.noteId.clone(), note_ids.len());
            }
            if !rater_ids.contains_key(&record.raterParticipantId) {
                rater_ids.insert(record.raterParticipantId.clone(), rater_ids.len());
            }
            records.push(record);
        }

        if rater_ids.len() >= max_raters {
            break;
        }
    }

    println!("Found {} records", records.len());
    println!("Number of unique notes: {}", note_ids.len());
    println!("Number of unique raters: {}", rater_ids.len());

    let mut matrix: Array2<f64> = Array2::from_elem((note_ids.len(), rater_ids.len()), f64::NAN);

    for record in records {
        let note_index = *note_ids.get(&record.noteId).unwrap();
        let rater_index = *rater_ids.get(&record.raterParticipantId).unwrap();
        matrix[[note_index, rater_index]] = match record.helpfulnessLevel.as_str() {
            "HELPFUL" => 1.0,
            "SOMEWHAT_HELPFUL" => 0.0,
            "NOT_HELPFUL" => -1.0,
            _ => panic!(), // Default case for unexpected values
        };
    }

    Ok(matrix)
}
