use std::error::Error;

mod matrix_completion_svd;
mod matrix_completion_svt;
mod print_array;

use crate::matrix_completion_svd::matrix_completion_svd;
use ndarray::{s, Array2, Zip};
use std::f64;

use crate::print_array::print_array;
use rusqlite::{params, Connection, Result};
use rustc_hash::FxHashMap;

#[derive(Debug)]
struct User {
    id: String,
    voted_at_millis: i64,
}

#[derive(Debug, Clone)]
struct Vote {
    user_id: String,
    item_id: i64,
    value: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let conn = Connection::open("dataset/ratings.db")?;

    let item_id = "1831373327648055733";
    let context_size = 20;

    let mut stmt_voters_on_note =
        conn.prepare("select raterParticipantId, createdAtMillis from ratings where noteId = ?1")?;
    // TODO: filter users with only one vote
    let mut stmt_votes_before =
        conn.prepare("select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 order by createdAtMillis desc limit ?3;")?;
    let mut stmt_votes_after =
        conn.prepare("select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 order by createdAtMillis asc limit ?3;")?;

    let user_iter = stmt_voters_on_note.query_map(params![item_id], |row| {
        Ok(User {
            id: row.get(0)?,
            voted_at_millis: row.get(1)?,
        })
    })?;

    let mut votes_before: Vec<Vote> = vec![];
    let mut votes_after: Vec<Vote> = vec![];
    let mut item_ids = FxHashMap::default();
    let mut user_ids = FxHashMap::default();

    for user in user_iter {
        let user = user?;
        if !user_ids.contains_key(&user.id) {
            user_ids.insert(user.id.clone(), user_ids.len());
        }

        let vote_iter = stmt_votes_before.query_map(
            params![user.id, user.voted_at_millis, context_size],
            |row| {
                Ok(Vote {
                    user_id: user.id.clone(),
                    item_id: row.get(0)?,
                    value: match row.get::<_, String>(1)?.as_str() {
                        "HELPFUL" => 1.0,
                        "SOMEWHAT_HELPFUL" => 0.0,
                        "NOT_HELPFUL" => -1.0,
                        _ => panic!(),
                    },
                })
            },
        )?;
        for vote in vote_iter {
            let vote = vote.unwrap();
            if !item_ids.contains_key(&vote.item_id) {
                item_ids.insert(vote.item_id.clone(), item_ids.len());
            }
            votes_before.push(vote.clone());
        }
        let vote_iter = stmt_votes_after.query_map(
            params![user.id, user.voted_at_millis, context_size],
            |row| {
                Ok(Vote {
                    user_id: user.id.clone(),
                    item_id: row.get(0)?,
                    value: match row.get::<_, String>(1)?.as_str() {
                        "HELPFUL" => 1.0,
                        "SOMEWHAT_HELPFUL" => 0.0,
                        "NOT_HELPFUL" => -1.0,
                        _ => panic!(),
                    },
                })
            },
        )?;
        for vote in vote_iter {
            let vote = vote.unwrap();
            if !item_ids.contains_key(&vote.item_id) {
                item_ids.insert(vote.item_id.clone(), item_ids.len());
            }
            votes_after.push(vote);
        }
    }

    println!("item id: {}", item_id);
    println!("context size {}", context_size);
    println!("Number of unique users: {}", user_ids.len());
    println!("Number of unique items: {}", item_ids.len());
    println!(
        "items/user: {}",
        item_ids.len() as f64 / user_ids.len() as f64
    );
    println!("votes before: {}", votes_before.len());
    println!("votes after:  {} ", votes_after.len());
    println!(
        "matrix density before: {}",
        (votes_before.len() as f64) / (item_ids.len() as f64 * user_ids.len() as f64)
    );
    println!(
        "matrix density after:  {}",
        (votes_after.len() as f64) / (item_ids.len() as f64 * user_ids.len() as f64)
    );

    let completion_rank = 3;
    let completion_tolerance = 1e-3;
    let completion_max_iterations = 100; // Limit max_iterations to 2 for debugging
    let _completion_lambda = 0.1; // Prefix with underscore to indicate intentional non-use
    let (observed_matrix_before, mental_model_before) = {
        let votes = votes_before;
        let mut observed_matrix: Array2<f64> =
            Array2::from_elem((item_ids.len(), user_ids.len()), f64::NAN);
        for vote in votes {
            let item_index = *item_ids.get(&vote.item_id).unwrap();
            let user_index = *user_ids.get(&vote.user_id).unwrap();
            observed_matrix[[item_index, user_index]] = vote.value;
        }

        let mental_model = matrix_completion_svd(
            observed_matrix.clone(),
            completion_rank,
            completion_tolerance,
            completion_max_iterations,
            None,
        );
        (observed_matrix, mental_model)
    };

    let (observed_matrix_after, mental_model_after) = {
        let votes = votes_after;
        let mut observed_matrix: Array2<f64> =
            Array2::from_elem((item_ids.len(), user_ids.len()), f64::NAN);
        let mut initial_guess = mental_model_before.clone();
        for vote in votes {
            let item_index = *item_ids.get(&vote.item_id).unwrap();
            let user_index = *user_ids.get(&vote.user_id).unwrap();
            observed_matrix[[item_index, user_index]] = vote.value;
            initial_guess[[item_index, user_index]] = vote.value; // Update initial_guess with observed values
        }
        let mental_model = matrix_completion_svd(
            observed_matrix.clone(),
            completion_rank,
            completion_tolerance,
            completion_max_iterations,
            Some(mental_model_before.clone()),
        );
        (observed_matrix, mental_model)
    };

    let top = 30;
    println!("Observed matrix before voting:");
    print_array(&observed_matrix_before.slice(s![..top, ..]).to_owned());

    println!("Observed matrix after voting:");
    print_array(&observed_matrix_after.slice(s![..top, ..]).to_owned());

    println!("Mental model before:");
    print_array(&mental_model_before.slice(s![..top, ..]).to_owned());
    println!("Mental model after:");
    print_array(&mental_model_after.slice(s![..top, ..]).to_owned());

    let rmse = root_mean_square_error(&mental_model_before, &mental_model_after);
    println!("RMSE: {}", rmse);

    Ok(())
}

fn root_mean_square_error(matrix1: &Array2<f64>, matrix2: &Array2<f64>) -> f64 {
    let diff = Zip::from(matrix1)
        .and(matrix2)
        .fold(0.0, |acc, &a, &b| acc + (a - b).powi(2));
    (diff / (matrix1.len() as f64)).sqrt()
}
