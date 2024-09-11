use std::{error::Error, time::Instant};

mod dataset;
mod matrix_completion_svd;
mod matrix_completion_svt;
mod print_array;

use crate::matrix_completion_svd::matrix_completion_svd;
use ndarray::{s, Array2};
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

    let item_id = "1830786640551284814";
    let context_size = 10;

    let mut stmt_voters_on_note =
        conn.prepare("select raterParticipantId, createdAtMillis from ratings where noteId = ?1")?;
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

    println!("item_id {}", item_id);
    println!("context size {}", context_size);
    println!("Found {} votes before", votes_before.len());
    println!("Found {} votes after", votes_after.len());
    println!("Number of unique users: {}", user_ids.len());
    println!("Number of unique items: {}", item_ids.len());
    println!(
        "items/user: {}",
        item_ids.len() as f64 / user_ids.len() as f64
    );
    println!(
        "matrix density before: {}",
        (votes_before.len() as f64) / (item_ids.len() as f64 * user_ids.len() as f64)
    );
    println!(
        "matrix density after: {}",
        (votes_after.len() as f64) / (item_ids.len() as f64 * user_ids.len() as f64)
    );

    let completion_rank = 3;
    let completion_tolerance = 1e-3;
    let completion_max_iterations = 100;
    let mental_model_before = {
        let votes = votes_before;
        let mut matrix: Array2<f64> = Array2::from_elem((item_ids.len(), user_ids.len()), f64::NAN);
        let mut observed: Vec<(usize, usize)> = vec![];
        for vote in votes {
            let item_index = *item_ids.get(&vote.item_id).unwrap();
            let user_index = *user_ids.get(&vote.user_id).unwrap();
            matrix[[item_index, user_index]] = vote.value;
            observed.push((item_index, user_index));
        }
        matrix_completion_svd(
            matrix,
            completion_rank,
            completion_tolerance,
            completion_max_iterations,
        )
    };
    let mental_model_after = {
        let votes = votes_after;
        let mut matrix: Array2<f64> = Array2::from_elem((item_ids.len(), user_ids.len()), f64::NAN);
        let mut observed: Vec<(usize, usize)> = vec![];
        for vote in votes {
            let item_index = *item_ids.get(&vote.item_id).unwrap();
            let user_index = *user_ids.get(&vote.user_id).unwrap();
            matrix[[item_index, user_index]] = vote.value;
            observed.push((item_index, user_index));
        }
        matrix_completion_svd(
            matrix,
            completion_rank,
            completion_tolerance,
            completion_max_iterations,
        )
    };

    let top = 20;
    println!("before");
    print_array(&mental_model_before.slice(s![..20, ..]).to_owned());
    println!("after");
    print_array(&mental_model_after.slice(s![..20, ..]).to_owned());

    // let matrix = arr2(&[[1., 1.], [1., f64::NAN]]);
    // let observed: Vec<(usize, usize)> = matrix
    //     .indexed_iter()
    //     .filter(|&(_, &value)| !value.is_nan())
    //     .map(|((i, j), _)| (i, j))
    //     .collect();
    // fill_missing_values(matrix, observed);
    Ok(())
}

// fn calculate_mind_change_score(noteId: String) -> f64 {
//     0.0
// }
