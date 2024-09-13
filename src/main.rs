use std::error::Error;

mod matrix_completion_svd;
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

#[derive(Debug)]
struct Item {
    id: i64,
}

#[derive(Debug, Clone)]
struct Vote {
    user_id: String,
    item_id: i64,
    value: f64,
}

const CONTEXT_SIZE: i32 = 500;

fn main() -> Result<(), Box<dyn Error>> {
    let conn = Connection::open("dataset/ratings.db")?;

    conn.execute("delete from scores", ())?;
    let mut stmt_note_ids =
        conn.prepare("select distinct noteId from ratings where noteId = '1354855483631423493' order by createdAtMillis asc")?;
    let item_count: i64 = conn
        .prepare("select count(distinct noteId) from ratings")?
        .query_map(params![], |row| Ok(row.get::<_, i64>(0)?))?
        .next()
        .unwrap()?;
    let item_iter = stmt_note_ids.query_map(params![], |row| {
        Ok(Item {
            id: row.get::<_, i64>(0)?,
        })
    })?;

    for (i, item) in item_iter.enumerate() {
        let item = item?;
        // let item = Item {
        //     id: 1360871260054503427,
        // };
        println!(
            "\n[Item {}/{} {:.4}%]",
            i,
            item_count,
            i as f64 / item_count as f64
        );
        match process_item(item.id.to_string().as_str(), &conn) {
            Ok(_) => {}
            Err(err) => println!("Error: {}", err),
        };
    }

    Ok(())
}

fn process_item(item_id: &str, conn: &Connection) -> Result<(), Box<dyn Error>> {
    let score = calculate_change_of_mind(item_id, &conn)?;
    println!("item: {}, change-of-mind: {}", item_id, score);
    if score != -1.0 {
        conn.execute(
            "insert or replace into scores (noteId, change) values (?1, ?2)",
            (item_id, score),
        )?;
    }
    Ok(())
}

fn calculate_change_of_mind(item_id: &str, conn: &Connection) -> Result<f64, Box<dyn Error>> {
    let context_size = CONTEXT_SIZE;
    let mut stmt_voters_on_note =
        conn.prepare("select raterParticipantId, createdAtMillis from ratings where noteId = ?1")?;
    // TODO: filter users/items with only one vote
    let mut stmt_votes_before =
        conn.prepare("select noteId, agree - disagree from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 and agree + disagree != 0 order by createdAtMillis desc limit ?3;")?;
    let mut stmt_votes_after =
        conn.prepare("select noteId, agree - disagree from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 and agree + disagree != 0 order by createdAtMillis asc limit ?3;")?;

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
                    value: row.get(1)?,
                })
            },
        )?;
        for vote in vote_iter {
            let vote = vote?;
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
                    value: row.get(1)?,
                })
            },
        )?;
        for vote in vote_iter {
            let vote = vote?;
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

    if item_ids.len() < 3 || user_ids.len() < 3 || votes_before.len() == 0 || votes_after.len() == 0
    {
        return Ok(-1.0);
    }

    let completion_rank = 4;
    let completion_tolerance = 0.001;
    let completion_max_iterations = 500; // Limit max_iterations to 2 for debugging

    let observed_matrix_before = compute_observed_matrix(&votes_before, &item_ids, &user_ids);
    let observed_matrix_after = compute_observed_matrix(&votes_after, &item_ids, &user_ids);

    let mental_model_before = compute_mental_model(
        &observed_matrix_before,
        completion_rank,
        completion_tolerance,
        completion_max_iterations,
        None,
    );

    let mental_model_after = compute_mental_model(
        &observed_matrix_after,
        completion_rank,
        completion_tolerance,
        completion_max_iterations,
        Some(&mental_model_before),
    );

    let top = 50.min(item_ids.len());
    println!("Observed matrix before voting:");
    print_array(&observed_matrix_before.slice(s![..top, ..]).to_owned());

    println!("Observed matrix after voting:");
    print_array(&observed_matrix_after.slice(s![..top, ..]).to_owned());

    println!("Mental model before:");
    print_array(&mental_model_before.slice(s![..top, ..]).to_owned());
    println!("Mental model after:");
    print_array(&mental_model_after.slice(s![..top, ..]).to_owned());
    println!("Mental model diff:");
    print_array(
        &(mental_model_after.clone() - mental_model_before.clone())
            .slice(s![..top, ..])
            .to_owned(),
    );

    let rmse = root_mean_square_error(&mental_model_before, &mental_model_after) / 2.0; // range is from 0 to 2
    println!("mind change: {}", rmse);

    Ok(rmse)
}

fn root_mean_square_error(matrix1: &Array2<f64>, matrix2: &Array2<f64>) -> f64 {
    let diff = Zip::from(matrix1)
        .and(matrix2)
        .fold(0.0, |acc, &a, &b| acc + (a - b).powi(2));
    (diff / (matrix1.len() as f64)).sqrt()
}

fn compute_observed_matrix(
    votes: &[Vote],
    item_ids: &FxHashMap<i64, usize>,
    user_ids: &FxHashMap<String, usize>,
) -> Array2<f64> {
    let mut observed_matrix: Array2<f64> =
        Array2::from_elem((item_ids.len(), user_ids.len()), f64::NAN);

    for vote in votes {
        let item_index = *item_ids.get(&vote.item_id).unwrap();
        let user_index = *user_ids.get(&vote.user_id).unwrap();
        observed_matrix[[item_index, user_index]] = vote.value;
    }

    observed_matrix
}

fn compute_mental_model(
    observed_matrix: &Array2<f64>,
    completion_rank: usize,
    completion_tolerance: f64,
    completion_max_iterations: usize,
    initial_guess: Option<&Array2<f64>>,
) -> Array2<f64> {
    matrix_completion_svd(
        observed_matrix.clone(),
        completion_rank,
        completion_tolerance,
        completion_max_iterations,
        initial_guess.cloned(),
    )
}
