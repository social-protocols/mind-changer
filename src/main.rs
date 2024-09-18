use std::error::Error;

mod initial_guess;
mod matrix_completion_gd;
mod matrix_completion_svd;
mod matrix_factorization_als;
mod matrix_factorization_svd;
mod print_array;

use matrix_factorization_svd::matrix_factorization_svd;
use ndarray::Array2;

use ndarray::s;
use ndarray_linalg::Norm;
use std::f64;

use crate::print_array::print_array;
use rusqlite::{params, Connection, Result, Statement};
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

const CONTEXT_SIZE: i32 = 30; // other votes from user before and after voting on target post
const AGREE: bool = true;

fn main() -> Result<(), Box<dyn Error>> {
    let conn = Connection::open("dataset/ratings.db")?;

    conn.execute("delete from scores", ())?;
    conn.execute("delete from user_change", ())?;
    //  where noteId = '1400247230330667008'
    //   where noteId = '1354864556552712194'
    // 10x133  where noteid = '1709553622751588430'
    // 39x544  where noteid = '1354855204005453826'
    let mut stmt_note_ids =
        conn.prepare("select distinct noteId from ratings where noteid = '1354855204005453826'")?;
    println!("counting notes...");
    let item_count: i64 = conn
        .prepare("select count(distinct noteId) from ratings")?
        .query_map(params![], |row| row.get::<_, i64>(0))?
        .next()
        .unwrap()?;
    let item_iter = stmt_note_ids.query_map(params![], |row| {
        Ok(Item {
            id: row.get::<_, i64>(0)?,
        })
    })?;

    for (i, item) in item_iter.enumerate() {
        let item = item?;
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

fn get_votes(
    user_id: &str,
    user_voted_at_millis: i64,
    statement: &mut Statement,
) -> Result<Vec<Vote>, Box<dyn Error>> {
    Ok(statement
        .query_map(
            params![user_id, user_voted_at_millis, CONTEXT_SIZE],
            |row| {
                Ok(Vote {
                    user_id: user_id.to_string(),
                    item_id: row.get(0)?,
                    value: if AGREE {
                        row.get(1)?
                    } else {
                        match row.get::<_, String>(1)?.as_str() {
                            "HELPFUL" => 1.0,
                            "SOMEWHAT_HELPFUL" => 0.0,
                            "NOT_HELPFUL" => -1.0,
                            _ => panic!(),
                        }
                    },
                })
            },
        )?
        .filter_map(Result::ok)
        .collect())
}

fn process_item(item_id: &str, conn: &Connection) -> Result<(), Box<dyn Error>> {
    let context_size = CONTEXT_SIZE;
    let mut stmt_voters_on_note =
        conn.prepare("select raterParticipantId, createdAtMillis from ratings where noteId = ?1")?;
    // TODO: filter users/items with only one vote
    let mut stmt_votes_before =
            conn.prepare(if AGREE {
        "select noteId, agree - disagree from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 and agree + disagree != 0 order by createdAtMillis desc limit ?3;"
        } else {"select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 and helpfulnesslevel != '' order by createdAtMillis desc limit ?3;"})?;
    let mut stmt_votes_after =
            conn.prepare(if AGREE {"select noteId, agree - disagree from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 and agree + disagree != 0 order by createdAtMillis asc limit ?3;"} else {"select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 and helpfulnesslevel != '' order by createdAtMillis asc limit ?3;" })?;

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

        let votes = get_votes(
            user.id.as_str(),
            user.voted_at_millis,
            &mut stmt_votes_before,
        )?;
        // println!("User {} votes before: {}", user.id, votes.len());
        for vote in votes.iter() {
            if !item_ids.contains_key(&vote.item_id) {
                item_ids.insert(vote.item_id, item_ids.len());
            }
            votes_before.push(vote.clone());
        }

        let votes = get_votes(
            user.id.as_str(),
            user.voted_at_millis,
            &mut stmt_votes_after,
        )?;
        // println!("User {} votes after: {}", user.id, votes.len());
        for vote in votes.iter() {
            if !item_ids.contains_key(&vote.item_id) {
                item_ids.insert(vote.item_id, item_ids.len());
            }
            votes_after.push(vote.clone());
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

    if item_ids.len() < 2 || user_ids.len() < 2 || votes_before.is_empty() || votes_after.is_empty()
    {
        println!("skipping");
        return Ok(());
    }

    let observed_matrix = {
        // we treat every user as two different users. One before voting on the current item and
        // one after voting. The columns of the observed matrix are interleaved, so that every user
        // gets two columns.
        let mut observed_matrix: Array2<f64> =
            Array2::from_elem((item_ids.len(), user_ids.len() * 2), f64::NAN);

        for vote in &votes_before {
            let item_index = *item_ids.get(&vote.item_id).unwrap();
            let user_index = *user_ids.get(&vote.user_id).unwrap();
            observed_matrix[[item_index, user_index * 2]] = vote.value;
        }
        for vote in &votes_after {
            let item_index = *item_ids.get(&vote.item_id).unwrap();
            let user_index = *user_ids.get(&vote.user_id).unwrap();
            observed_matrix[[item_index, user_index * 2 + 1]] = vote.value;
        }

        observed_matrix
    };

    let top = 30.min(item_ids.len());
    println!("Observed matrix:");
    print_array(&observed_matrix.slice(s![..top, ..]).to_owned());

    let factorization = matrix_factorization_svd(
        &observed_matrix,
        Some(3),
        Some(initial_guess(&observed_matrix)),
    );
    let user_embeddings = factorization.vt.t();
    assert!(user_embeddings.shape()[0] == user_ids.len() * 2);
    let mental_model = factorization
        .u
        .dot(&Array2::from_diag(&factorization.s))
        .dot(&factorization.vt);

    println!("Mental model:");
    print_array(&mental_model.slice(s![..top, ..]).to_owned());

    // Create a reverse map for user_ids
    let reverse_user_ids: FxHashMap<usize, String> =
        user_ids.iter().map(|(k, v)| (*v, k.clone())).collect();

    // Prepare the SQL statement
    let mut stmt = conn.prepare(
        "INSERT OR REPLACE INTO user_change (noteid, userid, change) VALUES (?1, ?2, ?3)",
    )?;

    let mut change_magnitude_sum = 0.0;
    for user_index in 0..user_ids.len() {
        // interleaved users x K
        let embedding_before = user_embeddings.row(user_index * 2);
        let embedding_after = user_embeddings.row(user_index * 2 + 1);
        println!("user {} embedding before {}", user_index, embedding_before);
        println!("user {} embedding after {}", user_index, embedding_after);
        let change_vector = embedding_after.to_owned() - embedding_before.to_owned();
        println!("user {} change vector {}", user_index, change_vector);
        let change_magnitude = change_vector.norm_l2();
        change_magnitude_sum += change_magnitude;
        println!(
            "user {} change magnitude {}\n",
            user_index, change_magnitude
        );
        // TODO: save item with largest change
        let user_id = reverse_user_ids.get(&user_index).unwrap();
        stmt.execute(params![item_id, user_id, change_magnitude])?;
    }

    let change_magnitude_avg = change_magnitude_sum / user_ids.len() as f64;

    println!("average mind change: {}", change_magnitude_avg);
    conn.execute(
    "insert or replace into scores (noteId, change, users, items, before, after) values (?1, ?2, ?3, ?4, ?5, ?6)",
    (item_id, change_magnitude_avg, user_ids.len(), item_ids.len(), votes_before.len(), votes_after.len()),
)?;

    Ok(())
}

fn initial_guess(incomplete_matrix: &Array2<f64>) -> Array2<f64> {
    let (rows, _) = incomplete_matrix.dim();

    let row_averages: Vec<f64> = (0..rows)
        .map(|i| {
            let row = incomplete_matrix.row(i);
            let known_values: Vec<f64> = row.iter().filter(|&&x| !x.is_nan()).cloned().collect();
            known_values.iter().sum::<f64>() / known_values.len() as f64
        })
        .collect();

    let mut completed_matrix = incomplete_matrix.clone();
    for ((i, j), value) in completed_matrix.indexed_iter_mut() {
        if value.is_nan() {
            let other_j = if j % 2 == 0 { j + 1 } else { j - 1 };
            let other_value = incomplete_matrix[[i, other_j]];
            *value = if other_value.is_nan() {
                row_averages[i]
            } else {
                other_value
            };
        }
    }
    completed_matrix
}
