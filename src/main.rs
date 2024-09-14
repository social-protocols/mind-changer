use std::error::Error;

mod initial_guess;
mod matrix_completion_svd;
mod print_array;

use crate::matrix_completion_svd::matrix_completion_svd;
use ndarray::{s, Array2, Axis, Zip};
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

const CONTEXT_SIZE: i32 = 8; // other votes from user before and after voting on target post
const COMPLETION_ENERGY_THRESHOLD: f64 = 0.99; // for adaptive rank selection
const COMPLETION_TOLERANCE: f64 = 0.0001;
const COMPLETION_MAX_ITERATIONS: usize = 2000;

fn main() -> Result<(), Box<dyn Error>> {
    let conn = Connection::open("dataset/ratings.db")?;

    conn.execute("delete from scores", ())?;
    conn.execute("delete from user_change", ())?;
    //  where noteId = '1400247230330667008'
    //   where noteId = '1354864556552712194'
    let mut stmt_note_ids =
        conn.prepare("select distinct noteId from ratings where noteId = '1354864556552712194'")?;
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
                    value: match row.get::<_, String>(1)?.as_str() {
                        "HELPFUL" => 1.0,
                        "SOMEWHAT_HELPFUL" => 0.0,
                        "NOT_HELPFUL" => -1.0,
                        _ => panic!(),
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
            conn.prepare("select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 and helpfulnesslevel != '' order by createdAtMillis desc limit ?3;")?;
    let mut stmt_votes_after =
            conn.prepare("select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 and helpfulnesslevel != '' order by createdAtMillis asc limit ?3;")?;

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
        println!("User {} votes before: {}", user.id, votes.len());
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
        println!("User {} votes after: {}", user.id, votes.len());
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

    let observed_matrix_before = compute_observed_matrix(&votes_before, &item_ids, &user_ids);
    let observed_matrix_after = compute_observed_matrix(&votes_after, &item_ids, &user_ids);

    let complete_matrix = |observed_matrix, initial_guess| {
        matrix_completion_svd(
            observed_matrix,
            COMPLETION_ENERGY_THRESHOLD,
            COMPLETION_TOLERANCE,
            COMPLETION_MAX_ITERATIONS,
            initial_guess,
        )
    };

    // just for initializing the guess.
    let mental_model_after_tmp = complete_matrix(observed_matrix_after.clone(), None);
    let mental_model_before_tmp = complete_matrix(
        observed_matrix_before.clone(),
        Some(mental_model_after_tmp.clone()),
    );

    let mental_model_after = complete_matrix(
        observed_matrix_after.clone(),
        Some(mental_model_before_tmp.clone()),
    );

    let mental_model_before = complete_matrix(
        observed_matrix_before.clone(),
        Some(mental_model_after.clone()),
    );

    let mental_model_after2 = complete_matrix(
        observed_matrix_after.clone(),
        Some(mental_model_before.clone()),
    );

    println!(
        "rmse before vs tmp: {}",
        root_mean_square_error(&mental_model_before, &mental_model_before_tmp)
    );
    println!(
        "rmse after  vs tmp: {}",
        root_mean_square_error(&mental_model_after, &mental_model_after_tmp)
    );
    println!(
        "rmse after2 vs tmp: {}",
        root_mean_square_error(&mental_model_after2, &mental_model_after_tmp)
    );

    let top = 5.min(item_ids.len());
    println!("Observed matrix before voting:");
    print_array(&observed_matrix_before.slice(s![..top, ..]).to_owned());

    println!("Observed matrix after voting:");
    print_array(&observed_matrix_after.slice(s![..top, ..]).to_owned());

    println!("Mental model before:");
    print_array(&mental_model_before.slice(s![..top, ..]).to_owned());
    println!("Mental model after:");
    print_array(&mental_model_after.slice(s![..top, ..]).to_owned());
    println!("Mental model diff:");
    let diff_matrix = mental_model_after.clone() - mental_model_before.clone();
    print_array(&diff_matrix.slice(s![..top, ..]).to_owned());

    // Calculate the sum of the absolute values of each column (user) in the difference matrix
    let column_sums = diff_matrix.mapv(|x| x.abs()).sum_axis(Axis(0));

    // Calculate the mean of the column sums per user
    let mean_column_sum_per_user = column_sums / diff_matrix.nrows() as f64;

    println!("Mean of column sums per user in the difference matrix:");
    print_array(
        &mean_column_sum_per_user
            .clone()
            .into_shape((1, user_ids.len()))
            .unwrap(),
    );

    // Create a reverse map for user_ids
    let reverse_user_ids: FxHashMap<usize, String> =
        user_ids.iter().map(|(k, v)| (*v, k.clone())).collect();

    // Prepare the SQL statement
    let mut stmt = conn.prepare(
        "INSERT OR REPLACE INTO user_change (noteid, userid, change) VALUES (?1, ?2, ?3)",
    )?;

    // Loop over mean_column_sum_per_user and insert into database
    for (index, &change) in mean_column_sum_per_user.iter().enumerate() {
        let user_id = reverse_user_ids.get(&index).unwrap();
        stmt.execute(params![item_id, user_id, change])?;
    }

    let rmse = root_mean_square_error(&mental_model_before, &mental_model_after); // range is from 0 to 2

    let mind_change = rmse / 2.0;
    println!("average mind change: {}", mind_change);
    conn.execute(
        "insert or replace into scores (noteId, change, users, items, before, after) values (?1, ?2, ?3, ?4, ?5, ?6)",
        (item_id, mind_change, user_ids.len(), item_ids.len(), votes_before.len(), votes_after.len()),
    )?;
    Ok(())
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
