use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;

mod initial_guess;
mod matrix_completion_gd;
mod matrix_completion_svd;
mod matrix_factorization_als;
mod matrix_factorization_svd;
mod print_array;

use itertools::Itertools;
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

// TODO: after one item changed someone's mind, other items which the user voted on right ifter
// have a similar context signature. How do we figure out which item actually changed the mind, so that no change is detected for the other items. Can we look at the target items in chronological order and detect patterns?
const CONTEXT_SIZE: i32 = 5; // other votes from user before and after voting on target post
const AGREE: bool = true;

fn main() -> Result<(), Box<dyn Error>> {
    let conn = Connection::open("dataset/ratings.db")?;

    conn.execute("delete from scores", ())?;
    conn.execute("delete from user_change", ())?;
    //  where noteId = '1400247230330667008'
    //   where noteId = '1354864556552712194'
    // 10x133  where noteid = '1709553622751588430'
    // 39x544  where noteid = '1354855204005453826'
    let mut stmt_target_note_ids = conn.prepare("select distinct noteId from ratings limit 50")?;
    println!("counting distinct items...");
    let item_count: i64 = conn
        .prepare("select count(distinct noteId) from ratings")?
        .query_map(params![], |row| row.get::<_, i64>(0))?
        .next()
        .unwrap()?;
    println!("total distinct items: {}", item_count);
    let mut stmt_voters_on_note =
        conn.prepare("select raterParticipantId, createdAtMillis from ratings where noteId = ?1")?;
    // TODO: filter users/items with only one vote
    let mut stmt_votes_before =
                    conn.prepare(if AGREE {
                "select noteId, agree - disagree from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 and agree + disagree != 0 order by createdAtMillis desc limit ?3;"
                } else {"select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 and helpfulnesslevel != '' order by createdAtMillis desc limit ?3;"})?;
    let mut stmt_votes_after =
                    conn.prepare(if AGREE {"select noteId, agree - disagree from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 and agree + disagree != 0 order by createdAtMillis asc limit ?3;"} else {"select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 and helpfulnesslevel != '' order by createdAtMillis asc limit ?3;" })?;

    // matrix where users appear two times per item: before and after voting on that item.
    // That means we get a user-item matrix for votes, where every user is a virtual user.
    // The goal is to detect change of voting behavior in the same user-embedding space.
    // target | item 0          | item 1
    //        +-----------------+------------
    //  users |1a 1b 2a 2b 3a 3b| 1a 1b 2a 2b ...
    //        +-----------------+------------
    // item 0 |
    // item 1 |              <votes>
    // item 2 |
    //          ...
    #[derive(Eq, PartialEq, Hash, Clone)]
    struct VirtualUser {
        user_id: String,     // TODO: track userids idx
        target_item_id: i64, // TODO: track target_item_idx
        before: bool,
    }
    let mut votes_by_virtual_users: HashMap<(VirtualUser, i64), f64, rustc_hash::FxBuildHasher> =
        FxHashMap::default();

    // track item_id mappings for the context matrix
    let mut context_item_idx: HashMap<i64, usize, rustc_hash::FxBuildHasher> = FxHashMap::default();

    // track VirtualUser mappings for the context matrix
    let mut virtual_user_idx: HashMap<VirtualUser, usize, rustc_hash::FxBuildHasher> =
        FxHashMap::default();

    // track target_item_id mappings for the final change-matrix
    let mut target_item_idx: HashMap<i64, usize, rustc_hash::FxBuildHasher> = FxHashMap::default();

    // track target_item_id mappings for the final change-matrix
    let mut user_idx: HashMap<String, usize, rustc_hash::FxBuildHasher> = FxHashMap::default();

    for (i, item) in stmt_target_note_ids
        .query_map(params![], |row| {
            Ok(Item {
                id: row.get::<_, i64>(0)?,
            })
        })?
        .enumerate()
    {
        let target_item = item?;
        println!(
            "\n[Item {}/{} {:.4}%]",
            i,
            item_count,
            i as f64 / item_count as f64
        );
        let target_item_id = target_item.id;

        if !target_item_idx.contains_key(&target_item_id) {
            target_item_idx.insert(target_item_id, target_item_idx.len());
        }

        let mut context_votes_before: Vec<Vote> = vec![];
        let mut context_votes_after: Vec<Vote> = vec![];

        // for users and items in context, we only need the unique counts
        let mut user_ids_in_context: HashSet<String> = HashSet::new();
        let mut item_ids_in_context: HashSet<i64> = HashSet::new();

        // for every user who voted on the target item
        for user in stmt_voters_on_note.query_map(params![target_item_id], |row| {
            Ok(User {
                id: row.get(0)?,
                voted_at_millis: row.get(1)?,
            })
        })? {
            let user = user?;

            if !user_idx.contains_key(&user.id) {
                user_idx.insert(user.id.clone(), user_idx.len());
            }

            // votes on other items BEFORE voting on target item
            let votes_before = get_votes(
                user.id.as_str(),
                user.voted_at_millis,
                &mut stmt_votes_before,
            )?;

            // votes on other items AFTER voting on target item
            let votes_after = get_votes(
                user.id.as_str(),
                user.voted_at_millis,
                &mut stmt_votes_after,
            )?;

            // Only process this user if they have both before and after votes
            if !votes_before.is_empty() && !votes_after.is_empty() {
                context_votes_before.extend(votes_before.iter().cloned());
                context_votes_after.extend(votes_after.iter().cloned());

                for vote in votes_before.iter().chain(votes_after.iter()) {
                    item_ids_in_context.insert(vote.item_id);
                    user_ids_in_context.insert(vote.user_id.clone());
                }
            }
        }

        println!("target item id: {}", target_item_id);
        // TODO: print item text
        println!("context size {}", CONTEXT_SIZE);
        println!("Number of unique users: {}", user_ids_in_context.len());
        println!("Number of unique items: {}", item_ids_in_context.len());
        println!(
            "items/user: {}",
            item_ids_in_context.len() as f64 / user_ids_in_context.len() as f64
        );
        println!("votes before: {}", context_votes_before.len());
        println!("votes after:  {} ", context_votes_after.len());

        if item_ids_in_context.len() < 2
            || user_ids_in_context.len() < 2
            || context_votes_before.is_empty()
            || context_votes_after.is_empty()
        {
            println!("skipping");
            continue;
        }

        let mut track_context_vote = |before: bool, vote: &Vote| {
            let virtual_user = VirtualUser {
                user_id: vote.user_id.clone(),
                target_item_id,
                before,
            };

            if !virtual_user_idx.contains_key(&virtual_user) {
                virtual_user_idx.insert(virtual_user.clone(), virtual_user_idx.len());
            }
            if !context_item_idx.contains_key(&vote.item_id) {
                context_item_idx.insert(vote.item_id, context_item_idx.len());
            }
            votes_by_virtual_users.insert((virtual_user, vote.item_id), vote.value);
        };

        for context_vote_before in context_votes_before.iter() {
            track_context_vote(true, context_vote_before);
        }
        for context_vote_after in context_votes_after.iter() {
            track_context_vote(false, context_vote_after);
        }
    }

    println!("\nvirtual users: {}", virtual_user_idx.len());
    println!("context items: {}", context_item_idx.len());
    assert!(
        virtual_user_idx.len() % 2 == 0,
        "virtual users must always come in pairs"
    );

    let observed_context_matrix = {
        // we treat every user as two different users for every post we want to analyze.
        let mut matrix: Array2<f64> =
            Array2::from_elem((context_item_idx.len(), virtual_user_idx.len()), f64::NAN);

        for ((virtual_user, item_id), vote_value) in votes_by_virtual_users {
            let context_item_index = *context_item_idx.get(&item_id).unwrap();
            let virtual_user_index = *virtual_user_idx.get(&virtual_user).unwrap();
            matrix[[context_item_index, virtual_user_index]] = vote_value;
        }

        matrix
    };

    let top = 30.min(context_item_idx.len());
    println!("Observed matrix:");
    print_array(&observed_context_matrix.slice(s![..top, ..]).to_owned());

    // println!("guess:");
    // print_array(
    //     &initial_guess(&observed_context_matrix)
    //         .slice(s![..top, ..])
    //         .to_owned(),
    // );

    let factorization = matrix_factorization_svd(
        &observed_context_matrix,
        Some(5),
        Some(initial_guess(&observed_context_matrix)),
    );
    let completed_context_matrix = factorization
        .u
        .dot(&Array2::from_diag(&factorization.s))
        .dot(&factorization.vt);

    println!("Completed context matrix:");
    print_array(&completed_context_matrix.slice(s![..top, ..]).to_owned());

    let virtual_user_embeddings = factorization.vt.t();

    // TODO: search virtual user with the biggest change, and show the texts of the before- and
    // after votes, and the text of the target_item

    let change_matrix = {
        let mut matrix: Array2<f64> =
            Array2::from_elem((target_item_idx.len(), user_idx.len()), f64::NAN);

        for ((virtual_user_before, &virtual_user_index_before), (_, &virtual_user_index_after)) in
            virtual_user_idx.iter().tuple_windows()
        {
            let embedding_before = virtual_user_embeddings.row(virtual_user_index_before);
            let embedding_after = virtual_user_embeddings.row(virtual_user_index_after);
            let change_vector = embedding_after.to_owned() - embedding_before.to_owned();
            let change_magnitude = change_vector.norm_l2();
            let target_item_index = target_item_idx[&virtual_user_before.target_item_id];
            let user_index = user_idx[&virtual_user_before.user_id];
            matrix[[target_item_index, user_index]] = change_magnitude;
        }
        matrix
    };

    println!("Change matrix:");
    print_array(&change_matrix.to_owned());

    let change_matrix_factorization = matrix_factorization_svd(&change_matrix, Some(3), None);
    let change_matrix_completion = change_matrix_factorization
        .u
        .dot(&Array2::from_diag(&change_matrix_factorization.s))
        .dot(&change_matrix_factorization.vt);

    println!("change matrix completion:");
    print_array(&change_matrix_completion);

    // Create a reverse map for user_ids
    let reverse_user_ids: FxHashMap<usize, VirtualUser> = virtual_user_idx
        .iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect();

    // TODO: print item texts which changed users minds on the target item
    // Prepare the SQL statement
    let mut stmt = conn.prepare(
        "INSERT OR REPLACE INTO user_change (noteid, userid, change) VALUES (?1, ?2, ?3)",
    )?;

    //     // TODO: save item with largest change
    //     let user_id = reverse_user_ids.get(&virtual_user_index).unwrap();
    //     stmt.execute(params![item_id, user_id, change_magnitude])?;
    // }
    //
    // let change_magnitude_avg = change_magnitude_sum / user_ids.len() as f64;
    //
    // println!("average mind change: {}", change_magnitude_avg);
    // conn.execute(
    //         "insert or replace into scores (noteId, change, users, items, before, after) values (?1, ?2, ?3, ?4, ?5, ?6)",
    //         (item_id, change_magnitude_avg, user_ids.len(), target_item_idx.len(), votes_before.len(), votes_after.len()),
    //     )?;

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

fn initial_guess(incomplete_matrix: &Array2<f64>) -> Array2<f64> {
    let (rows, _) = incomplete_matrix.dim();
    let mut completed_matrix = incomplete_matrix.clone();

    let row_averages: Vec<f64> = (0..rows)
        .map(|i| {
            let row = incomplete_matrix.row(i);
            let known_values: Vec<f64> = row.iter().filter(|&&x| !x.is_nan()).cloned().collect();
            known_values.iter().sum::<f64>() / known_values.len() as f64
        })
        .collect();

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
